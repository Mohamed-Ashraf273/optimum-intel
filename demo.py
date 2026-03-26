import argparse
import contextlib
import inspect
import types
from importlib.resources import files
from pathlib import Path

import torch
import whowhatbench
import yaml
from optimum.exporters.openvino import model_patcher as ov_model_patcher
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)


DEFAULT_MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
DEFAULT_MODEL_DIR = "./output_dir"
F32_CONFIG = {"INFERENCE_PRECISION_HINT": "f32"}
SHORT_PROMPTS = [
    "Who is the most famous programmer?",
    "Who is Leo Tolstoy?",
    "Explain what artificial intelligence is.",
    "What is deep learning?",
]
UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK = None
UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = None


def _load_default_prompts():
    prompt_path = files("whowhatbench.prompts").joinpath("text_prompts.yaml")
    prompt_data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    return prompt_data["en"]["prompts"]


def _resolve_prompts(prompt_set: str):
    if prompt_set == "default":
        return _load_default_prompts()
    return SHORT_PROMPTS


def _upstream_deepseek_v3_debug_trace(module, stage: str, **tensors):
    if UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK is None:
        return
    UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK(module=module, stage=stage, tensors=tensors)


def _upstream_deepseek_v3_moe_debug_trace(module, stage: str, **tensors):
    if UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK is None:
        return
    UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK(module=module, stage=stage, tensors=tensors)


def debug_deepseek_v3_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask=None,
    past_key_values=None,
    cache_position=None,
    **kwargs,
):
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    _upstream_deepseek_v3_debug_trace(
        self,
        "attn_proj",
        hidden_states=hidden_states,
        q=q_states,
        q_nope=q_pass,
        q_pe_pre_rope=q_rot,
        compressed_kv=compressed_kv,
        k_pass=k_pass,
        k_rot_pre_rope=k_rot,
        value_states=value_states,
    )

    cos, sin = position_embeddings
    if self.config.rope_interleave:
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    _upstream_deepseek_v3_debug_trace(
        self,
        "attn_rope",
        cos=cos,
        sin=sin,
        q_pe=q_rot,
        k_rot=k_rot,
    )
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)
    _upstream_deepseek_v3_debug_trace(
        self,
        "attn_inputs",
        q_nope=q_pass,
        q_pe=q_rot,
        k_pass=k_pass,
        k_rot=k_rot,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _upstream_deepseek_v3_debug_trace(
            self,
            "after_cache",
            key_states=key_states,
            value_states=value_states,
        )

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        value_states = torch.nn.functional.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )
    _upstream_deepseek_v3_debug_trace(
        self,
        "attn_sdpa_inputs",
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        attention_mask=attention_mask,
    )

    if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    _upstream_deepseek_v3_debug_trace(self, "attn_output_pre_transpose", attn_output=attn_output)
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    _upstream_deepseek_v3_debug_trace(self, "attn_output_pre_proj", attn_output=attn_output)
    attn_output = self.o_proj(attn_output)
    _upstream_deepseek_v3_debug_trace(self, "attn_output_post_proj", attn_output=attn_output)
    return attn_output, attn_weights


def debug_deepseek_v3_moe_forward(self, hidden_states):
    residuals = hidden_states
    orig_shape = hidden_states.shape
    topk_indices, topk_weights = self.gate(hidden_states)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    moe_output = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    shared_output = self.shared_experts(residuals)
    output = moe_output + shared_output
    _upstream_deepseek_v3_moe_debug_trace(
        self,
        "moe",
        topk_indices=topk_indices,
        topk_weights=topk_weights,
        moe_output=moe_output,
        shared_output=shared_output,
        output=output,
    )
    return output


def _select_debug_slice(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 4:
        return tensor[:, :, -1, :]
    if tensor.dim() == 3:
        return tensor[:, -1, :]
    return tensor


class AttentionTraceRecorder:
    def __init__(self):
        self.data = {}

    def __call__(self, module, stage: str, tensors):
        layer_idx = getattr(module, "layer_idx", -1)
        stage_store = self.data.setdefault(layer_idx, {}).setdefault(stage, {})
        for name, tensor in tensors.items():
            if tensor is None or not torch.is_tensor(tensor):
                continue
            stage_store[name] = {
                "shape": tuple(tensor.shape),
                "slice": _select_debug_slice(tensor),
            }


@contextlib.contextmanager
def _use_upstream_attention_debug(model, recorder: AttentionTraceRecorder):
    global UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK

    old_hook = UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK
    patched = []
    UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK = recorder
    try:
        for block in model.model.layers:
            patched.append((block.self_attn, block.self_attn.forward))
            block.self_attn.forward = types.MethodType(debug_deepseek_v3_attention_forward, block.self_attn)
        yield
    finally:
        for module, original_forward in patched:
            module.forward = original_forward
        UPSTREAM_DEEPSEEK_V3_DEBUG_HOOK = old_hook


@contextlib.contextmanager
def _use_upstream_moe_debug(model, recorder: AttentionTraceRecorder):
    global UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK

    old_hook = UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK
    patched = []
    UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = recorder
    try:
        for idx, block in enumerate(model.model.layers):
            if hasattr(block, "mlp") and hasattr(block.mlp, "moe") and hasattr(block.mlp, "shared_experts"):
                patched.append((block.mlp, block.mlp.forward))
                block.mlp.layer_idx = idx
                block.mlp.forward = types.MethodType(debug_deepseek_v3_moe_forward, block.mlp)
        yield
    finally:
        for module, original_forward in patched:
            module.forward = original_forward
        UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = old_hook


@contextlib.contextmanager
def _use_export_attention_debug(model, recorder: AttentionTraceRecorder):
    old_hook = ov_model_patcher.DEEPSEEK_V3_DEBUG_HOOK
    patched = []
    ov_model_patcher.DEEPSEEK_V3_DEBUG_HOOK = recorder
    try:
        ov_model_patcher.patch_cos_sin_cached_fp32(model)
        if hasattr(model, "model"):
            ov_model_patcher.patch_cos_sin_cached_fp32(model.model)
            for block in model.model.layers:
                patched.append((block.self_attn, block.self_attn.forward))
                block.self_attn.forward = types.MethodType(ov_model_patcher.deepseek_v3_attn_forward, block.self_attn)
        yield
    finally:
        for module, original_forward in patched:
            module.forward = original_forward
        ov_model_patcher.DEEPSEEK_V3_DEBUG_HOOK = old_hook


@contextlib.contextmanager
def _use_export_moe_debug(model, recorder: AttentionTraceRecorder):
    global UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK

    old_hook = UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK
    old_export_hook = getattr(ov_model_patcher, "DEEPSEEK_MOE_DEBUG_HOOK", None)
    patched_forward = []
    patched_moe = []
    UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = recorder
    ov_model_patcher.DEEPSEEK_MOE_DEBUG_HOOK = recorder
    try:
        if hasattr(model, "model"):
            for idx, block in enumerate(model.model.layers):
                if hasattr(block, "mlp") and hasattr(block.mlp, "moe") and hasattr(block.mlp, "experts"):
                    mlp = block.mlp
                    patched_forward.append((mlp, mlp.forward))
                    patched_moe.append((mlp, mlp.moe, [], []))
                    num_experts = len(mlp.experts)
                    gate_projs = torch.concat(
                        tuple(mlp.experts[i].gate_proj.weight.unsqueeze(0) for i in range(num_experts)),
                        dim=0,
                    )
                    up_projs = torch.concat(
                        tuple(mlp.experts[i].up_proj.weight.unsqueeze(0) for i in range(num_experts)),
                        dim=0,
                    )
                    down_projs = torch.concat(
                        tuple(mlp.experts[i].down_proj.weight.unsqueeze(0) for i in range(num_experts)),
                        dim=0,
                    )
                    if ov_model_patcher.is_openvino_version("<", "2026.1.0"):
                        mlp.gate_projs = gate_projs.float()
                        mlp.up_projs = up_projs.float()
                        mlp.down_projs = down_projs.float()
                    else:
                        mlp.gate_projs = gate_projs
                        mlp.up_projs = up_projs
                        mlp.down_projs = down_projs
                    mlp.layer_idx = idx
                    mlp.moe = types.MethodType(ov_model_patcher.deepseek_moe, mlp)
                    mlp.forward = types.MethodType(debug_deepseek_v3_moe_forward, mlp)
        yield
    finally:
        for mlp, original_forward in patched_forward:
            mlp.forward = original_forward
        for mlp, original_moe, _, _ in patched_moe:
            mlp.moe = original_moe
            for attr in ("gate_projs", "up_projs", "down_projs"):
                if hasattr(mlp, attr):
                    delattr(mlp, attr)
        UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = old_hook
        ov_model_patcher.DEEPSEEK_MOE_DEBUG_HOOK = old_export_hook


def _forward_logits(model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    inputs = {
        "input_ids": input_ids.to(getattr(model, "device", "cpu")),
        "attention_mask": attention_mask.to(getattr(model, "device", "cpu")),
    }
    with torch.no_grad():
        return model(**inputs).logits[0, -1].float().cpu()


def _build_teacher_forced_prefix(model, tokenizer, prompt: str, use_chat_template: bool, steps: int):
    shared_inputs = _prepare_inputs_cpu(tokenizer, prompt, use_chat_template)
    if "token_type_ids" in shared_inputs:
        shared_inputs.pop("token_type_ids")

    input_ids = shared_inputs["input_ids"].cpu()
    attention_mask = shared_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.cpu()

    for _ in range(steps):
        next_token = int(_forward_logits(model, input_ids, attention_mask).argmax().item())
        input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=input_ids.dtype)], dim=-1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)],
            dim=-1,
        )

    return input_ids, attention_mask


def _print_topk(title: str, logits: torch.Tensor, tokenizer, top_k: int):
    print(title)
    topk = torch.topk(logits, k=top_k)
    for token_id, score in zip(topk.indices.tolist(), topk.values.tolist()):
        print(f"  {token_id}: {repr(tokenizer.decode([token_id]))} -> {score:.6f}")


def _tensor_diff(left: torch.Tensor, right: torch.Tensor):
    diff = (left - right).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _capture_layer_mlp_input(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, layer_idx: int) -> torch.Tensor:
    captured = {}

    def hook(_module, args):
        captured["hidden_states"] = args[0].detach().clone()

    handle = model.model.layers[layer_idx].mlp.register_forward_pre_hook(hook)
    try:
        _forward_logits(model, input_ids, attention_mask)
    finally:
        handle.remove()

    if "hidden_states" not in captured:
        raise RuntimeError(f"Failed to capture MLP input for layer {layer_idx}.")
    return captured["hidden_states"]


def _replay_upstream_moe_layer(model, hidden_states: torch.Tensor, layer_idx: int, recorder: AttentionTraceRecorder):
    global UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK

    mlp = model.model.layers[layer_idx].mlp
    old_hook = UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK
    old_forward = mlp.forward
    old_layer_idx = getattr(mlp, "layer_idx", None)
    UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = recorder
    try:
        mlp.layer_idx = layer_idx
        mlp.forward = types.MethodType(debug_deepseek_v3_moe_forward, mlp)
        with torch.no_grad():
            return mlp(hidden_states.to(next(mlp.parameters()).device))
    finally:
        mlp.forward = old_forward
        if old_layer_idx is None:
            delattr(mlp, "layer_idx")
        else:
            mlp.layer_idx = old_layer_idx
        UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = old_hook


def _replay_export_moe_layer(model, hidden_states: torch.Tensor, layer_idx: int, recorder: AttentionTraceRecorder):
    global UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK

    mlp = model.model.layers[layer_idx].mlp
    old_hook = UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK
    old_export_hook = getattr(ov_model_patcher, "DEEPSEEK_MOE_DEBUG_HOOK", None)
    old_forward = mlp.forward
    old_moe = mlp.moe
    old_layer_idx = getattr(mlp, "layer_idx", None)
    added_attrs = []
    UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = recorder
    ov_model_patcher.DEEPSEEK_MOE_DEBUG_HOOK = recorder
    try:
        num_experts = len(mlp.experts)
        gate_projs = torch.concat(tuple(mlp.experts[i].gate_proj.weight.unsqueeze(0) for i in range(num_experts)), dim=0)
        up_projs = torch.concat(tuple(mlp.experts[i].up_proj.weight.unsqueeze(0) for i in range(num_experts)), dim=0)
        down_projs = torch.concat(tuple(mlp.experts[i].down_proj.weight.unsqueeze(0) for i in range(num_experts)), dim=0)
        mlp.gate_projs = gate_projs
        mlp.up_projs = up_projs
        mlp.down_projs = down_projs
        added_attrs.extend(["gate_projs", "up_projs", "down_projs"])
        mlp.layer_idx = layer_idx
        mlp.moe = types.MethodType(ov_model_patcher.deepseek_moe, mlp)
        mlp.forward = types.MethodType(debug_deepseek_v3_moe_forward, mlp)
        with torch.no_grad():
            return mlp(hidden_states.to(next(mlp.parameters()).device))
    finally:
        mlp.forward = old_forward
        mlp.moe = old_moe
        if old_layer_idx is None:
            delattr(mlp, "layer_idx")
        else:
            mlp.layer_idx = old_layer_idx
        for attr in added_attrs:
            if hasattr(mlp, attr):
                delattr(mlp, attr)
        UPSTREAM_DEEPSEEK_V3_MOE_DEBUG_HOOK = old_hook
        ov_model_patcher.DEEPSEEK_MOE_DEBUG_HOOK = old_export_hook


def compare_export_attention_paths(
    model_id: str,
    prompt: str,
    use_chat_template: bool,
    steps: int,
    top_k: int,
    layer_idx: int | None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    _model_snapshot("baseline", model, tokenizer, steps)

    input_ids, attention_mask = _build_teacher_forced_prefix(model, tokenizer, prompt, use_chat_template, steps)

    upstream_recorder = AttentionTraceRecorder()
    with _use_upstream_attention_debug(model, upstream_recorder):
        upstream_logits = _forward_logits(model, input_ids, attention_mask)

    patched_recorder = AttentionTraceRecorder()
    with _use_export_attention_debug(model, patched_recorder):
        patched_logits = _forward_logits(model, input_ids, attention_mask)

    print("=========================")
    print("Upstream vs export-patched PyTorch attention")
    print("Prompt:", prompt)
    print(f"Teacher-forced steps before compare: {steps}")
    print("argmax_upstream:", int(upstream_logits.argmax().item()))
    print("argmax_patched:", int(patched_logits.argmax().item()))
    print("logits_max_abs_diff:", float((upstream_logits - patched_logits).abs().max().item()))
    print("logits_mean_abs_diff:", float((upstream_logits - patched_logits).abs().mean().item()))
    print()
    _print_topk(f"Top-{top_k} upstream:", upstream_logits, tokenizer, top_k)
    print()
    _print_topk(f"Top-{top_k} export-patched:", patched_logits, tokenizer, top_k)
    print()

    layer_diffs = []
    for current_layer in sorted(set(upstream_recorder.data) & set(patched_recorder.data)):
        upstream_stage = upstream_recorder.data[current_layer].get("attn_output_post_proj", {})
        patched_stage = patched_recorder.data[current_layer].get("attn_output_post_proj", {})
        upstream_tensor = upstream_stage.get("attn_output", {}).get("slice")
        patched_tensor = patched_stage.get("attn_output", {}).get("slice")
        if upstream_tensor is None or patched_tensor is None:
            continue
        max_abs, mean_abs = _tensor_diff(upstream_tensor, patched_tensor)
        layer_diffs.append((current_layer, max_abs, mean_abs))

    print("Per-layer attn_output_post_proj diff:")
    for current_layer, max_abs, mean_abs in layer_diffs:
        print(f"  layer={current_layer} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    if layer_idx is None:
        layer_idx = next((current_layer for current_layer, max_abs, _ in layer_diffs if max_abs > 0), None)
        if layer_idx is None and layer_diffs:
            layer_idx = max(layer_diffs, key=lambda item: item[1])[0]

    if layer_idx is None:
        print("No attention-layer diff was captured.")
        return

    print()
    print(f"Detailed tensor diffs for layer {layer_idx}:")
    upstream_layer = upstream_recorder.data.get(layer_idx, {})
    patched_layer = patched_recorder.data.get(layer_idx, {})
    for stage in (
        "attn_proj",
        "attn_rope",
        "attn_inputs",
        "after_cache",
        "attn_sdpa_inputs",
        "attn_output_pre_transpose",
        "attn_output_pre_proj",
        "attn_output_post_proj",
    ):
        upstream_stage = upstream_layer.get(stage, {})
        patched_stage = patched_layer.get(stage, {})
        keys = sorted(set(upstream_stage) & set(patched_stage))
        if not keys:
            continue
        print(f"  {stage}:")
        for key in keys:
            max_abs, mean_abs = _tensor_diff(upstream_stage[key]["slice"], patched_stage[key]["slice"])
            print(
                f"    {key}: shape_upstream={upstream_stage[key]['shape']} "
                f"shape_patched={patched_stage[key]['shape']} "
                f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
            )


def compare_export_moe_paths(
    model_id: str,
    prompt: str,
    use_chat_template: bool,
    steps: int,
    top_k: int,
    layer_idx: int | None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    _model_snapshot("baseline", model, tokenizer, steps)

    input_ids, attention_mask = _build_teacher_forced_prefix(model, tokenizer, prompt, use_chat_template, steps)

    upstream_recorder = AttentionTraceRecorder()
    with _use_upstream_moe_debug(model, upstream_recorder):
        upstream_logits = _forward_logits(model, input_ids, attention_mask)

    patched_recorder = AttentionTraceRecorder()
    with _use_export_moe_debug(model, patched_recorder):
        patched_logits = _forward_logits(model, input_ids, attention_mask)

    print("=========================")
    print("Upstream vs export-patched PyTorch MoE")
    print("Prompt:", prompt)
    print(f"Teacher-forced steps before compare: {steps}")
    print("argmax_upstream:", int(upstream_logits.argmax().item()))
    print("argmax_patched:", int(patched_logits.argmax().item()))
    print("logits_max_abs_diff:", float((upstream_logits - patched_logits).abs().max().item()))
    print("logits_mean_abs_diff:", float((upstream_logits - patched_logits).abs().mean().item()))
    print()
    _print_topk(f"Top-{top_k} upstream:", upstream_logits, tokenizer, top_k)
    print()
    _print_topk(f"Top-{top_k} export-patched:", patched_logits, tokenizer, top_k)
    print()

    layer_diffs = []
    for current_layer in sorted(set(upstream_recorder.data) & set(patched_recorder.data)):
        upstream_stage = upstream_recorder.data[current_layer].get("moe", {})
        patched_stage = patched_recorder.data[current_layer].get("moe", {})
        upstream_tensor = upstream_stage.get("output", {}).get("slice")
        patched_tensor = patched_stage.get("output", {}).get("slice")
        if upstream_tensor is None or patched_tensor is None:
            continue
        max_abs, mean_abs = _tensor_diff(upstream_tensor, patched_tensor)
        layer_diffs.append((current_layer, max_abs, mean_abs))

    print("Per-layer MoE output diff:")
    for current_layer, max_abs, mean_abs in layer_diffs:
        print(f"  layer={current_layer} max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    if layer_idx is None:
        layer_idx = next((current_layer for current_layer, max_abs, _ in layer_diffs if max_abs > 0), None)
        if layer_idx is None and layer_diffs:
            layer_idx = max(layer_diffs, key=lambda item: item[1])[0]

    if layer_idx is None:
        print("No MoE-layer diff was captured.")
        return

    print()
    print(f"Detailed MoE diffs for layer {layer_idx}:")
    upstream_stage = upstream_recorder.data.get(layer_idx, {}).get("moe", {})
    patched_stage = patched_recorder.data.get(layer_idx, {}).get("moe", {})
    for key in ("topk_indices", "topk_weights", "moe_output", "shared_output", "output"):
        if key not in upstream_stage or key not in patched_stage:
            continue
        max_abs, mean_abs = _tensor_diff(upstream_stage[key]["slice"], patched_stage[key]["slice"])
        print(
            f"  {key}: shape_upstream={upstream_stage[key]['shape']} "
            f"shape_patched={patched_stage[key]['shape']} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )

    patched_inner = patched_recorder.data.get(layer_idx, {}).get("moe_inner", {})
    if patched_inner:
        print("  export-patched inner tensors:")
        for key in ("hidden_states", "topk_indices", "topk_weights", "routing", "gate", "up", "gate_up", "next_states_pre_routing"):
            if key not in patched_inner:
                continue
            print(f"    {key}: shape={patched_inner[key]['shape']}")

    patched_inner_post = patched_recorder.data.get(layer_idx, {}).get("moe_inner_post_routing", {})
    if "next_states" in patched_inner_post:
        print(f"  export-patched post-routing next_states: shape={patched_inner_post['next_states']['shape']}")

    patched_inner_output = patched_recorder.data.get(layer_idx, {}).get("moe_inner_output", {})
    if "next_states" in patched_inner_output:
        print(f"  export-patched reduced next_states: shape={patched_inner_output['next_states']['shape']}")

    print()
    print(f"Single-layer replay on shared hidden_states for layer {layer_idx}:")
    shared_hidden_states = _capture_layer_mlp_input(model, input_ids, attention_mask, layer_idx)
    shared_upstream_recorder = AttentionTraceRecorder()
    shared_patched_recorder = AttentionTraceRecorder()
    shared_upstream_output = _replay_upstream_moe_layer(model, shared_hidden_states, layer_idx, shared_upstream_recorder)
    shared_patched_output = _replay_export_moe_layer(model, shared_hidden_states, layer_idx, shared_patched_recorder)
    max_abs, mean_abs = _tensor_diff(
        shared_upstream_output.detach().float().cpu(),
        shared_patched_output.detach().float().cpu(),
    )
    print(f"  output max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    shared_upstream_stage = shared_upstream_recorder.data.get(layer_idx, {}).get("moe", {})
    shared_patched_stage = shared_patched_recorder.data.get(layer_idx, {}).get("moe", {})
    for key in ("topk_indices", "topk_weights", "moe_output", "shared_output", "output"):
        if key not in shared_upstream_stage or key not in shared_patched_stage:
            continue
        max_abs, mean_abs = _tensor_diff(shared_upstream_stage[key]["slice"], shared_patched_stage[key]["slice"])
        print(f"  shared-input {key}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

    shared_patched_inner = shared_patched_recorder.data.get(layer_idx, {}).get("moe_inner", {})
    if shared_patched_inner:
        print("  shared-input export-patched inner tensors:")
        for key in ("routing", "gate", "up", "gate_up", "next_states_pre_routing"):
            if key in shared_patched_inner:
                print(f"    {key}: shape={shared_patched_inner[key]['shape']}")


def export_model(model_id: str, model_dir: str, stateful: bool, fp32: bool) -> None:
    model_dir_path = Path(model_dir).expanduser().resolve()
    model_dir_path.mkdir(parents=True, exist_ok=True)

    export_kwargs = {
        "export": True,
        "use_cache": True,
        "load_in_8bit": False,
        "quantization_config": None,
        "stateful": stateful,
        "ov_config": F32_CONFIG if fp32 else None,
    }
    if fp32:
        export_kwargs["torch_dtype"] = torch.float32

    optimized_model = OVModelForCausalLM.from_pretrained(model_id, **export_kwargs)

    optimized_model.save_pretrained(str(model_dir_path))
    print(
        f"Saved OpenVINO model to {model_dir_path} "
        f"(stateful={optimized_model.stateful}, use_cache={optimized_model.use_cache}, fp32={fp32})"
    )


def _prepare_inputs(tokenizer, prompt: str, device: str, use_chat_template: bool):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
    return inputs.to(device)


def _prepare_inputs_cpu(tokenizer, prompt: str, use_chat_template: bool):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    return tokenizer(prompt, return_tensors="pt")


def _prepare_model_inputs(model, tokenizer, prompt: str, use_chat_template: bool):
    device = getattr(model, "device", "cpu")
    inputs = _prepare_inputs(tokenizer, prompt, device, use_chat_template)
    if "token_type_ids" in inputs and "token_type_ids" not in inspect.signature(model.forward).parameters:
        inputs.pop("token_type_ids")
    return inputs


def _build_generation_kwargs(model, tokenizer, max_new_tokens: int):
    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "use_model_defaults": False,
    }

    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id

    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        bos_token_id = getattr(getattr(model, "generation_config", None), "bos_token_id", None)
    if bos_token_id is not None:
        generation_kwargs["bos_token_id"] = bos_token_id

    return generation_kwargs


def _model_snapshot(name: str, model, tokenizer, max_new_tokens: int):
    generation_config = getattr(model, "generation_config", None)
    interesting_generation_fields = [
        "do_sample",
        "num_beams",
        "temperature",
        "top_p",
        "top_k",
        "typical_p",
        "repetition_penalty",
        "max_length",
        "max_new_tokens",
        "min_new_tokens",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "use_cache",
        "cache_implementation",
    ]
    generation_values = {}
    if generation_config is not None:
        for field in interesting_generation_fields:
            generation_values[field] = getattr(generation_config, field, None)

    config_values = {
        "model_type": getattr(model.config, "model_type", None),
        "torch_dtype": str(getattr(model.config, "torch_dtype", None)),
        "config_use_cache": getattr(model.config, "use_cache", None),
        "config_bos_token_id": getattr(model.config, "bos_token_id", None),
        "config_eos_token_id": getattr(model.config, "eos_token_id", None),
        "tokenizer_bos_token_id": tokenizer.bos_token_id,
        "tokenizer_eos_token_id": tokenizer.eos_token_id,
        "tokenizer_pad_token_id": tokenizer.pad_token_id,
        "stateful": getattr(model, "stateful", None),
        "model_use_cache": getattr(model, "use_cache", None),
    }

    print(f"--- {name} config snapshot ---")
    for key, value in config_values.items():
        print(f"{key}: {value}")
    print("generation_config:")
    for key, value in generation_values.items():
        print(f"  {key}: {value}")
    print("effective_generate_kwargs:")
    for key, value in _build_generation_kwargs(model, tokenizer, max_new_tokens).items():
        print(f"  {key}: {value}")
    if generation_config is not None and hasattr(generation_config, "to_diff_dict"):
        print("generation_config_diff:")
        for key, value in sorted(generation_config.to_diff_dict().items()):
            print(f"  {key}: {value}")
    print()


def _generate_answer(
    model,
    tokenizer,
    prompt,
    max_new_tokens,
    crop_question,
    use_chat_template=False,
    empty_adapters=False,
    num_assistant_tokens=0,
    assistant_confidence_threshold=0.0,
):
    del empty_adapters, num_assistant_tokens, assistant_confidence_threshold

    inputs = _prepare_model_inputs(model, tokenizer, prompt, use_chat_template)
    generation_kwargs = _build_generation_kwargs(model, tokenizer, max_new_tokens)
    tokens = model.generate(**inputs, **generation_kwargs)
    if crop_question:
        tokens = tokens[:, inputs["input_ids"].shape[-1] :]
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def _generate_tokens(model, tokenizer, prompt: str, max_new_tokens: int, use_chat_template: bool):
    inputs = _prepare_model_inputs(model, tokenizer, prompt, use_chat_template)
    generation_kwargs = _build_generation_kwargs(model, tokenizer, max_new_tokens)
    tokens = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[-1]
    return tokens[0, prompt_len:].cpu()


def compare_prompt_logits(
    model_id: str,
    model_dir: str,
    prompt: str,
    use_chat_template: bool,
    top_k: int,
) -> None:
    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model directory was not found: {model_dir_path}. "
            "Run `python demo.py export --model-dir ...` first or pass the correct exported model path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    optimized_model = OVModelForCausalLM.from_pretrained(
        str(model_dir_path),
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
        ov_config=F32_CONFIG,
    )
    _model_snapshot("baseline", base_model, tokenizer, 1)
    _model_snapshot("openvino", optimized_model, tokenizer, 1)

    base_inputs = _prepare_model_inputs(base_model, tokenizer, prompt, use_chat_template)
    ov_inputs = _prepare_model_inputs(optimized_model, tokenizer, prompt, use_chat_template)

    with torch.no_grad():
        base_outputs = base_model(**base_inputs)
        ov_outputs = optimized_model(**ov_inputs)

    base_logits = base_outputs.logits[0, -1].float().cpu()
    ov_logits = ov_outputs.logits[0, -1].float().cpu()
    abs_diff = (base_logits - ov_logits).abs()

    print("=========================")
    print("Prompt:", prompt)
    print("Last-token logits comparison")
    print("argmax_base:", int(base_logits.argmax().item()))
    print("argmax_ov:", int(ov_logits.argmax().item()))
    print("max_abs_diff:", float(abs_diff.max().item()))
    print("mean_abs_diff:", float(abs_diff.mean().item()))

    base_topk = torch.topk(base_logits, k=top_k)
    ov_topk = torch.topk(ov_logits, k=top_k)

    print()
    print(f"Top-{top_k} baseline:")
    for token_id, score in zip(base_topk.indices.tolist(), base_topk.values.tolist()):
        print(f"  {token_id}: {repr(tokenizer.decode([token_id]))} -> {score:.6f}")

    print()
    print(f"Top-{top_k} openvino:")
    for token_id, score in zip(ov_topk.indices.tolist(), ov_topk.values.tolist()):
        print(f"  {token_id}: {repr(tokenizer.decode([token_id]))} -> {score:.6f}")


def trace_teacher_forced_steps(
    model_id: str,
    model_dir: str,
    prompt: str,
    use_chat_template: bool,
    steps: int,
    top_k: int,
) -> None:
    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model directory was not found: {model_dir_path}. "
            "Run `python demo.py export --model-dir ...` first or pass the correct exported model path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    optimized_model = OVModelForCausalLM.from_pretrained(
        str(model_dir_path),
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
        ov_config=F32_CONFIG,
    )
    _model_snapshot("baseline", base_model, tokenizer, steps)
    _model_snapshot("openvino", optimized_model, tokenizer, steps)

    shared_inputs = _prepare_inputs_cpu(tokenizer, prompt, use_chat_template)
    if "token_type_ids" in shared_inputs:
        shared_inputs.pop("token_type_ids")

    input_ids = shared_inputs["input_ids"].cpu()
    attention_mask = shared_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.cpu()

    print("=========================")
    print("Teacher-forced trace")
    print("Prompt:", prompt)
    print(f"Steps: {steps}")
    print()

    for step_idx in range(steps):
        base_inputs = {
            "input_ids": input_ids.to(getattr(base_model, "device", "cpu")),
            "attention_mask": attention_mask.to(getattr(base_model, "device", "cpu")),
        }
        ov_inputs = {
            "input_ids": input_ids.to(getattr(optimized_model, "device", "cpu")),
            "attention_mask": attention_mask.to(getattr(optimized_model, "device", "cpu")),
        }

        with torch.no_grad():
            base_logits = base_model(**base_inputs).logits[0, -1].float().cpu()
            ov_logits = optimized_model(**ov_inputs).logits[0, -1].float().cpu()

        abs_diff = (base_logits - ov_logits).abs()
        base_next = int(base_logits.argmax().item())
        ov_next = int(ov_logits.argmax().item())
        same = base_next == ov_next

        print(
            f"step={step_idx} same={same} "
            f"base={base_next}:{repr(tokenizer.decode([base_next]))} "
            f"ov={ov_next}:{repr(tokenizer.decode([ov_next]))} "
            f"max_abs_diff={float(abs_diff.max().item()):.6f}"
        )

        if not same:
            base_topk = torch.topk(base_logits, k=top_k)
            ov_topk = torch.topk(ov_logits, k=top_k)
            print(f"Top-{top_k} baseline:")
            for token_id, score in zip(base_topk.indices.tolist(), base_topk.values.tolist()):
                print(f"  {token_id}: {repr(tokenizer.decode([token_id]))} -> {score:.6f}")
            print(f"Top-{top_k} openvino:")
            for token_id, score in zip(ov_topk.indices.tolist(), ov_topk.values.tolist()):
                print(f"  {token_id}: {repr(tokenizer.decode([token_id]))} -> {score:.6f}")
            return

        next_token = torch.tensor([[base_next]], dtype=input_ids.dtype)
        next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

    print("Trace completed with matching greedy next-token choices on the shared prefix.")


def evaluate_model(
    model_id: str,
    model_dir: str,
    prompts,
    max_new_tokens: int,
    use_chat_template: bool,
    top_k: int,
) -> None:
    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model directory was not found: {model_dir_path}. "
            "Run `python demo.py export --model-dir ...` first or pass the correct exported model path."
        )

    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    optimized_model = OVModelForCausalLM.from_pretrained(
        str(model_dir_path),
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
        ov_config=F32_CONFIG,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    _model_snapshot("baseline", base_model, tokenizer, max_new_tokens)
    _model_snapshot("openvino", optimized_model, tokenizer, max_new_tokens)

    gen_answer_fn = _generate_answer
    evaluator = whowhatbench.TextEvaluator(
        base_model=base_model,
        tokenizer=tokenizer,
        test_data=prompts,
        max_new_tokens=max_new_tokens,
        use_chat_template=use_chat_template,
        gen_answer_fn=gen_answer_fn,
    )
    metrics_per_prompt, metrics = evaluator.score(optimized_model, gen_answer_fn=gen_answer_fn)

    print(f"stateful={optimized_model.stateful} use_cache={optimized_model.use_cache} chat_template={use_chat_template}")
    print("similarity:", metrics["similarity"][0])
    print()
    print("Worst examples:")
    for example in evaluator.worst_examples(top_k=top_k, metric="similarity"):
        print("=========================")
        print("Prompt:", example["prompt"])
        print("Baseline:", example["source_model"])
        print("Optimized:", example["optimized_model"])
        print()

    if prompts is not None:
        print(metrics_per_prompt[["similarity"]])


def compare_model_outputs(
    model_id: str,
    model_dir: str,
    prompts,
    max_new_tokens: int,
    use_chat_template: bool,
) -> None:
    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model directory was not found: {model_dir_path}. "
            "Run `python demo.py export --model-dir ...` first or pass the correct exported model path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    optimized_model = OVModelForCausalLM.from_pretrained(
        str(model_dir_path),
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
        ov_config=F32_CONFIG,
    )
    _model_snapshot("baseline", base_model, tokenizer, max_new_tokens)
    _model_snapshot("openvino", optimized_model, tokenizer, max_new_tokens)

    exact_matches = 0
    for idx, prompt in enumerate(prompts):
        base_tokens = _generate_tokens(base_model, tokenizer, prompt, max_new_tokens, use_chat_template)
        ov_tokens = _generate_tokens(optimized_model, tokenizer, prompt, max_new_tokens, use_chat_template)
        same = base_tokens.shape == ov_tokens.shape and base_tokens.equal(ov_tokens)
        exact_matches += int(same)

        print("=========================")
        print(f"Prompt {idx + 1}/{len(prompts)}: {prompt}")
        print("Exact match:", same)
        if same:
            continue

        compare_len = min(base_tokens.shape[0], ov_tokens.shape[0])
        first_mismatch = None
        for token_idx in range(compare_len):
            if base_tokens[token_idx].item() != ov_tokens[token_idx].item():
                first_mismatch = token_idx
                break
        if first_mismatch is None:
            first_mismatch = compare_len

        print("First mismatch index:", first_mismatch)
        print("Base token id:", None if first_mismatch >= base_tokens.shape[0] else base_tokens[first_mismatch].item())
        print("OV token id:", None if first_mismatch >= ov_tokens.shape[0] else ov_tokens[first_mismatch].item())
        print("Base text:", tokenizer.decode(base_tokens, skip_special_tokens=True))
        print("OV text:", tokenizer.decode(ov_tokens, skip_special_tokens=True))

    print()
    print(
        f"Exact prompt matches: {exact_matches}/{len(prompts)} "
        f"(stateful={optimized_model.stateful}, use_cache={optimized_model.use_cache}, chat_template={use_chat_template})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export and evaluate GigaChat3 with OpenVINO.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export the model to OpenVINO IR.")
    export_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    export_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    export_parser.add_argument("--stateful", action=argparse.BooleanOptionalAction, default=True)
    export_parser.add_argument("--fp32", action=argparse.BooleanOptionalAction, default=True)

    eval_parser = subparsers.add_parser("eval", help="Run whowhatbench evaluation.")
    eval_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    eval_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    eval_parser.add_argument("--max-new-tokens", type=int, default=128)
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--use-chat-template", action="store_true")
    eval_parser.add_argument(
        "--prompt-set",
        choices=("default", "short"),
        default="default",
        help="Use whowhatbench default prompts or the short four-prompt sanity set.",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare exact HF and OV generated tokens.")
    compare_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    compare_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    compare_parser.add_argument("--max-new-tokens", type=int, default=128)
    compare_parser.add_argument("--use-chat-template", action="store_true")
    compare_parser.add_argument(
        "--prompt-set",
        choices=("default", "short"),
        default="short",
        help="Use whowhatbench default prompts or the short four-prompt sanity set.",
    )

    logits_parser = subparsers.add_parser("logits", help="Compare baseline and OV logits on a single prompt.")
    logits_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    logits_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    logits_parser.add_argument("--use-chat-template", action="store_true")
    logits_parser.add_argument("--top-k", type=int, default=10)
    logits_parser.add_argument(
        "--prompt",
        default=SHORT_PROMPTS[0],
        help="Prompt used for logits comparison.",
    )

    trace_parser = subparsers.add_parser("trace", help="Teacher-forced step-by-step logits trace on a single prompt.")
    trace_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    trace_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    trace_parser.add_argument("--use-chat-template", action="store_true")
    trace_parser.add_argument("--steps", type=int, default=32)
    trace_parser.add_argument("--top-k", type=int, default=10)
    trace_parser.add_argument(
        "--prompt",
        default=SHORT_PROMPTS[0],
        help="Prompt used for the teacher-forced trace.",
    )

    attn_debug_parser = subparsers.add_parser(
        "attn-debug",
        help="Compare upstream DeepSeek V3 attention against the export-patched PyTorch attention.",
    )
    attn_debug_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    attn_debug_parser.add_argument("--use-chat-template", action="store_true")
    attn_debug_parser.add_argument("--steps", type=int, default=10)
    attn_debug_parser.add_argument("--top-k", type=int, default=10)
    attn_debug_parser.add_argument("--layer-idx", type=int, default=None)
    attn_debug_parser.add_argument(
        "--prompt",
        default=SHORT_PROMPTS[0],
        help="Prompt used for the upstream-vs-patched attention comparison.",
    )

    moe_debug_parser = subparsers.add_parser(
        "moe-debug",
        help="Compare upstream DeepSeek V3 MoE against the export-patched PyTorch MoE.",
    )
    moe_debug_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    moe_debug_parser.add_argument("--use-chat-template", action="store_true")
    moe_debug_parser.add_argument("--steps", type=int, default=10)
    moe_debug_parser.add_argument("--top-k", type=int, default=10)
    moe_debug_parser.add_argument("--layer-idx", type=int, default=None)
    moe_debug_parser.add_argument(
        "--prompt",
        default=SHORT_PROMPTS[0],
        help="Prompt used for the upstream-vs-patched MoE comparison.",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "export":
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        export_model(args.model_id, args.model_dir, args.stateful, args.fp32)
        return

    if args.command == "eval":
        prompts = _resolve_prompts(args.prompt_set)
        eval_prompts = None if args.prompt_set == "default" else prompts
        evaluate_model(
            model_id=args.model_id,
            model_dir=args.model_dir,
            prompts=eval_prompts,
            max_new_tokens=args.max_new_tokens,
            use_chat_template=args.use_chat_template,
            top_k=args.top_k,
        )
        return

    if args.command == "logits":
        compare_prompt_logits(
            model_id=args.model_id,
            model_dir=args.model_dir,
            prompt=args.prompt,
            use_chat_template=args.use_chat_template,
            top_k=args.top_k,
        )
        return

    if args.command == "trace":
        trace_teacher_forced_steps(
            model_id=args.model_id,
            model_dir=args.model_dir,
            prompt=args.prompt,
            use_chat_template=args.use_chat_template,
            steps=args.steps,
            top_k=args.top_k,
        )
        return

    if args.command == "attn-debug":
        compare_export_attention_paths(
            model_id=args.model_id,
            prompt=args.prompt,
            use_chat_template=args.use_chat_template,
            steps=args.steps,
            top_k=args.top_k,
            layer_idx=args.layer_idx,
        )
        return

    if args.command == "moe-debug":
        compare_export_moe_paths(
            model_id=args.model_id,
            prompt=args.prompt,
            use_chat_template=args.use_chat_template,
            steps=args.steps,
            top_k=args.top_k,
            layer_idx=args.layer_idx,
        )
        return

    prompts = _resolve_prompts(args.prompt_set)
    compare_model_outputs(
        model_id=args.model_id,
        model_dir=args.model_dir,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=args.use_chat_template,
    )


if __name__ == "__main__":
    main()
