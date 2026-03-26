import argparse
from importlib.resources import files
from pathlib import Path

import torch
import whowhatbench
import yaml
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_ID = "ai-sage/GigaChat3-10B-A1.8B-bf16"
DEFAULT_MODEL_DIR = "./output_dir"
DEFAULT_MAX_NEW_TOKENS = 128
FAST_PROMPTS = [
    "Who is the most famous programmer?",
    "Who is Leo Tolstoy?",
    "Explain what artificial intelligence is.",
    "What is deep learning?",
]


def load_full_prompts():
    prompt_path = files("whowhatbench.prompts").joinpath("text_prompts.yaml")
    prompt_data = yaml.safe_load(prompt_path.read_text(encoding="utf-8"))
    return prompt_data["en"]["prompts"]


def build_generation_kwargs(tokenizer, max_new_tokens: int):
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    bos_token_id = tokenizer.bos_token_id

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
        "use_model_defaults": False,
    }

    if eos_token_id is not None:
        generation_kwargs["eos_token_id"] = eos_token_id
    if pad_token_id is not None:
        generation_kwargs["pad_token_id"] = pad_token_id
    if bos_token_id is not None:
        generation_kwargs["bos_token_id"] = bos_token_id

    return generation_kwargs


def prepare_inputs(model, tokenizer, prompt: str):
    device = getattr(model, "device", "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    inputs.pop("token_type_ids", None)
    return inputs


def generate_answer(
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
    del crop_question, use_chat_template, empty_adapters, num_assistant_tokens, assistant_confidence_threshold
    inputs = prepare_inputs(model, tokenizer, prompt)
    tokens = model.generate(**inputs, **build_generation_kwargs(tokenizer, max_new_tokens))
    prompt_len = inputs["input_ids"].shape[-1]
    return tokenizer.decode(tokens[0, prompt_len:], skip_special_tokens=True)


def load_models(model_id: str, model_dir: str):
    model_dir_path = Path(model_dir).expanduser().resolve()
    if not model_dir_path.exists():
        raise FileNotFoundError(
            f"OpenVINO model directory was not found: {model_dir_path}. "
            "Run `python demo.py export` first or pass the correct exported model path."
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    optimized_model = OVModelForCausalLM.from_pretrained(
        str(model_dir_path),
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
    )
    return tokenizer, base_model, optimized_model


def export_model(model_id: str, model_dir: str):
    model_dir_path = Path(model_dir).expanduser().resolve()
    model_dir_path.mkdir(parents=True, exist_ok=True)

    optimized_model = OVModelForCausalLM.from_pretrained(
        model_id,
        export=True,
        use_cache=True,
        load_in_8bit=False,
        quantization_config=None,
    )

    optimized_model.save_pretrained(str(model_dir_path))
    print(f"Saved OpenVINO model to {model_dir_path}")


def run_test(model_id: str, model_dir: str, prompts, max_new_tokens: int, top_k: int):
    tokenizer, base_model, optimized_model = load_models(model_id, model_dir)

    evaluator = whowhatbench.TextEvaluator(
        base_model=base_model,
        tokenizer=tokenizer,
        test_data=prompts,
        max_new_tokens=max_new_tokens,
        use_chat_template=False,
        gen_answer_fn=generate_answer,
    )
    _, metrics = evaluator.score(optimized_model, gen_answer_fn=generate_answer)

    print("similarity:", metrics["similarity"][0])
    print()
    print("Worst examples:")
    for example in evaluator.worst_examples(top_k=top_k, metric="similarity"):
        print("=========================")
        print("Prompt:", example["prompt"])
        print("Baseline:", example["source_model"])
        print("Optimized:", example["optimized_model"])
        print()


def build_parser():
    parser = argparse.ArgumentParser(description="Simple GigaChat3 OpenVINO demo.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export")
    export_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    export_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)

    test_parser = subparsers.add_parser("test")
    test_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    test_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    test_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    test_parser.add_argument("--top-k", type=int, default=5)

    fast_test_parser = subparsers.add_parser("fast-test")
    fast_test_parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    fast_test_parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    fast_test_parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    fast_test_parser.add_argument("--top-k", type=int, default=5)

    return parser


def main():
    args = build_parser().parse_args()

    if args.command == "export":
        export_model(args.model_id, args.model_dir)
        return

    if args.command == "test":
        run_test(
            model_id=args.model_id,
            model_dir=args.model_dir,
            prompts=load_full_prompts(),
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
        )
        return

    run_test(
        model_id=args.model_id,
        model_dir=args.model_dir,
        prompts=FAST_PROMPTS,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
