#!/usr/bin/env python3
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a causal LM on the ROCm GPU via Transformers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--prompt",
        default="Responda em portugues em duas frases: qual o estado atual da NPU AMD neste host Linux?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--quantization", default="none", choices=["none", "gptq", "bnb4"])
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple:
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "auto": "auto",
    }
    dtype = dtype_map[args.dtype]

    load_kwargs = {
        "device_map": "auto",
        "attn_implementation": "sdpa",
    }

    if args.quantization == "gptq":
        load_kwargs["torch_dtype"] = dtype if dtype != "auto" else torch.float16
        load_kwargs.pop("attn_implementation", None)
    elif args.quantization == "bnb4":
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kwargs["torch_dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **load_kwargs)
    load_s = time.time() - t0

    return tokenizer, model, load_s


def main() -> None:
    args = parse_args()

    print(f"[INFO] model={args.model_id}", flush=True)
    print(f"[INFO] quantization={args.quantization}", flush=True)
    print(f"[INFO] dtype={args.dtype}", flush=True)
    print(f"[INFO] torch={torch.__version__}", flush=True)
    print(f"[INFO] cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] device0={torch.cuda.get_device_name(0)}", flush=True)

    print("[INFO] loading model...", flush=True)
    tokenizer, model, load_s = load_model(args)
    print(f"[INFO] model loaded in {load_s:.2f}s", flush=True)

    messages = [{"role": "user", "content": args.prompt}]

    if tokenizer.chat_template is not None:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = args.prompt

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print("[INFO] generating...", flush=True)
    t1 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_s = time.time() - t1
    new_tokens = outputs.shape[1] - input_len

    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"[INFO] generation took {gen_s:.2f}s ({new_tokens} tokens, {new_tokens / gen_s:.2f} tok/s)", flush=True)
    print("--- OUTPUT ---")
    print(decoded)


if __name__ == "__main__":
    main()
