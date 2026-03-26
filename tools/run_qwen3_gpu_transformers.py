#!/usr/bin/env python3
import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Qwen3 model on the ROCm GPU via Transformers.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--prompt",
        default="Responda em portugues em duas frases: qual o estado atual da NPU AMD neste host Linux?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[INFO] model={args.model_id}", flush=True)
    print(f"[INFO] torch={torch.__version__}", flush=True)
    print(f"[INFO] cuda_available={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[INFO] device0={torch.cuda.get_device_name(0)}", flush=True)

    print("[INFO] loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    print("[INFO] loading model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        device_map="auto",
        attn_implementation="sdpa",
    )
    load_s = time.time() - t0
    print(f"[INFO] model loaded in {load_s:.2f}s", flush=True)

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print("[INFO] generating...", flush=True)
    t1 = time.time()
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    gen_s = time.time() - t1

    decoded = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    print(f"[INFO] generation took {gen_s:.2f}s", flush=True)
    print("--- OUTPUT ---")
    print(decoded)


if __name__ == "__main__":
    main()
