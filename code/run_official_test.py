"""
run_official_test.py — CLI wrapper for the official grading script logic.

Grading rule (identical to individual_project_test.py):
    is_correct = (true_answer.strip().lower()) in (pred_answer.strip().lower())

Default: BF16 (best accuracy). The official grading script uses float16 + 4-bit;
pass --load_in_4bit --fp16 to exactly replicate that environment.

Usage:
    # Standard (BF16, recommended):
    python run_official_test.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r8_ep3_bf16 \\
        --data_path    ../data/dataset.json

    # After teacher releases test set:
    python run_official_test.py \\
        --model_path     ../Llama-2-7b \\
        --adapter_path   ../experiments/r8_ep3_bf16 \\
        --test_data_path /path/to/teacher_test.json

    # Exactly replicate official grading (float16 + 4-bit):
    python run_official_test.py ... --load_in_4bit --fp16
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils import load_and_split_dataset

# ── Compatibility fix ─────────────────────────────────────────────────────────
# bitsandbytes >= 0.41.0 calls nn.Module.set_submodule during 4-bit loading.
# This shim ensures compatibility with PyTorch < 1.9.1 or edge-case installs.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.split(".")
        mod = self
        for part in parts[:-1]:
            mod = getattr(mod, part)
        setattr(mod, parts[-1], module)
    nn.Module.set_submodule = _set_submodule


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     required=True)
    p.add_argument("--adapter_path",   required=True)
    p.add_argument("--data_path",      default="../data/dataset.json")
    p.add_argument("--test_data_path", default=None,
                   help="Teacher-provided test JSON. Omit to use val split as proxy.")
    p.add_argument("--train_ratio",    type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int,   default=16)
    p.add_argument("--load_in_4bit",   action="store_true",
                   help="4-bit loading (low VRAM / replicate official grading env).")
    p.add_argument("--fp16",           action="store_true",
                   help="float16 instead of BF16 (pair with --load_in_4bit to mirror official script).")
    p.add_argument("--output_file",    default=None)
    return p.parse_args()


def load_model_and_tokenizer(model_path, adapter_path, load_in_4bit, fp16):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.float16 if fp16 else torch.bfloat16

    if load_in_4bit:
        print(f"Loading base model (4-bit NF4, compute_dtype={'float16' if fp16 else 'bfloat16'})...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
                torch_dtype=dtype,
            )
        except AttributeError as e:
            if "set_submodule" in str(e):
                print(
                    "\n[ERROR] set_submodule not found — requires PyTorch >= 1.9.1.\n"
                    "  Upgrade torch or drop --load_in_4bit.\n"
                )
                sys.exit(1)
            raise
    else:
        print(f"Loading base model ({'float16' if fp16 else 'BF16'}, recommended for RTX 4090)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto",
        )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def run_evaluation(model, tokenizer, test_data, max_new_tokens):
    correct = 0
    results = []

    for example in tqdm(test_data, desc="Evaluating"):
        question    = example["question"]
        true_answer = example["correct_answer"].strip().lower()

        # Prompt format identical to individual_project_test.py
        prompt = f"Question: {question} Answer:"
        encoding = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids      = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                attention_mask = attention_mask,
                pad_token_id   = tokenizer.eos_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        pred_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

        is_correct = true_answer in pred_answer
        if is_correct:
            correct += 1

        results.append({
            "question":    example["question"],
            "true_answer": example["correct_answer"],
            "pred_answer": pred_answer,
            "is_correct":  is_correct,
        })

    accuracy = correct / len(test_data)
    return accuracy, results


def main():
    args = parse_args()
    output_file = args.output_file or os.path.join(args.adapter_path, "official_test_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    if args.test_data_path:
        print(f"[INFO] Using teacher test set: {args.test_data_path}")
        with open(args.test_data_path, encoding="utf-8") as f:
            test_data = json.load(f)
        data_source = "teacher_test"
    else:
        print("[INFO] No --test_data_path provided — using val split as proxy.")
        _, test_data = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)
        data_source = "val_split_proxy"

    print(f"[INFO] Test samples: {len(test_data)}")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.adapter_path, args.load_in_4bit, args.fp16
    )

    accuracy, results = run_evaluation(model, tokenizer, test_data, args.max_new_tokens)

    print(f"\n{'='*50}")
    print(f"  Official Test Accuracy: {accuracy*100:.2f}%  ({int(accuracy*len(test_data))}/{len(test_data)})")
    print(f"  Data source: {data_source}")
    print(f"{'='*50}\n")

    output = {
        "adapter_path": args.adapter_path,
        "data_source":  data_source,
        "accuracy":     accuracy,
        "correct":      int(accuracy * len(test_data)),
        "total":        len(test_data),
        "results":      results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved → {output_file}")


if __name__ == "__main__":
    main()
