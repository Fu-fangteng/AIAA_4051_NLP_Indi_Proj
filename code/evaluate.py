"""
Evaluation Script — matches official test script logic exactly.
AIAA 4051 Individual Project

Evaluation rule (from individual_project_test.py):
    is_correct = (true_answer.strip().lower()) in (pred_answer.strip().lower())

Default: BF16 (same precision as training, best accuracy).
Use --load_in_4bit only if VRAM is insufficient or to replicate the official
grading environment (which uses 4-bit float16).

Usage:
    python evaluate.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r8_ep3_bf16 \\
        --data_path    ../data/dataset.json

    # 4-bit fallback (low VRAM or to mirror official script):
    python evaluate.py ... --load_in_4bit
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    p.add_argument("--model_path",     default="./Llama-2-7b")
    p.add_argument("--adapter_path",   default="../model")
    p.add_argument("--data_path",      default="../data/dataset.json")
    p.add_argument("--train_ratio",    type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int,   default=16)
    p.add_argument("--load_in_4bit",   action="store_true",
                   help="4-bit loading (low VRAM / official script parity). Default: BF16.")
    p.add_argument("--output_file",    default=None)
    return p.parse_args()


def load_model(model_path, adapter_path, load_in_4bit=False):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_in_4bit:
        print("Loading base model (4-bit NF4 + BF16 compute)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quant_config,
                device_map="auto",
            )
        except AttributeError as e:
            if "set_submodule" in str(e):
                print(
                    "\n[ERROR] set_submodule not found — requires PyTorch >= 1.9.1.\n"
                    "  Upgrade torch or drop --load_in_4bit (RTX 4090 does not need it).\n"
                )
                sys.exit(1)
            raise
    else:
        print("Loading base model (BF16, recommended for RTX 4090)...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def evaluate_split(model, tokenizer, data, split_name, max_new_tokens):
    correct = 0
    results = []

    for sample in tqdm(data, desc=f"Evaluating {split_name}"):
        question    = sample["question"].strip()
        true_answer = sample["correct_answer"].strip().lower()

        # Prompt format must match training exactly
        prompt = f"Question: {question} Answer:"
        encoding = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids      = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = tokenizer.eos_token_id,
                eos_token_id   = tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_ids.shape[1]:]
        pred_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

        is_correct = true_answer in pred_answer
        if is_correct:
            correct += 1

        results.append({
            "question":    sample["question"],
            "true_answer": sample["correct_answer"],
            "pred_answer": pred_answer,
            "is_correct":  is_correct,
        })

    accuracy = correct / len(data)
    print(f"{split_name:12s} Accuracy: {correct}/{len(data)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy, results


def main():
    args = parse_args()
    output_file = args.output_file or os.path.join(args.adapter_path, "eval_results.json")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    model, tokenizer = load_model(args.model_path, args.adapter_path, args.load_in_4bit)
    train_data, val_data = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)

    train_acc, train_results = evaluate_split(model, tokenizer, train_data, "Train",      args.max_new_tokens)
    val_acc,   val_results   = evaluate_split(model, tokenizer, val_data,   "Validation", args.max_new_tokens)

    output = {
        "adapter_path":   args.adapter_path,
        "train_accuracy": train_acc,
        "val_accuracy":   val_acc,
        "train_results":  train_results,
        "val_results":    val_results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_file}")


if __name__ == "__main__":
    main()
