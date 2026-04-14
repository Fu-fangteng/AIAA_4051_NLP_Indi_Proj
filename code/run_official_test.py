"""
run_official_test.py — CLI wrapper for the official evaluation script.

This replicates individual_project_test.py EXACTLY (same model loading,
same prompt format, same is_correct rule) but with:
  - configurable paths via CLI (no hardcoded "XXXX")
  - JSON output for downstream summary generation
  - support for a separate teacher-provided test set

When --test_data_path is NOT given, falls back to the val split of
--data_path as a local proxy (train_ratio controls the split).
When the teacher provides their official test file, pass it via
--test_data_path and results will reflect the true test accuracy.

Usage:
    # Local proxy (val split)
    python run_official_test.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r16_ep5 \\
        --data_path    ../data/dataset.json

    # With official teacher test data
    python run_official_test.py \\
        --model_path    ../Llama-2-7b \\
        --adapter_path  ../experiments/r16_ep5 \\
        --test_data_path /path/to/teacher_test.json
"""

import os, sys, json, argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils import load_and_split_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     required=True, help="Base Llama-2-7b path")
    p.add_argument("--adapter_path",   required=True, help="LoRA adapter directory")
    p.add_argument("--data_path",      default="../data/dataset.json",
                   help="Dataset used for train/val split (proxy when no test_data_path)")
    p.add_argument("--test_data_path", default=None,
                   help="Official teacher test file (JSON list). Overrides val-split proxy.")
    p.add_argument("--train_ratio",    type=float, default=0.9,
                   help="Only used when test_data_path is absent")
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--output_file",    default=None,
                   help="Where to save JSON results. "
                        "Defaults to <adapter_path>/official_test_results.json")
    return p.parse_args()


def load_model_and_tokenizer(model_path, adapter_path):
    """Load model exactly as the official script does."""
    # ── Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Base model with 4-bit quantization ────────────────────────────────
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,   # keep float16 to match official script
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # ── LoRA adapter ────────────────────────────────────────────────────────
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer


def run_evaluation(model, tokenizer, test_data, max_new_tokens):
    """
    Evaluate exactly as individual_project_test.py does.
    Rule: is_correct = (true_answer.strip().lower()) in (pred_answer.strip().lower())
    """
    correct = 0
    results = []

    for example in tqdm(test_data, desc="Evaluating"):
        question    = example["question"]
        true_answer = example["correct_answer"].strip().lower()

        # Prompt format — matches individual_project_test.py exactly
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

        # Official scoring rule
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

    # ── Load test data ────────────────────────────────────────────────────────
    if args.test_data_path:
        print(f"[INFO] Using official teacher test data: {args.test_data_path}")
        with open(args.test_data_path, encoding="utf-8") as f:
            test_data = json.load(f)
        data_source = "teacher_test"
    else:
        print(f"[INFO] No test_data_path given. Using val split of {args.data_path} as proxy.")
        _, test_data = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)
        data_source = "val_split_proxy"

    print(f"[INFO] Test samples: {len(test_data)}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[INFO] Loading base model: {args.model_path}")
    print(f"[INFO] Loading adapter:    {args.adapter_path}")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.adapter_path)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    accuracy, results = run_evaluation(model, tokenizer, test_data, args.max_new_tokens)

    print(f"\n{'='*50}")
    print(f"  Official Test Accuracy: {accuracy*100:.2f}%  ({int(accuracy*len(test_data))}/{len(test_data)})")
    print(f"  Data source: {data_source}")
    print(f"{'='*50}\n")

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "adapter_path":  args.adapter_path,
        "data_source":   data_source,
        "accuracy":      accuracy,
        "correct":       int(accuracy * len(test_data)),
        "total":         len(test_data),
        "results":       results,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Results saved → {output_file}")


if __name__ == "__main__":
    main()
