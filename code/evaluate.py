"""
Evaluation Script — compute accuracy on train / val splits.
AIAA 4051 Individual Project

Usage (on GPU server):
    python evaluate.py --model_path /path/to/Llama-2-7b \
                       --adapter_path ../model \
                       --data_path    ../data/dataset.json
"""

import os
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils import load_and_split_dataset, format_prompt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   default="./Llama-2-7b")
    parser.add_argument("--adapter_path", default="../model")
    parser.add_argument("--data_path",    default="../data/dataset.json")
    parser.add_argument("--train_ratio",  type=float, default=0.9)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output_file",  default="../model/eval_results.json")
    return parser.parse_args()


def compute_accuracy(model, tokenizer, data, split_name, max_new_tokens):
    correct = 0
    results = []

    for sample in tqdm(data, desc=f"Evaluating {split_name}"):
        # Question-only prompt — consistent with the official test script
        prompt = format_prompt(sample, include_answer=False)
        encoding = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids      = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask    = attention_mask,
                max_new_tokens    = max_new_tokens,
                do_sample         = False,
                pad_token_id      = tokenizer.eos_token_id,
                eos_token_id      = tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        generated_tokens = outputs[0][input_ids.shape[1]:]
        pred_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()
        true_answer = sample["correct_answer"].strip().lower()

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
    print(f"{split_name} Accuracy: {correct}/{len(data)} = {accuracy:.4f} ({accuracy*100:.2f}%)")
    return accuracy, results


def main():
    args = parse_args()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"   # left-padding for generation

    # ── Load base model + LoRA adapter ────────────────────────────────────────
    print("Loading base model with 4-bit quantization (bfloat16, RTX 4090)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # native bf16 on RTX 4090
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()

    # ── Load & split data ─────────────────────────────────────────────────────
    train_data, val_data = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)

    # ── Compute accuracy ──────────────────────────────────────────────────────
    train_acc, train_results = compute_accuracy(model, tokenizer, train_data, "Train",      args.max_new_tokens)
    val_acc,   val_results   = compute_accuracy(model, tokenizer, val_data,   "Validation", args.max_new_tokens)

    # ── Save results ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {
        "train_accuracy": train_acc,
        "val_accuracy":   val_acc,
        "train_results":  train_results,
        "val_results":    val_results,
    }
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {args.output_file}")


if __name__ == "__main__":
    main()
