"""
Evaluation Script — matches official test script logic exactly.
AIAA 4051 Individual Project

Evaluation rule (from individual_project_test.py):
    is_correct = (true_answer.strip().lower()) in (pred_answer.strip().lower())

默认用 BF16 加载（与训练保持一致，兼容性最好）。
如果服务器显存不足，加 --load_in_4bit。

Usage:
    python evaluate.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r16_ep5_bf16 \\
        --data_path    ../data/dataset.json

    # 显存不足时用 4-bit
    python evaluate.py ... --load_in_4bit
"""

import os, json, argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils import load_and_split_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     default="./Llama-2-7b")
    p.add_argument("--adapter_path",   default="../model")
    p.add_argument("--data_path",      default="../data/dataset.json")
    p.add_argument("--train_ratio",    type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int,   default=16)
    p.add_argument("--load_in_4bit",   action="store_true",
                   help="用 4-bit 加载（显存不足时使用）。默认 BF16。")
    p.add_argument("--output_file",    default=None)
    return p.parse_args()


def load_model(model_path, adapter_path, load_in_4bit=False):
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if load_in_4bit:
        print("Loading base model (4-bit QLoRA)...")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        print("Loading base model (BF16)...")
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
