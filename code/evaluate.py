"""
Evaluation Script — matches official test script logic exactly.
AIAA 4051 Individual Project

Evaluation rule (from individual_project_test.py):
    is_correct = (true_answer.strip().lower()) in (pred_answer.strip().lower())

Default: BF16, batched generation (eval_batch_size=16), val split only.
Use --val_only to skip the slow train-set evaluation (recommended after first run).

Usage:
    # Fast: val set only, batch=16 (~2 min on RTX 4090)
    python evaluate.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r8_ep3_bf16 \\
        --data_path    ../data/dataset.json \\
        --val_only

    # Full train+val (slow if train set is large):
    python evaluate.py \\
        --model_path   ../Llama-2-7b \\
        --adapter_path ../experiments/r8_ep3_bf16 \\
        --data_path    ../data/dataset.json
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from utils import load_and_split_dataset

# ── Compatibility fix ─────────────────────────────────────────────────────────
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
    p.add_argument("--model_path",      default="./Llama-2-7b")
    p.add_argument("--adapter_path",    default="../model")
    p.add_argument("--data_path",       default="../data/dataset.json")
    p.add_argument("--train_ratio",     type=float, default=0.9)
    p.add_argument("--max_new_tokens",  type=int,   default=32)
    p.add_argument("--eval_batch_size", type=int,   default=16,
                   help="Samples per generate() call. Larger = faster but more VRAM. "
                        "Default 16 uses ~4 GB extra on top of model.")
    p.add_argument("--val_only",        action="store_true",
                   help="Skip train-set evaluation (much faster; train acc is usually ~99%%).")
    p.add_argument("--load_in_4bit",    action="store_true",
                   help="4-bit loading (low VRAM / official script parity). Default: BF16.")
    p.add_argument("--output_file",     default=None)
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


def _clean_prediction(text: str) -> str:
    """
    Strip verbose tail from model generation.
    Llama 2 sometimes generates 'mirage\n\nQuestion: ...' or multi-sentence
    explanations. We keep only the core answer phrase.
    Rules (applied in order):
      1. Truncate at first double-newline (paragraph break = model rambling)
      2. Truncate when model loops back to generate another 'question:'
      3. Take only the first sentence if it is >= 2 words
    """
    text = text.split("\n\n")[0]
    lower = text.lower()
    for marker in ("\nquestion:", " question:"):
        pos = lower.find(marker)
        if pos != -1:
            text = text[:pos]
            lower = lower[:pos]
    first_sentence = text.split(". ")[0].strip()
    if len(first_sentence.split()) >= 2:
        text = first_sentence
    return text.strip()


def evaluate_split(model, tokenizer, data, split_name, max_new_tokens, batch_size):
    """
    Batched generation: process `batch_size` samples per model.generate() call.
    5-8x faster than the one-sample-at-a-time loop.
    """
    correct = 0
    results = []

    # Stop at newline as well as EOS — prevents the model from running past the
    # first answer line into repeated "Question: ..." generations.
    nl_ids   = tokenizer.encode("\n", add_special_tokens=False)
    stop_ids = [tokenizer.eos_token_id] + nl_ids

    for batch_start in tqdm(range(0, len(data), batch_size),
                            desc=f"Evaluating {split_name}",
                            unit="batch"):
        batch = data[batch_start : batch_start + batch_size]

        prompts      = [f"Question: {s['question'].strip()} Answer:" for s in batch]
        true_answers = [s["correct_answer"].strip().lower() for s in batch]

        # Left-pad so all prompts in the batch are the same length
        encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        input_ids      = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)
        prompt_len     = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask = attention_mask,
                max_new_tokens = max_new_tokens,
                do_sample      = False,
                pad_token_id   = tokenizer.eos_token_id,
                eos_token_id   = stop_ids,
            )

        # Decode only the newly generated tokens for each sample
        for i, sample in enumerate(batch):
            generated_tokens = outputs[i][prompt_len:]
            raw         = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            pred_answer = _clean_prediction(raw).lower()
            is_correct  = true_answers[i] in pred_answer
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

    train_acc, train_results = None, []
    if not args.val_only:
        train_acc, train_results = evaluate_split(
            model, tokenizer, train_data, "Train",
            args.max_new_tokens, args.eval_batch_size,
        )
    else:
        print("[INFO] --val_only: skipping train evaluation.")

    val_acc, val_results = evaluate_split(
        model, tokenizer, val_data, "Validation",
        args.max_new_tokens, args.eval_batch_size,
    )

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
