"""
Llama 2-7B QLoRA Fine-tuning Script  —  AIAA 4051 Individual Project
RTX 4090 optimized (bfloat16, all linear layers, answer-only loss)

Key fixes over v1:
  1. Answer-only loss (label masking): question tokens are masked out (-100)
     so the model learns to predict the *exact* answer string, not the prompt.
  2. All 7 LLaMA linear layers in LoRA target (was only q_proj + v_proj).
  3. Higher default rank (r=16) and more epochs (5) for better memorization.
  4. Gradient checkpointing support to trade speed for memory.

Usage:
    python train.py \\
        --model_path ../Llama-2-7b \\
        --data_path  ../data/dataset.json \\
        --output_dir ../experiments/exp_001 \\
        --load_in_4bit

For multi-experiment sweeps, use sweep.py instead.
"""

import os, json, argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

from utils import load_and_split_dataset


# ── All linear projection layers in LLaMA 2 ──────────────────────────────────
ALL_LINEAR_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",     default="./Llama-2-7b")
    p.add_argument("--data_path",      default="../data/dataset.json")
    p.add_argument("--output_dir",     default="../model")
    p.add_argument("--train_ratio",    type=float, default=0.9)
    p.add_argument("--epochs",         type=int,   default=5)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--grad_accum",     type=int,   default=2)
    p.add_argument("--lr",             type=float, default=2e-4)
    p.add_argument("--max_length",     type=int,   default=256)
    p.add_argument("--lora_r",         type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=32)
    p.add_argument("--lora_dropout",   type=float, default=0.05)
    p.add_argument("--load_in_4bit",   action="store_true")
    p.add_argument("--grad_ckpt",      action="store_true", help="Gradient checkpointing (saves VRAM, slower)")
    p.add_argument("--use_wandb",      action="store_true")
    p.add_argument("--wandb_project",  default="llama2-lora-nlp")
    p.add_argument("--resume",         action="store_true")
    return p.parse_args()


# ── Tokenization with answer-only loss ───────────────────────────────────────
def make_tokenize_fn(tokenizer, max_length):
    """
    Returns a tokenize function that masks out the prompt tokens in labels.

    Why this matters:
        Official eval: is_correct = (true_answer in pred_answer)
        If we train on the full sequence, the model wastes capacity learning
        to predict question tokens and never sharply learns the exact answer
        phrasing.  Masking prompt tokens concentrates the gradient entirely
        on the answer span → the model learns to reproduce the exact label.
    """
    def tokenize(sample):
        question = sample["question"].strip()
        answer   = sample["correct_answer"].strip()

        # Prompt that matches the official test script exactly
        prompt    = f"Question: {question} Answer:"
        full_text = f"Question: {question} Answer: {answer}"

        # Tokenize prompt alone to find its boundary in the full sequence.
        # add_special_tokens=True so BOS is counted.
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        # Tokenize full sequence (no padding here — DataCollatorForSeq2Seq pads)
        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

        # Build labels: -100 for prompt, real token IDs for answer
        labels = encoded["input_ids"].copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        encoded["labels"] = labels
        return encoded

    return tokenize


# ── Callback: JSON loss log + PNG ─────────────────────────────────────────────
class SaveLossCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir   = output_dir
        self.train_losses = []
        self.val_losses   = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": round(state.epoch or 0, 3)}
        if "loss"      in logs: self.train_losses.append({**entry, "loss": round(logs["loss"],      6)})
        if "eval_loss" in logs: self.val_losses.append(  {**entry, "loss": round(logs["eval_loss"], 6)})
        with open(os.path.join(self.output_dir, "loss_logs.json"), "w") as f:
            json.dump({"train_losses": self.train_losses, "val_losses": self.val_losses}, f, indent=2)

    def on_epoch_end(self, args, state, control, **kwargs):
        _plot_loss(self.train_losses, self.val_losses, self.output_dir)
        print(f"\n[Epoch {int(state.epoch or 0)}] loss_curve.png updated")


def _plot_loss(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    if train_losses:
        plt.plot([x["step"] for x in train_losses], [x["loss"] for x in train_losses],
                 label="Train Loss", alpha=0.8)
    if val_losses:
        plt.plot([x["step"] for x in val_losses], [x["loss"] for x in val_losses],
                 label="Val Loss", marker="o", linewidth=2)
    plt.xlabel("Steps"); plt.ylabel("Loss")
    plt.title("Training & Validation Loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()


# ── Checkpoint helper ─────────────────────────────────────────────────────────
def find_last_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    ckpts = [os.path.join(output_dir, d) for d in os.listdir(output_dir)
             if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    latest = max(ckpts, key=os.path.getmtime)
    print(f"Resuming from: {latest}")
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── wandb / tensorboard ───────────────────────────────────────────────────
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            report_to = "wandb"
        except ImportError:
            print("wandb not installed, falling back to tensorboard")
            report_to = "tensorboard"
    else:
        report_to = "tensorboard"

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── Base model ────────────────────────────────────────────────────────────
    print("Loading base model...")
    if args.load_in_4bit:
        # bfloat16 compute dtype: RTX 4090 Ada Lovelace has native bf16 support.
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        # Required for 4-bit + LoRA training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.grad_ckpt)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()

    model.config.use_cache = False

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    # All 7 linear projection layers — much stronger than q+v only.
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        bias           = "none",
        target_modules = ALL_LINEAR_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_raw, val_raw = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)

    def tokenize_batch(examples):
        # Dataset.map passes a dict of lists; process sample-by-sample
        results = [tokenize_fn({"question": q, "correct_answer": a})
                   for q, a in zip(examples["question"], examples["correct_answer"])]
        return {k: [r[k] for r in results] for k in results[0]}

    raw_train_ds = Dataset.from_list(train_raw)
    raw_val_ds   = Dataset.from_list(val_raw)
    train_dataset = raw_train_ds.map(tokenize_batch, batched=True,
                                     remove_columns=raw_train_ds.column_names)
    val_dataset   = raw_val_ds.map(tokenize_batch,   batched=True,
                                   remove_columns=raw_val_ds.column_names)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = True,   # RTX 4090 Ada Lovelace native bf16
        logging_steps               = 20,
        save_steps                  = 100,
        evaluation_strategy         = "epoch",
        save_strategy               = "steps",
        save_total_limit            = 2,
        load_best_model_at_end      = False,
        report_to                   = report_to,
        logging_dir                 = os.path.join(args.output_dir, "runs"),
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    loss_cb = SaveLossCallback(args.output_dir)
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        # DataCollatorForSeq2Seq pads and sets padding label tokens to -100
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                               padding=True, pad_to_multiple_of=8),
        callbacks     = [loss_cb],
    )

    resume_from = find_last_checkpoint(args.output_dir) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save final adapter ────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _plot_loss(loss_cb.train_losses, loss_cb.val_losses, args.output_dir)
    print(f"\nAdapter saved → {args.output_dir}")


if __name__ == "__main__":
    main()
