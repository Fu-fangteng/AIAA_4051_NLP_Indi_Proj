"""
Llama 2-7B LoRA Fine-tuning Script
AIAA 4051 Individual Project

Features:
  - wandb cloud visualization  (real-time loss curves, accessible from any device)
  - TensorBoard local fallback  (tensorboard --logdir ../model/runs)
  - Auto checkpoint resume      (safe to disconnect & reconnect)
  - Progress saved to JSON      (loss_logs.json, updated every epoch)

Usage (on GPU server inside tmux):
    python train.py --model_path /path/to/Llama-2-7b \\
                    --data_path  ../data/dataset.json \\
                    --output_dir ../model \\
                    --use_wandb                   # optional: wandb cloud logging
                    --resume                      # optional: resume from last checkpoint
"""

import os
import json
import argparse
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

from utils import load_and_split_dataset, format_prompt


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path",   default="./Llama-2-7b")
    parser.add_argument("--data_path",    default="../data/dataset.json")
    parser.add_argument("--output_dir",   default="../model")
    parser.add_argument("--train_ratio",  type=float, default=0.9)
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=2e-4)
    parser.add_argument("--max_length",   type=int,   default=128)
    parser.add_argument("--lora_r",       type=int,   default=8)
    parser.add_argument("--lora_alpha",   type=int,   default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--load_in_4bit", action="store_true", help="QLoRA: 4-bit quantization")
    parser.add_argument("--use_wandb",    action="store_true", help="Enable wandb cloud logging")
    parser.add_argument("--wandb_project",default="llama2-lora-nlp",  help="wandb project name")
    parser.add_argument("--resume",       action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


# ── Callback: saves loss JSON after every log step ────────────────────────────
class SaveLossCallback(TrainerCallback):
    """
    Writes loss_logs.json to disk after every logging step.
    This means you can `cat model/loss_logs.json` at any time to check progress,
    even mid-training, without waiting for the epoch to finish.
    """
    def __init__(self, output_dir):
        self.output_dir  = output_dir
        self.train_losses = []
        self.val_losses   = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if "loss" in logs:
            self.train_losses.append({
                "step":  state.global_step,
                "epoch": round(state.epoch, 3) if state.epoch else 0,
                "loss":  round(logs["loss"], 6),
            })
        if "eval_loss" in logs:
            self.val_losses.append({
                "step":  state.global_step,
                "epoch": round(state.epoch, 3) if state.epoch else 0,
                "loss":  round(logs["eval_loss"], 6),
            })
        # Write immediately so it's always up-to-date
        with open(os.path.join(self.output_dir, "loss_logs.json"), "w") as f:
            json.dump(
                {"train_losses": self.train_losses, "val_losses": self.val_losses},
                f, indent=2
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        """Also plot & save PNG at the end of every epoch."""
        _plot_loss(self.train_losses, self.val_losses, self.output_dir)
        epoch = int(state.epoch) if state.epoch else "?"
        print(f"\n[Epoch {epoch}] loss_curve.png updated → {self.output_dir}/loss_curve.png")


# ── Plotting helper ───────────────────────────────────────────────────────────
def _plot_loss(train_losses, val_losses, output_dir):
    plt.figure(figsize=(10, 5))
    if train_losses:
        steps  = [x["step"] for x in train_losses]
        losses = [x["loss"] for x in train_losses]
        plt.plot(steps, losses, label="Train Loss", alpha=0.8)
    if val_losses:
        steps  = [x["step"] for x in val_losses]
        losses = [x["loss"] for x in val_losses]
        plt.plot(steps, losses, label="Val Loss", marker="o", linewidth=2)
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150)
    plt.close()


# ── Resume helper ─────────────────────────────────────────────────────────────
def find_last_checkpoint(output_dir):
    """Return the path of the most recent checkpoint, or None."""
    if not os.path.isdir(output_dir):
        return None
    checkpoints = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-")
    ]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=os.path.getmtime)
    print(f"Found checkpoint: {latest}")
    return latest


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── wandb setup ───────────────────────────────────────────────────────────
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            report_to = "wandb"
            print(f"wandb enabled → project: {args.wandb_project}")
            print(f"View live training at: https://wandb.ai")
        except ImportError:
            print("wandb not installed. Falling back to TensorBoard. Run: pip install wandb")
            report_to = "tensorboard"
    else:
        report_to = "tensorboard"   # always keep tensorboard as fallback

    # ── 1. Tokenizer ──────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Base model ─────────────────────────────────────────────────────────
    print("Loading base model...")
    if args.load_in_4bit:
        # RTX 4090 (Ada Lovelace) has native bfloat16 support — more numerically
        # stable than float16 and recommended for QLoRA on Ampere/Ada GPUs.
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
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.config.use_cache = False

    # ── 3. LoRA adapter ───────────────────────────────────────────────────────
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        bias           = "none",
        target_modules = ["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 4. Dataset ────────────────────────────────────────────────────────────
    train_raw, val_raw = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)

    def tokenize(sample):
        text    = format_prompt(sample, include_answer=True)
        encoded = tokenizer(text, truncation=True, max_length=args.max_length, padding="max_length")
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    col_names     = list(Dataset.from_list(train_raw).column_names)
    train_dataset = Dataset.from_list(train_raw).map(tokenize, remove_columns=col_names)
    val_dataset   = Dataset.from_list(val_raw).map(tokenize,   remove_columns=col_names)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── 5. Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = True,   # RTX 4090 (Ada Lovelace) native bf16
        logging_steps               = 20,        # log every 20 steps
        save_steps                  = 100,       # checkpoint every 100 steps
        evaluation_strategy         = "epoch",
        save_strategy               = "steps",   # save more frequently than epoch
        save_total_limit            = 3,         # keep only last 3 checkpoints
        load_best_model_at_end      = False,     # incompatible with save_strategy="steps" + eval="epoch"
        report_to                   = report_to,
        logging_dir                 = os.path.join(args.output_dir, "runs"),
    )

    # ── 6. Callbacks ──────────────────────────────────────────────────────────
    loss_callback = SaveLossCallback(args.output_dir)

    # ── 7. Trainer ────────────────────────────────────────────────────────────
    trainer = Trainer(
        model         = model,
        args          = training_args,
        train_dataset = train_dataset,
        eval_dataset  = val_dataset,
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        callbacks     = [loss_callback],
    )

    # ── 8. Train (with optional resume) ───────────────────────────────────────
    resume_from = find_last_checkpoint(args.output_dir) if args.resume else None
    if resume_from:
        print(f"Resuming training from: {resume_from}")
    else:
        print("Starting training from scratch...")

    trainer.train(resume_from_checkpoint=resume_from)

    # ── 9. Save final LoRA adapter ────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nAdapter saved → {args.output_dir}")

    # ── 10. Final loss plot ───────────────────────────────────────────────────
    _plot_loss(loss_callback.train_losses, loss_callback.val_losses, args.output_dir)
    print(f"Final loss curve saved → {args.output_dir}/loss_curve.png")

    if args.use_wandb:
        import wandb
        wandb.log({"final_loss_curve": wandb.Image(os.path.join(args.output_dir, "loss_curve.png"))})
        wandb.finish()
        print("wandb run finished.")


if __name__ == "__main__":
    main()
