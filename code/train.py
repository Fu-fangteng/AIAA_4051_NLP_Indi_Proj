"""
Llama 2-7B LoRA Fine-tuning Script  —  AIAA 4051 Individual Project
RTX 4090 optimized (bfloat16, attention layers, answer-only loss)

Key design decisions:
  1. Answer-only loss (label masking): question tokens are masked out (-100)
     so the model learns to predict the *exact* answer string, not the prompt.
  2. 4 attention LoRA targets (q/k/v/o_proj) — balanced capacity vs. regularisation.
  3. EarlyStoppingCallback + load_best_model_at_end: saves the epoch with the
     lowest val loss instead of the last (overfit) checkpoint.
  4. Gradient checkpointing support to trade speed for memory on tight VRAM.

Default mode: BF16 (no quantization). RTX 4090 (24 GB) has plenty of VRAM.
Use --load_in_4bit only if VRAM < 20 GB or for quick smoke-tests.

Usage:
    python train.py \\
        --model_path ../Llama-2-7b \\
        --data_path  ../data/dataset.json \\
        --output_dir ../experiments/exp_001

    # With 4-bit (fallback / low-VRAM):
    python train.py ... --load_in_4bit

For multi-experiment sweeps, use sweep.py instead.
"""

import os, sys, json, argparse
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Compatibility fix ─────────────────────────────────────────────────────────
# set_submodule was added to nn.Module in PyTorch 1.9.1 and is required by
# bitsandbytes >= 0.41.0 when replacing linear layers during 4-bit loading.
# This shim makes older or edge-case PyTorch installs safe.
if not hasattr(nn.Module, "set_submodule"):
    def _set_submodule(self, target: str, module: nn.Module) -> None:
        parts = target.split(".")
        mod = self
        for part in parts[:-1]:
            mod = getattr(mod, part)
        setattr(mod, parts[-1], module)
    nn.Module.set_submodule = _set_submodule

from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig, TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset

from utils import load_and_split_dataset


# ── LoRA target modules ───────────────────────────────────────────────────────
# 4 attention projection layers — good balance between capacity and
# generalisation on small datasets. Using all 7 linear layers (incl. MLP)
# dramatically increases trainable params and worsens overfitting.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",                default="./Llama-2-7b")
    p.add_argument("--data_path",                 default="../data/dataset.json")
    p.add_argument("--output_dir",                default="../model")
    p.add_argument("--train_ratio",               type=float, default=0.9)
    # Training hyperparameters (tuned for small datasets, anti-overfitting)
    p.add_argument("--epochs",                    type=int,   default=3)
    p.add_argument("--batch_size",                type=int,   default=8)
    p.add_argument("--grad_accum",                type=int,   default=2)
    p.add_argument("--lr",                        type=float, default=1e-4)
    p.add_argument("--weight_decay",              type=float, default=0.01)
    p.add_argument("--max_length",                type=int,   default=256)
    # LoRA hyperparameters
    p.add_argument("--lora_r",                    type=int,   default=8)
    p.add_argument("--lora_alpha",                type=int,   default=16)
    p.add_argument("--lora_dropout",              type=float, default=0.1)
    # Early stopping
    p.add_argument("--early_stopping_patience",   type=int,   default=2)
    # Memory / precision options
    p.add_argument("--load_in_4bit",              action="store_true",
                   help="4-bit QLoRA (fallback). Default: BF16 (recommended for RTX 4090).")
    p.add_argument("--grad_ckpt",                 action="store_true",
                   help="Gradient checkpointing — saves ~3 GB VRAM, ~20%% slower.")
    # Logging
    p.add_argument("--use_wandb",                 action="store_true")
    p.add_argument("--wandb_project",             default="llama2-lora-nlp")
    p.add_argument("--resume",                    action="store_true")
    return p.parse_args()


# ── Tokenization with answer-only loss ───────────────────────────────────────
def make_tokenize_fn(tokenizer, max_length):
    """
    Masks prompt tokens in labels so cross-entropy is computed only on the
    answer span. This concentrates learning signal on the exact answer phrasing,
    which is what the official eval checks:
        is_correct = (true_answer.lower()) in (pred_answer.lower())
    """
    def tokenize(sample):
        question = sample["question"].strip()
        answer   = sample["correct_answer"].strip()

        prompt    = f"Question: {question} Answer:"
        full_text = f"Question: {question} Answer: {answer}"

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        prompt_len = len(prompt_ids)

        encoded = tokenizer(
            full_text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )

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


# ── 4-bit model loading (with error guidance) ─────────────────────────────────
def load_model_4bit(model_path, grad_ckpt, trust_remote_code=True):
    """
    Load model in 4-bit NF4 + BF16 compute. Includes compatibility checks and
    clear error messages for common failure modes.
    """
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,   # Ada Lovelace native precision
        bnb_4bit_use_double_quant=True,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=trust_remote_code,
        )
    except AttributeError as e:
        if "set_submodule" in str(e):
            print(
                "\n[ERROR] AttributeError: set_submodule not found.\n"
                "  This means bitsandbytes requires PyTorch >= 1.9.1 for 4-bit loading.\n"
                "  Fix: upgrade torch (pip install torch --upgrade) or drop --load_in_4bit.\n"
                "  RTX 4090 (24 GB) does NOT need 4-bit — BF16 is strongly preferred.\n"
            )
            sys.exit(1)
        raise

    # prepare_model_for_kbit_training:
    #   - casts LayerNorm layers to float32 (needed for stable gradient flow)
    #   - optionally enables gradient checkpointing
    try:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt
        )
    except TypeError:
        # Older PEFT (<0.5) does not accept use_gradient_checkpointing
        model = prepare_model_for_kbit_training(model)
        if grad_ckpt:
            model.gradient_checkpointing_enable()

    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Logging backend ───────────────────────────────────────────────────────
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
        print("[INFO] 4-bit mode: ~7-8 GB VRAM. For RTX 4090, BF16 is recommended.")
        model = load_model_4bit(args.model_path, args.grad_ckpt)
    else:
        print("[INFO] BF16 mode: ~14-18 GB VRAM (recommended for RTX 4090).")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        if args.grad_ckpt:
            model.gradient_checkpointing_enable()
            # Ensures the first non-embedding activation has requires_grad=True
            # so backward hooks fire correctly through frozen base layers.
            model.enable_input_require_grads()

    model.config.use_cache = False

    # ── LoRA adapter ──────────────────────────────────────────────────────────
    # 4 attention layers: fewer trainable params → less overfitting on small data.
    # r=8 + dropout=0.1 + weight_decay further regularise the adapter.
    lora_config = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = args.lora_r,
        lora_alpha     = args.lora_alpha,
        lora_dropout   = args.lora_dropout,
        bias           = "none",
        target_modules = TARGET_MODULES,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_raw, val_raw = load_and_split_dataset(args.data_path, train_ratio=args.train_ratio)
    tokenize_fn = make_tokenize_fn(tokenizer, args.max_length)

    def tokenize_batch(examples):
        results = [tokenize_fn({"question": q, "correct_answer": a})
                   for q, a in zip(examples["question"], examples["correct_answer"])]
        return {k: [r[k] for r in results] for k in results[0]}

    raw_train_ds  = Dataset.from_list(train_raw)
    raw_val_ds    = Dataset.from_list(val_raw)
    train_dataset = raw_train_ds.map(tokenize_batch, batched=True,
                                     remove_columns=raw_train_ds.column_names)
    val_dataset   = raw_val_ds.map(tokenize_batch,   batched=True,
                                   remove_columns=raw_val_ds.column_names)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # ── Training arguments ────────────────────────────────────────────────────
    # save_strategy must equal eval_strategy for load_best_model_at_end=True.
    # Both are "epoch" so the best checkpoint (lowest eval_loss) is kept.
    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        weight_decay                = args.weight_decay,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        bf16                        = True,
        logging_steps               = 20,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",   # must match eval_strategy
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
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
        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,
                                               padding=True, pad_to_multiple_of=8),
        callbacks     = [
            loss_cb,
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
        ],
    )

    resume_from = find_last_checkpoint(args.output_dir) if args.resume else None
    trainer.train(resume_from_checkpoint=resume_from)

    # ── Save best adapter ─────────────────────────────────────────────────────
    # trainer.model is the best checkpoint loaded by load_best_model_at_end=True
    best_model = trainer.model
    best_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    _plot_loss(loss_cb.train_losses, loss_cb.val_losses, args.output_dir)
    print(f"\nBest adapter saved → {args.output_dir}")


if __name__ == "__main__":
    main()
