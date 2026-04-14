"""
make_summary.py — Generate a per-model results.md for one experiment directory.

Reads the following files from <exp_dir> (all optional — gracefully skips missing):
  config.json                 training hyperparameters
  eval_results.json           train + val accuracy (our evaluator)
  official_test_results.json  accuracy matching official test script
  loss_logs.json              per-step loss values
  timing.json                 training wall-clock time

Writes:
  <exp_dir>/results.md

Usage:
    python make_summary.py --exp_dir ../experiments/r16_ep5
    python make_summary.py --exp_dir ../experiments/r16_ep5 --exp_name "Run 1 – r16 5ep"
"""

import os, json, argparse
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir",  required=True, help="Experiment directory")
    p.add_argument("--exp_name", default=None,  help="Override display name")
    return p.parse_args()


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt_pct(val):
    return f"{val*100:.2f}%" if val is not None else "N/A"


def wrong_sample_table(results, n=10):
    """Return a markdown table of up to n wrong predictions."""
    if not results:
        return "_No results available._"
    wrong = [r for r in results if not r.get("is_correct")][:n]
    if not wrong:
        return "_No wrong predictions found!_"
    lines = [
        "| True Answer | Model Prediction | Question (truncated) |",
        "|-------------|-----------------|----------------------|",
    ]
    for r in wrong:
        q    = r.get("question", "")[:60].replace("|", "\\|")
        true = r.get("true_answer", "").replace("|", "\\|")
        pred = r.get("pred_answer", "")[:40].replace("|", "\\|")
        lines.append(f"| `{true}` | `{pred}` | {q}… |")
    return "\n".join(lines)


def main():
    args   = parse_args()
    exp_dir = os.path.abspath(args.exp_dir)
    exp_name = args.exp_name or os.path.basename(exp_dir)

    # ── Load all available data ───────────────────────────────────────────────
    cfg      = load_json(os.path.join(exp_dir, "config.json"))          or {}
    eval_r   = load_json(os.path.join(exp_dir, "eval_results.json"))    or {}
    official = load_json(os.path.join(exp_dir, "official_test_results.json")) or {}
    losses   = load_json(os.path.join(exp_dir, "loss_logs.json"))       or {}
    timing   = load_json(os.path.join(exp_dir, "timing.json"))          or {}

    # ── Parse metrics ─────────────────────────────────────────────────────────
    train_acc    = eval_r.get("train_accuracy")
    val_acc      = eval_r.get("val_accuracy")
    official_acc = official.get("accuracy")
    data_source  = official.get("data_source", "unknown")

    train_losses = losses.get("train_losses", [])
    val_losses   = losses.get("val_losses", [])
    final_train_loss = train_losses[-1]["loss"] if train_losses else None
    final_val_loss   = val_losses[-1]["loss"]   if val_losses   else None

    duration     = timing.get("duration_str", "N/A")
    train_start  = timing.get("start_time",   "N/A")
    train_end    = timing.get("end_time",      "N/A")

    # ── Effective batch size ──────────────────────────────────────────────────
    bs   = cfg.get("batch_size",  "?")
    ga   = cfg.get("grad_accum",  "?")
    eff_bs = f"{bs} × {ga} = {bs*ga}" if isinstance(bs, int) and isinstance(ga, int) else "?"

    # ── Wrong prediction sample ───────────────────────────────────────────────
    val_results = eval_r.get("val_results", [])
    official_results = official.get("results", [])
    # Prefer official results for the wrong-prediction table
    sample_results = official_results if official_results else val_results

    # ── Build markdown ────────────────────────────────────────────────────────
    official_label = "Official Test Acc" if data_source == "teacher_test" else "Official Test Acc *(val-split proxy)*"

    md = f"""# Experiment: {exp_name}

> {cfg.get("description", "")}

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Accuracy Summary

| Metric | Score |
|--------|-------|
| **{official_label}** | **{fmt_pct(official_acc)}** |
| Train Accuracy (our eval) | {fmt_pct(train_acc)} |
| Val Accuracy (our eval)   | {fmt_pct(val_acc)} |

{"*Note: official test uses val split as proxy. Replace with teacher's test file for true score.*" if data_source != "teacher_test" else ""}

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| LoRA rank (r) | {cfg.get("lora_r", "?")} |
| LoRA alpha | {cfg.get("lora_alpha", "?")} |
| LoRA dropout | {cfg.get("lora_dropout", "?")} |
| Epochs | {cfg.get("epochs", "?")} |
| Per-device batch size | {bs} |
| Gradient accumulation | {ga} |
| Effective batch size | {eff_bs} |
| Learning rate | {cfg.get("lr", "?")} |
| LR scheduler | cosine |
| Max sequence length | {cfg.get("max_length", "?")} |
| 4-bit quantization | {"Yes (NF4 + double quant)" if cfg.get("load_in_4bit") else "No"} |
| Target modules | q/k/v/o proj + gate/up/down proj (all linear) |

---

## Training Loss

| Metric | Value |
|--------|-------|
| Final train loss | {final_train_loss if final_train_loss is not None else "N/A"} |
| Final val loss   | {final_val_loss   if final_val_loss   is not None else "N/A"} |
| Training start   | {train_start} |
| Training end     | {train_end} |
| Duration         | {duration} |

![Loss Curve](loss_curve.png)

---

## Wrong Predictions (sample from val/test split)

{wrong_sample_table(sample_results, n=10)}

---

## Files in This Experiment

| File | Description |
|------|-------------|
| `adapter_model.safetensors` | Trained LoRA adapter weights |
| `adapter_config.json` | LoRA configuration |
| `config.json` | Training hyperparameters |
| `eval_results.json` | Train + val accuracy (our evaluator) |
| `official_test_results.json` | Accuracy matching official test script |
| `loss_logs.json` | Per-step loss values |
| `loss_curve.png` | Loss curve plot |
| `experiment.log` | Full training + eval terminal output |
| `results.md` | This file |
"""

    out_path = os.path.join(exp_dir, "results.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"[make_summary] results.md written → {out_path}")


if __name__ == "__main__":
    main()
