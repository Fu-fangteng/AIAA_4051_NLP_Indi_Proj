"""
sweep.py — Multi-experiment trainer & comparator for AIAA 4051.

Trains each experiment config sequentially, evaluates each model,
and prints a ranked comparison table at the end.

Usage:
    # Run all experiments defined below
    python sweep.py --model_path ../Llama-2-7b --data_path ../data/dataset.json

    # Run only specific experiments by name
    python sweep.py --model_path ../Llama-2-7b --data_path ../data/dataset.json \\
                    --run r16_ep5 r32_ep5

    # Skip training, just re-evaluate existing experiments
    python sweep.py --model_path ../Llama-2-7b --data_path ../data/dataset.json --eval_only

Results are saved under ../experiments/<exp_name>/ and summarized in
../experiments/results_summary.json.
"""

import os, sys, json, argparse, subprocess
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRID  —  edit this to taste
# Each dict maps 1:1 to train.py CLI flags.
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    # ── Experiment 1: strong baseline (recommended first run) ──────────────
    {
        "name":         "r16_ep5",
        "description":  "All-linear LoRA r=16, 5 epochs — solid baseline",
        "lora_r":       16,
        "lora_alpha":   32,
        "epochs":       5,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           2e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    # ── Experiment 2: higher rank ──────────────────────────────────────────
    {
        "name":         "r32_ep5",
        "description":  "All-linear LoRA r=32, 5 epochs — more capacity",
        "lora_r":       32,
        "lora_alpha":   64,
        "epochs":       5,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           2e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    # ── Experiment 3: longer training ─────────────────────────────────────
    {
        "name":         "r16_ep10",
        "description":  "All-linear LoRA r=16, 10 epochs — more convergence",
        "lora_r":       16,
        "lora_alpha":   32,
        "epochs":       10,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           1e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    # ── Experiment 4: high rank + long training ────────────────────────────
    {
        "name":         "r32_ep10",
        "description":  "All-linear LoRA r=32, 10 epochs — max effort",
        "lora_r":       32,
        "lora_alpha":   64,
        "epochs":       10,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           1e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True, help="Path to base Llama-2-7b model")
    p.add_argument("--data_path",    default="../data/dataset.json")
    p.add_argument("--exp_root",     default="../experiments",
                   help="Root dir for all experiments")
    p.add_argument("--run",          nargs="*",
                   help="Only run these experiment names (default: all)")
    p.add_argument("--eval_only",    action="store_true",
                   help="Skip training; re-evaluate existing adapters")
    p.add_argument("--use_wandb",    action="store_true")
    return p.parse_args()


def config_to_flags(cfg, model_path, data_path, output_dir, use_wandb):
    """Convert an experiment dict to a train.py command-line."""
    flags = [
        sys.executable, "train.py",
        "--model_path",  model_path,
        "--data_path",   data_path,
        "--output_dir",  output_dir,
        "--lora_r",      str(cfg["lora_r"]),
        "--lora_alpha",  str(cfg["lora_alpha"]),
        "--epochs",      str(cfg["epochs"]),
        "--batch_size",  str(cfg["batch_size"]),
        "--grad_accum",  str(cfg["grad_accum"]),
        "--lr",          str(cfg["lr"]),
        "--max_length",  str(cfg["max_length"]),
    ]
    if cfg.get("load_in_4bit"):
        flags.append("--load_in_4bit")
    if use_wandb:
        flags += ["--use_wandb", "--wandb_project", f"nlp-sweep-{cfg['name']}"]
    return flags


def eval_flags(model_path, adapter_path, data_path, output_file):
    return [
        sys.executable, "evaluate.py",
        "--model_path",   model_path,
        "--adapter_path", adapter_path,
        "--data_path",    data_path,
        "--output_file",  output_file,
    ]


def load_result(result_file):
    if not os.path.exists(result_file):
        return None
    with open(result_file) as f:
        return json.load(f)


def print_summary(results):
    """Print a ranked comparison table."""
    print("\n" + "="*72)
    print(f"{'Experiment':<20} {'Train Acc':>10} {'Val Acc':>10}  Description")
    print("-"*72)
    sorted_r = sorted(results, key=lambda x: x.get("val_accuracy", 0), reverse=True)
    for r in sorted_r:
        ta = f"{r['train_accuracy']*100:.2f}%" if r.get("train_accuracy") is not None else "N/A"
        va = f"{r['val_accuracy']*100:.2f}%"   if r.get("val_accuracy")   is not None else "N/A"
        print(f"  {r['name']:<18} {ta:>10} {va:>10}  {r.get('description','')}")
    print("="*72)
    best = sorted_r[0] if sorted_r else None
    if best:
        print(f"\nBest model: {best['name']}  (val acc = {best['val_accuracy']*100:.2f}%)")
        print(f"Adapter   : {best['adapter_path']}")
    print()


def main():
    args = parse_args()
    os.makedirs(args.exp_root, exist_ok=True)

    # Filter experiments if --run is specified
    exps = EXPERIMENTS
    if args.run:
        exps = [e for e in EXPERIMENTS if e["name"] in args.run]
        if not exps:
            print(f"No experiments matched: {args.run}")
            sys.exit(1)

    summary_path = os.path.join(args.exp_root, "results_summary.json")
    all_results  = []

    for cfg in exps:
        name       = cfg["name"]
        output_dir = os.path.join(args.exp_root, name)
        result_file= os.path.join(output_dir, "eval_results.json")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  Experiment: {name}")
        print(f"  {cfg.get('description','')}")
        print(f"  Output dir: {output_dir}")
        print(f"{'='*60}\n")

        # Save config
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        # ── Training ──────────────────────────────────────────────────────
        if not args.eval_only:
            train_cmd = config_to_flags(cfg, args.model_path, args.data_path,
                                        output_dir, args.use_wandb)
            print(f"[TRAIN] {' '.join(train_cmd)}\n")
            t0 = datetime.now()
            ret = subprocess.run(train_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
            elapsed = datetime.now() - t0
            if ret.returncode != 0:
                print(f"[ERROR] Training failed for {name} (code {ret.returncode}). Skipping eval.")
                continue
            print(f"[DONE] Training finished in {elapsed}")

        # ── Evaluation ────────────────────────────────────────────────────
        # Check adapter exists
        adapter_ok = os.path.exists(os.path.join(output_dir, "adapter_model.safetensors")) or \
                     os.path.exists(os.path.join(output_dir, "adapter_model.bin"))
        if not adapter_ok:
            print(f"[SKIP] No adapter found in {output_dir}, skipping eval.")
            continue

        eval_cmd = eval_flags(args.model_path, output_dir, args.data_path, result_file)
        print(f"\n[EVAL] {' '.join(eval_cmd)}\n")
        subprocess.run(eval_cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

        # Collect result
        result = load_result(result_file)
        if result:
            all_results.append({
                "name":           name,
                "description":    cfg.get("description", ""),
                "adapter_path":   output_dir,
                "train_accuracy": result.get("train_accuracy"),
                "val_accuracy":   result.get("val_accuracy"),
                "config":         cfg,
            })

    # ── Summary ───────────────────────────────────────────────────────────────
    if all_results:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSummary saved → {summary_path}")
        print_summary(all_results)
    else:
        print("\nNo results to summarize.")


if __name__ == "__main__":
    main()
