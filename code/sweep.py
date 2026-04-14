"""
sweep.py — Multi-experiment trainer & comparator for AIAA 4051.

Trains each experiment config sequentially.  For each experiment:
  1. Runs train.py           → adapter saved to experiments/<name>/
  2. Runs evaluate.py        → eval_results.json  (train + val)
  3. Runs run_official_test.py → official_test_results.json
  4. Runs make_summary.py    → results.md

All stdout + stderr is tee'd to experiments/<name>/experiment.log
so SSH disconnects never lose output.

At the end, prints a ranked comparison table and writes
experiments/results_summary.json.

Usage:
    # Run all 4 predefined experiments
    python sweep.py --model_path ../Llama-2-7b --data_path ../data/dataset.json

    # Run specific experiments
    python sweep.py --model_path ../Llama-2-7b --run r16_ep5 r32_ep5

    # Re-evaluate existing experiments (no retraining)
    python sweep.py --model_path ../Llama-2-7b --eval_only

    # With official teacher test data
    python sweep.py --model_path ../Llama-2-7b \\
                    --test_data_path /path/to/teacher_test.json
"""

import os, sys, json, argparse, subprocess
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT GRID — edit freely to add / remove configs
# Keys map 1:1 to train.py CLI flags.
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "name":         "r16_ep5",
        "description":  "All-linear LoRA r=16, 5 epochs — solid baseline",
        "lora_r":       16,
        "lora_alpha":   32,
        "lora_dropout": 0.05,
        "epochs":       5,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           2e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    {
        "name":         "r32_ep5",
        "description":  "All-linear LoRA r=32, 5 epochs — more capacity",
        "lora_r":       32,
        "lora_alpha":   64,
        "lora_dropout": 0.05,
        "epochs":       5,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           2e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    {
        "name":         "r16_ep10",
        "description":  "All-linear LoRA r=16, 10 epochs — more convergence",
        "lora_r":       16,
        "lora_alpha":   32,
        "lora_dropout": 0.05,
        "epochs":       10,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           1e-4,
        "max_length":   256,
        "load_in_4bit": True,
    },
    {
        "name":         "r32_ep10",
        "description":  "All-linear LoRA r=32, 10 epochs — max effort",
        "lora_r":       32,
        "lora_alpha":   64,
        "lora_dropout": 0.05,
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
    p.add_argument("--model_path",     required=True)
    p.add_argument("--data_path",      default="../data/dataset.json")
    p.add_argument("--test_data_path", default=None,
                   help="Official teacher test JSON (overrides val-split proxy)")
    p.add_argument("--exp_root",       default="../experiments")
    p.add_argument("--run",            nargs="*",
                   help="Only run named experiments (default: all)")
    p.add_argument("--eval_only",      action="store_true",
                   help="Skip training, re-evaluate existing adapters")
    p.add_argument("--use_wandb",      action="store_true")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_logged(cmd, log_path, cwd=None):
    """
    Run *cmd* as a subprocess.
    All stdout + stderr is written to the terminal in real-time AND
    appended to *log_path* (like `tee -a`).
    Returns the exit code.
    """
    with open(log_path, "a", buffering=1) as log:
        log.write(f"\n{_ts()} CMD: {' '.join(str(c) for c in cmd)}\n")
        log.write("─" * 60 + "\n")
        log.flush()

        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            bufsize=1,
            text=True,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        proc.wait()
    return proc.returncode


def log(msg, log_path):
    line = f"{_ts()} {msg}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Command builders
# ─────────────────────────────────────────────────────────────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))


def train_cmd(cfg, model_path, data_path, output_dir, use_wandb):
    cmd = [
        sys.executable, "-u", os.path.join(HERE, "train.py"),
        "--model_path",  model_path,
        "--data_path",   data_path,
        "--output_dir",  output_dir,
        "--lora_r",      str(cfg["lora_r"]),
        "--lora_alpha",  str(cfg["lora_alpha"]),
        "--lora_dropout",str(cfg.get("lora_dropout", 0.05)),
        "--epochs",      str(cfg["epochs"]),
        "--batch_size",  str(cfg["batch_size"]),
        "--grad_accum",  str(cfg["grad_accum"]),
        "--lr",          str(cfg["lr"]),
        "--max_length",  str(cfg["max_length"]),
    ]
    if cfg.get("load_in_4bit"):
        cmd.append("--load_in_4bit")
    if use_wandb:
        cmd += ["--use_wandb", "--wandb_project", f"nlp-{cfg['name']}"]
    return cmd


def eval_cmd(model_path, adapter_path, data_path, output_file):
    return [
        sys.executable, "-u", os.path.join(HERE, "evaluate.py"),
        "--model_path",   model_path,
        "--adapter_path", adapter_path,
        "--data_path",    data_path,
        "--output_file",  output_file,
    ]


def official_test_cmd(model_path, adapter_path, data_path, output_file, test_data_path=None):
    cmd = [
        sys.executable, "-u", os.path.join(HERE, "run_official_test.py"),
        "--model_path",   model_path,
        "--adapter_path", adapter_path,
        "--data_path",    data_path,
        "--output_file",  output_file,
    ]
    if test_data_path:
        cmd += ["--test_data_path", test_data_path]
    return cmd


def summary_cmd(exp_dir, exp_name):
    return [
        sys.executable, "-u", os.path.join(HERE, "make_summary.py"),
        "--exp_dir",  exp_dir,
        "--exp_name", exp_name,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Results helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def print_table(all_results):
    print("\n" + "=" * 80)
    print(f"  {'Experiment':<20} {'Official':>10} {'Train':>10} {'Val':>10}  Description")
    print("-" * 80)
    sorted_r = sorted(all_results, key=lambda x: x.get("official_accuracy") or 0, reverse=True)
    for r in sorted_r:
        off = f"{r['official_accuracy']*100:.2f}%" if r.get("official_accuracy") is not None else "N/A"
        ta  = f"{r['train_accuracy']*100:.2f}%"    if r.get("train_accuracy")    is not None else "N/A"
        va  = f"{r['val_accuracy']*100:.2f}%"      if r.get("val_accuracy")      is not None else "N/A"
        print(f"  {r['name']:<20} {off:>10} {ta:>10} {va:>10}  {r.get('description','')}")
    print("=" * 80)
    best = sorted_r[0] if sorted_r else None
    if best:
        acc = best.get("official_accuracy")
        print(f"\n  Best: {best['name']}  (official acc = {acc*100:.2f}%)" if acc else f"\n  Best: {best['name']}")
        print(f"  Path: {best['adapter_path']}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.exp_root, exist_ok=True)

    # Sweep-level log (captures the inter-experiment progress)
    sweep_log = os.path.join(args.exp_root, "sweep.log")

    exps = EXPERIMENTS
    if args.run:
        exps = [e for e in EXPERIMENTS if e["name"] in args.run]
        if not exps:
            print(f"No experiments matched: {args.run}")
            sys.exit(1)

    log(f"Sweep starting — {len(exps)} experiment(s): {[e['name'] for e in exps]}", sweep_log)

    summary_path = os.path.join(args.exp_root, "results_summary.json")
    all_results  = []

    for cfg in exps:
        name       = cfg["name"]
        exp_dir    = os.path.join(args.exp_root, name)
        exp_log    = os.path.join(exp_dir, "experiment.log")
        os.makedirs(exp_dir, exist_ok=True)

        # Save config immediately
        with open(os.path.join(exp_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)

        log(f"{'='*60}", sweep_log)
        log(f"Experiment: {name}  —  {cfg.get('description','')}", sweep_log)
        log(f"Output dir: {exp_dir}", sweep_log)
        log(f"Log file  : {exp_log}", sweep_log)

        # Write experiment header to its own log
        with open(exp_log, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Experiment : {name}\n")
            f.write(f"Started    : {_ts()}\n")
            f.write(f"Config     : {json.dumps(cfg)}\n")
            f.write(f"{'='*60}\n")

        t_start = datetime.now()

        # ── 1. Train ─────────────────────────────────────────────────────────
        if not args.eval_only:
            log(f"[{name}] Step 1/4 — Training...", sweep_log)
            ret = run_logged(
                train_cmd(cfg, args.model_path, args.data_path, exp_dir, args.use_wandb),
                exp_log, cwd=HERE
            )
            elapsed = datetime.now() - t_start
            if ret != 0:
                log(f"[{name}] Training FAILED (exit {ret}). Skipping.", sweep_log)
                continue
            log(f"[{name}] Training done in {elapsed}", sweep_log)

            # Save timing
            with open(os.path.join(exp_dir, "timing.json"), "w") as f:
                json.dump({
                    "start_time":   t_start.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "duration_sec": int(elapsed.total_seconds()),
                    "duration_str": str(elapsed).split(".")[0],
                }, f, indent=2)
        else:
            log(f"[{name}] --eval_only: skipping training.", sweep_log)

        # ── Check adapter ─────────────────────────────────────────────────────
        has_adapter = (os.path.exists(os.path.join(exp_dir, "adapter_model.safetensors")) or
                       os.path.exists(os.path.join(exp_dir, "adapter_model.bin")))
        if not has_adapter:
            log(f"[{name}] No adapter found — skipping eval.", sweep_log)
            continue

        # ── 2. Internal eval (train + val) ────────────────────────────────────
        log(f"[{name}] Step 2/4 — Internal eval...", sweep_log)
        eval_result_file = os.path.join(exp_dir, "eval_results.json")
        run_logged(
            eval_cmd(args.model_path, exp_dir, args.data_path, eval_result_file),
            exp_log, cwd=HERE
        )

        # ── 3. Official test ──────────────────────────────────────────────────
        log(f"[{name}] Step 3/4 — Official test eval...", sweep_log)
        official_result_file = os.path.join(exp_dir, "official_test_results.json")
        run_logged(
            official_test_cmd(args.model_path, exp_dir, args.data_path,
                              official_result_file, args.test_data_path),
            exp_log, cwd=HERE
        )

        # ── 4. Make results.md ────────────────────────────────────────────────
        log(f"[{name}] Step 4/4 — Generating results.md...", sweep_log)
        run_logged(summary_cmd(exp_dir, name), exp_log, cwd=HERE)

        # ── Collect results for comparison table ──────────────────────────────
        ev  = load_json(eval_result_file)
        off = load_json(official_result_file)
        entry = {
            "name":              name,
            "description":       cfg.get("description", ""),
            "adapter_path":      exp_dir,
            "train_accuracy":    ev.get("train_accuracy")  if ev  else None,
            "val_accuracy":      ev.get("val_accuracy")    if ev  else None,
            "official_accuracy": off.get("accuracy")        if off else None,
            "official_source":   off.get("data_source")    if off else None,
            "config":            cfg,
        }
        all_results.append(entry)
        log(f"[{name}] official={fmt(entry['official_accuracy'])} "
            f"train={fmt(entry['train_accuracy'])} val={fmt(entry['val_accuracy'])}", sweep_log)

    # ── Final comparison table ────────────────────────────────────────────────
    if all_results:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        log(f"\nSummary saved → {summary_path}", sweep_log)
        print_table(all_results)
    else:
        log("No results to summarize.", sweep_log)

    log(f"Sweep complete.", sweep_log)


def fmt(val):
    return f"{val*100:.2f}%" if val is not None else "N/A"


if __name__ == "__main__":
    main()
