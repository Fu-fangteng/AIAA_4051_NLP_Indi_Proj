# AIAA 4051 — Llama-2-7B LoRA Fine-tuning

Llama-2-7B + LoRA (PEFT) fine-tuning on a 5000-sample QA dataset.
Optimised for RTX 4090 (BF16, all-linear LoRA, answer-only loss).

---

## Quick Start (on GPU server)

```bash
# 1. Install dependencies
pip install -r code/requirements.txt

# 2. Download Llama-2-7b
python -c "from modelscope import snapshot_download; snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')"

# 3. Check environment
python code/check_env.py --model_path ./Llama-2-7b --data_path data/dataset.json

# 4. Train (single experiment, full pipeline)
bash code/pipeline.sh --model_path ./Llama-2-7b --data_path data/dataset.json --exp_name r16_ep5

# 4b. OR run multiple experiments and compare (炼丹)
python code/sweep.py --model_path ./Llama-2-7b --data_path data/dataset.json
```

Results land in `experiments/<name>/` — see `experiment.log` if SSH disconnects.

---

## File Map

### `code/` — all runnable scripts

| File | What it does |
|------|-------------|
| `train.py` | Main training script. Loads Llama-2-7B, applies LoRA, trains, saves adapter. Answer-only loss + all 7 linear layers. |
| `evaluate.py` | Post-training accuracy on train + val split. Writes `eval_results.json`. |
| `run_official_test.py` | Exact replica of the official test script with CLI-configurable paths. Pass `--test_data_path` when teacher releases the test set. |
| `sweep.py` | Multi-experiment runner. Edit `EXPERIMENTS` list, run, get a ranked accuracy table. |
| `pipeline.sh` | **Recommended entry point.** One command: train → eval → official test → `results.md`. All output tee'd to `experiment.log`. |
| `run.sh` | Shortcut: opens tmux session and calls `pipeline.sh` with default args. |
| `check_env.py` | Environment sanity check (imports, CUDA, dataset, model smoke test). Run this first. |
| `make_summary.py` | Generates `results.md` per experiment. Called automatically by pipeline. |
| `utils.py` | `load_and_split_dataset` + `format_prompt`. |
| `requirements.txt` | Python dependencies. |

### Root files

| File | What it does |
|------|-------------|
| `individual_project_test.py` | **Official grading script** (provided by teacher). Do not modify. |
| `data/dataset.json` | 5000-sample QA dataset. |
| `GPU_TRAINING_GUIDE.md` | Detailed remote server guide: SSH, tmux, 4-bit vs BF16, monitoring, result download. |
| `PROJECT_REQUIREMENTS.txt` | Original project requirements from the course. |

### `experiments/<name>/` — output of each training run

| File | What it contains |
|------|-----------------|
| `adapter_config.json` + `adapter_model.safetensors` | LoRA adapter — **submit these two** |
| `eval_results.json` | Train + val accuracy |
| `official_test_results.json` | Accuracy from official test script logic |
| `loss_logs.json` | Per-step loss values |
| `loss_curve.png` | Loss curve plot (for report) |
| `experiment.log` | Full terminal output — check this after SSH reconnect |
| `results.md` | Auto-generated human-readable summary |
| `config.json` | Hyperparameters used for this run |

---

## 4-bit vs BF16 on RTX 4090

| Mode | VRAM | Notes |
|------|------|-------|
| BF16 + `--grad_ckpt` (default) | ~16 GB | Best accuracy, fits 4090 comfortably |
| 4-bit QLoRA (`--load_in_4bit`) | ~7 GB | Faster iteration, slightly lower precision |

RTX 4090 has 24 GB — BF16 is recommended. See `GPU_TRAINING_GUIDE.md` for details.

---

## Submission Structure (from project requirements)

```
studentID_name/
├── studentID_name.pdf
├── studentID_name_code/        ← contents of code/
└── studentID_name_model/       ← adapter_config.json + adapter_model.safetensors
```
