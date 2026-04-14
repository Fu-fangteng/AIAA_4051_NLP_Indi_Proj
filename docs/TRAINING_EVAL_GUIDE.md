# Training, Evaluation & Test Guide

> AIAA 4051 Individual Project — Llama 2-7B LoRA Fine-tuning

---

## 1. Dataset Splitting

### How splits are created

The provided `dataset.json` contains all labelled samples. Because there is **no separate test set** given to students, we split the data into:

| Split | Purpose | Default ratio |
|-------|---------|--------------|
| **Train** | Update LoRA weights | 90% |
| **Val** | Monitor overfitting, pick best checkpoint | 10% |
| **Test** | Final accuracy reported to the teacher | Held out by teacher |

Splitting is done in `utils.py`:

```python
def load_and_split_dataset(data_path, train_ratio=0.9, seed=42):
    data = json.load(open(data_path))
    random.seed(seed)
    random.shuffle(data)               # deterministic with seed=42
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]
```

Key properties:
- **Fixed random seed (42)**: the same split is produced every run, so train/val accuracy numbers are comparable across experiments.
- **Shuffle before split**: avoids any ordering bias in the original file.
- **All scripts use the same function**: `train.py`, `evaluate.py`, and `run_official_test.py` all call `load_and_split_dataset` with the same default `train_ratio=0.9`, ensuring that the validation set never leaks into training.

### Should you change the split ratio?

With ~1000 samples at 90/10, you get ~100 validation samples — enough for a stable accuracy estimate. If the dataset is smaller, consider `--train_ratio 0.85` to give the val set more samples:

```bash
python train.py ... --train_ratio 0.85
python evaluate.py ... --train_ratio 0.85   # must match!
```

> **Always pass the same `--train_ratio` to every script.** If `train.py` uses 0.9 and `evaluate.py` uses 0.85, the validation set will overlap with training data and inflate accuracy.

---

## 2. Training

### What training does

`train.py` fine-tunes only the LoRA adapter (≈2–8 M parameters) while keeping the 7B base model frozen. Loss is computed **on the answer span only** — prompt tokens are masked to `-100` so they contribute zero gradient:

```
Full sequence:  "Question: What is 2+2? Answer: 4"
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  masked (labels = -100)
                                                ^  trained (label = token for "4")
```

This teaches the model to produce the exact answer string, which is what the official evaluation checks.

### Running training

```bash
cd code

# Recommended: BF16, RTX 4090
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep3_bf16

# With custom hyperparameters:
python train.py \
    --model_path  ../Llama-2-7b \
    --data_path   ../data/dataset.json \
    --output_dir  ../experiments/r8_ep5_es \
    --epochs      5 \
    --lora_r      8 \
    --lora_alpha  16 \
    --lora_dropout 0.1 \
    --lr          1e-4 \
    --weight_decay 0.01 \
    --early_stopping_patience 2 \
    --grad_ckpt
```

### Key training settings

| Parameter | Default | Why |
|-----------|---------|-----|
| `--epochs` | 3 | Small dataset — 3 epochs is usually enough before overfitting |
| `--lora_r` | 8 | Lower rank → fewer trainable params → less overfitting |
| `--lora_dropout` | 0.1 | Regularises adapter weights during training |
| `--lr` | 1e-4 | Conservative; 2e-4 tends to overfit on small data |
| `--weight_decay` | 0.01 | L2 regularisation on adapter params |
| `--early_stopping_patience` | 2 | Stop if val loss doesn't improve for 2 epochs |

### What gets saved

After training completes, `output_dir` contains:

```
experiments/r8_ep3_bf16/
├── adapter_config.json        ← LoRA config (rank, alpha, target_modules …)
├── adapter_model.safetensors  ← trained weights (submit this)
├── tokenizer_config.json
├── tokenizer.json
├── train_config.json          ← hyperparameters used (reproducibility)
├── loss_logs.json             ← per-step train & val loss
├── loss_curve.png             ← loss plot (include in report)
└── runs/                      ← TensorBoard logs
```

Only `adapter_config.json` and `adapter_model.safetensors` are required for submission.

---

## 3. Validation (During Training)

Validation runs automatically at the end of every epoch via `evaluation_strategy="epoch"`. The Trainer:

1. Runs the val split through the **same loss computation** as training (teacher-forced, not generation).
2. Logs `eval_loss` to TensorBoard and `loss_logs.json`.
3. Saves the checkpoint with the **lowest `eval_loss`** (`load_best_model_at_end=True`).
4. Triggers `EarlyStoppingCallback` if `eval_loss` has not improved for `patience` epochs.

### Checking training progress

```bash
# Live loss log (updated every 20 steps):
tail -f experiments/r8_ep3_bf16/loss_logs.json

# TensorBoard:
tensorboard --logdir experiments/r8_ep3_bf16/runs/

# Watch GPU:
watch -n 2 nvidia-smi
```

---

## 4. Post-training Evaluation (Accuracy)

After training, run `evaluate.py` to compute **generation accuracy** on both splits:

```bash
python evaluate.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/r8_ep3_bf16 \
    --data_path    ../data/dataset.json
```

This uses **greedy generation** (not teacher-forced loss) to compute accuracy:

```
For each sample:
    prompt     = "Question: {question} Answer:"
    prediction = model.generate(prompt, max_new_tokens=16, do_sample=False)
    is_correct = true_answer.lower() in prediction.lower()
```

Output is saved to `experiments/r8_ep3_bf16/eval_results.json`:

```json
{
  "train_accuracy": 0.9982,
  "val_accuracy":   0.5720,
  "train_results":  [...],
  "val_results":    [...]
}
```

> **Train accuracy vs. Val accuracy gap** indicates overfitting. A gap larger than 30 pp (e.g., 99% vs. 57%) means the model has memorised training data and needs stronger regularisation (lower rank, higher dropout, fewer epochs).

---

## 5. Official Test (Teacher-held Set)

The teacher will run your submitted adapter on a held-out test set using `individual_project_test.py` (in `docs/`). To pre-check your model against this exact logic:

```bash
# Using val split as a proxy (before teacher releases test data):
python run_official_test.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/r8_ep3_bf16 \
    --data_path    ../data/dataset.json

# After teacher releases test.json:
python run_official_test.py \
    --model_path     ../Llama-2-7b \
    --adapter_path   ../experiments/r8_ep3_bf16 \
    --test_data_path /path/to/teacher_test.json

# Exact replication of official grading environment (float16 + 4-bit):
python run_official_test.py \
    --model_path     ../Llama-2-7b \
    --adapter_path   ../experiments/r8_ep3_bf16 \
    --test_data_path /path/to/teacher_test.json \
    --load_in_4bit --fp16
```

### Why the official script uses float16 + 4-bit

`individual_project_test.py` was written with `torch.float16` and 4-bit quantization. This is a lower-precision environment than our BF16 training. Accuracy can differ by 1–3 pp. To minimize the gap:
- Train and evaluate in BF16 (our default).
- Before final submission, run `run_official_test.py --load_in_4bit --fp16` on the val split and compare the accuracy to your BF16 result.

---

## 6. Full Pipeline

For a complete automated run (train → evaluate → official test):

```bash
cd code
bash pipeline.sh \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --exp_name   r8_ep3_bf16 \
    --lora_r     8 \
    --lora_alpha 16 \
    --epochs     3 \
    --lr         1e-4
```

Results summary appears in `experiments/r8_ep3_bf16/results.md`.

---

## 7. Choosing the Best Model

If you ran multiple experiments with `sweep.py`:

```bash
python sweep.py --model_path ../Llama-2-7b --data_path ../data/dataset.json
```

Compare all runs:

```bash
python3 -c "
import json, glob
rows = []
for f in glob.glob('experiments/*/eval_results.json'):
    d = json.load(open(f))
    name = f.split('/')[1]
    rows.append((name, d.get('val_accuracy', 0), d.get('train_accuracy', 0)))
rows.sort(key=lambda x: -x[1])
print(f'{'Experiment':<25} {'Val Acc':>10} {'Train Acc':>10}')
for name, va, ta in rows:
    print(f'{name:<25} {va*100:>9.2f}% {ta*100:>9.2f}%')
"
```

Choose the experiment with the **highest val accuracy** (not train accuracy) as your submission model.

---

## 8. What to Report

The project report must include (per `PROJECT_REQUIREMENTS.txt`):

1. **Dataset splitting** — ratio used, number of train/val samples, how the split is deterministic (seed=42).
2. **Model setup** — base model, LoRA config (rank, alpha, dropout, target modules).
3. **Training procedure** — optimizer, learning rate, epochs, batch size, gradient accumulation, early stopping.
4. **Evaluation steps** — how train/val accuracy is computed (generation, not loss), what the official eval metric is.
5. **Loss curves** — include `loss_curve.png` in the report.
6. **Accuracy table** — train accuracy, val accuracy, and (if available) official test accuracy.
