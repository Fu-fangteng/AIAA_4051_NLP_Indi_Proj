# GPU Remote Training Guide

> Target hardware: RTX 4090 (24 GB VRAM) — remote SSH / Diandong platform

---

## RTX 4090 Memory Planning

### VRAM breakdown — Llama 2-7B LoRA training

| Component | BF16 (no quant) | 4-bit QLoRA |
|-----------|----------------|-------------|
| Base model weights | ~13.5 GB | ~3.9 GB |
| LoRA adapter (r=8, 4 layers) | ~30 MB | ~30 MB |
| Optimizer states (AdamW) | ~0.5 GB | ~0.5 GB |
| Activations (batch=8, len=256) | ~2.5 GB | ~2.5 GB |
| Gradient checkpointing saving | −3 GB | −1 GB |
| **Total (no grad ckpt)** | **~16.5 GB** | **~7.0 GB** |
| **Total (with --grad_ckpt)** | **~13.5 GB** | **~6.0 GB** |

**Conclusion for RTX 4090:**
- BF16 without grad checkpointing: ~16.5 GB → **fits comfortably** (7.5 GB free)
- BF16 with `--grad_ckpt`: ~13.5 GB → **very comfortable** (10.5 GB free)
- 4-bit QLoRA: ~7 GB → **overkill for 4090**, only needed on ≤16 GB GPUs

### Recommended mode by GPU

| GPU | VRAM | Recommended mode |
|-----|------|-----------------|
| RTX 4090 | 24 GB | **BF16** (default, no flags needed) |
| A100 40G | 40 GB | BF16 |
| A100 80G | 80 GB | BF16, can increase batch size |
| V100 32G | 32 GB | BF16 |
| V100 16G | 16 GB | 4-bit QLoRA (`--load_in_4bit`) |
| RTX 3090 | 24 GB | BF16 (but no native BF16; minor perf loss) |
| T4 | 16 GB | 4-bit QLoRA |

> **Why BF16 over 4-bit on 4090?**
> Ada Lovelace (RTX 4090) has hardware-accelerated BF16 tensor cores. The LoRA adapter trains in full BF16 precision, which gives better gradient signal and higher final accuracy than 4-bit compute. There is no VRAM reason to use 4-bit on a 4090.

---

## Common 4-bit Error: `AttributeError: set_submodule`

```
AttributeError: 'LlamaForCausalLM' object has no attribute 'set_submodule'
```

**Root cause:** `bitsandbytes >= 0.41.0` calls `nn.Module.set_submodule()` when replacing linear layers with quantized ones during `from_pretrained()`. This method requires PyTorch ≥ 1.9.1. If the installed PyTorch is older, or if there is a version conflict, the call fails.

**Fixes (in order of preference):**

1. **Don't use 4-bit on 4090** (strongly recommended):
   ```bash
   # Remove --load_in_4bit from your command — BF16 is better on 4090
   python train.py --model_path ../Llama-2-7b ...
   ```

2. **Upgrade PyTorch** (if you must use 4-bit):
   ```bash
   pip install "torch>=2.1.0" --upgrade
   ```

3. **Pin a compatible bitsandbytes version**:
   ```bash
   pip install bitsandbytes==0.41.3
   ```

The code in `train.py`, `evaluate.py`, and `run_official_test.py` already includes a compatibility shim that patches `set_submodule` onto `nn.Module` if it is missing. If you still see this error, it means the error is happening inside PyTorch's `from_pretrained` before our patch runs — fix (1) or (2) above is the right path.

---

## Step 0: Connect and Start tmux

```bash
# 1. SSH into server
ssh username@your-server-ip

# 2. Start a tmux session immediately (keeps training alive if SSH drops)
tmux new-session -s nlp

# Key tmux commands:
#   Ctrl+B, D          → detach (training keeps running)
#   tmux attach -t nlp → re-attach after reconnecting SSH
#   Ctrl+B, [          → scroll mode (q to exit)
#   Ctrl+B, c          → new window in session
```

> **Always run training inside tmux.** If your SSH connection drops without tmux, the training process is killed.

---

## Step 1: Environment Setup

```bash
# Verify GPU is visible
nvidia-smi
# Expected: NVIDIA GeForce RTX 4090, 24564 MiB

# Create conda environment (Python 3.10 recommended)
conda create -n nlp python=3.10 -y
conda activate nlp

# Navigate to project
cd /path/to/individual

# Install dependencies
pip install -r code/requirements.txt

# Sanity check
python code/check_env.py
# Should print: torch OK, cuda OK, transformers OK, peft OK, bitsandbytes OK
```

### Version requirements (important for 4090)

| Package | Minimum | Recommended | Notes |
|---------|---------|-------------|-------|
| torch | 2.0.0 | 2.1.0+ | 2.1+ has better BF16 performance |
| transformers | 4.36.0 | 4.38.0+ | Llama 2 support |
| peft | 0.7.0 | 0.9.0+ | EarlyStopping compat |
| bitsandbytes | 0.41.0 | 0.43.0+ | 4-bit quantization |
| accelerate | 0.24.0 | 0.27.0+ | device_map support |

---

## Step 2: Download Llama 2-7B

```bash
# Method 1: ModelScope (fast from mainland China)
pip install modelscope
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')
print('Downloaded to:', model_dir)
"

# Confirm the model files exist:
ls ./Llama-2-7b/
# Must contain: config.json  tokenizer.json  tokenizer_config.json  pytorch_model-*.bin (or model.safetensors)

# Method 2: If the server already has the model at a known path
# Just pass that path as --model_path
```

---

## Step 3A: Single Experiment (BF16, RTX 4090 recommended)

```bash
cd code

# === Simplest: direct python call ===
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep3_bf16 \
    --lora_r     8 \
    --lora_alpha 16 \
    --epochs     3 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr         1e-4

# === With gradient checkpointing (saves ~3 GB, ~20% slower) ===
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep3_gradckpt \
    --lora_r     8 \
    --epochs     3 \
    --grad_ckpt

# === Full pipeline (train + eval + test, auto-generates results.md) ===
bash pipeline.sh \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --exp_name   r8_ep3_bf16 \
    --lora_r     8 \
    --lora_alpha 16 \
    --epochs     3 \
    --lr         1e-4

# === Quick run with defaults (auto-tmux) ===
bash run.sh
```

Expected VRAM usage (RTX 4090, BF16, batch=8):
- Without `--grad_ckpt`: ~16–17 GB
- With `--grad_ckpt`: ~13–14 GB

Expected training time per epoch (RTX 4090, ~900 train samples, batch=8):
- ~3–5 minutes per epoch → ~10–15 min for 3 epochs

---

## Step 3B: Multiple Experiments (Sweep)

```bash
cd code

# Run all experiments defined in sweep.py
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json

# Run only specific experiments
python sweep.py \
    --model_path ../Llama-2-7b \
    --run r8_ep3 r4_ep3

# Re-evaluate already-trained models (skip training)
python sweep.py \
    --model_path ../Llama-2-7b \
    --eval_only
```

Edit the `EXPERIMENTS` list in `sweep.py` to define your configs:

```python
EXPERIMENTS = [
    {
        "name":         "r8_ep3_bf16",
        "description":  "BF16, r=8, 3 epochs (anti-overfit defaults)",
        "lora_r":       8,
        "lora_alpha":   16,
        "lora_dropout": 0.1,
        "epochs":       3,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           1e-4,
        "weight_decay": 0.01,
        "max_length":   256,
        "load_in_4bit": False,   # BF16 for 4090
    },
    {
        "name":         "r4_ep3_bf16",
        "description":  "BF16, r=4, stronger regularisation",
        "lora_r":       4,
        "lora_alpha":   8,
        "lora_dropout": 0.1,
        "epochs":       3,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           1e-4,
        "weight_decay": 0.01,
        "max_length":   256,
        "load_in_4bit": False,
    },
]
```

---

## Step 4: Monitor Training

```bash
# Live experiment log (recommended — all output in one place)
tail -f experiments/r8_ep3_bf16/experiment.log

# Current loss values
python3 -c "
import json
d = json.load(open('experiments/r8_ep3_bf16/loss_logs.json'))
tl = d['train_losses']; vl = d['val_losses']
if tl: print(f'Train  → step {tl[-1][\"step\"]}, loss {tl[-1][\"loss\"]:.4f}')
if vl: print(f'Val    → step {vl[-1][\"step\"]}, loss {vl[-1][\"loss\"]:.4f}')
"

# Real-time GPU utilisation and VRAM
watch -n 2 nvidia-smi

# More detailed GPU stats
nvidia-smi dmon -s u

# TensorBoard
tensorboard --logdir experiments/r8_ep3_bf16/runs/ --port 6006
# Then: ssh -L 6006:localhost:6006 username@server  (on local machine)
# Open: http://localhost:6006
```

---

## Step 5: Reconnect After SSH Drop

```bash
# On local machine: reconnect SSH
ssh username@your-server-ip

# Re-attach to tmux (training is still running)
tmux attach -t nlp

# If tmux session is gone (server restarted), check if training finished
cat experiments/r8_ep3_bf16/experiment.log | tail -30

# Resume from checkpoint if training was interrupted
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep3_bf16 \
    --lora_r 8 --epochs 3 \
    --resume
# train.py --resume automatically finds the latest checkpoint in output_dir
```

---

## Step 6: Evaluate After Training

```bash
cd code

# Train + val accuracy (generation-based, matches official eval logic)
python evaluate.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/r8_ep3_bf16 \
    --data_path    ../data/dataset.json

# Official test script logic (val split as proxy, before teacher releases test set)
python run_official_test.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/r8_ep3_bf16

# After teacher releases test.json
python run_official_test.py \
    --model_path     ../Llama-2-7b \
    --adapter_path   ../experiments/r8_ep3_bf16 \
    --test_data_path /path/to/teacher_test.json
```

---

## Step 7: Download Results to Local Mac

```bash
# On your local Mac:

# Download all experiment outputs (excluding large checkpoint folders and TF event files)
rsync -avz \
    --exclude='checkpoint-*/optimizer.pt' \
    --exclude='checkpoint-*/rng_state.pth' \
    --exclude='runs/' \
    username@server:/path/to/individual/experiments/ \
    ~/Downloads/experiments/

# Download only a specific experiment's adapter + results
rsync -avz \
    username@server:/path/to/individual/experiments/r8_ep3_bf16/ \
    ~/Downloads/r8_ep3_bf16/

# Download with scp (single file)
scp username@server:/path/to/individual/experiments/r8_ep3_bf16/results.md ./
```

---

## Step 8: Prepare Submission

```bash
# On the server (replace ID and NAME):
STUDENT_ID="12345678"
NAME="YourName"
FOLDER="${STUDENT_ID}_${NAME}"
BEST_EXP="r8_ep3_bf16"   # replace with your best experiment name

mkdir -p "$FOLDER/${FOLDER}_code" "$FOLDER/${FOLDER}_model"

# Copy source code
cp code/train.py code/evaluate.py code/utils.py \
   code/run_official_test.py code/requirements.txt \
   "$FOLDER/${FOLDER}_code/"

# Copy adapter only (do NOT include base model weights)
cp experiments/$BEST_EXP/adapter_config.json \
   experiments/$BEST_EXP/adapter_model.safetensors \
   "$FOLDER/${FOLDER}_model/"

echo "Done. Add your PDF: ${FOLDER}/${FOLDER}.pdf"
ls -lh "$FOLDER/${FOLDER}_model/"
```

Submission structure:
```
12345678_YourName/
├── 12345678_YourName.pdf
├── 12345678_YourName_code/
│   ├── train.py
│   ├── evaluate.py
│   ├── run_official_test.py
│   ├── utils.py
│   └── requirements.txt
└── 12345678_YourName_model/
    ├── adapter_config.json        (~1 KB)
    └── adapter_model.safetensors  (~50–200 MB depending on rank)
```

---

## Troubleshooting

### OOM (out of memory)

```bash
# Option 1: Enable gradient checkpointing (~3 GB saving, 20% slower)
python train.py ... --grad_ckpt

# Option 2: Reduce batch size + increase grad_accum to keep effective batch same
python train.py ... --batch_size 4 --grad_accum 4

# Option 3: Reduce max_length (shorter sequences = less activation memory)
python train.py ... --max_length 128

# Option 4: 4-bit QLoRA (last resort on 4090 — accuracy trade-off)
python train.py ... --load_in_4bit
```

### GPU not used / training is CPU-bound

```bash
# Confirm CUDA is available
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# Check GPU utilisation during training (should be 80–99%)
nvidia-smi dmon -s u

# If GPU utilisation is low (<50%):
#   - Increase batch_size (more work per GPU step)
#   - Reduce max_length (shorter sequences → smaller kernel launches)
#   - Check that model loaded to GPU: print(next(model.parameters()).device)
```

### AttributeError: set_submodule

See the dedicated section at the top of this guide. Short answer: **remove `--load_in_4bit`**. The 4090 does not need it.

### Training loss not decreasing

```bash
# Check learning rate — 1e-4 is safe, 2e-4 can oscillate
# Check that bf16=True in TrainingArguments (enabled by default in train.py)
# Confirm GPU is actually being used (see above)
# Try increasing warmup_ratio to 0.1 if loss spikes early
```

### Comparing multiple experiments

```bash
python3 -c "
import json, glob, os
rows = []
for f in glob.glob('experiments/*/eval_results.json'):
    name = f.split(os.sep)[1]
    d = json.load(open(f))
    rows.append((name, d.get('val_accuracy', 0), d.get('train_accuracy', 0)))
rows.sort(key=lambda x: -x[1])
print(f'{'Experiment':<28} {'Val':>8} {'Train':>8}')
for n, v, t in rows:
    print(f'{n:<28} {v*100:>7.2f}% {t*100:>7.2f}%')
"
```
