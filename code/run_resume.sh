#!/bin/bash
# ============================================================
#  断点续训脚本 — 续训最近一次实验
# ============================================================

MODEL_PATH="./Llama-2-7b"
DATA_PATH="../data/dataset.json"
SESSION="nlp_train"

# 找到最近的实验目录
LATEST_EXP=$(ls -td ../experiments/run_* 2>/dev/null | head -1)
if [ -z "$LATEST_EXP" ]; then
    echo "[ERROR] No experiment dirs found in ../experiments/run_*"
    exit 1
fi

CKPT_COUNT=$(ls -d ${LATEST_EXP}/checkpoint-* 2>/dev/null | wc -l)
if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "[WARNING] No checkpoints in $LATEST_EXP. Starting from scratch."
    RESUME_FLAG=""
else
    LATEST_CKPT=$(ls -td ${LATEST_EXP}/checkpoint-* | head -1)
    echo "[INFO] Resuming from: $LATEST_CKPT"
    RESUME_FLAG="--resume"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Session '$SESSION' is still running. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "[INFO] Restarting training in tmux session: $SESSION"
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd $(pwd) && python train.py \
    --model_path  $MODEL_PATH \
    --data_path   $DATA_PATH \
    --output_dir  $LATEST_EXP \
    --lora_r      16 \
    --lora_alpha  32 \
    --epochs      5 \
    --batch_size  8 \
    --grad_accum  2 \
    --lr          2e-4 \
    --max_length  256 \
    --load_in_4bit \
    --use_wandb \
    $RESUME_FLAG" Enter

echo "Training resumed in $LATEST_EXP. Attaching..."
tmux attach -t "$SESSION"
