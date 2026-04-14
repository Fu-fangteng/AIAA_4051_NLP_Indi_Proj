#!/bin/bash
# ============================================================
#  断点续训脚本 — 断连后重新执行此脚本即可接着训练
# ============================================================

MODEL_PATH="./Llama-2-7b"
DATA_PATH="../data/dataset.json"
OUTPUT_DIR="../model"
SESSION="nlp_train"

# 检查是否有 checkpoint 可以恢复
CKPT_COUNT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | wc -l)
if [ "$CKPT_COUNT" -eq 0 ]; then
    echo "[WARNING] No checkpoints found in $OUTPUT_DIR. Starting from scratch."
    RESUME_FLAG=""
else
    LATEST=$(ls -td ${OUTPUT_DIR}/checkpoint-* | head -1)
    echo "[INFO] Found $CKPT_COUNT checkpoint(s). Will resume from: $LATEST"
    RESUME_FLAG="--resume"
fi

# 如果 session 仍在运行，直接 attach
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Session '$SESSION' is still running. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "[INFO] Restarting training in new tmux session: $SESSION"
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd $(pwd) && python train.py \
    --model_path  $MODEL_PATH \
    --data_path   $DATA_PATH \
    --output_dir  $OUTPUT_DIR \
    --batch_size  8 \
    --use_wandb \
    --load_in_4bit \
    $RESUME_FLAG" Enter

echo ""
echo "Training resumed. Attaching to session..."
tmux attach -t "$SESSION"
