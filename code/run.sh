#!/bin/bash
# ============================================================
#  run.sh — 单实验快速启动（in tmux, SSH-safe）
#  等价于手动调用 pipeline.sh，全程日志写入 experiment.log
# ============================================================

MODEL_PATH="./Llama-2-7b"
DATA_PATH="../data/dataset.json"
EXP_NAME="r16_ep5_$(date +%Y%m%d_%H%M%S)"
SESSION="nlp_train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CMD="bash $SCRIPT_DIR/pipeline.sh \
    --model_path  $MODEL_PATH \
    --data_path   $DATA_PATH \
    --exp_name    $EXP_NAME \
    --lora_r      16 \
    --lora_alpha  32 \
    --epochs      5 \
    --batch_size  8 \
    --grad_accum  2 \
    --lr          2e-4 \
    --max_length  256 \
    --grad_ckpt \
    --use_wandb"
    # No --load_in_4bit: RTX 4090 (24 GB) runs BF16 LoRA comfortably (~18 GB).
    # Add --load_in_4bit if your GPU has < 20 GB VRAM.

if ! command -v tmux &> /dev/null; then
    echo "[WARNING] tmux not found. Running in foreground (SSH disconnect = loss of session)."
    eval "$CMD"
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "[INFO] Starting pipeline in tmux session: $SESSION"
echo "[INFO] Experiment: $EXP_NAME"
echo "[INFO] Log will be at: ../experiments/$EXP_NAME/experiment.log"

tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd $SCRIPT_DIR && $CMD" Enter

echo ""
echo "==========================================================="
echo "  Session   : $SESSION"
echo "  Experiment: $EXP_NAME"
echo ""
echo "  Commands:"
echo "    tmux attach -t $SESSION                   # 查看实时输出"
echo "    Ctrl+B, D                                 # 断开但保持运行"
echo "    tail -f ../experiments/$EXP_NAME/experiment.log  # 查看日志"
echo ""
echo "  多实验炼丹:"
echo "    python sweep.py --model_path $MODEL_PATH"
echo "==========================================================="
tmux attach -t "$SESSION"
