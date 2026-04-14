#!/bin/bash
# ============================================================
#  单实验启动脚本（RTX 4090 优化配置）
#  用法：bash run.sh
# ============================================================

MODEL_PATH="./Llama-2-7b"
DATA_PATH="../data/dataset.json"
OUTPUT_DIR="../experiments/run_$(date +%Y%m%d_%H%M%S)"
SESSION="nlp_train"

if ! command -v tmux &> /dev/null; then
    echo "[WARNING] tmux not found. Running in foreground..."
    python train.py \
        --model_path  "$MODEL_PATH" \
        --data_path   "$DATA_PATH" \
        --output_dir  "$OUTPUT_DIR" \
        --lora_r      16 \
        --lora_alpha  32 \
        --epochs      5 \
        --batch_size  8 \
        --grad_accum  2 \
        --lr          2e-4 \
        --max_length  256 \
        --load_in_4bit \
        --use_wandb
    exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

echo "[INFO] Starting training in tmux session: $SESSION"
echo "[INFO] Output dir: $OUTPUT_DIR"
tmux new-session -d -s "$SESSION" -x 220 -y 50
tmux send-keys -t "$SESSION" "cd $(pwd) && python train.py \
    --model_path  $MODEL_PATH \
    --data_path   $DATA_PATH \
    --output_dir  $OUTPUT_DIR \
    --lora_r      16 \
    --lora_alpha  32 \
    --epochs      5 \
    --batch_size  8 \
    --grad_accum  2 \
    --lr          2e-4 \
    --max_length  256 \
    --load_in_4bit \
    --use_wandb" Enter

echo ""
echo "============================================================"
echo "  Training started: tmux session '$SESSION'"
echo "  Output dir      : $OUTPUT_DIR"
echo ""
echo "  Commands:"
echo "    tmux attach -t $SESSION         # 查看训练输出"
echo "    Ctrl+B, D                       # 断开但保持运行"
echo "    cat $OUTPUT_DIR/loss_logs.json  # 实时查看 loss"
echo ""
echo "  多实验炼丹："
echo "    python sweep.py --model_path $MODEL_PATH --data_path $DATA_PATH"
echo "============================================================"
tmux attach -t "$SESSION"
