#!/bin/bash
# ============================================================
#  一键启动训练脚本（在服务器上执行）
#  用法：bash run.sh
# ============================================================

# ── 路径配置（根据服务器实际路径修改） ──────────────────────
MODEL_PATH="./Llama-2-7b"        # Llama-2-7b 模型路径
DATA_PATH="../data/dataset.json"  # 数据集路径
OUTPUT_DIR="../model"             # LoRA adapter 保存目录
SESSION="nlp_train"               # tmux session 名称

# ── 检测 tmux 是否安装 ────────────────────────────────────────
if ! command -v tmux &> /dev/null; then
    echo "[WARNING] tmux not found. Training will run in foreground (unsafe for remote SSH)."
    echo "  Install tmux: sudo apt install tmux  or  conda install -c conda-forge tmux"
    echo ""
    echo "Starting training in foreground..."
    python train.py \
        --model_path  "$MODEL_PATH" \
        --data_path   "$DATA_PATH" \
        --output_dir  "$OUTPUT_DIR" \
        --batch_size  8 \
        --use_wandb \
        --load_in_4bit
    exit 0
fi

# ── 如果 session 已存在，直接 attach ─────────────────────────
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "[INFO] Session '$SESSION' already exists. Attaching..."
    tmux attach -t "$SESSION"
    exit 0
fi

# ── 创建新 tmux session 并在后台运行训练 ──────────────────────
echo "[INFO] Creating tmux session: $SESSION"
tmux new-session -d -s "$SESSION" -x 220 -y 50

# 在 tmux 中执行训练命令
tmux send-keys -t "$SESSION" "cd $(pwd) && python train.py \
    --model_path  $MODEL_PATH \
    --data_path   $DATA_PATH \
    --output_dir  $OUTPUT_DIR \
    --batch_size  8 \
    --use_wandb \
    --load_in_4bit" Enter

echo ""
echo "============================================================"
echo "  Training started in tmux session: '$SESSION'"
echo ""
echo "  Useful commands:"
echo "    tmux attach -t $SESSION    # 重新连接查看实时输出"
echo "    tmux detach                # 断开但保持训练运行 (Ctrl+B, D)"
echo "    cat $OUTPUT_DIR/loss_logs.json  # 随时查看当前 loss"
echo ""
echo "  如果用了 --use_wandb，打开浏览器查看实时曲线："
echo "    https://wandb.ai"
echo ""
echo "  断点续训（断连后重新运行）："
echo "    bash run_resume.sh"
echo "============================================================"
echo ""

# Attach to the session so user can see it immediately
tmux attach -t "$SESSION"
