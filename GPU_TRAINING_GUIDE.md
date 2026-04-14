# GPU 远程训练指引

> 适用：RTX 4090 (24 GB) 远程 SSH 服务器 / Diandong 平台

---

## 关于 4-bit 量化：4090 用不用？

| 模式 | 显存占用 | 训练速度 | 数值精度 | 推荐场景 |
|------|---------|---------|---------|---------|
| **BF16（无量化）** | ~18 GB（需开梯度检查点）| 较慢 | 最优 | **追求最高准确率** |
| **4-bit QLoRA** | ~7-8 GB | 较快 | 略低 | 显存不足 / 快速迭代 |

**RTX 4090 推荐：BF16 + 梯度检查点**
- 4090 有 24 GB 显存，BF16 LoRA 大约占 14 GB（模型） + 4 GB（激活值）= ~18 GB，装得下
- BF16 是 Ada Lovelace 架构的原生精度，LoRA 权重精度更高
- 加上 `--grad_ckpt` 可以降到约 16 GB，余量充足，不需要 4-bit

**何时用 4-bit：**
- 服务器 GPU 显存 < 20 GB（如 A100 40G 也够，V100 16G 需要 4-bit）
- 想在同一张卡上同时跑多个实验
- 快速验证代码是否正常运行

---

## Step 0：连接服务器并启动 tmux

```bash
# 1. SSH 连接服务器
ssh username@your-server-ip

# 2. 立即创建 tmux session（防止 SSH 断联丢失训练）
tmux new-session -s nlp

# 常用 tmux 快捷键：
#   Ctrl+B, D    → 断开 session（训练继续在后台跑）
#   tmux attach -t nlp  → 重新连接 session（SSH 断联后重连）
#   Ctrl+B, [    → 进入滚动模式，可查看历史输出（q 退出）
```

> **重要：** 所有训练命令必须在 tmux 里运行。如果 SSH 断联后 session 消失，训练也会中断。

---

## Step 1：环境准备

```bash
# 检查 GPU
nvidia-smi

# 安装依赖（建议在 conda 环境中）
conda create -n nlp python=3.10 -y
conda activate nlp

# 进入项目目录
cd /path/to/individual

pip install -r code/requirements.txt

# 验证 CUDA 和 PyTorch
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 2：下载 Llama-2-7b

```bash
# 方法 1：ModelScope（推荐，国内快）
pip install modelscope
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')
print('Downloaded to:', model_dir)
"
# 下载完成后，模型在 ./Llama-2-7b/shakechen/Llama-2-7b/ 或 ./Llama-2-7b/

# 确认路径
ls ./Llama-2-7b/
# 应看到: config.json  tokenizer.json  tokenizer_config.json  pytorch_model-*.bin

# 方法 2：如果服务器已有模型，直接指定路径
# 将 MODEL_PATH 改为已有路径即可
```

---

## Step 3A：单实验训练（BF16，RTX 4090 推荐）

```bash
cd code

# === 方式 1：直接调用 pipeline.sh（最完整，自动记录日志）===
bash pipeline.sh \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --exp_name   r16_ep5_bf16 \
    --lora_r     16 \
    --lora_alpha 32 \
    --epochs     5 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr         2e-4 \
    --max_length 256 \
    --grad_ckpt
    # 注意：不加 --load_in_4bit → 使用 BF16 全精度

# === 方式 2：如果想用 4-bit（保守方案）===
bash pipeline.sh \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --exp_name   r16_ep5_4bit \
    --lora_r     16 \
    --lora_alpha 32 \
    --epochs     5 \
    --batch_size 8 \
    --load_in_4bit

# === 方式 3：快捷脚本（自动 tmux）===
bash run.sh    # 默认参数，自动开 tmux session
```

pipeline.sh 自动完成：训练 → 内部评估 → 官方测试逻辑 → 生成 results.md

---

## Step 3B：多实验炼丹（推荐正式训练时使用）

```bash
cd code

# 跑 sweep.py 中定义的 4 个实验（顺序执行）
# 每个实验都有独立日志：experiments/<name>/experiment.log
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json

# 只跑部分实验
python sweep.py \
    --model_path ../Llama-2-7b \
    --run r16_ep5 r32_ep5

# 训练完成，只重新评估
python sweep.py \
    --model_path ../Llama-2-7b \
    --eval_only
```

**自定义实验配置**：编辑 `code/sweep.py` 顶部的 `EXPERIMENTS` 列表：

```python
EXPERIMENTS = [
    {
        "name":         "bf16_r16_ep5",        # 实验名（文件夹名）
        "description":  "BF16, r=16, 5 epochs",
        "lora_r":       16,
        "lora_alpha":   32,
        "lora_dropout": 0.05,
        "epochs":       5,
        "batch_size":   8,
        "grad_accum":   2,
        "lr":           2e-4,
        "max_length":   256,
        "load_in_4bit": False,    # ← BF16（4090 推荐）
    },
    # 加更多 ...
]
```

---

## Step 4：监控训练进度（SSH 断联后也能查）

```bash
# 1. 实时查看完整日志（推荐，包含所有输出）
tail -f experiments/r16_ep5/experiment.log

# 2. 只看最新 loss（每 20 步更新）
python3 -c "
import json
d = json.load(open('experiments/r16_ep5/loss_logs.json'))
tl, vl = d['train_losses'], d['val_losses']
if tl: print(f'Train  → step {tl[-1][\"step\"]}, loss {tl[-1][\"loss\"]:.4f}')
if vl: print(f'Val    → step {vl[-1][\"step\"]}, loss {vl[-1][\"loss\"]:.4f}')
"

# 3. 查看 GPU 使用情况
watch -n 2 nvidia-smi

# 4. 查看 sweep 总进度
tail -f experiments/sweep.log

# 5. 查看生成的 results.md（实验结束后）
cat experiments/r16_ep5/results.md
```

---

## Step 5：SSH 断联后恢复

```bash
# SSH 断联后重连到服务器
ssh username@your-server-ip

# 重新连接 tmux session（训练仍在跑）
tmux attach -t nlp

# 如果 session 没了（意外断联），查看日志确认训练是否完成
cat experiments/r16_ep5/experiment.log | tail -50

# 如果训练中断，使用断点续训
bash run_resume.sh

# 或手动续训
bash pipeline.sh \
    --model_path ../Llama-2-7b \
    --exp_name   r16_ep5 \
    --lora_r 16 --epochs 5 --load_in_4bit \
    --resume
```

---

## Step 6：用教师测试集评估（发放后）

```bash
cd code

# 教师发放测试 JSON 后，直接评估最好的模型
python run_official_test.py \
    --model_path     ../Llama-2-7b \
    --adapter_path   ../experiments/r32_ep10 \
    --test_data_path /path/to/teacher_test.json

# 或通过 pipeline 重新跑完整评估（生成新 results.md）
bash pipeline.sh \
    --model_path     ../Llama-2-7b \
    --exp_name       r32_ep10 \
    --test_data_path /path/to/teacher_test.json \
    --eval_only
```

---

## Step 7：下载结果到本地（在本地 Mac 执行）

```bash
# 下载整个 experiments 目录（含 results.md、eval_results.json 等）
rsync -avz --exclude='checkpoint-*' --exclude='runs/' \
    username@server:/path/to/individual/experiments/ \
    ~/Downloads/experiments/

# 只下载最好模型的 adapter 文件
rsync -avz \
    username@server:/path/to/individual/experiments/r32_ep10/ \
    ~/Downloads/adapter/

# 也可用 scp
scp username@server:/path/to/individual/experiments/r16_ep5/results.md ./
```

---

## Step 8：准备提交文件

按照项目要求，提交结构为：

```
studentID_name/                    ← 顶层文件夹
├── studentID_name.pdf             ← PDF 报告
├── studentID_name_code/           ← 源代码
│   ├── train.py
│   ├── evaluate.py
│   ├── run_official_test.py
│   ├── utils.py
│   └── requirements.txt
└── studentID_name_model/          ← 只需 adapter 文件（不含基座模型）
    ├── adapter_config.json
    └── adapter_model.safetensors
```

快速准备：

```bash
# 在服务器上运行（替换 ID 和 NAME）
STUDENT_ID="12345678"
NAME="YourName"
FOLDER="${STUDENT_ID}_${NAME}"
BEST_EXP="r32_ep10"   # 换成你最好的实验名

mkdir -p "$FOLDER/${FOLDER}_code" "$FOLDER/${FOLDER}_model"

# 复制源码
cp code/train.py code/evaluate.py code/utils.py \
   code/run_official_test.py code/requirements.txt \
   "$FOLDER/${FOLDER}_code/"

# 复制 adapter（只要这两个文件）
cp experiments/$BEST_EXP/adapter_config.json \
   experiments/$BEST_EXP/adapter_model.safetensors \
   "$FOLDER/${FOLDER}_model/"

echo "Submission folder ready: $FOLDER/"
echo "Now add your PDF: ${FOLDER}/${FOLDER}.pdf"
```

---

## 常见问题

**Q: OOM（显存不足）怎么办？**
```bash
# 方案 1：开启梯度检查点（省 3-4 GB，速度慢 ~30%）
bash pipeline.sh ... --grad_ckpt

# 方案 2：改用 4-bit 量化
bash pipeline.sh ... --load_in_4bit

# 方案 3：减小 batch_size
bash pipeline.sh ... --batch_size 4 --grad_accum 4
```

**Q: 训练速度太慢怎么办？**
```bash
# 检查是否真的在用 GPU
python -c "import torch; print(torch.cuda.current_device())"

# 查看 GPU 利用率
nvidia-smi dmon -s u

# 如果 GPU 利用率低，增大 batch_size 或减小 max_length
```

**Q: 能否同时跑多个实验？**
```bash
# 同一张 GPU 不能同时跑（显存不够）
# 用 CUDA_VISIBLE_DEVICES 区分 GPU（如果有多卡）
CUDA_VISIBLE_DEVICES=0 bash pipeline.sh --exp_name exp_a ... &
CUDA_VISIBLE_DEVICES=1 bash pipeline.sh --exp_name exp_b ...
```

**Q: 如何查看所有实验的对比结果？**
```bash
python3 -c "
import json
try:
    data = json.load(open('experiments/results_summary.json'))
    print(f'{'Name':<20} {'Official':>10} {'Val':>10}')
    for r in sorted(data, key=lambda x: x.get('official_accuracy') or 0, reverse=True):
        off = f\"{r['official_accuracy']*100:.2f}%\" if r.get('official_accuracy') else 'N/A'
        val = f\"{r['val_accuracy']*100:.2f}%\" if r.get('val_accuracy') else 'N/A'
        print(f\"{r['name']:<20} {off:>10} {val:>10}\")
except: print('No results_summary.json yet.')
"
```
