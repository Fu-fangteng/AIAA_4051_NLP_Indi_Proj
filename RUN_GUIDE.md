# 项目运行指南（修订版 v2）

## 核心修复说明

v1 → v2 的关键改动，直接影响准确率：

| 问题 | v1（错误）| v2（修正）|
|------|-----------|-----------|
| **Loss 计算** | 整个序列（Question+Answer）都参与 loss | **只对 Answer token 计算 loss**（prompt 部分 label=-100）|
| **LoRA 覆盖** | 仅 `q_proj + v_proj`（2个）| **全部 7 个线性层**（q/k/v/o + gate/up/down）|
| **LoRA rank** | r=8 | **r=16 起步** |
| **训练轮数** | 3 epochs | **5~10 epochs** |
| **数值精度** | float16 | **bfloat16**（RTX 4090 原生支持）|

---

## 目录结构

```
individual/
├── data/
│   └── dataset.json
├── code/
│   ├── train.py              # 训练（已修复 label masking + LoRA）
│   ├── evaluate.py           # 评估（与官方测试脚本完全对齐）
│   ├── sweep.py              # 多实验炼丹器
│   ├── utils.py
│   ├── requirements.txt
│   ├── run.sh                # 单次训练快捷脚本
│   └── run_resume.sh         # 断点续训脚本
├── experiments/              # 每个实验独立保存
│   ├── r16_ep5/
│   │   ├── config.json
│   │   ├── adapter_model.safetensors
│   │   ├── eval_results.json
│   │   └── loss_logs.json
│   └── results_summary.json  # 所有实验对比
└── RUN_GUIDE.md
```

---

## Step 1：安装依赖

```bash
pip install -r code/requirements.txt
```

---

## Step 2：下载 Llama-2-7b

```bash
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')
"
```

---

## Step 3A：单次训练（推荐先跑这个）

```bash
cd code
bash run.sh
```

或手动指定参数（RTX 4090 推荐配置）：

```bash
cd code
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r16_ep5 \
    --lora_r     16 \
    --lora_alpha 32 \
    --epochs     5 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr         2e-4 \
    --max_length 256 \
    --load_in_4bit \
    --use_wandb
```

---

## Step 3B：多实验炼丹（推荐正式训练）

```bash
cd code

# 运行所有预定义实验（r16_ep5, r32_ep5, r16_ep10, r32_ep10）
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json

# 只跑指定实验
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --run r16_ep5 r32_ep5

# 只重新评估（不重新训练）
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --eval_only
```

sweep.py 结束后自动打印对比表：

```
========================================================================
Experiment           Train Acc    Val Acc  Description
------------------------------------------------------------------------
  r32_ep10            92.40%      88.60%  All-linear LoRA r=32, 10 epochs
  r32_ep5             89.20%      86.40%  All-linear LoRA r=32, 5 epochs
  r16_ep10            88.60%      85.80%  All-linear LoRA r=16, 10 epochs
  r16_ep5             85.40%      83.20%  All-linear LoRA r=16, 5 epochs
========================================================================
Best model: r32_ep10  (val acc = 88.60%)
```

---

## Step 4：评估单个模型

```bash
cd code
python evaluate.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/r16_ep5 \
    --data_path    ../data/dataset.json
```

---

## Step 5：断点续训

```bash
cd code
bash run_resume.sh
```

---

## 超参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--lora_r` | 16 / 32 | LoRA rank，越高学习能力越强，显存占用增加 |
| `--lora_alpha` | = r×2 | 通常设为 r 的 2 倍 |
| `--epochs` | 5 ~ 10 | 数据集 5000 条，需要更多轮次充分记忆 |
| `--batch_size` | 8 | RTX 4090 24GB，可用 8 |
| `--grad_accum` | 2 | 等效 batch = 16 |
| `--lr` | 1e-4 ~ 2e-4 | 长训练用小 lr |
| `--max_length` | 256 | 覆盖所有问题长度 |
| `--load_in_4bit` | 推荐开启 | QLoRA 4-bit，显存占用约 6GB，余量充足 |

---

## 炼丹策略建议

1. **先跑 `r16_ep5`**（约 30 分钟）验证代码正常运行
2. **再跑 `r32_ep10`**（约 2 小时）取最高准确率
3. 如准确率仍不满意，在 `sweep.py` 的 `EXPERIMENTS` 列表中添加更多配置继续炼丹
4. 用最高 val_accuracy 的模型提交
