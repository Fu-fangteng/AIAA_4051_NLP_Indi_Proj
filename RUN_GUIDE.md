# 项目运行指南 v3

## 文件结构

```
individual/
├── data/
│   └── dataset.json
├── code/
│   ├── train.py              # 训练（answer-only loss + all-linear LoRA）
│   ├── evaluate.py           # 内部评估（train + val 准确率）
│   ├── run_official_test.py  # 官方测试脚本封装（CLI 可配路径）
│   ├── make_summary.py       # 生成每个实验的 results.md
│   ├── sweep.py              # 多实验炼丹器
│   ├── pipeline.sh           # 单实验完整流程（推荐入口）
│   ├── run.sh                # 快捷启动（tmux + pipeline.sh）
│   ├── run_resume.sh         # 断点续训
│   └── utils.py
├── experiments/              # 每个实验独立保存
│   ├── r16_ep5/
│   │   ├── config.json                  训练超参数
│   │   ├── adapter_model.safetensors    LoRA adapter 权重
│   │   ├── eval_results.json            train/val 准确率
│   │   ├── official_test_results.json   官方测试脚本结果
│   │   ├── loss_logs.json               逐步 loss 记录
│   │   ├── loss_curve.png               loss 曲线图
│   │   ├── timing.json                  训练时间
│   │   ├── experiment.log               ★ 完整终端输出（SSH 安全）
│   │   └── results.md                   ★ 人类可读汇总
│   ├── r32_ep10/
│   │   └── ...
│   ├── sweep.log                        sweep 进度日志
│   └── results_summary.json             所有实验对比
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

## Step 3A：单实验完整流程（推荐）

```bash
cd code

# 方式 1：通过 run.sh（自动创建 tmux session，SSH 安全）
bash run.sh

# 方式 2：直接调用 pipeline.sh（完整参数控制）
bash pipeline.sh \
    --model_path ./Llama-2-7b \
    --data_path  ../data/dataset.json \
    --exp_name   r16_ep5 \
    --lora_r     16 \
    --lora_alpha 32 \
    --epochs     5 \
    --batch_size 8 \
    --grad_accum 2 \
    --lr         2e-4 \
    --max_length 256 \
    --load_in_4bit
```

`pipeline.sh` 自动执行：
1. **训练** → `adapter_model.safetensors`
2. **内部评估** → `eval_results.json`（train + val 准确率）
3. **官方测试** → `official_test_results.json`（与 `individual_project_test.py` 完全一致）
4. **生成报告** → `results.md`

**全程输出同时写入 `experiment.log`**，SSH 断联不丢日志。

---

## Step 3B：多实验炼丹（推荐正式训练）

```bash
cd code

# 跑 sweep.py 中预定义的 4 个实验
python sweep.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json

# 只跑指定实验
python sweep.py \
    --model_path ../Llama-2-7b \
    --run r16_ep5 r32_ep5

# 跳过训练，只重新评估
python sweep.py \
    --model_path ../Llama-2-7b \
    --eval_only
```

每个实验单独记录日志，结束后打印排名表：

```
================================================================================
  Experiment           Official      Train        Val    Description
--------------------------------------------------------------------------------
  r32_ep10            91.80%      95.20%      88.60%    r=32, 10 epochs
  r32_ep5             88.40%      92.10%      85.20%    r=32, 5 epochs
  r16_ep10            87.60%      91.80%      84.40%    r=16, 10 epochs
  r16_ep5             84.20%      88.60%      82.00%    r=16, 5 epochs
================================================================================
  Best: r32_ep10  (official acc = 91.80%)
```

---

## Step 4：用官方测试集评估（教师发放时）

```bash
cd code

# 有了教师的测试集 → 直接跑
python run_official_test.py \
    --model_path    ./Llama-2-7b \
    --adapter_path  ../experiments/r32_ep10 \
    --test_data_path /path/to/teacher_test.json

# 或者通过 pipeline.sh 的 --eval_only + --test_data_path
bash pipeline.sh \
    --model_path     ./Llama-2-7b \
    --exp_name       r32_ep10 \
    --test_data_path /path/to/teacher_test.json \
    --eval_only
```

---

## Step 5：查看单个实验报告

每个实验目录下都有 `results.md`：

```bash
cat ../experiments/r16_ep5/results.md
```

内容包含：准确率、超参数、loss、错误样例等。

---

## 关于日志（SSH 安全）

所有训练和评估输出同时写入 `experiment.log`：

```bash
# 实时查看日志（即使 SSH 断联后重连也能看到完整历史）
tail -f ../experiments/r16_ep5/experiment.log

# 查看 sweep 总进度
tail -f ../experiments/sweep.log

# 快速看最新 loss
python -c "
import json
d = json.load(open('../experiments/r16_ep5/loss_logs.json'))
tl = d['train_losses']
print('Last train loss:', tl[-1])
"
```

---

## 超参数说明

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--lora_r` | 16 / 32 | LoRA rank，越高效果越好，显存占用越多 |
| `--lora_alpha` | = r×2 | 通常设为 rank 的 2 倍 |
| `--epochs` | 5 ~ 10 | 数据集5000条，需要足够轮次记忆精确答案 |
| `--batch_size` | 8 | RTX 4090 24GB VRAM 可用 8 |
| `--grad_accum` | 2 | 等效 batch=16 |
| `--lr` | 2e-4（短训）/ 1e-4（长训）| 长训练用小 lr |
| `--max_length` | 256 | 覆盖所有问题+答案长度 |
| `--load_in_4bit` | 推荐 | QLoRA 4-bit，显存约 6GB，余量充足 |

---

## 核心修复（v1 → v3）

| 问题 | v1 | v3 |
|------|----|----|
| Loss 计算 | 全序列（Question+Answer）| **只对 Answer token 计算**（label masking）|
| LoRA 覆盖 | 2 层（q/v） | **7 层**（q/k/v/o + gate/up/down）|
| LoRA rank | r=8 | **r=16 起步** |
| 训练轮数 | 3 | **5~10** |
| 日志 | 终端输出（SSH断联丢失）| **tee 写入 experiment.log** |
| 评估 | 仅内部 | **内部 + 官方脚本逻辑** |
| 报告 | 无 | **每个实验自动生成 results.md** |
