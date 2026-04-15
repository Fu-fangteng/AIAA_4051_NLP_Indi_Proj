# Llama 2-7B LoRA 过拟合优化指南

> 原始状况：Train Acc 99.82% / Val Acc 57.20% — 严重过拟合  
> 目标：将 Val Acc 提升至 65%+  
> **全部优化已应用至代码（2026-04-15）**

---

## 一、问题诊断

| 指标 | 数值 | 状态 |
|------|------|------|
| Train Acc | 99.82% | 模型已"背题" |
| Val / Test Acc | 57.20% | 泛化能力差 |

**根本原因（已确认两处）：**
1. **Loss masking 有 off-by-1 bug**：手动数 `prompt_len` 在 SentencePiece tokenizer 下存在 token 边界不匹配风险，导致部分 answer token 被错误 mask 成 -100，训练信号不准确
2. **超参数过于激进**：高 rank、低 dropout、保存最后而非最优 checkpoint

---

## 二、所有优化（已全部应用）

### ✅ 核心修复：SFTTrainer + DataCollatorForCompletionOnly

**这是最重要的一项。** 替换手动 token masking，改用 `trl` 的 `DataCollatorForCompletionOnly`：

```python
# 旧代码（有 off-by-1 风险）
prompt_ids = tokenizer(prompt)["input_ids"]
prompt_len = len(prompt_ids)
labels[:prompt_len] = -100   # 可能与 full_text 中的 token 边界不对齐

# 新代码（精确）
collator = DataCollatorForCompletionOnly(
    response_template=" Answer:",
    tokenizer=tokenizer,
)
# 在完整 token 序列里直接搜索 " Answer:" token 序列，mask 之前全部 token
```

### ✅ 修复 load_best_model_at_end Bug

```python
load_best_model_at_end  = True
evaluation_strategy     = "epoch"
save_strategy           = "epoch"   # 必须与 eval_strategy 对齐
metric_for_best_model   = "eval_loss"
greater_is_better       = False
```

### ✅ EarlyStoppingCallback

```python
EarlyStoppingCallback(early_stopping_patience=2)
```

### ✅ 学习率 + Weight Decay

```python
lr           = 1e-4    # 原 2e-4
weight_decay = 0.01
max_grad_norm = 1.0    # 梯度裁剪
```

### ✅ LoRA Dropout 增加（抑制过拟合）

```python
lora_dropout = 0.1    # 原 0.05
```

### ✅ target_modules 保持 4 个注意力层

```python
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

### ✅ 数据清洗（Data Cleaning）

`utils.py` 中加入：
- 删除 null/空字段
- 删除问题长度 < 3 的样本
- 解决 6 个已知冲突标注

### ✅ 数据划分调整

```python
train_ratio = 0.85   # 原 0.9
# 4250 train / 750 val（更多 val 样本，准确率估计更稳定）
```

### ✅ Prompt 格式统一

所有脚本（`train.py`, `evaluate.py`, `run_official_test.py`）统一调用 `format_prompt()`：

```python
# utils.py — 单一事实来源
def format_prompt(question, answer=None):
    question = question.strip()
    if answer is not None:
        return f"Question: {question} Answer: {answer.strip()}"
    return f"Question: {question} Answer:"
```

---

## 三、推荐实验命令

```bash
cd code

# 标准训练（推荐）
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/sft_r16_ep3 \
    --lora_r 16 --lora_alpha 32 --epochs 3

# 更激进正则（若仍过拟合）
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/sft_r8_ep5 \
    --lora_r 8 --lora_alpha 16 --epochs 5

# 训练结束后评估（val only，约 2 分钟）
python evaluate.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../experiments/sft_r16_ep3 \
    --data_path    ../data/dataset.json \
    --val_only
```

---

## 四、验证 DataCollatorForCompletionOnly 是否生效

训练开始后，日志中的 train loss 在 epoch 1 应该明显低于之前（旧代码 epoch 1 loss ≈ 0.94，新代码应 < 0.8）。如果 loss 更低，说明 masking 正确工作，模型真正在学习 answer token。

---

## 五、时间规划（截止 2026-04-15 23:55）

| 时间段 | 任务 |
|--------|------|
| 12:30 | 拉取代码，`pip install trl scikit-learn`，启动训练 |
| 13:00 | 训练结束（约 15 min），运行 evaluate.py |
| 13:30 | 查看结果，必要时启动第二个实验 |
| 下午   | 整理 loss 曲线、准确率数据 |
| 下午晚  | 撰写报告，整理提交文件夹 |
| 22:00 | 最晚完成提交 |
