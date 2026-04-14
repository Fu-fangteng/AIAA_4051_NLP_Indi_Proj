# Llama 2-7B LoRA 过拟合优化指南

> 当前状况：Train Acc 99.82% / Val Acc 57.20% — 严重过拟合  
> 目标：将 Val Acc 提升至 65~75%+  
> **状态：所有高优先级修改已应用至 `train.py`（2026-04-14）**

---

## 一、问题诊断

| 指标 | 数值 | 状态 |
|------|------|------|
| Train Acc | 99.82% | 模型已"背题" |
| Val / Test Acc | 57.20% | 泛化能力差 |

**根本原因：模型容量远超数据量，训练策略偏激进。**

---

## 二、已应用的修改（train.py 当前默认值）

### ✅ 修改1：`load_best_model_at_end` Bug 已修复

训练结束保存的是**验证 loss 最低的 checkpoint**，而非最后一个（过拟合最严重）。

```python
# 当前 train.py
load_best_model_at_end  = True
evaluation_strategy     = "epoch"
save_strategy           = "epoch"   # 必须与 eval_strategy 对齐
metric_for_best_model   = "eval_loss"
greater_is_better       = False
```

### ✅ 修改2：LoRA Rank 已降低

```python
# 当前默认
--lora_r     8    # 原来 16；参数量约减半
--lora_alpha 16   # 原来 32；保持 alpha/r 比例不变
```

### ✅ 修改3：LoRA Dropout 已增加

```python
--lora_dropout 0.1   # 原来 0.05
```

### ✅ 修改4：EarlyStopping 已加入

```python
# 训练自动在 val_loss 不再下降 2 个 epoch 后停止
--early_stopping_patience 2
```

### ✅ 修改5：学习率已降低 + Weight Decay

```python
--lr           1e-4   # 原来 2e-4
--weight_decay 0.01   # 原来无
```

### ✅ 修改6：target_modules 缩减至 4 个注意力层

```python
# 当前 train.py
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
# 原来是 7 层（含 gate_proj, up_proj, down_proj）
```

---

## 三、推荐实验矩阵

| 实验名 | r | epoch | dropout | lr | weight_decay | EarlyStopping | 预期 Val Acc |
|--------|---|-------|---------|-----|--------------|---------------|-------------|
| `r8_ep3_fixed` | 8 | 3 | 0.1 | 1e-4 | 0.01 | ❌ | 65~70% |
| `r4_ep3_wd` | 4 | 3 | 0.1 | 1e-4 | 0.01 | ❌ | 68~73% |
| `r8_ep5_es` | 8 | 5* | 0.1 | 1e-4 | 0.01 | ✅ patience=2 | 70~75% |

> `r8_ep5_es` 设 epoch=5 但靠 EarlyStopping 自动提前停止，保存最优 checkpoint

**建议执行顺序：先跑 `r8_ep3_fixed` 验证方向（约 15 分钟），再跑 `r8_ep5_es`。**

---

## 四、运行命令

```bash
cd code

# 实验 1：r8, 3 epochs（先跑这个验证方向）
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep3_fixed \
    --lora_r 8 --lora_alpha 16 --epochs 3

# 实验 2：r4, 3 epochs（更强正则）
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r4_ep3_wd \
    --lora_r 4 --lora_alpha 8 --epochs 3

# 实验 3：r8, 5 epochs + EarlyStopping（最终推荐）
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../experiments/r8_ep5_es \
    --lora_r 8 --lora_alpha 16 --epochs 5
```

所有参数均已更新为上方推荐默认值，直接运行即可。

---

## 五、验证修改是否生效

训练日志中应出现：

```
# ✅ EarlyStopping 生效
"Early stopping — patience exhausted"  或  "No improvement"

# ✅ Best model 已保存
"Saving best model checkpoint to ..."
"Best model at epoch X with eval_loss=..."

# ✅ 可训练参数量（r=8 + 4 层）
trainable params: ~6M   (原来 r=16 + 7 层约 42M)
```

---

## 六、时间规划

> 截止：2026-04-15 23:55 | 剩余约 **29 小时**

| 时间段 | 任务 |
|--------|------|
| 今晚 18:30 | 启动 `r8_ep3_fixed`（约 15 min） |
| 今晚 19:00 | 查看结果，启动 `r8_ep5_es`（约 30 min） |
| 今晚 20:00 | 根据两组结果选最优，启动最终实验 |
| 明天上午 | 整理 loss 曲线、准确率数据 |
| 明天下午 | 撰写报告，整理提交文件夹 |
| 明天 22:00 | 最晚完成提交，留 buffer |

---

## 七、过拟合诊断快速对照

| 现象 | 可能原因 | 调整方向 |
|------|---------|---------|
| Train 99%+, Val < 60% | LoRA rank 过大 / epoch 过多 | 降 r，开 EarlyStopping |
| Train 90%+, Val 60~70% | 轻微过拟合，正常 | 尝试 r=4 或更大 dropout |
| Train < 80% | 欠拟合 | 升 lr 或增加 epoch |
| Val loss 震荡不收敛 | lr 过高 | 降 lr 至 5e-5 |
