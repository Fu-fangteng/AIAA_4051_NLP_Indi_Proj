# Llama 2-7B LoRA 微调项目实现指南

> AIAA 4051 Individual Project | Deadline: April 15, 23:55

---

## Step 0：准备工作

申请 **Diandong 平台 GPU 资源**，确保有至少 16GB VRAM 的 GPU 可用。

---

## Step 1：环境配置

```bash
pip install torch transformers peft datasets accelerate bitsandbytes scipy matplotlib
```

确认 CUDA 可用：

```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))
```

---

## Step 2：项目结构

```
studentID_name/
├── studentID_name.pdf
├── studentID_name_code/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
└── studentID_name_model/
    └── (adapter files saved here)
```

---

## Step 3：下载模型

```python
# 从 ModelScope 下载
from modelscope import snapshot_download
model_dir = snapshot_download('shakechen/Llama-2-7b')
```

---

## Step 4：数据预处理 `utils.py`

```python
from datasets import load_dataset
import json

def load_and_split_dataset(data_path, train_ratio=0.9):
    # 读取数据集（根据实际格式调整）
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    return train_data, val_data

def format_prompt(sample):
    """
    Llama 2 标准 prompt 格式
    根据数据集实际字段名调整 instruction / input / output
    """
    instruction = sample.get("instruction", "")
    input_text  = sample.get("input", "")
    output      = sample.get("output", "")

    if input_text:
        prompt = (
            f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
            f"{instruction}\n\nInput: {input_text} [/INST] {output}"
        )
    else:
        prompt = (
            f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n"
            f"{instruction} [/INST] {output}"
        )
    return prompt
```

---

## Step 5：训练主脚本 `train.py`

```python
import os, json, torch, matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from utils import load_and_split_dataset, format_prompt

# ── 路径配置 ──────────────────────────────────────────────
MODEL_PATH   = "./Llama-2-7b"          # ModelScope 下载路径
DATA_PATH    = "./dataset.json"         # Canvas 下载的数据集
OUTPUT_DIR   = "./studentID_name_model"
LOG_PATH     = "./logs.json"

# ── 1. 加载 Tokenizer & 模型 ──────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",              # 自动分配 GPU
    trust_remote_code=True
)
model.config.use_cache = False

# ── 2. LoRA 配置 ──────────────────────────────────────────
lora_config = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = 8,               # rank，可尝试 8 / 16
    lora_alpha   = 32,              # alpha，通常 = 2 * r
    lora_dropout = 0.05,
    bias         = "none",
    target_modules = ["q_proj", "v_proj"]   # Llama 2 标准目标层
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # 确认可训练参数量

# ── 3. 数据集准备 ─────────────────────────────────────────
train_raw, val_raw = load_and_split_dataset(DATA_PATH, train_ratio=0.9)

def tokenize(samples):
    prompts = [format_prompt(s) for s in samples]
    return tokenizer(
        prompts,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

train_dataset = Dataset.from_list(train_raw).map(
    lambda x: tokenize([x]), batched=False,
    remove_columns=Dataset.from_list(train_raw).column_names
)
val_dataset = Dataset.from_list(val_raw).map(
    lambda x: tokenize([x]), batched=False,
    remove_columns=Dataset.from_list(val_raw).column_names
)

# ── 4. 训练参数 ───────────────────────────────────────────
training_args = TrainingArguments(
    output_dir              = OUTPUT_DIR,
    num_train_epochs        = 3,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 4,
    gradient_accumulation_steps = 4,    # 等效 batch_size = 16
    learning_rate           = 2e-4,
    lr_scheduler_type       = "cosine",
    warmup_ratio            = 0.05,
    fp16                    = True,
    logging_steps           = 10,
    evaluation_strategy     = "epoch",
    save_strategy           = "epoch",
    load_best_model_at_end  = True,
    report_to               = "none"
)

# ── 5. 自定义 Trainer（记录 loss 曲线）────────────────────
class LoggingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_losses = []
        self.val_losses   = []

    def log(self, logs):
        super().log(logs)
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
        if "eval_loss" in logs:
            self.val_losses.append(logs["eval_loss"])

trainer = LoggingTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    data_collator   = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
)

# ── 6. 开始训练 ───────────────────────────────────────────
trainer.train()

# ── 7. 保存 LoRA Adapter ──────────────────────────────────
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Adapter saved to {OUTPUT_DIR}")

# ── 8. 绘制 Loss 曲线 ─────────────────────────────────────
plt.figure(figsize=(10, 5))
plt.plot(trainer.train_losses, label="Train Loss")
plt.plot(
    [int(len(trainer.train_losses) / len(trainer.val_losses)) * i
     for i in range(len(trainer.val_losses))],
    trainer.val_losses, label="Val Loss", marker='o'
)
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.savefig("loss_curve.png", dpi=150)
print("Loss curve saved.")
```

---

## Step 6：评估脚本 `evaluate.py`

```python
import torch, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import load_and_split_dataset, format_prompt

MODEL_PATH   = "./Llama-2-7b"
ADAPTER_PATH = "./studentID_name_model"
DATA_PATH    = "./dataset.json"

# 加载模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

def compute_accuracy(data, split_name=""):
    correct = 0
    for sample in data:
        # 构造只含问题的 prompt（不含答案）
        question_prompt = format_prompt({**sample, "output": ""})
        inputs = tokenizer(question_prompt, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 判断答案是否出现在生成文本中
        answer = str(sample.get("output", "")).strip().lower()
        if answer in response.lower():
            correct += 1

    accuracy = correct / len(data)
    print(f"{split_name} Accuracy: {correct}/{len(data)} = {accuracy:.4f}")
    return accuracy

train_data, val_data = load_and_split_dataset(DATA_PATH, train_ratio=0.9)
train_acc = compute_accuracy(train_data, "Train")
val_acc   = compute_accuracy(val_data,   "Validation")
```

---

## Step 7：超参数参考表（填入报告）

| 超参数 | 推荐值 | 说明 |
|--------|--------|------|
| LoRA rank `r` | 8 | 越大表达能力越强，但显存占用越高 |
| LoRA alpha | 32 | 通常设为 2×r |
| LoRA dropout | 0.05 | 防止过拟合 |
| Target modules | q_proj, v_proj | Llama 2 注意力层 |
| Learning rate | 2e-4 | LoRA 微调常用值 |
| Epochs | 3 | 数据量小可适当增加 |
| Batch size | 4 × 4 = 16 | per_device × gradient_accumulation |
| Max length | 512 | 根据数据长度调整 |
| Scheduler | cosine | 收敛更稳定 |

---

## Step 8：报告所需材料 Checklist

- [ ] 训练流程文字描述
- [ ] 超参数汇总表格
- [ ] `loss_curve.png` 插入报告
- [ ] 训练集准确率（来自 `evaluate.py`）
- [ ] 验证集准确率（来自 `evaluate.py`）

---

## ⚠️ 重要提示

1. **先检查数据集字段名**，`format_prompt()` 中的字段名需与实际数据对齐
2. 如果显存不足，将 `per_device_train_batch_size` 改为 `2`，`gradient_accumulation_steps` 改为 `8`
3. 训练完成后**务必检查 adapter 文件**是否存在 `adapter_config.json` 和 `adapter_model.bin`
4. **现在只剩约2天**，建议今天立刻开始跑训练
5. 代码需理解透彻，**期末考试会考**
