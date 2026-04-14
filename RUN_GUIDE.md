# 项目运行指南

## 目录结构

```
individual/
├── data/
│   └── dataset.json          # 数据集（5000条 QA）
├── code/
│   ├── utils.py              # 数据加载 & prompt格式化
│   ├── train.py              # 训练主脚本
│   ├── evaluate.py           # 评估脚本
│   └── requirements.txt      # 依赖列表
├── model/                    # 训练后保存 LoRA adapter 的目录
└── individual_project_test.py # 官方测试脚本（教师使用）
```

---

## Step 1：上传到服务器后，安装依赖

```bash
pip install -r code/requirements.txt
```

---

## Step 2：下载 Llama-2-7b 模型（在服务器上执行）

```python
from modelscope import snapshot_download
model_dir = snapshot_download('shakechen/Llama-2-7b')
print(model_dir)   # 记下这个路径，填入下面的 --model_path
```

或者直接指定下载目录：

```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')"
```

---

## Step 3：训练

```bash
cd code

# RTX 4090（24GB VRAM）推荐配置：QLoRA 4-bit + bfloat16
# bfloat16 在 Ada Lovelace 上原生支持，数值更稳定
python train.py \
    --model_path ../Llama-2-7b \
    --data_path  ../data/dataset.json \
    --output_dir ../model \
    --epochs 3 \
    --batch_size 8 \
    --grad_accum 4 \
    --lr 2e-4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --load_in_4bit

# 如果使用其他 GPU（VRAM < 16GB），同样加 --load_in_4bit 即可
# 如果 VRAM >= 24GB 且不需要量化，去掉 --load_in_4bit
```

训练完成后，`model/` 目录下会生成：
- `adapter_config.json`
- `adapter_model.bin` 或 `adapter_model.safetensors`
- `loss_curve.png`（训练/验证 loss 曲线图）
- `loss_logs.json`（原始 loss 数值）

---

## Step 4：评估（计算准确率）

```bash
cd code

python evaluate.py \
    --model_path   ../Llama-2-7b \
    --adapter_path ../model \
    --data_path    ../data/dataset.json
```

输出示例：
```
Train Accuracy:      4500/4500 = 0.7823 (78.23%)
Validation Accuracy:  500/500  = 0.7460 (74.60%)
```

评估结果保存在 `model/eval_results.json`。

---

## Step 5：确认官方测试脚本可用

```bash
# 修改 individual_project_test.py 中的路径
base_model_path = "../Llama-2-7b"
adapter_path    = "../model"
test_data_path  = "../data/dataset.json"   # 替换为教师给的测试集路径

python individual_project_test.py
```

---

## 关键超参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--lora_r` | 8 | LoRA rank，越大参数量越多 |
| `--lora_alpha` | 32 | 缩放系数，通常 = 4×r |
| `--lora_dropout` | 0.05 | 防止过拟合 |
| `--epochs` | 3 | 数据集5000条，3轮足够 |
| `--batch_size` | 8 | per device batch size（RTX 4090 / 24GB） |
| `--grad_accum` | 4 | 等效 batch = 32 |
| `--lr` | 2e-4 | LoRA 标准学习率 |
| `--max_length` | 128 | 序列最大长度（QA较短，128够用）|

---

## 提交文件结构

```
studentID_name/
├── studentID_name.pdf
├── studentID_name_code/
│   ├── utils.py
│   ├── train.py
│   ├── evaluate.py
│   └── requirements.txt
└── studentID_name_model/
    ├── adapter_config.json
    └── adapter_model.bin
```
