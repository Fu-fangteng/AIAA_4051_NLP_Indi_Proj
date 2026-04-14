import torch
import json
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ========== 路径配置 ==========
base_model_path = "XXXX"
adapter_path = "XXXX"
test_data_path = "XXXX"


# ========== 加载模型 ==========
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 第一步：加载基座模型
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# 第二步：加载 LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# ========== 加载测试数据 ==========
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"Test samples: {len(test_data)}")

# ========== 评估 ==========
correct = 0
results = []

for example in tqdm(test_data):
    question = example['question']
    true_answer = example['correct_answer'].strip().lower()

    prompt = f"Question: {question} Answer:"
    encoding = tokenizer(prompt, return_tensors="pt", padding=True)
    input_ids = encoding.input_ids.to(model.device)
    attention_mask = encoding.attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=16,
            do_sample=False,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_ids.shape[1]:]
    pred_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().lower()

    is_correct = true_answer in pred_answer
    if is_correct:
        correct += 1

    results.append({
        "question": question,
        "true_answer": example['correct_answer'],
        "pred_answer": pred_answer,
        "is_correct": is_correct,
    })

accuracy = 100 * correct / len(test_data)
print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_data)})")
