import json
import random


def load_and_split_dataset(data_path, train_ratio=0.9, seed=42):
    """
    Load dataset.json and split into train / val sets.
    Dataset format: [{"question": "...", "correct_answer": "..."}, ...]
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(seed)
    random.shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    print(f"Dataset loaded: {len(data)} total | {len(train_data)} train | {len(val_data)} val")
    return train_data, val_data


def format_prompt(sample, include_answer=True):
    """
    Build the prompt string for a single sample.
    Kept consistent with the official test script:
        prompt = f"Question: {question} Answer:"

    During training  → include_answer=True  → full sequence for loss computation
    During inference → include_answer=False → question-only prefix for generation
    """
    question = sample["question"].strip()
    if include_answer:
        answer = sample["correct_answer"].strip()
        return f"Question: {question} Answer: {answer}"
    else:
        return f"Question: {question} Answer:"
