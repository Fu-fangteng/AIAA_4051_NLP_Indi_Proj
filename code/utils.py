import json
import random
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split


# Known conflicting annotations in the dataset — unified ground truth
CONFLICT_RESOLUTIONS = {
    "Where are protons and neutrons located?": "nucleus",
    "What is the main function of the cardiovascular system?": "transporting substances around the body",
    "What is the simplest life cycle?": "haploid life cycle",
    "What is the basic unit of matter?": "atom",
    "What occurs when a parent cell splits into two identical daughter cells of the same size?": "binary fission",
    "What is the first part of the large intestine, where wastes enter from the small intestine?": "cecum",
}


def load_and_clean_data(data_path):
    """
    Load dataset.json, remove noise, and resolve annotation conflicts.

    Cleaning steps:
      1. Drop samples with null/empty question or answer
      2. Drop questions shorter than 3 characters
      3. Resolve 6 known conflicting annotations with unified answers
    """
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_count = len(data)

    # 1. Remove null / empty fields
    data = [
        d for d in data
        if d.get("question") and d.get("correct_answer")
        and d["question"].strip() and d["correct_answer"].strip()
    ]

    # 2. Remove questions shorter than 3 characters
    data = [d for d in data if len(d["question"].strip()) >= 3]

    # Diagnostic stats
    q_counts = Counter(d["question"] for d in data)
    dup_q    = sum(1 for c in q_counts.values() if c > 1)
    qa_map   = defaultdict(set)
    for d in data:
        qa_map[d["question"]].add(d["correct_answer"])
    conflict_count = sum(1 for ans in qa_map.values() if len(ans) > 1)

    # 3. Resolve annotation conflicts
    for d in data:
        if d["question"] in CONFLICT_RESOLUTIONS:
            d["correct_answer"] = CONFLICT_RESOLUTIONS[d["question"]]

    print(f"Dataset: {original_count} → {len(data)} samples "
          f"(dup_questions={dup_q}, resolved_conflicts={conflict_count})")
    return data


def load_and_split_dataset(data_path, train_ratio=0.9, seed=42):
    """
    Load, clean, and split into train / val sets.
    train_ratio=0.9  → 90% train, 10% val
    Dataset format: [{"question": "...", "correct_answer": "..."}, ...]
    """
    data = load_and_clean_data(data_path)

    # sklearn's train_test_split gives a reproducible, stratified-compatible split
    val_ratio = 1.0 - train_ratio
    train_data, val_data = train_test_split(
        data, test_size=val_ratio, random_state=seed
    )

    print(f"Split: {len(train_data)} train | {len(val_data)} val")
    return train_data, val_data


def format_prompt(question, answer=None):
    """
    Build prompt string.  Must match the official test script exactly:
        prompt = f"Question: {question} Answer:"

    During training  (answer given) → full sequence for loss computation.
    During inference (answer=None)  → question-only prefix for generation.
    """
    question = question.strip()
    if answer is not None:
        return f"Question: {question} Answer: {answer.strip()}"
    return f"Question: {question} Answer:"
