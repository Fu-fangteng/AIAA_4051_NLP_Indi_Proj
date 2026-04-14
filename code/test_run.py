"""
Smoke Test Script — run this BEFORE full training to verify the environment works.

Checks:
  1. CUDA / GPU available
  2. All packages importable
  3. Dataset loads correctly
  4. Model + LoRA loads (with 4-bit quantization)
  5. One forward pass succeeds
  6. One training step succeeds
  7. Adapter can be saved

Usage:
    python test_run.py --model_path /path/to/Llama-2-7b --data_path ../data/dataset.json
"""

import os
import sys
import argparse
import time

def check(label):
    """Simple progress printer."""
    print(f"\n{'='*55}")
    print(f"  CHECK: {label}")
    print(f"{'='*55}")

def ok(msg=""):
    print(f"  [PASS] {msg}")

def fail(msg):
    print(f"  [FAIL] {msg}")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to Llama-2-7b")
    parser.add_argument("--data_path",  default="../data/dataset.json")
    parser.add_argument("--output_dir", default="../model_test")
    return parser.parse_args()


def main():
    args = parse_args()
    t0 = time.time()

    # ── 1. CUDA check ─────────────────────────────────────────────────────────
    check("CUDA / GPU")
    import torch
    if not torch.cuda.is_available():
        fail("CUDA not available. Check your GPU driver and CUDA installation.")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"GPU: {gpu_name}  |  VRAM: {vram_gb:.1f} GB")

    # ── 2. Package imports ────────────────────────────────────────────────────
    check("Package imports")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset
        ok("transformers, peft, datasets: OK")
    except ImportError as e:
        fail(f"Import error: {e}\nRun: pip install -r requirements.txt")

    # ── 3. Dataset ────────────────────────────────────────────────────────────
    check("Dataset loading")
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from utils import load_and_split_dataset, format_prompt
        train_raw, val_raw = load_and_split_dataset(args.data_path, train_ratio=0.9)
        sample = train_raw[0]
        ok(f"Loaded {len(train_raw)+len(val_raw)} samples | Keys: {list(sample.keys())}")
        prompt = format_prompt(sample, include_answer=True)
        ok(f"Prompt example: {prompt[:80]}...")
    except Exception as e:
        fail(f"Dataset error: {e}")

    # ── 4. Tokenizer ──────────────────────────────────────────────────────────
    check("Tokenizer")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "right"
        ids = tokenizer("Hello world", return_tensors="pt")
        ok(f"Tokenizer loaded | vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        fail(f"Tokenizer error: {e}")

    # ── 5. Model loading (4-bit) ──────────────────────────────────────────────
    check("Model loading (4-bit QLoRA)")
    try:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model.config.use_cache = False
        ok("Base model loaded")
    except Exception as e:
        fail(f"Model loading error: {e}")

    # ── 6. LoRA ───────────────────────────────────────────────────────────────
    check("LoRA adapter")
    try:
        lora_config = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = 8,
            lora_alpha     = 32,
            lora_dropout   = 0.05,
            bias           = "none",
            target_modules = ["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        ok("LoRA applied")
    except Exception as e:
        fail(f"LoRA error: {e}")

    # ── 7. Mini training run (5 samples, 1 step) ──────────────────────────────
    check("Mini training run (5 samples, 1 step)")
    try:
        mini_data = train_raw[:5]

        def tokenize(s):
            enc = tokenizer(format_prompt(s, include_answer=True),
                            truncation=True, max_length=128, padding="max_length")
            enc["labels"] = enc["input_ids"].copy()
            return enc

        col_names     = list(Dataset.from_list(mini_data).column_names)
        mini_dataset  = Dataset.from_list(mini_data).map(tokenize, remove_columns=col_names)

        os.makedirs(args.output_dir, exist_ok=True)
        training_args = TrainingArguments(
            output_dir                  = args.output_dir,
            num_train_epochs            = 1,
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 1,
            max_steps                   = 2,          # only 2 steps
            fp16                        = True,
            logging_steps               = 1,
            evaluation_strategy         = "no",
            save_strategy               = "no",
            report_to                   = "none",
        )
        trainer = Trainer(
            model         = model,
            args          = training_args,
            train_dataset = mini_dataset,
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        )
        trainer.train()
        ok("2 training steps completed without error")
    except Exception as e:
        fail(f"Training step error: {e}")

    # ── 8. Save adapter ───────────────────────────────────────────────────────
    check("Save LoRA adapter")
    try:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        saved_files = os.listdir(args.output_dir)
        ok(f"Saved files: {saved_files}")
        # Clean up test output
        import shutil
        shutil.rmtree(args.output_dir)
        ok("Test output cleaned up")
    except Exception as e:
        fail(f"Save error: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*55}")
    print(f"  ALL CHECKS PASSED in {elapsed:.1f}s")
    print(f"  Environment is ready. Run full training with:")
    print(f"    bash run.sh")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()
