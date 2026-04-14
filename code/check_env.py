#!/usr/bin/env python3
"""
check_env.py — Environment sanity check before training.

Runs a series of checks and prints PASS / FAIL for each one.
If --model_path is given, also loads the model and runs 2 training
steps + 1 generation to confirm the full pipeline works end-to-end.

Usage:
    # Quick check (no model download needed)
    python check_env.py --data_path ../data/dataset.json

    # Full end-to-end check (takes ~2 min, uses the real model)
    python check_env.py --model_path ./Llama-2-7b --data_path ../data/dataset.json

    # Full check with 4-bit (if VRAM < 20 GB)
    python check_env.py --model_path ./Llama-2-7b --data_path ../data/dataset.json --load_in_4bit
"""

import sys
import argparse
import traceback

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✗ FAIL{RESET}  {msg}"); _failures.append(msg)
def warn(msg):  print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{'─'*50}\n  {title}\n{'─'*50}{RESET}")

_failures = []

# ── Argument parsing ──────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   default=None,
                   help="Path to Llama-2-7b. If omitted, skips model checks.")
    p.add_argument("--data_path",    default="../data/dataset.json")
    p.add_argument("--load_in_4bit", action="store_true",
                   help="Use 4-bit quantization for the model smoke test.")
    return p.parse_args()


# ── Check 1: Imports ──────────────────────────────────────────────────────────
def check_imports():
    section("1 / 5  Package imports")
    packages = [
        ("torch",          "PyTorch"),
        ("transformers",   "Transformers"),
        ("peft",           "PEFT (LoRA)"),
        ("datasets",       "Datasets"),
        ("accelerate",     "Accelerate"),
        ("bitsandbytes",   "BitsAndBytes (4-bit)"),
        ("scipy",          "SciPy"),
        ("tqdm",           "tqdm"),
        ("matplotlib",     "Matplotlib"),
    ]
    for module, label in packages:
        try:
            m = __import__(module)
            version = getattr(m, "__version__", "?")
            ok(f"{label:<20} {version}")
        except ImportError as e:
            fail(f"{label:<20} NOT FOUND — pip install {module}")


# ── Check 2: CUDA / GPU ───────────────────────────────────────────────────────
def check_cuda():
    section("2 / 5  CUDA & GPU")
    import torch

    if not torch.cuda.is_available():
        fail("CUDA not available — training will run on CPU (extremely slow)")
        return None

    n = torch.cuda.device_count()
    ok(f"CUDA available — {n} GPU(s) detected")

    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024**3
        ok(f"GPU {i}: {props.name}  |  VRAM: {vram_gb:.1f} GB")
        if vram_gb >= 20:
            ok(f"GPU {i}: VRAM >= 20 GB → BF16 LoRA fits (no 4-bit needed)")
        elif vram_gb >= 12:
            warn(f"GPU {i}: VRAM {vram_gb:.0f} GB → use --load_in_4bit")
        else:
            fail(f"GPU {i}: VRAM {vram_gb:.0f} GB — too small for Llama-2-7B even in 4-bit")

    # BF16 support (Ada Lovelace / Ampere)
    if torch.cuda.is_bf16_supported():
        ok("BF16 supported natively (good for RTX 4090 / A100)")
    else:
        warn("BF16 not supported — training will fall back to FP16")

    # Quick tensor op
    try:
        x = torch.randn(4, 4, device="cuda")
        _ = x @ x.T
        ok("CUDA tensor op (matmul) succeeded")
    except Exception as e:
        fail(f"CUDA tensor op failed: {e}")

    return torch.cuda.get_device_properties(0).total_memory / 1024**3


# ── Check 3: Data file ────────────────────────────────────────────────────────
def check_data(data_path):
    section("3 / 5  Dataset")
    import json, os

    if not os.path.exists(data_path):
        fail(f"Dataset not found: {data_path}")
        return None

    try:
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        fail(f"Cannot parse JSON: {e}")
        return None

    ok(f"Loaded {len(data)} samples from {data_path}")

    required_keys = {"question", "correct_answer"}
    sample = data[0]
    missing = required_keys - set(sample.keys())
    if missing:
        fail(f"Missing keys in sample: {missing}")
    else:
        ok(f"Sample keys OK: {list(sample.keys())}")
        ok(f"Example Q: {sample['question'][:60]}...")
        ok(f"Example A: {sample['correct_answer']}")

    # Answer length stats
    lengths = [len(s["correct_answer"].split()) for s in data]
    avg = sum(lengths) / len(lengths)
    ok(f"Answer length: avg {avg:.1f} words, max {max(lengths)} words")

    return data


# ── Check 4: utils.py ─────────────────────────────────────────────────────────
def check_utils(data):
    section("4 / 5  utils.py")
    try:
        from utils import load_and_split_dataset, format_prompt
        ok("utils.py imports OK")
    except Exception as e:
        fail(f"Cannot import utils.py: {e}")
        return

    if data is None:
        warn("No data loaded — skipping format_prompt check")
        return

    sample = data[0]
    try:
        p_train = format_prompt(sample, include_answer=True)
        p_infer = format_prompt(sample, include_answer=False)
        ok(f"format_prompt (train): {repr(p_train[:60])}...")
        ok(f"format_prompt (infer): {repr(p_infer[:60])}...")

        # Verify inference prompt matches official test script format
        q = sample["question"].strip()
        expected = f"Question: {q} Answer:"
        if p_infer.strip() == expected.strip():
            ok("Prompt format matches official test script ✓")
        else:
            fail(f"Prompt mismatch!\n    Got     : {repr(p_infer)}\n    Expected: {repr(expected)}")
    except Exception as e:
        fail(f"format_prompt error: {e}")


# ── Check 5: Model smoke test ─────────────────────────────────────────────────
def check_model(model_path, data, vram_gb, load_in_4bit):
    section("5 / 5  Model smoke test (2 training steps + 1 generation)")

    if model_path is None:
        warn("--model_path not given — skipping model checks")
        warn("To run the full check: python check_env.py --model_path ./Llama-2-7b")
        return

    import os
    if not os.path.isdir(model_path):
        fail(f"Model directory not found: {model_path}")
        warn("Download with: python -c \"from modelscope import snapshot_download; "
             "snapshot_download('shakechen/Llama-2-7b', cache_dir='./Llama-2-7b')\"")
        return

    # Check model files exist
    has_weights = any(
        f.endswith((".bin", ".safetensors"))
        for f in os.listdir(model_path)
        if "model" in f.lower()
    )
    if not has_weights:
        # Maybe nested directory (ModelScope layout)
        for sub in os.listdir(model_path):
            subdir = os.path.join(model_path, sub)
            if os.path.isdir(subdir):
                if any(f.endswith((".bin", ".safetensors"))
                       for f in os.listdir(subdir) if "model" in f.lower()):
                    model_path = subdir
                    has_weights = True
                    ok(f"Found model weights in subdirectory: {model_path}")
                    break

    if not has_weights:
        fail(f"No model weight files found in {model_path}")
        return

    ok(f"Model directory found: {model_path}")

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

    # Tokenizer
    print("  Loading tokenizer...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.padding_side = "right"
        ok(f"Tokenizer loaded  vocab_size={tokenizer.vocab_size}")
    except Exception as e:
        fail(f"Tokenizer load failed: {e}")
        return

    # Base model
    use_4bit = load_in_4bit or (vram_gb is not None and vram_gb < 20)
    print(f"  Loading model ({'4-bit' if use_4bit else 'BF16'})...", flush=True)
    try:
        if use_4bit:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=quant,
                device_map="auto", trust_remote_code=True,
            )
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16,
                device_map="auto", trust_remote_code=True,
            )
        model.config.use_cache = False
        ok(f"Base model loaded ({'4-bit' if use_4bit else 'BF16'})")
    except Exception as e:
        fail(f"Model load failed: {e}\n{traceback.format_exc()}")
        return

    # LoRA
    try:
        lora_cfg = LoraConfig(
            task_type      = TaskType.CAUSAL_LM,
            r              = 8,
            lora_alpha     = 16,
            lora_dropout   = 0.05,
            bias           = "none",
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        trainable, total = model.get_nb_trainable_parameters()
        ok(f"LoRA applied — trainable params: {trainable:,} / {total:,} "
           f"({100*trainable/total:.2f}%)")
    except Exception as e:
        fail(f"LoRA setup failed: {e}")
        return

    # Mini training (2 steps on 4 synthetic samples)
    try:
        from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
        from datasets import Dataset

        # Build 4 minimal samples matching our training format
        mini_samples = [
            {"question": "What is the capital of France?",  "correct_answer": "Paris"},
            {"question": "What is 2 plus 2?",               "correct_answer": "four"},
            {"question": "What color is the sky?",          "correct_answer": "blue"},
            {"question": "What is water made of?",          "correct_answer": "hydrogen and oxygen"},
        ]

        def tokenize(sample):
            prompt    = f"Question: {sample['question']} Answer:"
            full_text = f"Question: {sample['question']} Answer: {sample['correct_answer']}"
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            enc = tokenizer(full_text, truncation=True, max_length=64, add_special_tokens=True)
            labels = enc["input_ids"].copy()
            for i in range(min(len(prompt_ids), len(labels))):
                labels[i] = -100
            enc["labels"] = labels
            return enc

        raw_ds   = Dataset.from_list(mini_samples)
        train_ds = raw_ds.map(
            lambda ex: tokenize({"question": ex["question"], "correct_answer": ex["correct_answer"]}),
            remove_columns=raw_ds.column_names,
        )

        train_args = TrainingArguments(
            output_dir               = "/tmp/check_env_smoke",
            num_train_epochs         = 1,
            per_device_train_batch_size = 2,
            max_steps                = 2,   # only 2 steps — quick!
            bf16                     = not use_4bit,
            fp16                     = False,
            logging_steps            = 1,
            report_to                = "none",
            save_strategy            = "no",
        )
        trainer = Trainer(
            model         = model,
            args          = train_args,
            train_dataset = train_ds,
            data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        )
        trainer.train()
        ok("2-step training run completed without errors")
    except Exception as e:
        fail(f"Training smoke test failed: {e}\n{traceback.format_exc()}")
        return

    # Generation check
    try:
        model.eval()
        tokenizer.padding_side = "left"
        prompt = "Question: What is the capital of France? Answer:"
        enc    = tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids      = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_tokens = out[0][input_ids.shape[1]:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        ok(f"Generation works — output: {repr(answer)}")
    except Exception as e:
        fail(f"Generation failed: {e}")

    # VRAM used
    try:
        used  = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        ok(f"VRAM allocated: {used:.1f} GB  |  reserved: {reserved:.1f} GB")
    except Exception:
        pass

    # Cleanup
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    print(f"\n{BOLD}{'='*50}")
    print("  Environment Check — AIAA 4051 NLP")
    print(f"{'='*50}{RESET}")

    check_imports()
    vram_gb = check_cuda()
    data    = check_data(args.data_path)
    check_utils(data)
    check_model(args.model_path, data, vram_gb, args.load_in_4bit)

    # ── Final verdict ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*50}")
    if _failures:
        print(f"  {RED}RESULT: {len(_failures)} check(s) FAILED{RESET}")
        print(f"{'='*50}{RESET}")
        for f in _failures:
            print(f"  {RED}✗{RESET} {f}")
        print()
        sys.exit(1)
    else:
        print(f"  {GREEN}RESULT: ALL CHECKS PASSED — ready to train!{RESET}")
        print(f"{'='*50}{RESET}\n")
        if args.model_path:
            print("  Next step:")
            print("    bash pipeline.sh --model_path ./Llama-2-7b --exp_name r16_ep5\n")
        else:
            print("  Run with --model_path to also test model loading:")
            print(f"    python check_env.py --model_path ./Llama-2-7b --data_path {args.data_path}\n")


if __name__ == "__main__":
    main()
