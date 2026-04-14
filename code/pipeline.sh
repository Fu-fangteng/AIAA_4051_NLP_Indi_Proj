#!/bin/bash
# =============================================================================
#  pipeline.sh — Complete train → eval → official_test → summary pipeline
#  All output is tee'd to <exp_dir>/experiment.log (SSH-disconnect safe).
#
#  Usage:
#    bash pipeline.sh --model_path ./Llama-2-7b [OPTIONS]
#
#  Required:
#    --model_path PATH     Path to base Llama-2-7b model directory
#
#  Optional (training):
#    --exp_name    NAME    Experiment name  (default: timestamped)
#    --data_path   PATH    Dataset JSON     (default: ../data/dataset.json)
#    --exp_root    PATH    Experiments root (default: ../experiments)
#    --lora_r      INT     LoRA rank        (default: 16)
#    --lora_alpha  INT     LoRA alpha       (default: 32)
#    --epochs      INT     Epochs           (default: 5)
#    --batch_size  INT     Per-device batch (default: 8)
#    --grad_accum  INT     Grad accumulation(default: 2)
#    --lr          FLOAT   Learning rate    (default: 2e-4)
#    --max_length  INT     Max seq length   (default: 256)
#    --load_in_4bit        Enable QLoRA 4-bit (default: ON)
#    --no_4bit             Disable QLoRA 4-bit
#    --grad_ckpt           Enable gradient checkpointing
#    --use_wandb           Enable wandb logging
#    --resume              Resume from latest checkpoint
#
#  Optional (eval only):
#    --eval_only           Skip training, go straight to eval
#    --test_data_path PATH Official teacher test JSON (overrides val-split proxy)
#
#  Examples:
#    # Train + eval with defaults (RTX 4090)
#    bash pipeline.sh --model_path ./Llama-2-7b
#
#    # Custom experiment
#    bash pipeline.sh --model_path ./Llama-2-7b --exp_name r32_ep10 \
#                     --lora_r 32 --lora_alpha 64 --epochs 10 --lr 1e-4
#
#    # Eval only on existing experiment
#    bash pipeline.sh --model_path ./Llama-2-7b \
#                     --exp_name r16_ep5 --eval_only
#
#    # With official teacher test data
#    bash pipeline.sh --model_path ./Llama-2-7b \
#                     --exp_name r16_ep5 --eval_only \
#                     --test_data_path /path/to/test.json
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL_PATH=""
DATA_PATH="../data/dataset.json"
EXP_ROOT="../experiments"
EXP_NAME=""
TEST_DATA_PATH=""

LORA_R=16
LORA_ALPHA=32
EPOCHS=5
BATCH_SIZE=8
GRAD_ACCUM=2
LR="2e-4"
MAX_LENGTH=256
# RTX 4090 (24 GB): BF16 LoRA fits (~18 GB). 4-bit is off by default.
# Add --load_in_4bit if VRAM < 20 GB or you want larger batch sizes.
LOAD_4BIT=false
# Gradient checkpointing: saves ~3 GB VRAM at ~30% speed cost.
# Recommended for BF16 on 4090 to leave comfortable headroom.
GRAD_CKPT=true
USE_WANDB=false
RESUME=false
EVAL_ONLY=false

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)    MODEL_PATH="$2";    shift 2 ;;
        --data_path)     DATA_PATH="$2";     shift 2 ;;
        --exp_root)      EXP_ROOT="$2";      shift 2 ;;
        --exp_name)      EXP_NAME="$2";      shift 2 ;;
        --test_data_path) TEST_DATA_PATH="$2"; shift 2 ;;
        --lora_r)        LORA_R="$2";        shift 2 ;;
        --lora_alpha)    LORA_ALPHA="$2";    shift 2 ;;
        --epochs)        EPOCHS="$2";        shift 2 ;;
        --batch_size)    BATCH_SIZE="$2";    shift 2 ;;
        --grad_accum)    GRAD_ACCUM="$2";    shift 2 ;;
        --lr)            LR="$2";            shift 2 ;;
        --max_length)    MAX_LENGTH="$2";    shift 2 ;;
        --no_4bit)       LOAD_4BIT=false;    shift   ;;
        --load_in_4bit)  LOAD_4BIT=true;     shift   ;;
        --grad_ckpt)     GRAD_CKPT=true;     shift   ;;
        --use_wandb)     USE_WANDB=true;     shift   ;;
        --resume)        RESUME=true;        shift   ;;
        --eval_only)     EVAL_ONLY=true;     shift   ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ -z "$MODEL_PATH" ]]; then
    echo "[ERROR] --model_path is required."
    echo "Usage: bash pipeline.sh --model_path ./Llama-2-7b [OPTIONS]"
    exit 1
fi

# ── Set experiment name and directory ─────────────────────────────────────────
if [[ -z "$EXP_NAME" ]]; then
    EXP_NAME="r${LORA_R}_ep${EPOCHS}_$(date +%Y%m%d_%H%M%S)"
fi
EXP_DIR="$EXP_ROOT/$EXP_NAME"
mkdir -p "$EXP_DIR"

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE="$EXP_DIR/experiment.log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Helper: run a command, send stdout+stderr to BOTH terminal and log file.
# Uses 'python -u' for unbuffered Python output (critical for remote servers).
run_logged() {
    # First arg is a display label, rest is the command.
    local label="$1"; shift
    echo "" | tee -a "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') ── $label" | tee -a "$LOG_FILE"
    echo "$(date '+%Y-%m-%d %H:%M:%S') CMD: $*"   | tee -a "$LOG_FILE"
    echo "──────────────────────────────────────────" | tee -a "$LOG_FILE"
    # -u = unbuffered; 2>&1 merges stderr into stdout; tee appends to log
    "$@" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

# ── Header ────────────────────────────────────────────────────────────────────
{
echo "============================================================"
echo "  AIAA 4051 NLP — Training Pipeline"
echo "  Experiment : $EXP_NAME"
echo "  Started    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Log file   : $LOG_FILE"
echo "  Model      : $MODEL_PATH"
echo "  Data       : $DATA_PATH"
echo "  Output     : $EXP_DIR"
if [[ -n "$TEST_DATA_PATH" ]]; then
echo "  Test data  : $TEST_DATA_PATH"
fi
echo "============================================================"
} | tee -a "$LOG_FILE"

PIPELINE_START=$(date +%s)

# ── Step 1: Training ──────────────────────────────────────────────────────────
if [[ "$EVAL_ONLY" == "false" ]]; then
    TRAIN_ARGS=(
        python -u "$SCRIPT_DIR/train.py"
        --model_path  "$MODEL_PATH"
        --data_path   "$DATA_PATH"
        --output_dir  "$EXP_DIR"
        --lora_r      "$LORA_R"
        --lora_alpha  "$LORA_ALPHA"
        --epochs      "$EPOCHS"
        --batch_size  "$BATCH_SIZE"
        --grad_accum  "$GRAD_ACCUM"
        --lr          "$LR"
        --max_length  "$MAX_LENGTH"
    )
    [[ "$LOAD_4BIT" == "true" ]] && TRAIN_ARGS+=(--load_in_4bit)
    [[ "$GRAD_CKPT"  == "true" ]] && TRAIN_ARGS+=(--grad_ckpt)
    [[ "$USE_WANDB"  == "true" ]] && TRAIN_ARGS+=(--use_wandb)
    [[ "$RESUME"     == "true" ]] && TRAIN_ARGS+=(--resume)

    TRAIN_START=$(date +%s)
    if ! run_logged "STEP 1/4 — Training" "${TRAIN_ARGS[@]}"; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] Training failed. Pipeline aborted." | tee -a "$LOG_FILE"
        exit 1
    fi
    TRAIN_END=$(date +%s)
    TRAIN_DURATION=$(( TRAIN_END - TRAIN_START ))
    TRAIN_DURATION_STR="$(( TRAIN_DURATION/3600 ))h $(( (TRAIN_DURATION%3600)/60 ))m $(( TRAIN_DURATION%60 ))s"

    # Save timing info for make_summary.py (use Python for cross-platform timestamp)
    python -u -c "
import json, datetime
ts_start = datetime.datetime.fromtimestamp($TRAIN_START)
ts_end   = datetime.datetime.fromtimestamp($TRAIN_END)
timing = {
    'start_time':   ts_start.strftime('%Y-%m-%d %H:%M:%S'),
    'end_time':     ts_end.strftime('%Y-%m-%d %H:%M:%S'),
    'duration_sec': $TRAIN_DURATION,
    'duration_str': '$TRAIN_DURATION_STR',
}
with open('$EXP_DIR/timing.json', 'w') as f:
    json.dump(timing, f, indent=2)
print('Timing saved.')
" 2>&1 | tee -a "$LOG_FILE"

    echo "$(date '+%Y-%m-%d %H:%M:%S') Training complete. Duration: $TRAIN_DURATION_STR" | tee -a "$LOG_FILE"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [--eval_only] Skipping training." | tee -a "$LOG_FILE"
fi

# ── Check adapter exists ───────────────────────────────────────────────────────
if [[ ! -f "$EXP_DIR/adapter_model.safetensors" && ! -f "$EXP_DIR/adapter_model.bin" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ERROR] No adapter found in $EXP_DIR" | tee -a "$LOG_FILE"
    echo "  Expected: adapter_model.safetensors or adapter_model.bin" | tee -a "$LOG_FILE"
    exit 1
fi

# ── Step 2: Internal evaluation (train + val split) ───────────────────────────
EVAL_RESULT_FILE="$EXP_DIR/eval_results.json"
EVAL_ARGS=(
    python -u "$SCRIPT_DIR/evaluate.py"
    --model_path   "$MODEL_PATH"
    --adapter_path "$EXP_DIR"
    --data_path    "$DATA_PATH"
    --output_file  "$EVAL_RESULT_FILE"
)
[[ "$LOAD_4BIT" == "true" ]] && EVAL_ARGS+=(--load_in_4bit)
if ! run_logged "STEP 2/4 — Internal Evaluation (train+val split)" "${EVAL_ARGS[@]}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] Internal eval failed. Continuing." | tee -a "$LOG_FILE"
fi

# ── Step 3: Official test script (matches individual_project_test.py) ─────────
OFFICIAL_RESULT_FILE="$EXP_DIR/official_test_results.json"
OFFICIAL_TEST_ARGS=(
    python -u "$SCRIPT_DIR/run_official_test.py"
    --model_path   "$MODEL_PATH"
    --adapter_path "$EXP_DIR"
    --data_path    "$DATA_PATH"
    --output_file  "$OFFICIAL_RESULT_FILE"
)
[[ "$LOAD_4BIT" == "true" ]] && OFFICIAL_TEST_ARGS+=(--load_in_4bit)
[[ -n "$TEST_DATA_PATH" ]] && OFFICIAL_TEST_ARGS+=(--test_data_path "$TEST_DATA_PATH")

if ! run_logged "STEP 3/4 — Official Test Evaluation" "${OFFICIAL_TEST_ARGS[@]}"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] Official test eval failed. Continuing." | tee -a "$LOG_FILE"
fi

# ── Step 4: Generate results.md ───────────────────────────────────────────────
if ! run_logged "STEP 4/4 — Generating results.md" \
    python -u "$SCRIPT_DIR/make_summary.py" \
        --exp_dir  "$EXP_DIR" \
        --exp_name "$EXP_NAME"; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WARN] make_summary.py failed. Continuing." | tee -a "$LOG_FILE"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL_SEC=$(( PIPELINE_END - PIPELINE_START ))
TOTAL_STR="$(( TOTAL_SEC/3600 ))h $(( (TOTAL_SEC%3600)/60 ))m $(( TOTAL_SEC%60 ))s"

{
echo ""
echo "============================================================"
echo "  Pipeline Complete"
echo "  Experiment : $EXP_NAME"
echo "  Finished   : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  Total time : $TOTAL_STR"
echo ""
echo "  Output files:"
echo "    $EXP_DIR/adapter_model.safetensors  — LoRA adapter"
echo "    $EXP_DIR/eval_results.json          — train/val accuracy"
echo "    $EXP_DIR/official_test_results.json — official test accuracy"
echo "    $EXP_DIR/results.md                 — human-readable summary"
echo "    $EXP_DIR/experiment.log             — this full log"
echo ""

# Quick accuracy printout from JSON
python3 -c "
import json, os
e = '$EXP_DIR'
try:
    off = json.load(open(os.path.join(e,'official_test_results.json')))
    print(f'  Official Test Acc : {off[\"accuracy\"]*100:.2f}%')
except: pass
try:
    ev = json.load(open(os.path.join(e,'eval_results.json')))
    print(f'  Train Acc         : {ev[\"train_accuracy\"]*100:.2f}%')
    print(f'  Val Acc           : {ev[\"val_accuracy\"]*100:.2f}%')
except: pass
" 2>/dev/null || true

echo "============================================================"
} | tee -a "$LOG_FILE"
