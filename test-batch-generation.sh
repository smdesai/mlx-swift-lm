#!/bin/zsh
#
# Test harness for batch generation across supported model architectures (≤7B).
# Tests both single-prompt and batch (multi-prompt) generation for each model.
#
# Usage:
#   ./test-batch-generation.sh              # Run all models
#   ./test-batch-generation.sh llama qwen3  # Run specific model types only
#

set -euo pipefail

BINARY=".build/release/BatchGenerate"
MAX_TOKENS=200
SYSTEM_PROMPT="Answer the users questions concisely."
SINGLE_PROMPT="Why is the sky blue?"
BATCH_PROMPT='Why is the sky blue?\nWhy are apples red?'
LOG_DIR="test-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${LOG_DIR}/summary_${TIMESTAMP}.txt"

# ── Model registry: parallel arrays (type → model-id) ────────────────────────

MODEL_TYPES=(
    llama
    phi3
    gemma
    gemma2
    gemma3_text
    gemma3n
    qwen2
    qwen3
    granite
    bitnet
    smollm3
    ernie4_5
    lfm2
    exaone4
    lille
    openelm
    olmo2
    olmoe
    mimo
    internlm2
    falcon
    #jamba
    nanochat
    granite_4_0
)

MODEL_IDS=(
    "mlx-community/Llama-3.2-1B-Instruct-4bit"
    "mlx-community/Phi-3.5-mini-instruct-4bit"
    "mlx-community/quantized-gemma-2b-it"
    "mlx-community/gemma-2-2b-it-4bit"
    "mlx-community/gemma-3-1b-it-qat-4bit"
    "mlx-community/gemma-3n-E2B-it-lm-4bit"
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    "mlx-community/Qwen3-0.6B-4bit"
    "mlx-community/granite-3.3-2b-instruct-4bit"
    "mlx-community/bitnet-b1.58-2B-4T-4bit"
    "mlx-community/SmolLM3-3B-4bit"
    "mlx-community/ERNIE-4.5-0.3B-PT-bf16-ft"
    "mlx-community/LFM2-1.2B-4bit"
    "mlx-community/exaone-4.0-1.2b-4bit"
    "mlx-community/lille-130m-instruct-bf16"
    "mlx-community/OpenELM-270M-Instruct"
    "mlx-community/OLMo-2-1124-7B-Instruct-4bit"
    "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit"
    "mlx-community/MiMo-7B-SFT-4bit"
    "mlx-community/internlm2_5-7b-chat-4bit"
    "mlx-community/Falcon3-1B-Instruct-4bit"
    #"mlx-community/AI21-Jamba-Reasoning-3B-bf16"
    "dnakov/nanochat-d20-mlx"
    "mlx-community/granite-4.0-1b-4bit"
)

# Models that need --no-think
NO_THINK_MODELS=" qwen3 "

# ── Helpers ───────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

pass_count=0
fail_count=0
skip_count=0

# Results stored as "type:mode:result" entries
typeset -a RESULT_ENTRIES

log() { echo "${CYAN}[$(date +%H:%M:%S)]${RESET} $*"; }

get_model_id() {
    local search_type="$1"
    local i
    for i in {1..${#MODEL_TYPES[@]}}; do
        if [[ "${MODEL_TYPES[$i]}" == "$search_type" ]]; then
            echo "${MODEL_IDS[$i]}"
            return 0
        fi
    done
    return 1
}

store_result() {
    RESULT_ENTRIES+=("$1:$2:$3")
}

get_result() {
    local search_type="$1"
    local search_mode="$2"
    local entry
    for entry in "${RESULT_ENTRIES[@]}"; do
        if [[ "$entry" == "${search_type}:${search_mode}:"* ]]; then
            echo "${entry#*:*:}"
            return 0
        fi
    done
    echo "SKIP"
}

run_test() {
    local label="$1"
    local model_type="$2"
    local model_id="$3"
    local mode="$4"  # "single" or "batch"
    local log_file="$5"

    local extra_flags=""
    if [[ "$NO_THINK_MODELS" == *" ${model_type} "* ]]; then
        extra_flags="--no-think"
    fi

    local prompt
    if [[ "$mode" == "single" ]]; then
        prompt="$SINGLE_PROMPT"
    else
        prompt="$BATCH_PROMPT"
    fi

    local cmd="$BINARY --model-id $model_id --prompt \"$prompt\" --system-prompt \"$SYSTEM_PROMPT\" --max-tokens $MAX_TOKENS $extra_flags"

    echo "$ $cmd" >> "$log_file"
    echo "---" >> "$log_file"

    local exit_code=0
    gtimeout 300 $BINARY \
        --model-id "$model_id" \
        --prompt "$prompt" \
        --system-prompt "$SYSTEM_PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        ${=extra_flags} \
        >> "$log_file" 2>&1 || exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo "  ${GREEN}PASS${RESET} $label"
        pass_count=$((pass_count + 1))
        store_result "$model_type" "$mode" "PASS"
    elif [[ $exit_code -eq 124 ]]; then
        echo "  ${RED}FAIL${RESET} $label (timeout after 300s)"
        fail_count=$((fail_count + 1))
        store_result "$model_type" "$mode" "FAIL(timeout)"
    else
        echo "  ${RED}FAIL${RESET} $label (exit code: $exit_code)"
        fail_count=$((fail_count + 1))
        store_result "$model_type" "$mode" "FAIL(exit=$exit_code)"
    fi

    echo "" >> "$log_file"
    echo "=== EXIT CODE: $exit_code ===" >> "$log_file"
    echo "" >> "$log_file"
}

# ── Pre-flight checks ────────────────────────────────────────────────────────

if [[ ! -x "$BINARY" ]]; then
    echo "Error: $BINARY not found. Run: swift build -c release"
    exit 1
fi

# Check for gtimeout (from coreutils)
if ! command -v gtimeout &>/dev/null; then
    echo "Warning: gtimeout not found. Install with: brew install coreutils"
    echo "Running without timeout protection."
    gtimeout() { shift; "$@"; }
fi

mkdir -p "$LOG_DIR"

# ── Determine which models to test ───────────────────────────────────────────

typeset -a test_types

if [[ $# -gt 0 ]]; then
    test_types=("$@")
else
    # All models, sorted
    test_types=(${(o)MODEL_TYPES[@]})
fi

# ── Run tests ────────────────────────────────────────────────────────────────

total=${#test_types[@]}
current=0

echo ""
echo "${BOLD}Batch Generation Test Harness${RESET}"
echo "${BOLD}=============================${RESET}"
echo "Binary:     $BINARY"
echo "Models:     $total"
echo "Max tokens: $MAX_TOKENS"
echo "Log dir:    $LOG_DIR"
echo ""

for model_type in "${test_types[@]}"; do
    current=$((current + 1))

    model_id=$(get_model_id "$model_type" 2>/dev/null) || {
        echo "  ${YELLOW}SKIP${RESET} [$current/$total] $model_type — unknown model type"
        skip_count=$((skip_count + 1))
        continue
    }

    log_file="${LOG_DIR}/${model_type}_${TIMESTAMP}.log"

    echo ""
    log "${BOLD}[$current/$total] $model_type${RESET} → $model_id"
    echo "# Test: $model_type ($model_id)" > "$log_file"
    echo "# Date: $(date)" >> "$log_file"
    echo "" >> "$log_file"

    # Single prompt test
    echo "== SINGLE PROMPT ==" >> "$log_file"
    run_test "single-prompt" "$model_type" "$model_id" "single" "$log_file"

    # Batch prompt test
    echo "== BATCH PROMPT ==" >> "$log_file"
    run_test "batch-prompt" "$model_type" "$model_id" "batch" "$log_file"
done

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "${BOLD}═══════════════════════════════════════${RESET}"
echo "${BOLD}Summary${RESET}"
echo "${BOLD}═══════════════════════════════════════${RESET}"
echo ""

# Print results table
printf "%-15s %-12s %-12s\n" "MODEL TYPE" "SINGLE" "BATCH"
printf "%-15s %-12s %-12s\n" "──────────" "──────" "─────"

for model_type in "${test_types[@]}"; do
    model_id=$(get_model_id "$model_type" 2>/dev/null) || continue

    single_result=$(get_result "$model_type" "single")
    batch_result=$(get_result "$model_type" "batch")

    # Colorize
    if [[ "$single_result" == "PASS" ]]; then
        single_display="${GREEN}PASS${RESET}"
    else
        single_display="${RED}${single_result}${RESET}"
    fi
    if [[ "$batch_result" == "PASS" ]]; then
        batch_display="${GREEN}PASS${RESET}"
    else
        batch_display="${RED}${batch_result}${RESET}"
    fi

    printf "%-15s %-23b %-23b\n" "$model_type" "$single_display" "$batch_display"
done

echo ""
echo "${GREEN}Passed: $pass_count${RESET}  ${RED}Failed: $fail_count${RESET}  ${YELLOW}Skipped: $skip_count${RESET}"
echo ""

# Save summary to file
{
    echo "Batch Generation Test Summary"
    echo "Date: $(date)"
    echo "Binary: $BINARY"
    echo ""
    printf "%-15s %-12s %-12s\n" "MODEL TYPE" "SINGLE" "BATCH"
    printf "%-15s %-12s %-12s\n" "----------" "------" "-----"
    for model_type in "${test_types[@]}"; do
        model_id=$(get_model_id "$model_type" 2>/dev/null) || continue
        printf "%-15s %-12s %-12s\n" \
            "$model_type" \
            "$(get_result "$model_type" "single")" \
            "$(get_result "$model_type" "batch")"
    done
    echo ""
    echo "Passed: $pass_count  Failed: $fail_count  Skipped: $skip_count"
} > "$SUMMARY_FILE"

echo "Summary saved to: $SUMMARY_FILE"
echo "Detailed logs in: $LOG_DIR/"

# Exit with failure if any tests failed
[[ $fail_count -eq 0 ]]
