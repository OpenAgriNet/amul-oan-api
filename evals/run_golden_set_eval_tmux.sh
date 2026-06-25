#!/usr/bin/env bash
# Run Langfuse eval on deduped GoldenSet (679 questions) inside tmux on vm5.
#
# Prereqs on server:
#   - repo at REPO_DIR with eval scripts + .env (LANGFUSE_*, EVAL_CHAT_*)
#   - eval_outputs/golden_dedup/GoldenSet_deduped.csv
#   - chat API reachable at EVAL_CHAT_BASE_URL (default http://127.0.0.1:8000)
#
# Usage:
#   chmod +x evals/run_golden_set_eval_tmux.sh
#   ./evals/run_golden_set_eval_tmux.sh
#
# Attach later:  tmux attach -t eval-golden
# Detach:        Ctrl+B then D

set -euo pipefail

SESSION_NAME="${SESSION_NAME:-eval-golden}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
INPUT_CSV="${INPUT_CSV:-$REPO_DIR/eval_outputs/golden_dedup/GoldenSet_deduped.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-$REPO_DIR/eval_outputs/golden_set_eval_full.csv}"
RAW_JSON_DIR="${RAW_JSON_DIR:-$REPO_DIR/eval_outputs/golden_langfuse_raw_json}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/eval_outputs/golden_set_eval.log}"
LANGFUSE_WAIT="${LANGFUSE_WAIT:-180}"
MAX_INDEX="${MAX_INDEX:-679}"

cd "$REPO_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install: sudo apt install tmux"
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Missing .env in $REPO_DIR"
  exit 1
fi

if [[ ! -f "$INPUT_CSV" ]]; then
  echo "Missing input CSV: $INPUT_CSV"
  echo "Run dedup first: python evals/deduplicate_golden_set.py evals/GoldenSet.csv"
  exit 1
fi

mkdir -p eval_outputs "$(dirname "$RAW_JSON_DIR")"

PYTHON="python3"
if [[ -x "$REPO_DIR/venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/venv/bin/python"
fi

RUN_CMD="cd '$REPO_DIR' && export PYTHONUNBUFFERED=1 && '$PYTHON' evals/batch_langfuse_queries.py '$INPUT_CSV' \
  --shareable-csv '$OUTPUT_CSV' \
  --output '$REPO_DIR/eval_outputs/golden_set_eval_detail.csv' \
  --raw-json-dir '$RAW_JSON_DIR' \
  --csv-columns full \
  --max-index $MAX_INDEX \
  --langfuse-wait $LANGFUSE_WAIT \
  2>&1 | tee -a '$LOG_FILE'"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "  attach: tmux attach -t $SESSION_NAME"
  echo "  kill:   tmux kill-session -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$RUN_CMD"
echo "Started tmux session: $SESSION_NAME"
echo "  input:  $INPUT_CSV"
echo "  output: $OUTPUT_CSV"
echo "  raw:    $RAW_JSON_DIR"
echo "  log:    $LOG_FILE"
echo "  attach: tmux attach -t $SESSION_NAME"
echo "  tail:   tail -f $LOG_FILE"
