#!/usr/bin/env bash
# Re-run failed/partial Langfuse eval queries inside a tmux session on the dev server.
#
# Prereqs on server:
#   - repo at REPO_DIR (default: ~/amul-oan-api) on branch eval-pipeline
#   - Python venv with deps (langfuse, httpx, python-dotenv)
#   - .env with LANGFUSE_* and EVAL_CHAT_* (see below)
#   - eval_outputs/langfuse_raw_json/ + langfuse_eval_all_queries.csv from prior run
#   - evals/queries.csv
#
# Usage:
#   chmod +x evals/run_retry_failed_tmux.sh
#   ./evals/run_retry_failed_tmux.sh
#
# Attach later:  tmux attach -t eval-retry
# Detach:        Ctrl+B then D

set -euo pipefail

SESSION_NAME="${SESSION_NAME:-eval-retry}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/eval_outputs/batch_retry_server.log}"
LANGFUSE_WAIT="${LANGFUSE_WAIT:-180}"

cd "$REPO_DIR"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found. Install: sudo apt install tmux"
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Missing .env in $REPO_DIR"
  echo "Required: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_BASE_URL,"
  echo "          EVAL_CHAT_BASE_URL, EVAL_CHAT_JWT"
  exit 1
fi

if [[ ! -f evals/queries.csv ]]; then
  echo "Missing evals/queries.csv"
  exit 1
fi

mkdir -p eval_outputs

PYTHON="python3"
if [[ -x "$REPO_DIR/venv/bin/python" ]]; then
  PYTHON="$REPO_DIR/venv/bin/python"
fi

RUN_CMD="cd '$REPO_DIR' && export PYTHONUNBUFFERED=1 && '$PYTHON' evals/batch_langfuse_queries.py evals/queries.csv --retry-failed --langfuse-wait $LANGFUSE_WAIT --max-index 200 2>&1 | tee -a '$LOG_FILE'"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session '$SESSION_NAME' already exists."
  echo "  attach: tmux attach -t $SESSION_NAME"
  echo "  kill:   tmux kill-session -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "$RUN_CMD"
echo "Started tmux session: $SESSION_NAME"
echo "  log:    $LOG_FILE"
echo "  attach: tmux attach -t $SESSION_NAME"
echo "  tail:   tail -f $LOG_FILE"
