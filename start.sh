#!/usr/bin/env bash

set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$APP_DIR/.streamlit.pid"
LOG_FILE="$APP_DIR/streamlit.log"

cd "$APP_DIR"

# Stop the instance previously started by this script, if its PID file exists.
if [[ -s "$PID_FILE" ]]; then
    old_pid="$(<"$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        kill "$old_pid" 2>/dev/null || true
        for _ in {1..10}; do
            kill -0 "$old_pid" 2>/dev/null || break
            sleep 0.2
        done
        kill -9 "$old_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
fi

# Also catch instances started manually or before the PID file was created.
pkill -f '[s]treamlit run .*app\.py' 2>/dev/null || true

if [[ -x "$APP_DIR/.venv/bin/streamlit" ]]; then
    streamlit_bin="$APP_DIR/.venv/bin/streamlit"
elif command -v streamlit >/dev/null 2>&1; then
    streamlit_bin="$(command -v streamlit)"
else
    echo "streamlit was not found; install dependencies first" >&2
    exit 1
fi

nohup "$streamlit_bin" run app.py >"$LOG_FILE" 2>&1 < /dev/null &
echo $! >"$PID_FILE"

echo "Started Streamlit (PID $(<"$PID_FILE")); logs: $LOG_FILE"
