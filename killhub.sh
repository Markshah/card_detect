#!/bin/bash
# Kill any Poker Hub process listening on a given port (default: 8888)
# Usage:
#   ./kill_hub.sh           → kills anything on port 8888
#   ./kill_hub.sh 8890      → kills anything on port 8890

PORT=${1:-8888}

echo "=== Checking for processes using port $PORT ==="

# Find all PIDs listening on that port
PIDS=$(lsof -ti tcp:$PORT)

if [ -z "$PIDS" ]; then
    echo "No process found on port $PORT."
    exit 0
fi

echo "Found process(es): $PIDS"
echo "Killing..."

# Try graceful kill first
kill $PIDS 2>/dev/null

# Wait a moment
sleep 1

# Recheck if still alive
PIDS_LEFT=$(lsof -ti tcp:$PORT)
if [ -n "$PIDS_LEFT" ]; then
    echo "Force killing remaining process(es): $PIDS_LEFT"
    kill -9 $PIDS_LEFT 2>/dev/null
else
    echo "All processes terminated."
fi

# Final confirmation
sleep 0.5
if lsof -i tcp:$PORT >/dev/null 2>&1; then
    echo "⚠️  Something is still bound to port $PORT."
else
    echo "✅ Port $PORT is now free."
fi

