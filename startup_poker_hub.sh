#!/usr/bin/env bash
# Launch Poker Hub a new Terminal window 

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"

# ===== Launch in Terminal.app (new window) =====
osascript <<EOF
tell application "Terminal"
    # Create a new window (not a tab)
    do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}"
    delay 0.5
    set custom title of front window to "PokerHub"
    activate
end tell
EOF
