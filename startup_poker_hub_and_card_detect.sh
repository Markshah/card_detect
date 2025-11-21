#!/usr/bin/env bash
# Launch Poker Hub and Card Detector in separate macOS Terminal tabs

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
DETECT_CMD="source .venv/bin/activate; export HUB_WS_URL=ws://192.168.1.54:8888; python card_detect.py"

# ===== Launch in Terminal.app (same window, new tabs) =====
osascript <<EOF
tell application "Terminal"
    # --- Tab 1: Card Detector ---
    do script "cd ${PROJECT_DIR}; echo '=== CARD DETECTOR ==='; ${DETECT_CMD}"
    delay 1
    set custom title of front window to "PokerHub"
    set custom title of selected tab of front window to "CardDetector"

    # --- Tab 2: Poker Hub (will be active since it's created last) ---
    tell application "System Events" to tell process "Terminal" to keystroke "t" using command down
    delay 0.5
    do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}" in selected tab of front window
    delay 1
    set custom title of selected tab of front window to "PokerHub"

    activate
end tell
EOF

