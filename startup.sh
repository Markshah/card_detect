#!/usr/bin/env bash
# Launch Poker Hub and Card Detector in separate macOS Terminal tabs

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python app/poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
DETECT_CMD="source .venv/bin/activate; export HUB_WS_URL=ws://192.168.1.54:8888; python app/card_detect.py"

# ===== Launch in Terminal.app (same window, new tabs) =====
osascript <<EOF
tell application "Terminal"
    # --- Tab 1: Poker Hub (start first so Card Detector can connect) ---
    do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}"
    delay 2
    set custom title of front window to "PokerHub"
    set custom title of selected tab of front window to "PokerHub"

    # --- Tab 2: Card Detector (start after Hub is running) ---
    tell application "System Events" to tell process "Terminal" to keystroke "t" using command down
    delay 0.5
    do script "cd ${PROJECT_DIR}; echo '=== CARD DETECTOR ==='; sleep 2; ${DETECT_CMD}" in selected tab of front window
    delay 1
    set custom title of selected tab of front window to "CardDetector"

    activate
end tell
EOF

