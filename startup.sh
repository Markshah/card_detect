#!/usr/bin/env bash
# Launch Poker Hub and Card Detector in separate macOS Terminal windows
cd "$(dirname "$0")"

# Adjust IP and serial port if needed
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
DETECT_CMD="source .venv/bin/activate; export HUB_WS_URL=ws://192.168.1.54:8888; python card_detect.py"

osascript <<EOF
tell application "Terminal"
    # First window: Poker Hub
    do script "cd $(pwd); echo '=== POKER HUB ==='; $HUB_CMD"
    delay 1
    # Second window: Card Detector
    do script "cd $(pwd); echo '=== CARD DETECTOR ==='; $DETECT_CMD"
    delay 0.5
    activate
end tell
EOF

