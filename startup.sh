#!/usr/bin/env bash
# Launch Poker Hub, Card Detector, and Cloudflare Tunnel in separate macOS Terminal tabs

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
DETECT_CMD="source .venv/bin/activate; export HUB_WS_URL=ws://192.168.1.54:8888; python card_detect.py"
TUNNEL_CMD="pkill cloudflared || true; cloudflared tunnel run pokerhub"

# ===== Launch in Terminal.app (same window, new tabs) =====
osascript <<EOF
tell application "Terminal"
    # --- Tab 1: Poker Hub ---
    do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}"
    delay 1
    set custom title of front window to "PokerHub"

    # --- Tab 2: Card Detector ---
    tell application "System Events" to tell process "Terminal" to keystroke "t" using command down
    delay 0.5
    do script "cd ${PROJECT_DIR}; echo '=== CARD DETECTOR ==='; ${DETECT_CMD}" in selected tab of front window
    delay 1
    set custom title of selected tab of front window to "CardDetector"

    # --- Tab 3: Cloudflare Tunnel ---
    tell application "System Events" to tell process "Terminal" to keystroke "t" using command down
    delay 0.5
    do script "cd ${PROJECT_DIR}; echo '=== CLOUDFLARE TUNNEL ==='; ${TUNNEL_CMD}" in selected tab of front window
    delay 1
    set custom title of selected tab of front window to "CloudflareTunnel"

    activate
end tell
EOF

