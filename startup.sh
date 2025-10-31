#!/usr/bin/env bash
# Launch Poker Hub, Card Detector, and Cloudflare Tunnel in separate macOS Terminal windows

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
DETECT_CMD="source .venv/bin/activate; export HUB_WS_URL=ws://192.168.1.54:8888; python card_detect.py"
TUNNEL_CMD="pkill cloudflared || true; cloudflared tunnel run pokerhub"

# ===== Launch in Terminal.app =====
osascript <<EOF
tell application "Terminal"
    # --- Window 1: Poker Hub ---
    set hubWin to do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}"
    delay 1
    set custom title of hubWin to "PokerHub"

    # --- Window 2: Card Detector ---
    set detectWin to do script "cd ${PROJECT_DIR}; echo '=== CARD DETECTOR ==='; ${DETECT_CMD}"
    delay 1
    set custom title of detectWin to "CardDetector"

    # --- Window 3: Cloudflare Tunnel ---
    set tunnelWin to do script "cd ${PROJECT_DIR}; echo '=== CLOUDFLARE TUNNEL ==='; ${TUNNEL_CMD}"
    delay 1
    set custom title of tunnelWin to "CloudflareTunnel"

    activate
end tell
EOF

