#!/usr/bin/env bash
# Launch Poker Hub and Cloudflare Tunnel in a new Terminal window with 2 tabs
# Poker Hub will be in the 2nd tab and will be the active tab when it launches

cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# ===== Commands =====
HUB_CMD="source .venv/bin/activate; python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire"
TUNNEL_CMD="pkill cloudflared || true; cloudflared tunnel run pokerhub"

# ===== Launch in Terminal.app (new window, 2 tabs) =====
osascript <<EOF
tell application "Terminal"
    # Create new window with first tab
    set newWindow to do script "cd ${PROJECT_DIR}; echo '=== CLOUDFLARE TUNNEL ==='; ${TUNNEL_CMD}"
    delay 1
    set custom title of front window to "PokerHub"
    set custom title of selected tab of front window to "CloudflareTunnel"

    # --- Tab 2: Poker Hub (will be selected/active) ---
    tell application "System Events" to tell process "Terminal" to keystroke "t" using command down
    delay 0.5
    do script "cd ${PROJECT_DIR}; echo '=== POKER HUB ==='; ${HUB_CMD}" in selected tab of front window
    delay 1
    set custom title of selected tab of front window to "PokerHub"
    
    # Ensure the Poker Hub tab is active (it should be since it's the selected tab)
    activate

end tell
EOF
