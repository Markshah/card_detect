#!/usr/bin/env bash
# Kill poker_hub, card_detect, and cloudflared processes,
# then close ONLY the Terminal windows running poker_hub/card_detect.
# Works on macOS default Bash 3.2 (no mapfile).

echo "🛑 Stopping Poker Hub, Card Detector, and Cloudflare Tunnel..."

# ---- graceful kill helper ----
kill_grace() {
  for pid in "$@"; do
    [ -n "$pid" ] && kill "$pid" 2>/dev/null || true
  done
  sleep 0.3
  for pid in "$@"; do
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
  done
}

# ---- collect PIDs for pattern (space-separated) ----
collect_pids() {
  local pattern="$1"
  local pids
  pids=$(pgrep -f "$pattern" 2>/dev/null)
  echo "$pids"
}

# 1) Kill python scripts
hub_pids="$(collect_pids 'poker_hub\.py')"
det_pids="$(collect_pids 'card_detect\.py')"
tunnel_pids="$(collect_pids 'cloudflared.*tunnel.*pokerhub')"
[ -z "$tunnel_pids" ] && tunnel_pids="$(collect_pids 'cloudflared')"  # fallback

# Kill all three kinds
kill_grace $hub_pids $det_pids $tunnel_pids

# 2) Close ONLY windows for hub & detector (leave tunnel window open)
#    We match by:
#      - tab contents containing "poker_hub.py" or "card_detect.py"
#      - OR the window/tab name containing "PokerHub" or "CardDetector" (from startup titles)
#      - OR the contents containing our header lines "=== POKER HUB ===" / "=== CARD DETECTOR ==="
osascript <<'APPLESCRIPT'
tell application "Terminal"
  set targets to {"poker_hub.py", "card_detect.py", "=== POKER HUB ===", "=== CARD DETECTOR ===", "PokerHub", "CardDetector"}
  set windowsToClose to {}

  repeat with w in windows
    set shouldClose to false

    -- Check window name/title (custom title or auto title)
    try
      set wname to (name of w) as text
      repeat with key in targets
        if wname contains (key as text) then
          set shouldClose to true
          exit repeat
        end if
      end repeat
    end try

    if not shouldClose then
      -- Check each tab: name AND contents
      repeat with t in tabs of w
        try
          set tname to (name of t) as text
        on error
          set tname to ""
        end try
        set tcontents to ""
        try
          set tcontents to (contents of t)
        end try

        repeat with key in targets
          set needle to (key as text)
          if (tname contains needle) or (tcontents contains needle) then
            set shouldClose to true
            exit repeat
          end if
        end repeat

        if shouldClose then exit repeat
      end repeat
    end if

    if shouldClose then
      set end of windowsToClose to w
    end if
  end repeat

  -- Close targeted windows
  repeat with w in windowsToClose
    try
      close w
    end try
  end repeat
end tell
APPLESCRIPT

echo "✅ Killed poker_hub.py, card_detect.py, cloudflared."
echo "✅ Closed only the poker_hub/card_detect Terminal windows; tunnel window left open."

