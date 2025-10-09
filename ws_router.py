# wsrouter.py
import os
from wsmgr import WSManager

TABLET_WS_URL  = os.getenv("TABLET_WS_URL",  "ws://192.168.1.246:8765")  # your tablet app
ARDUINO_WS_URL = os.getenv("ARDUINO_WS_URL", "ws://192.168.1.245:8888")  # your Arduino

# two independent persistent connections
tablet_ws  = WSManager(url=TABLET_WS_URL)
arduino_ws = WSManager(url=ARDUINO_WS_URL)

def start():
    """Start both background connections."""
    tablet_ws.start()
    arduino_ws.start()

def stop():
    """Cleanly stop both connections (on program exit)."""
    tablet_ws.stop()
    arduino_ws.stop()

def wait_ready(timeout=2.0):
    """Optionally wait for either/both sockets to be up."""
    t_ok = tablet_ws.wait_connected(timeout)
    a_ok = arduino_ws.wait_connected(timeout)
    return t_ok, a_ok

# ---- Routed sends ----
def send_cards_detected(count: int) -> bool:
    """Goes to TABLET only."""
    return tablet_ws.send_cards_detected(count)

def send_move_dealer_forward() -> bool:
    """Goes to ARDUINO only."""
    return arduino_ws.send_move_dealer_forward()

def send_move_dealer_forward_burst(burst: int = 1, spacing_ms: int = 0) -> bool:
    """Still only to ARDUINO."""
    return arduino_ws.send_move_dealer_forward_burst(burst=burst, spacing_ms=spacing_ms)

