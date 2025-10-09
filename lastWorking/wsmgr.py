# websocket_manager.py
import json, os, time, threading, logging
from websocket import WebSocketApp

logging.basicConfig(level=logging.INFO)

DEFAULT_WS_URL = os.getenv("WS_URL", "ws://192.168.1.245:8888")
PING_INTERVAL  = int(os.getenv("WS_PING_INTERVAL_SEC", "20"))
PING_TIMEOUT   = int(os.getenv("WS_PING_TIMEOUT_SEC", "10"))

class WSManager:
    def __init__(self, url: str = DEFAULT_WS_URL, max_retries: int = 30, retry_delay: float = 3.0):
        self.url = url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._ws = None
        self._thread = None
        self._lock = threading.RLock()
        self._should_run = threading.Event()
        self._connected = threading.Event()
        self._retry_count = 0

        # heartbeat
        self._hb_interval = int(os.getenv("WS_HEARTBEAT_SEC", "0"))
        self._hb_thread = None
        self._hb_stop = threading.Event()

    # ---- Lifecycle ---------------------------------------------------------
    def start(self):
        """Start background connection (auto-reconnect, keepalive)."""
        if self._thread and self._thread.is_alive():
            return
        self._should_run.set()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

        # optional heartbeat (app-level keepalive)
        if self._hb_interval > 0 and (not self._hb_thread or not self._hb_thread.is_alive()):
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

    def stop(self):
        """Close and stop retries/heartbeat."""
        self._should_run.clear()
        self._hb_stop.set()
        with self._lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass
                self._ws = None
        if self._thread:
            self._thread.join(timeout=2)
        if self._hb_thread:
            self._hb_thread.join(timeout=2)
        self._connected.clear()

    # ---- Public API --------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def wait_connected(self, timeout: float = 1.5) -> bool:
        """Block up to timeout seconds for a connection to be established."""
        return self._connected.wait(timeout)

    def send_cards_detected(self, count: int) -> bool:
        print("SEND CARDS DETECTED.")

        """Send: {"command":"cards_detected", "data": {"count": N}}"""
        return self.send_json({
            "command": "cards_detected",
            "data": {
                "count": int(count)
            }
        })

    def send_move_dealer_forward(self) -> bool:
        """Send: {"command":"move_dealer_forward"}"""
        return self.send_json({"command": "move_dealer_forward"})

    def send_move_dealer_forward_burst(self, burst: int = 1, spacing_ms: int = 0) -> bool:
        """Send the command multiple times in a tiny burst; True if any succeed."""
        ok_any = False
        for i in range(max(1, burst)):
            ok = self.send_move_dealer_forward()
            ok_any = ok_any or ok
            if i + 1 < burst and spacing_ms > 0:
                time.sleep(spacing_ms / 1000.0)
        return ok_any

    def send_json(self, payload: dict) -> bool:
        """Thread-safe send; returns True if delivered."""
        data = json.dumps(payload)
        with self._lock:
            if self._ws and self.is_connected:
                try:
                    self._ws.send(data)
                    logging.info("Sent WS: %s", data)
                    return True
                except Exception as e:
                    logging.error("WS send failed: %s", e)
            else:
                logging.warning("WS not connected; can't send: %s", data)
        return False

    # ---- Internals ---------------------------------------------------------
    def _run_forever(self):
        while self._should_run.is_set():
            try:
                self._ws = WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Keepalive ping/pong at the WebSocket layer
                self._ws.run_forever(ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT)
            except Exception as e:
                logging.exception("WS run_forever crashed: %s", e)

            self._connected.clear()
            if not self._should_run.is_set():
                break
            if self._retry_count >= self.max_retries:
                logging.error("Max retry attempts reached; giving up.")
                break

            self._retry_count += 1
            delay = self._jitter(self.retry_delay, self._retry_count)
            logging.info("Retrying websocket in %.1fs (attempt %d/%d)...", delay, self._retry_count, self.max_retries)
            time.sleep(delay)

    def _heartbeat_loop(self):
        # Optional app-level heartbeat JSON (useful if your server drops idle sockets even with pings)
        while not self._hb_stop.is_set():
            if self.is_connected:
                try:
                    self.send_json({"type": "heartbeat", "ts": time.time()})
                except Exception:
                    pass
            self._hb_stop.wait(max(1, self._hb_interval))

    @staticmethod
    def _jitter(base: float, n: int) -> float:
        # small backoff + jitter (prevents reconnect storms)
        import random
        return base * min(5, 1 + 0.25 * n) * (0.75 + 0.5 * random.random())

    # ---- Event handlers ----------------------------------------------------
    def _on_open(self, ws):
        self._retry_count = 0
        self._connected.set()
        logging.info("WebSocket opened: %s", self.url)

    def _on_message(self, ws, msg):
        logging.debug("WS message: %s", msg)

    def _on_error(self, ws, err):
        logging.error("WS error: %s", err)

    def _on_close(self, ws, status, msg):
        logging.info("WS closed (%s): %s", status, msg)

