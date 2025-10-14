# ws_mgr.py
import json, os, time, threading, logging
from websocket import WebSocketApp

# ---- Tunables (env) ---------------------------------------------------------
DEFAULT_WS_URL = os.getenv("WS_URL", "ws://192.168.1.245:8888")

# Shorter pings help detect half-open sockets quickly
PING_INTERVAL  = int(os.getenv("WS_PING_INTERVAL_SEC", "5"))
PING_TIMEOUT   = int(os.getenv("WS_PING_TIMEOUT_SEC", "3"))

# App-level heartbeat JSON; set to 0 to disable
HB_INTERVAL    = int(os.getenv("WS_HEARTBEAT_SEC", "5"))

# If no messages/pongs/hearbeats are observed for this many seconds, we force-reconnect
IDLE_RECON_SEC = int(os.getenv("WS_IDLE_RECONNECT_SEC", "12"))

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("WSManager")


class WSManager:
    def __init__(self, url: str = DEFAULT_WS_URL, max_retries: int = 30, retry_delay: float = 3.0):
        self.url = url
        self.max_retries = max_retries   # 0 = retry forever
        self.retry_delay = retry_delay

        self._ws = None
        self._thread = None
        self._lock = threading.RLock()
        self._should_run = threading.Event()
        self._connected = threading.Event()
        self._retry_count = 0

        # heartbeat + idle watchdog
        self._hb_interval = HB_INTERVAL
        self._hb_thread = None
        self._hb_stop = threading.Event()
        self._last_rx_ts = time.time()
        self._last_tx_ts = time.time()

    # ---- Lifecycle ---------------------------------------------------------
    def start(self):
        """Start background connection (auto-reconnect, keepalive)."""
        if self._thread and self._thread.is_alive():
            return
        self._should_run.set()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

        # optional heartbeat (app-level keepalive + idle watchdog)
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

    def send_cards_detected(self, count, codes=None):
        """
        Send current count and optional list of codes.
        Example: {"type": "cards_detected", "count": 3, "codes": ["KD","JS"]}
        """
        try:
            payload = {"command": "cards_detected", "count": int(count)}
            if codes:
                payload["codes"] = [c.upper() for c in codes if c]
            self._send_json(payload)
            return True
        except Exception:
            return False

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
                    self._last_tx_ts = time.time()
                    log.info("Sent WS: %s", data)
                    return True
                except Exception as e:
                    log.error("WS send failed: %s", e)
            else:
                log.warning("WS not connected; can't send: %s", data)
        return False

    # ---- Internals ---------------------------------------------------------
    def _run_forever(self):
        import random
        while self._should_run.is_set():
            try:
                self._ws = WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # Also wire pong handler to update last-rx
                self._ws.on_pong = self._on_pong

                # Keepalive ping/pong at the WebSocket layer
                self._ws.run_forever(
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                )
            except Exception as e:
                log.exception("WS run_forever crashed: %s", e)

            self._connected.clear()
            if not self._should_run.is_set():
                break

            # retry policy
            if self.max_retries and self._retry_count >= self.max_retries:
                log.error("Max retry attempts reached; giving up.")
                break

            self._retry_count += 1
            delay = self._jitter(self.retry_delay, self._retry_count, random)
            log.info("Retrying websocket in %.1fs (attempt %d%s)...",
                     delay, self._retry_count,
                     f"/{self.max_retries}" if self.max_retries else "")
            time.sleep(delay)

    def _heartbeat_loop(self):
        # App-level heartbeat + idle watchdog
        while not self._hb_stop.is_set():
            now = time.time()

            # App-level heartbeat JSON (optional)
            if self._hb_interval > 0 and self.is_connected:
                try:
                    # light heartbeat the server can ignore if it wants
                    self.send_json({"type": "heartbeat", "ts": now})
                except Exception:
                    pass

            # Idle watchdog: force reconnect if no RX for too long
            idle = now - self._last_rx_ts
            if IDLE_RECON_SEC > 0 and idle > IDLE_RECON_SEC:
                log.warning("Idle %.1fs > %ds; forcing reconnect on %s",
                            idle, IDLE_RECON_SEC, self.url)
                self._force_reconnect()

            # sleep until next check
            self._hb_stop.wait(max(1, self._hb_interval if self._hb_interval > 0 else 1))

    def _force_reconnect(self):
        with self._lock:
            try:
                if self._ws:
                    self._ws.close()
            except Exception:
                pass
            self._connected.clear()

    @staticmethod
    def _jitter(base: float, n: int, rnd) -> float:
        # small backoff + jitter (prevents reconnect storms)
        return base * min(5, 1 + 0.25 * n) * (0.75 + 0.5 * rnd.random())

    # ---- Event handlers ----------------------------------------------------
    def _on_open(self, ws):
        self._retry_count = 0
        self._connected.set()
        self._last_rx_ts = time.time()
        log.info("WebSocket opened: %s", self.url)

    def _on_message(self, ws, msg):
        self._last_rx_ts = time.time()
        log.debug("WS message: %s", msg)

    def _on_pong(self, ws, frame_data):
        # Any pong means the peer is alive
        self._last_rx_ts = time.time()

    def _on_error(self, ws, err):
        log.error("WS error: %s", err)

    def _on_close(self, ws, status, msg):
        log.info("WS closed (%s): %s", status, msg)
        # last_rx stays as-is; heartbeat loop may force reconnect if needed

