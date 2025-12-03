# poker_hub_client.py
import json, os, time, threading, logging
from websocket import WebSocketApp

# ---- Tunables (env) ---------------------------------------------------------
DEFAULT_WS_URL = os.getenv("WS_URL", "ws://192.168.1.54:8888")  # default to Mac hub

PING_INTERVAL  = int(os.getenv("WS_PING_INTERVAL_SEC", "20"))
PING_TIMEOUT   = int(os.getenv("WS_PING_TIMEOUT_SEC", "10"))
IDLE_RECON_SEC = int(os.getenv("WS_IDLE_RECONNECT_SEC", "120"))

HB_INTERVAL    = int(os.getenv("WS_HEARTBEAT_SEC", "0"))
HB_COMMAND     = os.getenv("WS_HEARTBEAT_CMD", "heartbeat").strip()  # "heartbeat" or "ping"

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("WSManager")


class WSManager:
    def __init__(
        self,
        url: str = DEFAULT_WS_URL,
        max_retries: int = 30,
        retry_delay: float = 3.0,
        on_event=None,
        name: str | None = None,
        role: str | None = None,
        client_name: str | None = None,
    ):
        self.url = url
        self.max_retries = max_retries   # 0 = retry forever
        self.retry_delay = retry_delay

        self._on_event = on_event
        self._name = name or url
        self._client_role = role  # Role for hello message (e.g., "detector", "tablet")
        self._client_name = client_name  # Client name for hello message (e.g., "emulator", "tablet")

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
        if self._thread and self._thread.is_alive():
            return
        self._should_run.set()
        self._thread = threading.Thread(target=self._run_forever, daemon=True)
        self._thread.start()

        if self._hb_interval > 0 and (not self._hb_thread or not self._hb_thread.is_alive()):
            self._hb_stop.clear()
            self._hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._hb_thread.start()

    def stop(self):
        self._should_run.clear()
        self._hb_stop.set()
        with self._lock:
            if self._ws:
                try: self._ws.close()
                except Exception: pass
                self._ws = None
        if self._thread: self._thread.join(timeout=2)
        if self._hb_thread: self._hb_thread.join(timeout=2)
        self._connected.clear()

    # ---- Public API --------------------------------------------------------
    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    def wait_connected(self, timeout: float = 1.5) -> bool:
        return self._connected.wait(timeout)

    def send_cards_detected(self, count, codes=None) -> bool:
        try:
            data = {"count": int(count)}
            if codes is not None:
                if isinstance(codes, (list, tuple)):
                    data["codes"] = [str(c).upper() for c in codes if c is not None]
                else:
                    data["codes"] = [str(codes).upper()]
            payload = {"command": "cards_detected", "data": data}
            return self.send_json(payload)
        except Exception:
            logging.exception("send_cards_detected failed")
            return False

    def send_ping(self) -> bool:
        return self.send_json({"command": "ping", "ts": time.time()})

    def send_json(self, payload: dict) -> bool:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            if self._ws and self.is_connected:
                try:
                    self._ws.send(data)
                    self._last_tx_ts = time.time()
                    log.info("Sent WS: %s", data)
                    return True
                except Exception as e:
                    log.error("WS send failed: %s", e)
                    self._emit(f"[WS {self._name}] send failed: {e}")
            else:
                log.warning("WS not connected; can't send: %s", data)
                self._emit(f"[WS {self._name}] not connected; send skipped")
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
                self._ws.on_pong = self._on_pong

                self._emit(f"[WS {self._name}] connecting…")
                self._ws.run_forever(
                    ping_interval=PING_INTERVAL,
                    ping_timeout=PING_TIMEOUT,
                )
            except Exception as e:
                log.exception("WS run_forever crashed: %s", e)
                self._emit(f"[WS {self._name}] run_forever crashed: {e}")

            self._connected.clear()
            if not self._should_run.is_set():
                break

            if self.max_retries and self._retry_count >= self.max_retries:
                msg = "Max retry attempts reached; giving up."
                log.error(msg)
                self._emit(f"[WS {self._name}] {msg}")
                break

            self._retry_count += 1
            delay = self._jitter(self.retry_delay, self._retry_count, random)
            log.info("Retrying websocket in %.1fs (attempt %d%s)...",
                     delay, self._retry_count,
                     f"/{self.max_retries}" if self.max_retries else "")
            self._emit(f"[WS {self._name}] reconnect in {delay:.1f}s "
                       f"(attempt {self._retry_count}{'/' + str(self.max_retries) if self.max_retries else ''})")
            time.sleep(delay)

    def _heartbeat_loop(self):
        while not self._hb_stop.is_set():
            now = time.time()

            if self._hb_interval > 0 and self.is_connected:
                try:
                    cmd = HB_COMMAND if HB_COMMAND else "heartbeat"
                    self.send_json({"command": cmd, "ts": now})
                except Exception:
                    pass

            idle = now - self._last_rx_ts
            if IDLE_RECON_SEC > 0 and idle > IDLE_RECON_SEC:
                log.warning("Idle %.1fs > %ds; forcing reconnect on %s",
                            idle, IDLE_RECON_SEC, self.url)
                self._emit(f"[WS {self._name}] idle {idle:.1f}s > {IDLE_RECON_SEC}s → force reconnect")
                self._force_reconnect()

            self._hb_stop.wait(max(1, self._hb_interval if self._hb_interval > 0 else 1))

    def _force_reconnect(self):
        with self._lock:
            try:
                if self._ws: self._ws.close()
            except Exception:
                pass
            self._connected.clear()

    @staticmethod
    def _jitter(base: float, n: int, rnd) -> float:
        return base * min(5, 1 + 0.25 * n) * (0.75 + 0.5 * rnd.random())

    # ---- Event handlers ----------------------------------------------------
    def _on_open(self, ws):
        self._retry_count = 0
        self._connected.set()
        self._last_rx_ts = time.time()
        log.info("WebSocket opened: %s", self.url)
        self._emit(f"[WS {self._name}] OPEN")
        # Send hello message with role and name if configured
        if hasattr(self, '_client_role') or hasattr(self, '_client_name'):
            hello_msg = {"command": "hello"}
            if hasattr(self, '_client_role'):
                hello_msg["role"] = self._client_role
            if hasattr(self, '_client_name'):
                hello_msg["name"] = self._client_name
            try:
                self.send_json(hello_msg)
            except Exception:
                pass  # Ignore errors, connection just opened

    def _on_message(self, ws, msg):
        self._last_rx_ts = time.time()
        log.debug("WS message: %s", msg)
        # If you want to surface tablet->app messages, emit here:
        # self._emit(f"[WS {self._name}] RX: {msg}")

    def _on_pong(self, ws, frame_data):
        self._last_rx_ts = time.time()

    def _on_error(self, ws, err):
        log.error("WS error: %s", err)
        self._emit(f"[WS {self._name}] ERROR: {err}")

    def _on_close(self, ws, status, msg):
        log.info("WS closed (%s): %s", status, msg)
        self._emit(f"[WS {self._name}] CLOSE ({status}): {msg}")

    # ---- Helper ------------------------------------------------------------
    def _emit(self, s: str):
        try:
            if self._on_event:
                self._on_event(s)
        except Exception:
            pass

