#!/usr/bin/env python3
"""
Poker Table Hub (Mac mini)
- WebSocket <-> USB Serial bridge with role-based routing

Routing rules:
- DETECTOR  -> TABLET only
- TABLET    -> ARDUINO (Serial) only   (except special "seat_name" messages handled internally)
- ARDUINO   -> TABLET only

Run examples:
  python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire
  python poker_hub.py --ws-port 8888 --baud 115200 --wire

Env (optional):
  WS_HOST=0.0.0.0
  WS_PORT=8888
  BAUD=115200
  HTTP_BRIDGE_PORT=8787
  FIXED_REBUY_AMOUNT=20
  FIXED_HALF_REBUY_AMOUNT=10
  SEAT_NAME_TIMEOUT_MS=2500                      # optional override (default 2500 ms)
"""

import asyncio, json, logging, argparse, os, sys, glob, threading, time
import serial            # pip install pyserial
import websockets        # pip install websockets
from aiohttp import web  # pip install aiohttp
from typing import Optional, List, Dict

# ------------ optional dotenv ------------
try:
    from dotenv import load_dotenv
    if os.path.exists("env"):
        load_dotenv("env")
except Exception:
    pass

# --------------------------
# Configuration defaults
# --------------------------
DEFAULT_WS_HOST = os.getenv("WS_HOST", "0.0.0.0")
DEFAULT_WS_PORT = int(os.getenv("WS_PORT", "8888"))
DEFAULT_BAUD    = int(os.getenv("BAUD", "115200"))

DEFAULT_HTTP_PORT     = int(os.getenv("HTTP_BRIDGE_PORT", "8787"))
FIXED_REBUY           = int(os.getenv("FIXED_REBUY_AMOUNT", "20"))
FIXED_HALF_REBUY      = int(os.getenv("FIXED_HALF_REBUY_AMOUNT", "10"))
SEAT_NAME_TIMEOUT_MS  = int(os.getenv("SEAT_NAME_TIMEOUT_MS", "2500"))

SEAT_MAX = int(os.getenv("SEAT_MAX", "9"))  # valid seats are 0..SEAT_MAX (inclusive)

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("poker_hub")

RED = "\033[91m"
RESET = "\033[0m"

def trunc(s: str, n: int = 220) -> str:
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + "…"

# --------------------------
# Serial helpers
# --------------------------
def auto_find_serial():
    """Pick the first macOS UNO R4 style modem if available."""
    cands = sorted(glob.glob("/dev/tty.usbmodem*"))
    return cands[0] if cands else None

class SerialBridge:
    """
    Threaded serial reader/writer.
    - write_line(str) to send a line to Arduino (adds trailing \n)
    - reader thread parses lines and enqueues them onto an asyncio.Queue
      using loop.call_soon_threadsafe(...)   <-- thread-safe!
    """
    def __init__(self, port: Optional[str], baud: int, out_queue: asyncio.Queue, wire: bool = False):
        self.port = port
        self.baud = baud
        self.q = out_queue
        self.loop: asyncio.AbstractEventLoop | None = None   # set by HubServer.start()
        self.ser = None
        self._stop = threading.Event()
        self._thr = None
        self._wire = wire
        self._warned_no_serial = False
        self._device_not_configured_logged = False

    def open(self):
        if not self.port:
            log.warning(f"{RED}⚠️  No serial device specified/found. Continuing without Arduino.{RESET}")
            return
        log.info(f"Opening serial: {self.port} @ {self.baud}")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2, write_timeout=1.0)
            time.sleep(1.8)  # UNO R4 typically resets on open
            self._thr = threading.Thread(target=self._reader_loop, name="serial-reader", daemon=True)
            self._thr.start()
            log.info("Serial opened.")
        except Exception as e:
            log.warning(f"{RED}⚠️  Could not open serial port {self.port}: {e}{RESET}")
            log.warning(f"{RED}Continuing without Arduino (Hub will still serve WS clients).{RESET}")
            self.ser = None

    def close(self):
        self._stop.set()
        if self._thr and self._thr.is_alive():
            self._thr.join(timeout=1.0)
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass
        log.info("Serial closed.")

    def write_line(self, s: str):
        if not self.ser:
            if not self._warned_no_serial:
                log.warning(f"{RED}⚠️  Arduino serial unavailable; dropping message to Serial. Is the UNO R4 connected?{RESET}")
                self._warned_no_serial = True
            return
        if not s.endswith("\n"):
            s = s + "\n"
        try:
            self.ser.write(s.encode())
            if self._wire:
                log.info("WS→SER  %s", trunc(s.strip()))
        except Exception as e:
            log.warning(f"{RED}Serial write failed: {e!r}{RESET}")

    def _reader_loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                if not self.ser:
                    break
                chunk = self.ser.read(256)
                if chunk:
                    buf.extend(chunk)
                    while True:
                        nl = buf.find(b"\n")
                        if nl == -1:
                            break
                        line = buf[:nl].decode(errors="ignore").strip()
                        del buf[:nl + 1]
                        if line:
                            if self._wire:
                                log.info("SER→HUB %s", trunc(line))
                            if self.loop is not None:
                                self.loop.call_soon_threadsafe(self.q.put_nowait, line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                # Check if device is not configured - log once and stop trying
                error_str = str(e)
                if "Device not configured" in error_str or "[Errno 6]" in error_str:
                    if not self._device_not_configured_logged:
                        log.warning(f"{RED}Serial device not configured (Arduino may be off). Stopping serial reads.{RESET}")
                        self._device_not_configured_logged = True
                    # Break out of loop - stop trying to read
                    break
                else:
                    log.warning(f"{RED}Serial read error: {e!r}{RESET}")
                    time.sleep(0.2)

# --------------------------
# WebSocket server with roles
# --------------------------
ROLE_UNKNOWN  = "unknown"
ROLE_TABLET   = "tablet"
ROLE_DETECTOR = "detector"

class HubServer:
    def __init__(self, ws_host: str, ws_port: int, serial_bridge: SerialBridge, wire: bool = False):
        self.host = ws_host
        self.port = ws_port
        self.sb = serial_bridge
        self.clients_all = set()
        self.role_by_client = {}
        self.clients_lock = asyncio.Lock()
        self.serial_to_ws_q: asyncio.Queue[str] = asyncio.Queue(maxsize=512)
        self._wire = wire
        self.loop: asyncio.AbstractEventLoop | None = None

        # dedupe window for rebuys
        self._last_rebuy_ts_by_seat: dict[int, int] = {}
        self._dedupe_ms = 1500

        # Waiters for seat-name replies (keyed by seat)
        self._seat_name_waiters: Dict[int, List[asyncio.Future]] = {}
        
        # Waiters for seat-name replies by player name (keyed by normalized name)
        self._seat_name_by_name_waiters: Dict[str, List[asyncio.Future]] = {}

        self._http_runner: web.AppRunner | None = None

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.sb.q = self.serial_to_ws_q
        self.sb.loop = self.loop

        self.sb.open()
        broadcaster = asyncio.create_task(self._broadcast_serial_lines_to_tablets())
        await self._start_http_bridge()

        async with websockets.serve(self._handler, self.host, self.port, ping_interval=20, ping_timeout=20):
            log.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()  # run forever
            finally:
                broadcaster.cancel()
                await self._stop_http_bridge()
                self.sb.close()

    # ---------- HTTP bridge ----------
    async def _start_http_bridge(self):
        app = web.Application()

        async def _health(_req):
            return web.Response(text="ok")

        app.add_routes([
            web.get("/health", _health),
            web.get("/healthz", _health),
        ])

        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", DEFAULT_HTTP_PORT)
        await site.start()
        log.info(f"HTTP bridge listening on http://0.0.0.0:{DEFAULT_HTTP_PORT}  (POST /alexa)")

    async def _stop_http_bridge(self):
        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None

    async def _wait_for_seat_name(self, seat: int, timeout_ms: int = SEAT_NAME_TIMEOUT_MS) -> tuple[Optional[str], Optional[int]]:
        """Wait for seat_name response. Returns (name, total_rebuys) tuple."""
        fut: asyncio.Future = self.loop.create_future()  # type: ignore
        self._seat_name_waiters.setdefault(seat, []).append(fut)
        try:
            result = await asyncio.wait_for(fut, timeout=timeout_ms / 1000.0)
            return result  # type: ignore
        except asyncio.TimeoutError:
            return (None, None)
        finally:
            try:
                lst = self._seat_name_waiters.get(seat, [])
                if fut in lst:
                    lst.remove(fut)
            except Exception:
                pass

    async def _fulfill_seat_name_waiters(self, seat: int, name: str, total_rebuys: Optional[int] = None, waiter_name: Optional[str] = None):
        """Fulfill waiters with (name, total_rebuys) tuple.
        
        Args:
            seat: Seat number
            name: Display name (returned to caller)
            total_rebuys: Total rebuys amount
            waiter_name: Name to use for matching waiters (usually the matched name from Alexa)
        """
        # Fulfill seat-based waiters
        lst = self._seat_name_waiters.get(seat)
        if lst:
            while lst:
                fut = lst.pop(0)
                if not fut.done():
                    fut.set_result((name, total_rebuys))
        
        # Fulfill name-based waiters (use waiter_name if provided, otherwise use display name)
        name_for_matching = waiter_name if waiter_name else name
        if name_for_matching:
            normalized_name = name_for_matching.strip().lower()
            log.info("Looking for name-based waiters with normalized name: '%s' (from waiter_name=%r, name=%r)", 
                    normalized_name, waiter_name, name)
            name_lst = self._seat_name_by_name_waiters.get(normalized_name)
            if name_lst:
                log.info("Found %d waiter(s) for name '%s', fulfilling...", len(name_lst), normalized_name)
                while name_lst:
                    fut = name_lst.pop(0)
                    if not fut.done():
                        fut.set_result((name, total_rebuys))
                        log.info("Fulfilled waiter for name '%s' with name=%r, total_rebuys=%r", normalized_name, name, total_rebuys)
            else:
                log.warning("No waiters found for normalized name '%s'. Active waiters: %s", 
                           normalized_name, list(self._seat_name_by_name_waiters.keys()))
    
    async def _wait_for_seat_name_by_name(self, player_name: str, timeout_ms: int = SEAT_NAME_TIMEOUT_MS) -> tuple[Optional[str], Optional[int]]:
        """Wait for seat_name response by player name. Returns (name, total_rebuys) tuple."""
        normalized_name = player_name.strip().lower()
        log.info("Setting up waiter for seat_name by name: '%s' (normalized: '%s')", player_name, normalized_name)
        fut: asyncio.Future = self.loop.create_future()  # type: ignore
        self._seat_name_by_name_waiters.setdefault(normalized_name, []).append(fut)
        try:
            result = await asyncio.wait_for(fut, timeout=timeout_ms / 1000.0)
            log.info("Waiter fulfilled for name '%s': name=%r, total_rebuys=%r", normalized_name, result[0], result[1])
            return result  # type: ignore
        except asyncio.TimeoutError:
            log.warning("Waiter timed out for name '%s' after %dms", normalized_name, timeout_ms)
            return (None, None)
        finally:
            try:
                lst = self._seat_name_by_name_waiters.get(normalized_name, [])
                if fut in lst:
                    lst.remove(fut)
            except Exception:
                pass
    
    # ---------- seat & chips helpers ----------
    def _parse_seat_from_slots(self, slots) -> tuple[bool, Optional[int], Optional[str]]:
        """Parse seat number from slots. Returns (ok, seat, error_message)."""
        val = None
        try:
            val = slots.get("seatNumber", {}).get("value")
        except Exception:
            pass
        try:
            seat = int(val)
        except (TypeError, ValueError):
            return False, None, "I need a seat number, like seat four."
        if not (0 <= seat <= SEAT_MAX):
            return False, None, f"Seat must be between zero and {SEAT_MAX}."
        return True, seat, None
    
    def _parse_player_name_from_slots(self, slots) -> tuple[bool, Optional[str], Optional[str]]:
        """Parse player name from slots. Returns (ok, name, error_message).
        Prefers resolved value from Alexa's custom slot type if available.
        """
        pname_slot = slots.get("pname", {})
        if not pname_slot:
            return False, None, None
        
        # Try to get resolved value first (canonical value from custom slot type)
        pname_val = None
        try:
            resolutions = pname_slot.get("resolutions", {}).get("resolutionsPerAuthority", [])
            if resolutions:
                authority = resolutions[0]
                if authority.get("status", {}).get("code") == "ER_SUCCESS_MATCH":
                    values = authority.get("values", [])
                    if values:
                        pname_val = values[0].get("value", {}).get("name")
        except Exception:
            pass
        
        # Fall back to raw value if no resolution
        if not pname_val:
            try:
                pname_val = pname_slot.get("value")
            except Exception:
                pass
        
        if pname_val:
            pname = str(pname_val).strip()
            if pname:
                return True, pname, None
        return False, None, None  # No error if missing, just not provided

    def _parse_chips_from_slots(self, slots) -> tuple[bool, Optional[int], Optional[str]]:
        val = None
        try:
            val = slots.get("chips", {}).get("value")
        except Exception:
            pass
        try:
            chips = int(val)
        except (TypeError, ValueError):
            return False, None, "I need a chip amount in dollars, like two hundred."
        if chips < 0:
            return False, None, "Chip amount must be zero or more."
        return True, chips, None

    # ---------- role helpers ----------

    # ---------- role helpers ----------
    async def _set_role(self, ws, role: str):
        role = (role or ROLE_UNKNOWN).lower()
        async with self.clients_lock:
            self.role_by_client[ws] = role

    async def _get_role(self, ws) -> str:
        async with self.clients_lock:
            return self.role_by_client.get(ws, ROLE_UNKNOWN)

    async def _clients_with_role(self, role: str):
        async with self.clients_lock:
            return [c for c, r in self.role_by_client.items() if r == role]

    # ---------- WS handler ----------
    async def _handler(self, ws, path=None):
        peer = getattr(ws, "remote_address", None)
        log.info(f"Client connected: {peer}")
        async with self.clients_lock:
            self.clients_all.add(ws)
            self.role_by_client[ws] = ROLE_UNKNOWN

        try:
            async for text in ws:
                text = text.strip()
                if not text:
                    continue
                if self._wire:
                    log.info("WS→HUB %s", trunc(text))

                try:
                    j = json.loads(text)
                except json.JSONDecodeError:
                    continue

                if j.get("command") == "hello":
                    await self._set_role(ws, j.get("role", ROLE_UNKNOWN))
                    if self._wire:
                        log.info("HUB: role set to %s for %s", await self._get_role(ws), peer)
                    continue

                role = await self._get_role(ws)
                if role == ROLE_UNKNOWN:
                    role = self._infer_role_from_message(j)
                    await self._set_role(ws, role)
                    if self._wire:
                        log.info("HUB: role inferred as %s for %s", role, peer)

                if j.get("command") == "ping":
                    await ws.send(json.dumps({"command": "pong"}))
                    continue

                cmd = j.get("command")

                if role == ROLE_DETECTOR:
                    if cmd == "cards_detected":
                        if self._wire:
                            log.info("DET→HUB: forwarding cards_detected to tablets")
                        await self._forward_to_tablets(j)
                    continue

                if role == ROLE_TABLET:
                    if cmd == "seat_name":
                        seat = j.get("seat")
                        pname_raw = j.get("name")
                        pname = "" if pname_raw is None else str(pname_raw).strip()
                        matched_name_raw = j.get("matched_name")  # The name Alexa sent (for waiter matching)
                        matched_name = "" if matched_name_raw is None else str(matched_name_raw).strip()
                        total_rebuys = j.get("total_rebuys")  # Extract total_rebuys if present
                        if isinstance(seat, int):
                            if self._wire:
                                log.info("TAB→HUB seat_name seat=%s name=%r matched_name=%r total_rebuys=%r", seat, pname, matched_name, total_rebuys)
                            # Use matched_name for waiter fulfillment if available, otherwise use display name
                            waiter_name = matched_name if matched_name else pname
                            log.info("Fulfilling seat_name waiters: seat=%d, name=%r, waiter_name=%r, total_rebuys=%r, active_name_waiters=%s", 
                                    seat, pname, waiter_name, total_rebuys, list(self._seat_name_by_name_waiters.keys()))
                            await self._fulfill_seat_name_waiters(seat, pname, total_rebuys, waiter_name)
                        continue
                    
                        continue

                    # TABLET -> ARDUINO (Serial) only
                    self.sb.write_line(json.dumps(j, separators=(",", ":")))
                    continue

                # Unknown role: conservative default = like tablet (to serial)
                self.sb.write_line(json.dumps(j, separators=(",", ":")))

        except websockets.ConnectionClosed:
            pass
        finally:
            async with self.clients_lock:
                self.clients_all.discard(ws)
                self.role_by_client.pop(ws, None)
            log.info(f"Client disconnected: {peer}")

    def _infer_role_from_message(self, j: dict) -> str:
        if j.get("command") == "cards_detected":
            return ROLE_DETECTOR
        return ROLE_TABLET

    # ---------- Broadcast helpers ----------
    async def _forward_to_tablets(self, payload: dict):
        line = json.dumps(payload, separators=(",", ":"))
        tablets = await self._clients_with_role(ROLE_TABLET)
        if tablets:
            if self._wire:
                log.info("HUB→TAB x%d %s", len(tablets), trunc(line))
            dead = []
            for ws in tablets:
                try:
                    await ws.send(line)
                except Exception as e:
                    log.warning("Failed to send to tablet: %s", e)
                    dead.append(ws)
            if dead:
                async with self.clients_lock:
                    for ws in dead:
                        self.clients_all.discard(ws)
                        self.role_by_client.pop(ws, None)
        else:
            if self._wire:
                log.warning("HUB→TAB: No tablets connected, dropping %s", trunc(line))

    async def _broadcast_serial_lines_to_tablets(self):
        """Serial → TABLET only."""
        while True:
            line = await self.serial_to_ws_q.get()
            try:
                _ = json.loads(line)
            except json.JSONDecodeError:
                continue

            tablets = await self._clients_with_role(ROLE_TABLET)
            if self._wire:
                if tablets:
                    log.info("HUB→TAB x%d %s", len(tablets), trunc(line))
                else:
                    log.info("HUB→TAB x0 (no tablet connected) %s", trunc(line))

            dead = []
            for ws in tablets:
                try:
                    await ws.send(line)
                except Exception:
                    dead.append(ws)
            if dead:
                async with self.clients_lock:
                    for ws in dead:
                        self.clients_all.discard(ws)
                        self.role_by_client.pop(ws, None)

# --------------------------
# Entrypoint
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Poker Table Hub (WS <-> Serial)")
    p.add_argument("--serial", help="Serial device (e.g., /dev/tty.usbmodem48CA435C84242)")
    p.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    p.add_argument("--ws-host", default=DEFAULT_WS_HOST)
    p.add_argument("--ws-port", type=int, default=DEFAULT_WS_PORT)
    p.add_argument("--wire", action="store_true", help="Enable wire-level logging of messages")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()

async def async_main():
    args = parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    port = args.serial or os.getenv("SERIAL_PORT") or auto_find_serial()
    if not port:
        log.warning(f"{RED}⚠️  No serial device found. Hub will start without Arduino Serial.{RESET}")
    else:
        log.info(f"Using serial device: {port}")

    dummy_q = asyncio.Queue()
    sb = SerialBridge(port, args.baud, dummy_q, wire=args.wire)
    server = HubServer(args.ws_host, args.ws_port, sb, wire=args.wire)
    await server.start()

if __name__ == "__main__":
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass

