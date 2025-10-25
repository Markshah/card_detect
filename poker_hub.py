#!/usr/bin/env python3
"""
Poker Table Hub (Mac) — WebSocket <-> USB Serial bridge with role-based routing

Routing rules:
- DETECTOR  -> TABLET only
- TABLET    -> ARDUINO (Serial) only
- ARDUINO   -> TABLET only

Examples:
  python poker_hub.py --serial /dev/tty.usbmodem48CA435C84242 --wire
  python poker_hub.py --ws-port 8888 --baud 115200 --wire
"""

import asyncio, json, logging, argparse, os, sys, glob, threading, time
import serial            # pip install pyserial
import websockets        # pip install websockets

# --------------------------
# Configuration defaults
# --------------------------
DEFAULT_WS_HOST = "0.0.0.0"
DEFAULT_WS_PORT = 8888
DEFAULT_BAUD    = 115200

LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("poker_hub")

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
    def __init__(self, port: str, baud: int, out_queue: asyncio.Queue, wire: bool = False):
        self.port = port
        self.baud = baud
        self.q = out_queue
        self.loop: asyncio.AbstractEventLoop | None = None   # set by HubServer.start()
        self.ser = None
        self._stop = threading.Event()
        self._thr = None
        self._wire = wire


    def open(self):
        log.info(f"Opening serial: {self.port} @ {self.baud}")
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.2, write_timeout=1.0)
            # UNO R4 typically resets on open; give it a moment
            time.sleep(1.8)
            self._thr = threading.Thread(target=self._reader_loop, name="serial-reader", daemon=True)
            self._thr.start()
            log.info("Serial opened.")
        except serial.SerialException as e:
            log.warning(f"⚠️  Could not open serial port {self.port}: {e}.")
            log.warning("Continuing without Arduino (Hub will still serve WS clients).")
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
            return
        if not s.endswith("\n"):
            s = s + "\n"
        try:
            self.ser.write(s.encode())
            if self._wire:
                log.info("WS→SER  %s", trunc(s.strip()))
        except Exception as e:
            log.warning(f"Serial write failed: {e!r}")

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
                            # THREAD-SAFE enqueue into asyncio loop
                            if self.loop is not None:
                                self.loop.call_soon_threadsafe(self.q.put_nowait, line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                log.warning(f"Serial read error: {e!r}")
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
        self.clients_all = set()             # all websockets
        self.role_by_client = {}             # ws -> role
        self.clients_lock = asyncio.Lock()
        self.serial_to_ws_q: asyncio.Queue[str] = asyncio.Queue(maxsize=512)
        self._wire = wire
        self.loop: asyncio.AbstractEventLoop | None = None

    async def start(self):
        # Bind queue + loop to the serial bridge
        self.loop = asyncio.get_running_loop()
        self.sb.q = self.serial_to_ws_q
        self.sb.loop = self.loop

        # Start serial
        self.sb.open()

        # Start serial→tablet broadcaster
        broadcaster = asyncio.create_task(self._broadcast_serial_lines_to_tablets())

        async with websockets.serve(self._handler, self.host, self.port, ping_interval=20, ping_timeout=20):
            log.info(f"WebSocket server listening on ws://{self.host}:{self.port}")
            try:
                await asyncio.Future()  # run forever
            finally:
                broadcaster.cancel()
                self.sb.close()

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

                # Parse JSON
                try:
                    j = json.loads(text)
                except json.JSONDecodeError:
                    # ignore non-JSON
                    continue

                # Hello sets role explicitly
                if j.get("command") == "hello":
                    await self._set_role(ws, j.get("role", ROLE_UNKNOWN))
                    if self._wire:
                        log.info("HUB: role set to %s for %s", await self._get_role(ws), peer)
                    continue

                # Infer role if unknown
                role = await self._get_role(ws)
                if role == ROLE_UNKNOWN:
                    role = self._infer_role_from_message(j)
                    await self._set_role(ws, role)
                    if self._wire:
                        log.info("HUB: role inferred as %s for %s", role, peer)

                # Ping/pong convenience
                if j.get("command") == "ping":
                    await ws.send(json.dumps({"command": "pong"}))
                    continue


                # -------- Routing rules (with detector's move_dealer_forward -> SERIAL) --------
                cmd = j.get("command")

                if role == ROLE_DETECTOR:
                    if cmd == "cards_detected":
                        # DETECTOR -> TABLET
                        await self._forward_to_tablets(j)
                    elif cmd == "move_dealer_forward":
                        # DETECTOR -> ARDUINO (Serial)  <<< fixed per your note
                        self.sb.write_line(json.dumps(j, separators=(",", ":")))
                    else:
                        # ignore or treat as detector->tablet if you prefer:
                        # await self._forward_to_tablets(j)
                        pass
                    continue

                if role == ROLE_TABLET:
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
        # If it's clearly detector output
        if j.get("command") == "cards_detected":
            return ROLE_DETECTOR
        # If it's a control (e.g., move_dealer_forward) and came from WS, we treat it as tablet by default
        return ROLE_TABLET

    # ---------- Broadcast helpers ----------
    async def _forward_to_tablets(self, payload: dict):
        line = json.dumps(payload, separators=(",", ":"))
        tablets = await self._clients_with_role(ROLE_TABLET)
        if self._wire and tablets:
            log.info("HUB→TAB x%d %s", len(tablets), trunc(line))
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

    async def _broadcast_serial_lines_to_tablets(self):
        """Serial → TABLET only."""
        while True:
            line = await self.serial_to_ws_q.get()
            # sanity: only forward JSON-looking lines
            try:
                _ = json.loads(line)
            except json.JSONDecodeError:
                continue

            tablets = await self._clients_with_role(ROLE_TABLET)
            if self._wire and tablets:
                log.info("HUB→TAB x%d %s", len(tablets), trunc(line))

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
    p = argparse.ArgumentParser(description="Poker Table Hub (WebSocket <-> Serial)")
    p.add_argument("--serial", help="Serial device (e.g., /dev/tty.usbmodem48CA435C84242)")
    p.add_argument("--baud", type=int, default=int(os.getenv("BAUD", DEFAULT_BAUD)))
    p.add_argument("--ws-host", default=os.getenv("WS_HOST", DEFAULT_WS_HOST))
    p.add_argument("--ws-port", type=int, default=int(os.getenv("WS_PORT", DEFAULT_WS_PORT)))
    p.add_argument("--wire", action="store_true", help="Enable wire-level logging of messages")
    p.add_argument("--debug", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()

async def async_main():
    args = parse_args()
    if args.debug:
        log.setLevel(logging.DEBUG)

    port = args.serial or os.getenv("SERIAL_PORT") or auto_find_serial()
    if not port:
        log.error("No serial device found. Plug in the UNO R4 or pass --serial /dev/tty.usbmodemXXXX")
        sys.exit(2)
    log.info(f"Using serial device: {port}")

    dummy_q = asyncio.Queue()
    sb = SerialBridge(port, args.baud, dummy_q, wire=args.wire)
    server = HubServer(args.ws_host, args.ws_port, sb, wire=args.wire)
    await server.start()

if __name__ == "__main__":
    asyncio.run(async_main())

