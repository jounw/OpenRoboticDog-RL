"""Phone web joystick: a small HTTP server run on the Pi (or Mac). Standard library only.

    python joystick_web.py            # standalone test: open http://<this computer IP>:8080 on the phone
Imported by run_policy.py. The phone page POSTs JSON to /cmd at 20 Hz and reads /status.
Left stick: forward (up) / sideways (left-right); right stick left-right: turn. Release = 0. 'Deadman' is active only while a finger is touching.
"""
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PAGE = (Path(__file__).resolve().parent / "joystick.html").read_text(encoding="utf-8") if \
    (Path(__file__).resolve().parent / "joystick.html").exists() else "<h1>joystick.html not found</h1>"


class Command:
    """Latest command from the phone. Zero if not updated within timeout."""

    def __init__(self, timeout_s=0.5):
        self.lock = threading.Lock()
        self.timeout = timeout_s
        self.vx = self.vy = self.yaw = 0.0
        self.deadman = False
        self.estop = False
        self.enable = False          # 'START' button: allow the policy to run
        self.t = 0.0
        self.status = {}

    def update(self, d):
        with self.lock:
            self.vx = float(d.get("vx", 0.0)); self.vy = float(d.get("vy", 0.0)); self.yaw = float(d.get("yaw", 0.0))
            self.deadman = bool(d.get("deadman", False))
            if d.get("estop"):
                self.estop = True
            if "enable" in d:
                self.enable = bool(d["enable"])
                if self.enable:
                    self.estop = False   # START also clears E-STOP
            self.t = time.monotonic()

    def get(self):
        """(vx, vy, yaw) in [-1, 1], zero if deadman is not held or the command is stale"""
        with self.lock:
            fresh = (time.monotonic() - self.t) < self.timeout
            if not (fresh and self.deadman):
                return 0.0, 0.0, 0.0
            return self.vx, self.vy, self.yaw

    def clear_enable(self):
        """After a halt/resume: require START again before the policy runs (the web button syncs via /status)."""
        with self.lock:
            self.enable = False

    def set_status(self, **kw):
        with self.lock:
            self.status.update(kw)

    def get_status(self):
        with self.lock:
            return dict(self.status, enable=self.enable, estop=self.estop,
                        age_ms=int((time.monotonic() - self.t) * 1000) if self.t else None)


def make_handler(cmd):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body.encode("utf-8")
            self.send_response(code); self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path.startswith("/status"):
                self._send(200, json.dumps(cmd.get_status()))
            else:
                self._send(200, PAGE, "text/html")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                d = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                d = {}
            cmd.update(d)
            self._send(200, json.dumps(cmd.get_status()))
    return H


class DualStackServer(ThreadingHTTPServer):
    """Accepts both IPv4 and IPv6 (so the phone can reach the Mac/Pi on an IPv6-only hotspot)"""
    address_family = socket.AF_INET6

    def server_bind(self):
        self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        super().server_bind()


def start_server(cmd, port=8080):
    try:
        srv = DualStackServer(("::", port), make_handler(cmd))
    except OSError:
        srv = ThreadingHTTPServer(("0.0.0.0", port), make_handler(cmd))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    c = Command()
    start_server(c, 8080)
    print("Open http://<IP>:8080. Ctrl+C to quit")
    try:
        while True:
            time.sleep(0.2)
            c.set_status(loop_hz=50, voltage=12.3, tilt_deg=1.0)
            print(f"cmd={c.get()} enable={c.enable} estop={c.estop}", end="\r")
    except KeyboardInterrupt:
        pass
