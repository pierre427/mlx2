"""A supported HTTP burst must reach bounded handler admission intact."""

from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
import threading
import time
from urllib.request import Request, urlopen

from mlx2.server import BoundedHTTPServer


def test_sixteen_request_burst_survives_a_busy_accept_loop():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")

    server = BoundedHTTPServer(("127.0.0.1", 0), Handler, max_connections=32)
    barrier = threading.Barrier(17)
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    url = f"http://127.0.0.1:{server.server_address[1]}"

    def request(_):
        barrier.wait(timeout=5)
        with urlopen(Request(url, data=b"x" * 100), timeout=5) as response:
            return response.status, response.read()

    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(request, i) for i in range(16)]
            barrier.wait(timeout=5)
            # Model work or thread scheduling can briefly delay accept. The
            # connection cohort must remain queued until handlers can run.
            time.sleep(0.2)
            thread.start()
            assert [future.result(timeout=6) for future in futures] == [(200, b"OK")] * 16
    finally:
        if thread.is_alive():
            server.shutdown()
            thread.join(timeout=5)
        server.server_close()
