"""Standalone HTTP service for typed decision models."""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ..batch_metrics import HttpRuntimeMetrics, http_metric_route
from ..http_security import authorize, load_api_key, policy_for_bind
from ..prometheus import CONTENT_TYPE
from .qualification import install_qualification, serving_settings
from .runtime import load_decision_engine
from .schema import DecisionExecutionFailure, DecisionRequestError, normalize_request

LOG = logging.getLogger("mlx2.decisions")


def _reject_json_constant(value):
    raise ValueError(f"non-finite JSON number {value!r} is not accepted")


class DuplicateKeyError(ValueError):
    """A request object repeats a key; every artifact reader rejects this too."""


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DuplicateKeyError(f"duplicate object key {key!r} in request body")
        value[key] = item
    return value


def decode_request_body(raw: bytes):
    """Parse one request body; duplicate keys and non-finite numbers raise."""
    return json.loads(
        raw, parse_constant=_reject_json_constant, object_pairs_hook=_unique_pairs
    )


class _HeaderPhaseReader:
    """Bound the whole request-line/header phase, not each socket receive."""

    def __init__(self, rfile, connection, idle_timeout, deadline_seconds):
        self._rfile = rfile
        self._connection = connection
        self._idle_timeout = idle_timeout
        self._deadline_seconds = deadline_seconds
        self.deadline = None
        self.expired = False

    def begin_request(self):
        self.deadline = time.monotonic() + float(self._deadline_seconds())
        self.expired = False

    def readline(self, limit=-1):
        if limit == 0:
            return b""
        chunks = []
        size = 0
        while True:
            if self.deadline is None:
                self._connection.settimeout(self._idle_timeout)
            else:
                budget = self.deadline - time.monotonic()
                if budget <= 0:
                    self.expired = True
                    raise TimeoutError("request headers were not received in time")
                self._connection.settimeout(min(self._idle_timeout, budget))
            try:
                buffered = self._rfile.peek(1)
            except TimeoutError:
                if self.deadline is not None:
                    self.expired = True
                raise
            if not buffered:
                break
            if self.deadline is None:
                self.deadline = time.monotonic() + float(self._deadline_seconds())
            end = buffered.find(b"\n")
            take = len(buffered) if end < 0 else end + 1
            if limit is not None and limit >= 0:
                take = min(take, limit - size)
            chunk = self._rfile.read(take)
            chunks.append(chunk)
            size += len(chunk)
            if chunk.endswith(b"\n") or (limit is not None and 0 <= limit <= size):
                break
        return b"".join(chunks)

    def __getattr__(self, name):
        return getattr(self._rfile, name)


class DecisionApplication:
    """Transport-neutral dispatcher, small enough to exercise without a socket."""

    def __init__(self, engine):
        self.engine = engine
        if not hasattr(engine, "http_metrics"):
            engine.http_metrics = HttpRuntimeMetrics()

    def get(self, path: str):
        if path == "/health":
            return 200, {"status": "ok"}
        if path == "/v1/status":
            return 200, self.engine.status()
        if path == "/v1/models":
            return 200, {
                "object": "list",
                "data": [
                    {
                        "id": self.engine.model_name,
                        "object": "model",
                        "owned_by": "mlx2",
                        "capabilities": list(self.engine.capabilities),
                        "qualification": self.engine.route_receipt(
                            observed_used=False
                        )["qualification"],
                    }
                ],
            }
        return 404, self.error(
            "route not found", code="not_found", error_type="invalid_request_error"
        )

    def post(self, path: str, payload):
        if path != "/v1/systemone":
            return 404, self.error(
                "route not found", code="not_found", error_type="invalid_request_error"
            )
        try:
            request = normalize_request(
                payload,
                default_model=self.engine.model_name,
                reserved_tokens=self.engine.reserved_tokens,
            )
        except DecisionRequestError as error:
            self.engine.record_refusal()
            return error.status, self.error(str(error), code=error.code)
        try:
            return 200, self.engine.predict(request)
        except DecisionRequestError as error:
            return error.status, self.error(str(error), code=error.code)

    def error(
        self,
        message,
        *,
        code="invalid_request",
        error_type="invalid_request_error",
        observed_used=False,
    ):
        return {
            "error": {"message": message, "type": error_type, "code": code},
            "mlx2": self.engine.route_receipt(observed_used=observed_used),
        }


class BoundedDecisionServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args, max_connections=8, **kwargs):
        self.connections = threading.BoundedSemaphore(max_connections)
        super().__init__(*args, **kwargs)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except BaseException:
            self.connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


def make_handler(
    application,
    *,
    security_policy=None,
    max_request_bytes=4 << 20,
    header_timeout_seconds=30.0,
    body_timeout_seconds=30.0,
):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        REQUEST_HEADER_DEADLINE_SECONDS = float(header_timeout_seconds)
        REQUEST_BODY_DEADLINE_SECONDS = float(body_timeout_seconds)

        def setup(self):
            super().setup()
            self.connection.settimeout(30)
            self.rfile = _HeaderPhaseReader(
                self.rfile,
                self.connection,
                30,
                lambda: self.REQUEST_HEADER_DEADLINE_SECONDS,
            )

        def parse_request(self):
            self._http_started_at = application.engine.http_metrics.started()
            self._http_recorded = False
            return super().parse_request()

        def _record_http(self, status):
            if getattr(self, "_http_recorded", False):
                return
            self._http_recorded = True
            started_at = getattr(
                self,
                "_http_started_at",
                application.engine.http_metrics.started(),
            )
            application.engine.http_metrics.completed(
                getattr(self, "command", "other"),
                http_metric_route(getattr(self, "path", "other")),
                int(status),
                started_at,
            )

        def send_error(self, code, message=None, explain=None):
            self._record_http(code)
            return super().send_error(code, message, explain)

        def handle_one_request(self):
            self.rfile.begin_request()
            super().handle_one_request()
            if self.rfile.expired:
                for attribute, value in (
                    ("requestline", ""),
                    ("command", None),
                    ("request_version", self.protocol_version),
                ):
                    if not hasattr(self, attribute):
                        setattr(self, attribute, value)
                self.close_connection = True
                try:
                    self.connection.settimeout(30)
                    self._json(
                        408,
                        application.error(
                            "request headers were not received in time",
                            code="request_timeout",
                        ),
                    )
                except OSError:
                    pass

        def _json(self, status, value, *, headers=None):
            payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
            self._record_http(status)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            for name, content in (headers or {}).items():
                self.send_header(name, content)
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def _text(self, status, value, *, content_type):
            payload = value.encode()
            self._record_http(status)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True

        def _authorize(self):
            try:
                authorize(
                    security_policy,
                    self.command,
                    self.path.split("?", 1)[0],
                    self.headers,
                    local_address=self.connection.getsockname()[0],
                )
                return True
            except Exception as error:
                if not all(hasattr(error, name) for name in ("status", "payload")):
                    LOG.exception("decision HTTP authorization failed")
                    self._json(
                        500,
                        application.error(
                            "authorization failed",
                            code="internal_error",
                            error_type="server_error",
                        ),
                    )
                    return False
                self._json(error.status, error.payload(), headers=error.headers)
                return False

        def do_GET(self):
            self.connection.settimeout(30)
            if not self._authorize():
                return
            path = self.path.split("?", 1)[0]
            if path == "/metrics":
                try:
                    payload = application.engine.prometheus_metrics()
                except Exception:
                    LOG.exception("decision Prometheus exposition failed")
                    self._text(
                        500,
                        "# decision metrics unavailable\n",
                        content_type=CONTENT_TYPE,
                    )
                    return
                self._text(200, payload, content_type=CONTENT_TYPE)
                return
            status, body = application.get(path)
            self._json(status, body)

        def do_POST(self):
            self.connection.settimeout(30)
            if not self._authorize():
                return
            if self.headers.get_all("Transfer-Encoding"):
                self._json(
                    400,
                    application.error(
                        "Transfer-Encoding is unsupported", code="invalid_request"
                    ),
                )
                return
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1:
                status = 411 if not lengths else 400
                self._json(
                    status,
                    application.error(
                        "Content-Length is required exactly once",
                        code="invalid_request",
                    ),
                )
                return
            # int() also accepts a sign, underscores, surrounding spaces and
            # non-ASCII digits, and refuses thousands of digits with an
            # exception; HTTP defines one ASCII decimal byte count, and a
            # proxy may frame the same bytes differently (main server:
            # Handler._content_length).  Twenty digits already exceed any
            # max_request_bytes this service accepts.
            value = lengths[0].strip(" \t")
            if not (value.isascii() and value.isdigit() and len(value) <= 20):
                self._json(
                    400,
                    application.error("Content-Length must be one decimal byte count"),
                )
                return
            length = int(value)
            if length > max_request_bytes:
                self._json(
                    413,
                    application.error(
                        f"request body exceeds {max_request_bytes} bytes",
                        code="request_too_large",
                    ),
                )
                return
            try:
                deadline = time.monotonic() + self.REQUEST_BODY_DEADLINE_SECONDS
                chunks = []
                remaining = length
                while remaining:
                    budget = deadline - time.monotonic()
                    if budget <= 0:
                        raise TimeoutError("request body was not received in time")
                    self.connection.settimeout(min(30, budget))
                    chunk = self.rfile.read1(min(remaining, 1 << 20))
                    if not chunk:
                        raise ValueError("request body ended early")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b"".join(chunks)
                self.connection.settimeout(30)
                payload = decode_request_body(raw)
            except TimeoutError:
                self.connection.settimeout(30)
                self._json(
                    408,
                    application.error(
                        "request body was not received in time",
                        code="request_timeout",
                    ),
                )
                return
            except DuplicateKeyError as error:
                self._json(400, application.error(str(error)))
                return
            except (
                OSError,
                ValueError,
                UnicodeDecodeError,
                json.JSONDecodeError,
                RecursionError,
            ):
                self._json(400, application.error("request body must be valid JSON"))
                return
            try:
                status, body = application.post(self.path.split("?", 1)[0], payload)
                self._json(status, body)
            except DecisionExecutionFailure as error:
                LOG.exception("decision request failed after engine dispatch")
                self._json(
                    500,
                    application.error(
                        "decision execution failed",
                        code="internal_error",
                        error_type="server_error",
                        observed_used=error.observed_used,
                    ),
                )
                return
            except Exception:
                LOG.exception("decision request failed")
                self._json(
                    500,
                    application.error(
                        "decision execution failed",
                        code="internal_error",
                        error_type="server_error",
                    ),
                )
                return

        def log_message(self, format, *args):
            LOG.info("%s - %s", self.address_string(), format % args)

    return Handler


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, help="local prepared decision-model artifact"
    )
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8291)
    parser.add_argument("--max-connections", type=int, default=8)
    parser.add_argument("--max-request-bytes", type=int, default=4 << 20)
    parser.add_argument(
        "--qualification",
        help="revision-bound decision qualification receipt to validate and install",
    )
    parser.add_argument("--allowed-host", action="append", default=[])
    key = parser.add_mutually_exclusive_group()
    key.add_argument("--api-key-file")
    key.add_argument("--api-key-env")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if not 1 <= args.max_connections <= 128:
        parser.error("--max-connections must be between 1 and 128")
    if not 1024 <= args.max_request_bytes <= 16 << 20:
        parser.error("--max-request-bytes must be between 1024 and 16777216")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        security = policy_for_bind(
            args.host,
            allowed_hosts=args.allowed_host,
            api_key=load_api_key(
                key_file=args.api_key_file,
                key_env=args.api_key_env,
            ),
            logger=LOG,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    engine = load_decision_engine(args.model, served_model_name=args.served_model_name)
    settings = serving_settings(
        engine,
        max_connections=args.max_connections,
        max_request_bytes=args.max_request_bytes,
    )
    if args.qualification:
        try:
            state = install_qualification(
                args.qualification,
                engine=engine,
                settings=settings,
            )
        except (OSError, TypeError, ValueError) as error:
            engine.close()
            parser.error(str(error))
        LOG.info(
            "installed qualified decision route receipt %s",
            state["receipt_sha256"],
        )
    else:
        LOG.warning(
            "decision serving is implemented but unqualified; media is unsupported"
        )
    application = DecisionApplication(engine)
    handler = make_handler(
        application,
        security_policy=security,
        max_request_bytes=args.max_request_bytes,
    )
    server = BoundedDecisionServer(
        (args.host, args.port), handler, max_connections=args.max_connections
    )
    try:
        LOG.info("serving %s on http://%s:%d", engine.model_name, args.host, args.port)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
