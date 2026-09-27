"""Research-only HTTP wire gate for an exact-model external supervisor.

This deliberately uses no MLX runtime or production server integration. It
checks the wire/lease boundary left open by the policy model.
"""

from __future__ import annotations

import http.client
import json
from dataclasses import dataclass
from typing import Iterator, Mapping

from scripts.research.mtplx520_supervisor_preflight import (
    Refused,
    RequestLease,
    SupervisorPreflight,
)


@dataclass
class HTTPLeaseStream:
    supervisor: SupervisorPreflight
    lease: RequestLease
    connection: http.client.HTTPConnection
    response: http.client.HTTPResponse
    terminal_seen: bool = False
    done_seen: bool = False
    closed: bool = False

    def events(self) -> Iterator[bytes]:
        """Yield complete SSE frames; require a matching terminal receipt."""
        frame = bytearray()
        try:
            while True:
                line = self.response.readline()
                if not line:
                    raise Refused("child_stream_missing_terminal_receipt")
                frame.extend(line)
                if line not in (b"\n", b"\r\n"):
                    continue
                payload = bytes(frame)
                frame.clear()
                data = [part[5:].strip() for part in payload.splitlines() if part.startswith(b"data:")]
                if not data:
                    continue
                if b"\n".join(data) == b"[DONE]":
                    if not self.terminal_seen:
                        raise Refused("child_stream_missing_terminal_receipt")
                    self.done_seen = True
                    yield payload
                    return
                try:
                    event = json.loads(b"\n".join(data))
                except (ValueError, UnicodeDecodeError) as exc:
                    raise Refused("invalid_child_event") from exc
                if not isinstance(event, dict) or event.get("model") != self.lease.model_id:
                    raise Refused("child_response_model_mismatch")
                if "mlx2" in event:
                    self.supervisor.validate_response(
                        self.lease,
                        response_model=event["model"],
                        child_receipt=event["mlx2"],
                    )
                    self.terminal_seen = True
                yield payload
        finally:
            self.close()

    def close(self) -> None:
        """A cancellation releases only this generation's lease, once."""
        if self.closed:
            return
        self.closed = True
        self.connection.close()
        self.supervisor.finish(self.lease)


def forward_stream(
    supervisor: SupervisorPreflight,
    model_id: str,
    *,
    credential: str,
    body: bytes,
    headers: Mapping[str, str],
    path: str = "/v1/chat/completions",
    max_body_bytes: int = 2 << 20,
) -> HTTPLeaseStream:
    """Admit before touching the child, then pin until terminal/cancel/failure."""
    if path not in ("/v1/chat/completions", "/v1/completions"):
        raise Refused("unsupported_endpoint")
    lease = supervisor.admit(
        model_id,
        credential=credential,
        body_reader=lambda limit: body[:limit],
        max_body_bytes=max_body_bytes,
        required_capabilities=frozenset({"text"}),
    )
    connection = None
    try:
        host, port = lease.child.endpoint
        connection = http.client.HTTPConnection(host, port, timeout=5)
        # Never trust a caller-supplied tenant or authorization header. The
        # single-tenant candidate injects its authenticated identity here.
        child_headers = {
            "Authorization": f"Bearer {credential}",
            "X-Tenant-ID": "default",
            "Content-Type": "application/json",
        }
        connection.request("POST", path, body=body, headers=child_headers)
        response = connection.getresponse()
        if response.status != 200:
            raise Refused("child_http_failure")
        if response.getheader("Content-Type", "").split(";", 1)[0] != "text/event-stream":
            raise Refused("child_content_type_mismatch")
        return HTTPLeaseStream(supervisor, lease, connection, response)
    except BaseException:
        if connection is not None:
            connection.close()
        supervisor.finish(lease)
        raise
