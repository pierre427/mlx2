"""Correction cases and residual-ablation overlays behind the desktop control plane.

A *correction case* is one flagged assistant reply from a chat (session and
branch identifiers as the chat archive uses them, the history up to the
flagged turn, the reply itself, the model checkpoint and the reason). It is a
must-fix test for an edit, never training data: one example cannot estimate a
direction.

An *overlay* is a content-addressed, checkpoint-bound description of a runtime
edit: a reference to a capture's refusal directions plus strength parameters in
``AblationParameters.as_dict`` shape. A server applies it with exact projection
hooks; nothing here writes a model. The serving API is specified in
``docs/research/heretic-kimi-linear-20261105/OVERLAY-API.md``; every call that
needs it probes the capability first and fails closed.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

from .desktop import atomic_json
from .desktop_residual_ablation import ablation_parameters, plan_preview, runner_facts

REASONS = ("refused", "redirected", "lectured", "other")
ROLES = ("system", "user", "assistant")
OVERLAY_API_VERSION = 1
OVERLAY_SCHEMA = "mlx2.residual-ablation-overlay.v1"
CASE_SCHEMA = "mlx2.residual-ablation-case.v1"
MAX_HISTORY = 400
MAX_CASE_BYTES = 4 << 20
CONTRAST_FLAG = "--extra-bad-prompts"
_ID = re.compile(r"^[a-z0-9-]{1,64}$")


def _text(value, name, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be text of at most {limit} characters")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value or len(value.encode()) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a nonempty bounded identifier")
    return value


def fingerprint_of(checkpoint):
    if not isinstance(checkpoint, str) or "@" not in checkpoint or checkpoint.startswith("*@"):
        return None
    return checkpoint.rsplit("@", 1)[1] or None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def loopback(url):
    parsed = urlparse(url or "")
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1") or not parsed.port:
        raise ValueError("Overlay calls go only to a loopback server (http://127.0.0.1:<port>)")
    return f"http://{parsed.hostname}:{parsed.port}"


def _http(method, url, body=None, token=None, timeout=5.0):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")
    if token:
        request.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 -- loopback only, checked by caller
        return json.loads(response.read() or b"{}")


def probe_capability(url, *, http=_http):
    """Does the server at ``url`` implement the overlay API? Fails closed with a reason."""
    try:
        base = loopback(url)
    except ValueError as error:
        return {"supported": False, "reason": str(error)}
    try:
        value = http("GET", base + "/v1/overlays", timeout=2.0)
    except urllib.error.HTTPError as error:
        reason = ("This server has no overlay API yet (GET /v1/overlays → 404)" if error.code == 404
                  else f"Overlay API probe failed (HTTP {error.code})")
        return {"supported": False, "reason": reason}
    except (OSError, ValueError) as error:
        return {"supported": False, "reason": f"Server not reachable for the overlay probe: {type(error).__name__}"}
    if not isinstance(value, dict) or value.get("object") != "overlay_list" or value.get("api_version") != OVERLAY_API_VERSION:
        return {"supported": False, "reason": "Server answered /v1/overlays without the v1 overlay contract"}
    return {"supported": True, "reason": None, "fingerprint": value.get("checkpoint_fingerprint"),
            "loaded": [row.get("id") for row in value.get("data", []) if isinstance(row, dict)],
            "preview": bool(value.get("preview"))}


class Corrections:
    def __init__(self, controller, *, http=_http):
        self.controller = controller
        self.root = Path(controller.home) / "residual-ablation"
        self.cases_dir = self.root / "cases"
        self.overlays_dir = self.root / "overlays"
        self.lock = threading.RLock()
        self.http = http
        self.previews = {}  # case id -> running thread
        self._summaries = {}
        self.closed = False

    # ------------------------------------------------------------- cases

    def create_case(self, request):
        history = request.get("history")
        if not isinstance(history, list) or len(history) > MAX_HISTORY:
            raise ValueError(f"History must be a list of at most {MAX_HISTORY} messages")
        clean = []
        for row in history:
            if not isinstance(row, dict) or row.get("role") not in ROLES:
                raise ValueError("History messages need a system, user or assistant role")
            clean.append({"role": row["role"], "text": _text(row.get("text"), "Message text", 1 << 20),
                          **({"id": _identifier(row["id"], "Message id")} if row.get("id") else {})})
        if not clean or clean[-1]["role"] != "user":
            raise ValueError("The flagged reply must answer a user message")
        reason = request.get("reason")
        if reason not in REASONS:
            raise ValueError("Reason must be refused, redirected, lectured or other")
        model_path = _text(request.get("model_path") or "", "Model path", 4096)
        checkpoint = request.get("checkpoint")
        case = {
            "schema": CASE_SCHEMA, "id": "case-" + uuid.uuid4().hex[:16], "created": time.time(),
            "session_id": _identifier(request.get("session_id"), "Session id"),
            "branch_id": _identifier(request.get("branch_id"), "Branch id"),
            "flagged_message_id": _identifier(request.get("flagged_message_id"), "Flagged message id"),
            "history": clean, "flagged_text": _text(request.get("flagged_text"), "Flagged reply", 1 << 20),
            "model_path": model_path, "checkpoint": checkpoint if fingerprint_of(checkpoint) else None,
            "fingerprint": fingerprint_of(checkpoint), "reason": reason,
            "note": _text(request.get("note") or "", "Note", 4000), "replays": [], "previews": [],
            "use": "must-fix test (not training data)",
        }
        if len(canonical(case).encode()) > MAX_CASE_BYTES:
            raise ValueError("This conversation is too long to flag (4 MiB case limit)")
        with self.lock:
            self.cases_dir.mkdir(parents=True, exist_ok=True)
            atomic_json(self.cases_dir / f"{case['id']}.json", case)
        return self.summary(case)

    def _case_path(self, identity):
        if not isinstance(identity, str) or not identity.startswith("case-") or not _ID.match(identity):
            raise ValueError("Unknown correction case")
        return self.cases_dir / f"{identity}.json"

    def get_case(self, identity):
        path = self._case_path(identity)
        if not path.is_file():
            raise ValueError("Unknown correction case")
        return json.loads(path.read_text())

    @staticmethod
    def summary(case):
        prompt = next((m["text"] for m in reversed(case["history"]) if m["role"] == "user"), "")
        return {key: case.get(key) for key in ("id", "created", "session_id", "branch_id", "flagged_message_id",
                                               "checkpoint", "fingerprint", "model_path", "reason", "note")} | {
            "prompt_excerpt": prompt[:240], "flagged_excerpt": case["flagged_text"][:240],
            "turns": len(case["history"]), "replays": case.get("replays", []),
            "previews": len(case.get("previews", []))}

    def list_cases(self):
        rows = []
        if self.cases_dir.is_dir():
            for path in sorted(self.cases_dir.glob("case-*.json")):
                # Summaries are cached on mtime: the snapshot polls every 2 s
                # and a case may hold megabytes of history.
                stamp = path.stat().st_mtime_ns
                cached = self._summaries.get(path.name)
                if cached and cached[0] == stamp:
                    row = dict(cached[1])
                else:
                    try:
                        row = self.summary(json.loads(path.read_text()))
                    except ValueError:
                        continue
                    self._summaries[path.name] = (stamp, dict(row))
                case = row
                thread = self.previews.get(case["id"])
                row["previewing"] = bool(thread and thread.is_alive())
                rows.append(row)
        return sorted(rows, key=lambda r: -r["created"])

    def delete_case(self, identity):
        path = self._case_path(identity)
        with self.lock:
            if not path.is_file():
                raise ValueError("Unknown correction case")
            path.unlink()
            self.previews.pop(identity, None)
        return {"deleted": identity}

    def record_replay(self, request):
        """The chat created a replay branch for this case (original branch untouched)."""
        with self.lock:
            case = self.get_case(request.get("case_id"))
            overlay = self.get_overlay(request.get("overlay_id"))
            if case.get("fingerprint") and overlay["binding"]["fingerprint"] != case["fingerprint"]:
                raise ValueError("Overlay is bound to a different checkpoint than this case")
            case["replays"].append({"branch_id": _identifier(request.get("branch_id"), "Branch id"),
                                    "overlay_id": overlay["id"], "created": time.time()})
            atomic_json(self._case_path(case["id"]), case)
            return self.summary(case)

    # ----------------------------------------------------------- overlays

    def _index_path(self):
        return self.overlays_dir / "index.json"

    def _index(self):
        try:
            value = json.loads(self._index_path().read_text())
            return value if isinstance(value, dict) else {}
        except (OSError, ValueError):
            return {}

    def create_overlay(self, request):
        """Content-addressed overlay: capture directions ref + strength parameters + checkpoint binding."""
        capture = Path(_text(request.get("capture") or "", "Capture folder", 4096)).expanduser()
        if not capture.is_absolute():
            raise ValueError("Capture folder must be an absolute path")
        capture = capture.resolve(strict=True)
        report = json.loads((capture / "report.json").read_text())
        meta = json.loads((capture / "directions.json").read_text())
        if report.get("status") != "ok" or report.get("phase") != "capture":
            raise ValueError("Overlays need a completed capture")
        if meta.get("sha256") != report.get("directions_sha256"):
            raise ValueError("Capture report and directions file disagree")
        npz = capture / "directions.npz"
        checkpoint = report.get("checkpoint")
        expected = fingerprint_of(request.get("checkpoint"))
        if expected and expected != fingerprint_of(checkpoint):
            raise ValueError(f"The capture was measured on {checkpoint}, not on {request.get('checkpoint')}")
        params = ablation_parameters(request.get("params")).as_dict()
        layers = len(report.get("separation_d") or [])
        preview = plan_preview(params, layers)
        source = request.get("source") or {"kind": "custom"}
        if not isinstance(source, dict) or source.get("kind") not in ("trial", "custom", "probe"):
            raise ValueError("Overlay source must be a trial, custom or probe profile")
        body = {
            "schema": OVERLAY_SCHEMA, "api_version": OVERLAY_API_VERSION,
            "binding": {"checkpoint": checkpoint, "fingerprint": fingerprint_of(checkpoint)},
            "directions": {"path": str(npz), "vectors_sha256": meta["sha256"],
                           "file_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                           "shape": meta.get("shape"), "capture": str(capture)},
            "parameters": params, "row_normalization": "none", "layers": layers, "edits": preview["edits"],
        }
        identity = "ovl-" + hashlib.sha256(canonical(body).encode()).hexdigest()[:20]
        label = _text(request.get("label") or f"{source.get('kind')} {source.get('trial', '')}".strip(), "Label", 120)
        overlay = {**body, "id": identity, "label": label, "source": source, "created": time.time(),
                   "qualification": "unqualified"}
        with self.lock:
            folder = self.overlays_dir / identity
            folder.mkdir(parents=True, exist_ok=True)
            if not (folder / "overlay.json").is_file():
                atomic_json(folder / "overlay.json", overlay)
            index = self._index()
            index[identity] = {"id": identity, "label": label, "fingerprint": body["binding"]["fingerprint"],
                               "checkpoint": checkpoint, "created": overlay["created"], "edits": preview["edits"],
                               "source": source}
            atomic_json(self._index_path(), index)
        return json.loads((self.overlays_dir / identity / "overlay.json").read_text())

    def get_overlay(self, identity):
        if not isinstance(identity, str) or not identity.startswith("ovl-") or not _ID.match(identity):
            raise ValueError("Unknown overlay")
        path = self.overlays_dir / identity / "overlay.json"
        if not path.is_file():
            raise ValueError("Unknown overlay")
        return json.loads(path.read_text())

    def list_overlays(self):
        return sorted(self._index().values(), key=lambda r: -r["created"])

    def delete_overlay(self, identity):
        overlay = self.get_overlay(identity)
        with self.lock:
            (self.overlays_dir / identity / "overlay.json").unlink()
            try:
                (self.overlays_dir / identity).rmdir()
            except OSError:
                pass
            index = self._index()
            index.pop(identity, None)
            atomic_json(self._index_path(), index)
        return {"deleted": overlay["id"]}

    # ------------------------------------------------------ server side (API)

    def _admin_token(self, base):
        """Admin token of the app-managed worker listening at ``base``; fail closed otherwise."""
        from .http_security import load_secret_file

        for engine in getattr(self.controller, "engines", {}).values():
            worker = getattr(engine, "worker", None) or {}
            if worker.get("url") and loopback(worker["url"]) == base and worker.get("admin_token_file"):
                return load_secret_file(Path(worker["admin_token_file"]), flag="--admin-token-file")
        raise ValueError("Overlays load only on an app-managed server (its admin key is needed)")

    def capability(self, url):
        return probe_capability(url, http=self.http)

    def _require(self, url, overlay=None):
        state = self.capability(url)
        if not state["supported"]:
            raise ValueError(state["reason"])
        if overlay is not None and state.get("fingerprint") and state["fingerprint"] != overlay["binding"]["fingerprint"]:
            raise ValueError("The server's model is a different checkpoint than this overlay is bound to")
        return state

    def load_overlay(self, request):
        overlay = self.get_overlay(request.get("overlay_id"))
        base = loopback(request.get("url"))
        self._require(base, overlay)
        token = self._admin_token(base)
        return self.http("POST", base + "/v1/admin/overlays", {"overlay": overlay}, token=token, timeout=60.0)

    def unload_overlay(self, request):
        base = loopback(request.get("url"))
        self._require(base)
        identity = request.get("overlay_id")
        self.get_overlay(identity)
        return self.http("DELETE", base + f"/v1/admin/overlays/{identity}", token=self._admin_token(base), timeout=30.0)

    def start_preview(self, request):
        """Regenerate the flagged reply under an overlay (or ad-hoc parameters) as a background job."""
        case = self.get_case(request.get("case_id"))
        base = loopback(request.get("url"))
        overlay = self.get_overlay(request["overlay_id"]) if request.get("overlay_id") else None
        state = self._require(base, overlay)
        if not state.get("preview"):
            raise ValueError("This server's overlay API has no preview call")
        if overlay is None:
            params = ablation_parameters(request.get("params")).as_dict()
            capture = _text(request.get("capture") or "", "Capture folder", 4096)
            selection = {"parameters": params, "directions": {"capture": capture}}
        else:
            # Server contract (validate_overlay_selection): {"id", "branch_id"}.
            selection = {"id": overlay["id"]}
        # Ad-hoc previews read host files on the server and need its admin token.
        preview_token = self._admin_token(base) if overlay is None else None
        if case.get("fingerprint") and state.get("fingerprint") and state["fingerprint"] != case["fingerprint"]:
            raise ValueError("The server's model is a different checkpoint than the flagged chat used")
        with self.lock:
            running = self.previews.get(case["id"])
            if running and running.is_alive():
                raise ValueError("A preview for this case is already running")
            body = {"messages": [{"role": m["role"], "content": m["text"]} for m in case["history"]],
                    "max_tokens": int(request.get("max_tokens") or 256), "temperature": 0,
                    "mlx2_overlay": selection, "checkpoint_fingerprint": case.get("fingerprint")}
            key = hashlib.sha256(canonical(selection).encode()).hexdigest()[:16]

            def work():
                entry = {"key": key, "selection": selection, "started": time.time()}
                try:
                    reply = self.http("POST", base + "/v1/overlays/preview", body, token=preview_token,
                                      timeout=600.0)
                    entry.update(state="completed", text=reply.get("text"), receipt=reply.get("route_receipt"))
                except Exception as error:  # noqa: BLE001 -- preview failure is recorded on the case
                    entry.update(state="failed", error=f"{type(error).__name__}: {error}"[:400])
                entry["finished"] = time.time()
                with self.lock:
                    try:
                        current = self.get_case(case["id"])
                    except ValueError:
                        return
                    current["previews"] = (current.get("previews", []) + [entry])[-20:]
                    atomic_json(self._case_path(case["id"]), current)

            thread = threading.Thread(target=work, daemon=True, name="mlx2-ablation-preview")
            self.previews[case["id"]] = thread
            thread.start()
        return {"started": True, "key": key}

    def export_contrast(self, request):
        """Write the flagged prompts as an extra harmful-prompt file for a future capture."""
        identities = request.get("case_ids")
        if not isinstance(identities, list) or not identities:
            raise ValueError("Choose flagged cases to add")
        workspace = Path(_text(request.get("workspace") or "", "Evidence folder", 4096)).expanduser()
        if not workspace.is_absolute():
            raise ValueError("Evidence folder must be an absolute path")
        rows = []
        for identity in identities:
            case = self.get_case(identity)
            prompt = next(m["text"] for m in reversed(case["history"]) if m["role"] == "user")
            rows.append({"text": prompt, "case_id": case["id"], "reason": case["reason"]})
        folder = workspace / "contrast"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"flagged-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{uuid.uuid4().hex[:6]}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        supported = CONTRAST_FLAG in runner_facts()["flags"]
        return {"path": str(path), "prompts": len(rows), "runner_flag": CONTRAST_FLAG, "runner_supports": supported,
                "note": None if supported else f"Saved. The capture runner needs {CONTRAST_FLAG} before Measure can use it."}

    def snapshot(self):
        return {"cases": self.list_cases(), "overlays": self.list_overlays(), "reasons": list(REASONS),
                "contrast_flag_supported": CONTRAST_FLAG in runner_facts()["flags"]}

    def dispatch(self, request):
        command = request["command"]
        handlers = {
            "ablation_case_create": lambda: self.create_case(request),
            "ablation_case_get": lambda: self.get_case(request.get("case_id")),
            "ablation_case_delete": lambda: self.delete_case(request.get("case_id")),
            "ablation_case_replay": lambda: self.record_replay(request),
            "ablation_case_preview": lambda: self.start_preview(request),
            "ablation_case_contrast": lambda: self.export_contrast(request),
            "ablation_overlay_create": lambda: self.create_overlay(request),
            "ablation_overlay_get": lambda: self.get_overlay(request.get("overlay_id")),
            "ablation_overlay_delete": lambda: self.delete_overlay(request.get("overlay_id")),
            "ablation_overlay_capability": lambda: self.capability(request.get("url")),
            "ablation_overlay_load": lambda: self.load_overlay(request),
            "ablation_overlay_unload": lambda: self.unload_overlay(request),
        }
        if command not in handlers:
            raise ValueError("Unknown correction operation")
        return handlers[command]()

    def close(self):
        self.closed = True
        for thread in list(self.previews.values()):
            thread.join(timeout=2)
