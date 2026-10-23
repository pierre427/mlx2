"""Choice requests convey preferences, never action approval or authorization."""
from __future__ import annotations

import json
import threading
import time
import uuid

from .desktop import atomic_json


class Choices:
    def __init__(self, controller):
        self.home = controller.home / "choices"
        self.home.mkdir(mode=0o700, exist_ok=True)
        self.policy_path = self.home / "policy.json"
        self.policy = {"auto_pick": True, "timeout_seconds": 60}
        if self.policy_path.exists():
            self.policy.update(json.loads(self.policy_path.read_text()))
        self.lock = threading.RLock()
        self.pending = {}

    def snapshot(self):
        return {"policy": dict(self.policy)}

    def dispatch(self, request):
        with self.lock:
            if request["command"] == "choice_policy":
                changes = request["changes"]
                value = {**self.policy, **changes}
                if set(value) != {"auto_pick", "timeout_seconds"} or type(value["auto_pick"]) is not bool or type(value["timeout_seconds"]) is not int or not 5 <= value["timeout_seconds"] <= 3600:
                    raise ValueError("Choice timeout must be 5–3600 seconds with an explicit auto-pick toggle")
                atomic_json(self.policy_path, value)
                self.policy = value
                return self.snapshot()
            if request["command"] == "choice_panel":
                for key, maximum in (("title", 200), ("question", 2000)):
                    if not isinstance(request.get(key), str) or not 1 <= len(request[key]) <= maximum:
                        raise ValueError("Choice panels need a bounded title and question")
                rows = request.get("choices")
                if not isinstance(rows, list) or not 2 <= len(rows) <= 8:
                    raise ValueError("Offer between two and eight choices")
                for row in rows:
                    if not isinstance(row, dict) or set(row) != {"id", "label", "description"}:
                        raise ValueError("Invalid choice fields")
                    for key, maximum in (("id", 64), ("label", 200), ("description", 1000)):
                        if not isinstance(row[key], str) or not (0 if key == "description" else 1) <= len(row[key]) <= maximum:
                            raise ValueError("Choice fields exceed their budget")
                if len({r["id"] for r in rows}) != len(rows) or request.get("recommended_id") not in {r["id"] for r in rows}:
                    raise ValueError("Choices need unique IDs and a valid recommendation")
                if request.get("presentation") not in {"auto", "popup", "object"}:
                    raise ValueError("Unknown choice presentation")
                identity = uuid.uuid4().hex
                value = {k: request[k] for k in ("title", "question", "choices", "recommended_id", "presentation")}
                value.update(panel_id=identity, policy=dict(self.policy), created=time.time(), native_action="choice_panel")
                self.pending = {k: v for k, v in self.pending.items() if time.time() - v["created"] < 7200}
                self.pending[identity] = value
                return value
            if request["command"] == "choice_resolve":
                value = self.pending.get(request["panel_id"])
                if value is None:
                    raise ValueError("Choice panel expired or was already resolved")
                selected, source = request.get("choice_id"), request.get("source")
                if source not in {"user", "timeout", "cancelled"} or (source != "cancelled" and selected not in {r["id"] for r in value["choices"]}):
                    raise ValueError("Invalid choice response")
                if source == "timeout" and (not self.policy["auto_pick"] or selected != value["recommended_id"] or time.time() - value["created"] < value["policy"]["timeout_seconds"]):
                    raise ValueError("Automatic choice is disabled or its deadline has not elapsed")
                receipt = {"panel_id": request["panel_id"], "choice_id": selected, "source": source,
                           "resolved_at": time.time(), "is_authorization": False}
                atomic_json(self.home / (request["panel_id"] + ".json"), receipt)
                del self.pending[request["panel_id"]]
                return receipt
            raise ValueError("Unknown choice operation")
