"""Original, versioned MLX2 harness instructions, installed as editable saved prompts."""
from __future__ import annotations

import hashlib
import math
import uuid
from pathlib import Path

PRESETS = {
    "full": ("MLX2 Agent · Full", "mlx2-agent-full.md"),
    "compact": ("MLX2 Agent · Compact", "mlx2-agent-compact.md"),
}


def load(preset):
    if preset not in PRESETS:
        raise ValueError("Choose the full or compact MLX2 agent preset")
    name, filename = PRESETS[preset]
    text = (Path(__file__).parent / "prompts" / filename).read_text(encoding="utf-8")
    if not text.strip() or len(text) > 65536:
        raise ValueError("The bundled prompt is empty or exceeds the saved prompt budget")
    fingerprint = hashlib.sha256(text.encode()).hexdigest()
    return {"id": "mlx2-agent-" + preset + "-" + fingerprint[:12], "name": name, "text": text}


def catalog():
    result = []
    for preset in PRESETS:
        prompt = load(preset)
        result.append({"preset": preset, "name": prompt["name"], "characters": len(prompt["text"]),
                       "estimated_tokens": math.ceil(len(prompt["text"]) / 4),
                       "estimate_basis": "characters / 4; exact count depends on the model tokenizer",
                       "sha256": hashlib.sha256(prompt["text"].encode()).hexdigest()})
    return {"presets": result, "activation": "Installed copies are editable saved prompts. Context is checked before generation; choose compact for smaller windows."}


def install(controller, preset, activate):
    if type(activate) is not bool:
        raise ValueError("activate must be true or false")
    prompt = load(preset)
    rows = list(controller.preferences["system_prompts"])
    existing = next((row for row in rows if row["id"] == prompt["id"]), None)
    if existing and existing != prompt:
        # Preserve edits to a previously imported preset.
        prompt = {**prompt, "id": str(uuid.uuid4())}
        existing = None
    if existing is None:
        if len(rows) >= 64:
            raise ValueError("Save at most 64 system prompts; remove an unused prompt before importing a preset")
        rows.append(prompt)
    changes = {"system_prompts": rows}
    if activate:
        changes["base_system_prompt_id"] = prompt["id"]
    controller.configure_preferences(changes)
    return {"prompt_id": prompt["id"], "name": prompt["name"], "activated": activate,
            "applies_to": "future chat turns", "custom_prompts_preserved": True}
