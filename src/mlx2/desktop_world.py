"""Portable declarative native scenes. Scene documents never contain executable code."""
from __future__ import annotations

import base64
import copy
import hashlib
import html
import io
import json
import math
import os
import re
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from .desktop_lane import LANE

KINDS = {"chat", "memory", "knowledge", "web", "file", "folder", "avatar", "media"}
SHAPES = {"panel", "sphere", "cube", "ring"}
ANIMATIONS = {"none", "pulse", "spin", "float"}
OBJECT_FIELDS = ("id", "kind", "label", "position", "rotation", "scale", "color", "shape", "text", "resource", "animation")
# World documents are stored as <id>.json beside these application files.
RESERVED_WORLD_IDS = {"state", "assets"}
MAX_SAVED_WORLDS = 256


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Invalid world or object identifier")
    return value


def color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Colors must be #RRGGBB")
    return value


def vector(value, low, high):
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) or not low <= v <= high or not math.isfinite(v) for v in value):
        raise ValueError("Invalid bounded three dimensional transform")
    return value


def validate_object(value):
    if not isinstance(value, dict):
        raise ValueError("A scene object must be an object")
    missing, unexpected = set(OBJECT_FIELDS) - set(value), set(value) - set(OBJECT_FIELDS)
    if missing or unexpected:
        raise ValueError("Unexpected or missing scene object fields"
                         + (f"; missing: {', '.join(sorted(missing))}" if missing else "")
                         + (f"; unexpected: {', '.join(sorted(map(str, unexpected)))}" if unexpected else ""))
    identifier(value["id"])
    if not isinstance(value["kind"], str) or not isinstance(value["shape"], str) or value["kind"] not in KINDS or value["shape"] not in SHAPES:
        raise ValueError("Unsupported scene object kind or shape")
    for key, maximum in (("label", 200), ("text", 8192), ("resource", 4096)):
        if not isinstance(value[key], str) or len(value[key]) > maximum:
            raise ValueError("Scene object text exceeds its budget")
    resource = value["resource"]
    if resource and not (resource.startswith(("https://", "/", "memory:", "capsule:", "knowledge:"))):
        raise ValueError("Resources must be explicit references, local paths or HTTPS links")
    vector(value["position"], -100, 100)
    vector(value["rotation"], -360, 360)
    vector(value["scale"], 0.01, 20)
    color(value["color"])
    animation = value["animation"]
    if not isinstance(animation, dict) or set(animation) != {"kind", "speed"} or not isinstance(animation["kind"], str) or animation["kind"] not in ANIMATIONS:
        raise ValueError("Unsupported declarative animation")
    # Bounds before isfinite: an integer beyond float range must be a refusal, not OverflowError.
    if type(animation["speed"]) not in (int, float) or not 0.05 <= animation["speed"] <= 5 or not math.isfinite(animation["speed"]):
        raise ValueError("Animation speed must be between 0.05 and 5")
    return copy.deepcopy(value)


def object_defaults(identity, kind, accent="#38CBFF"):
    return {"id": identity, "kind": kind, "label": identity, "position": [0, 0, 0], "rotation": [0, 0, 0],
            "scale": [1, 1, 1], "color": accent, "shape": "sphere" if kind == "avatar" else "panel",
            "text": "", "resource": "", "animation": {"kind": "none", "speed": 1}}


def merge_object(existing, patch, accent="#38CBFF"):
    """Apply a partial object edit; a new object needs an id and kind, other fields default."""
    if not isinstance(patch, dict):
        raise ValueError("A scene object must be an object")
    unexpected = set(patch) - set(OBJECT_FIELDS)
    if unexpected:
        raise ValueError(f"Unexpected scene object fields: {', '.join(sorted(map(str, unexpected)))}")
    identifier(patch.get("id"))
    if existing is None:
        if "kind" not in patch:
            raise ValueError(f"Scene object {patch['id']} does not exist; a new object needs a kind")
        base = object_defaults(patch["id"], patch["kind"], accent)
    else:
        base = copy.deepcopy(existing)
    merged = {**base, **copy.deepcopy(patch)}
    if isinstance(base["animation"], dict) and isinstance(patch.get("animation"), dict):
        merged["animation"] = {**base["animation"], **patch["animation"]}
    return validate_object(merged)


def default_world(name="My world"):
    return {"schema_version": 1, "id": uuid.uuid4().hex, "name": name,
            "theme": {"background": "#071321", "accent": "#38CBFF", "exposure": 1},
            "transitions": {"duration": 0.6, "style": "ease"},
            "objects": [{"id": "chat", "kind": "chat", "label": "Chat", "position": [0, 0, 0],
                         "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#38CBFF", "shape": "panel",
                         "text": "Your conversation", "resource": "", "animation": {"kind": "none", "speed": 1}},
                        {"id": "agent", "kind": "avatar", "label": "Agent", "position": [3, 0.4, 0],
                         "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#38CBFF", "shape": "sphere",
                         "text": "", "resource": "", "animation": {"kind": "pulse", "speed": 1}}]}


def migrate_world(value):
    """Repair stored scenes that earlier releases accepted but current rules refuse."""
    if not isinstance(value, dict):
        return value
    value = copy.deepcopy(value)
    if isinstance(value.get("name"), str) and not value["name"].strip():
        value["name"] = "My world"
    if isinstance(value.get("id"), str) and value["id"].casefold() in RESERVED_WORLD_IDS:
        value["id"] = uuid.uuid4().hex
    return value


def validate_world(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "id", "name", "theme", "transitions", "objects"} or value["schema_version"] != 1:
        raise ValueError("Unsupported world document schema")
    identifier(value["id"])
    # Worlds are stored as <id>.json on a case-insensitive volume: STATE is state.
    if value["id"].casefold() in RESERVED_WORLD_IDS:
        raise ValueError("That world identifier is reserved")
    if not isinstance(value["name"], str) or not value["name"].strip() or len(value["name"]) > 200:
        raise ValueError("A world needs a non-blank name of at most 200 characters")
    theme = value["theme"]
    if not isinstance(theme, dict) or set(theme) != {"background", "accent", "exposure"}:
        raise ValueError("Invalid scene theme")
    color(theme["background"])
    color(theme["accent"])
    if type(theme["exposure"]) not in (int, float) or not 0.1 <= theme["exposure"] <= 4 or not math.isfinite(theme["exposure"]):
        raise ValueError("Exposure must be between 0.1 and 4")
    transitions = value["transitions"]
    if not isinstance(transitions, dict) or set(transitions) != {"duration", "style"} or not isinstance(transitions["style"], str) or transitions["style"] not in {"ease", "linear", "instant"}:
        raise ValueError("Invalid scene transitions")
    if type(transitions["duration"]) not in (int, float) or not 0 <= transitions["duration"] <= 5 or not math.isfinite(transitions["duration"]):
        raise ValueError("Transition duration must be between zero and five seconds")
    if not isinstance(value["objects"], list) or not 1 <= len(value["objects"]) <= 128:
        raise ValueError("A world needs between one and 128 objects")
    objects = [validate_object(obj) for obj in value["objects"]]
    if len({o["id"] for o in objects}) != len(objects) or not any(o["kind"] == "chat" for o in objects):
        raise ValueError("World object identities must be unique and include a chat object")
    if len(json.dumps(value, allow_nan=False).encode()) > 4 * 1024 * 1024:
        raise ValueError("World scene metadata exceeds four MiB")
    return copy.deepcopy(value)


class Worlds:
    def __init__(self, controller):
        self.automatic_image_roots = [(controller.home / name).resolve() for name in ("shared-media", "canonical-media")]
        self.home = controller.home / "worlds"
        self.home.mkdir(mode=0o700, exist_ok=True)
        self.lock = threading.RLock()
        self.assets = self.home / "assets"
        self.assets.mkdir(mode=0o700, exist_ok=True)
        self.search_results = {}
        self.seen_artifacts = {}
        self.policy = {"autonomous_scene": False}
        state = self.home / "state.json"
        self.startup_notice = None
        if state.is_file():
            try:
                value = json.loads(state.read_text())
                policy = value.get("policy")
                # Anything but an explicit boolean grant leaves autonomous editing off.
                if isinstance(policy, dict) and set(policy) == {"autonomous_scene"} and type(policy["autonomous_scene"]) is bool:
                    self.policy = dict(policy)
                seen = value.get("seen_artifacts", {})
                self.seen_artifacts = seen if isinstance(seen, dict) and all(isinstance(v, list) for v in seen.values()) else {}
                self.active = validate_world(migrate_world(value["active"]))
            except (ValueError, TypeError, KeyError, AttributeError, ArithmeticError, RecursionError) as error:
                # A world the current rules reject must not stop the application.
                # Keep the old state for inspection and start from a fresh scene.
                aside = self.home / f"state.json.invalid-{uuid.uuid4().hex[:8]}"
                os.replace(state, aside)
                self.startup_notice = f"The saved active world could not be loaded ({error}); it was kept as {aside.name}."
                self.active = default_world()
            self.persist()
        else:
            self.active = default_world()
            self.persist()

    def write(self, path, document):
        fd, name = tempfile.mkstemp(dir=self.home, prefix=".world-")
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(document, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    def persist(self):
        self.write(self.home / (self.active["id"] + ".json"), self.active)
        self.write(self.home / "state.json", {"active": self.active, "policy": self.policy, "seen_artifacts": self.seen_artifacts})

    def sync_artifacts(self, jobs):
        """Create each completed artifact's representation once, then leave edits to its owner."""
        with self.lock:
            seen = self.seen_artifacts.setdefault(self.active["id"], [])
            # An artifact is represented once, in the world active when it completed;
            # creating or switching worlds must not replay the job history.
            # Entries are finished job ids and the artifact object identities already
            # created, so removing an object (or switching worlds) never re-creates it.
            everywhere = {entry for rows in self.seen_artifacts.values() for entry in rows}
            changed = False
            for job in jobs:
                if job.get("kind") not in {"media", "helper", "adapter", "document"} or job.get("state") not in {"completed", "succeeded", "partial"} or job["id"] in everywhere:
                    continue
                if len(self.active["objects"]) >= 128:
                    break
                result = job.get("result") or {}
                paths = []
                def visit(value, depth=0, paths=paths):
                    if depth > 5:
                        return
                    if isinstance(value, dict):
                        for key, item in value.items():
                            if key in {"path", "artifact_path", "output_path", "audio_path", "video_path", "proposal_path"} and isinstance(item, str) and item.startswith("/"):
                                paths.append(item)
                            elif isinstance(item, (dict, list)):
                                visit(item, depth + 1)
                    elif isinstance(value, list):
                        for item in value[:16]:
                            visit(item, depth + 1)
                visit(result)
                capability = job.get("spec", {}).get("capability", "document")
                for index, path in enumerate(dict.fromkeys(paths)):
                    if len(self.active["objects"]) >= 128:
                        break
                    # Object identities must stay loadable whatever a job record's id contains.
                    identity = "artifact-" + re.sub(r"[^A-Za-z0-9_-]", "_", str(job["id"]))[:40] + "-" + str(index)
                    if identity in everywhere or any(o["id"] == identity for o in self.active["objects"]):
                        continue
                    resource = path
                    if capability == "image":
                        try:
                            file = Path(path)
                            resolved = file.resolve(strict=True)
                            if any(resolved.is_relative_to(root) for root in self.automatic_image_roots) and resolved.is_file() and not file.is_symlink() and resolved.stat().st_size <= 8 * 1024 * 1024:
                                resource = self.store_image(resolved.read_bytes())
                        except Exception:
                            # A damaged image stays a file reference; it must not
                            # break every later snapshot.
                            pass
                    count = len(self.active["objects"])
                    self.active["objects"].append({"id": identity, "kind": "media" if capability in {"image", "video", "music", "voice"} else "file",
                        "label": Path(path).name[:200], "position": [((count % 5) - 2) * 2.5, 0, -2 - (count // 5) * 2],
                        "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#38CBFF", "shape": "panel",
                        "text": (capability + " · " + job["state"] + "\nJob: " + job["id"] + "\n" + path)[:8192],
                        "resource": resource[:4096], "animation": {"kind": "none", "speed": 1}})
                    seen.append(identity)
                    everywhere.add(identity)
                    changed = True
                # Partial jobs and jobs cut off by the object cap are revisited;
                # per-path identities keep their existing objects unique.
                if job.get("state") != "partial" and len(self.active["objects"]) < 128:
                    seen.append(job["id"])
                    changed = True
            if changed:
                self.seen_artifacts[self.active["id"]] = seen[-8192:]
                self.persist()
            return changed

    def snapshot(self):
        with self.lock:
            worlds, errors = [], []
            for path in sorted(self.home.glob("*.json")):
                if path.name.casefold() == "state.json":
                    continue
                try:
                    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                        raise ValueError("Invalid saved world file")
                    value = validate_world(migrate_world(json.loads(path.read_text())))
                    if value["id"] != path.stem:
                        raise ValueError("Saved world identity differs from its filename")
                    worlds.append({"id": value["id"], "name": value["name"]})
                except (ValueError, TypeError, OSError, ArithmeticError, RecursionError) as error:
                    errors.append(f"Could not read {path.name}: {error}")
            if self.startup_notice:
                errors.insert(0, self.startup_notice)
            return {"active": copy.deepcopy(self.active), "worlds": worlds, "policy": dict(self.policy), "asset_root": str(self.assets), "errors": errors}

    def store_image(self, data):
        from PIL import Image

        from .image_decoding import UNTRUSTED_IMAGE_FORMATS
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("World textures must fit within eight MiB")
        try:
            with Image.open(io.BytesIO(data), formats=UNTRUSTED_IMAGE_FORMATS) as image:
                if image.width * image.height > 16_000_000:
                    raise ValueError("World textures must be bounded PNG, JPEG or WebP images")
                image.verify()
        except (OSError, SyntaxError, ValueError, TypeError, Image.DecompressionBombError) as error:
            # Pillow reports damaged chunks as SyntaxError and some formats as ValueError.
            if isinstance(error, ValueError) and str(error).startswith("World textures"):
                raise
            raise ValueError("World texture is not a valid bounded image") from error
        name = hashlib.sha256(data).hexdigest() + ".image"
        path = self.assets / name
        if path.is_symlink():
            raise ValueError("Stored world texture must not be a link")
        if path.exists():
            if not path.is_file() or path.stat().st_size != len(data) or path.read_bytes() != data:
                raise ValueError("Stored world texture digest mismatch")
            os.utime(path)  # Recently stored or reused textures are protected from reclaim.
            return str(path)
        if self._asset_bytes() + len(data) > 64 * 1024 * 1024:
            self.reclaim_assets()
        if self._asset_bytes() + len(data) > 64 * 1024 * 1024:
            raise ValueError("World texture storage exceeds its 64 MiB budget")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        return str(path)

    def _asset_bytes(self):
        return sum(p.stat().st_size for p in self.assets.iterdir() if not p.is_symlink() and p.is_file())

    def reclaim_assets(self, grace_seconds=24 * 3600):
        """Delete stored textures nothing references and nothing stored recently.

        References are found textually in saved worlds, set-aside states and user
        profiles. The grace period covers textures that were imported or fetched
        but not yet applied, and images stored earlier in an in-progress bundle.
        """
        with self.lock:
            candidates = {p.name: p for p in self.assets.iterdir()
                          if re.fullmatch(r"[0-9a-f]{64}\.image", p.name) and p.is_file() and not p.is_symlink()}
            referenced = {Path(o["resource"]).name for o in self.active["objects"]}
            profiles = self.home.parent / "user-profiles"
            sources = [*self.home.glob("*.json"), *self.home.glob("state.json.invalid-*"),
                       *(profiles.rglob("*.json") if profiles.is_dir() else [])]
            for path in sources:
                try:
                    if path.is_symlink() or not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
                        continue
                    text = path.read_text(errors="ignore")
                except OSError:
                    return 0  # An unreadable source might reference anything; keep every texture.
                referenced.update(name for name in candidates if name in text)
            now, removed = time.time(), 0
            for name, path in candidates.items():
                if name not in referenced and now - path.stat().st_mtime >= grace_seconds:
                    path.unlink()
                    removed += 1
            return removed

    def bundle(self, world):
        value, assets = copy.deepcopy(world), {}
        for obj in value["objects"]:
            path = Path(obj["resource"])
            if path.is_absolute() and path.parent == self.assets and path.is_file() and not path.is_symlink():
                assets[path.name] = base64.b64encode(path.read_bytes()).decode("ascii")
                obj["resource"] = "asset:" + path.name
        if len(assets) > 32 or sum(len(v) for v in assets.values()) > 24 * 1024 * 1024:
            raise ValueError("This world's textures exceed the portable bundle budget (32 images, 24 MiB encoded). Reduce its textures before exporting.")
        return {"world": value, "assets": assets}

    def import_bundle(self, value):
        with self.lock:
            before = {path.name for path in self.assets.iterdir()}
            try:
                return self._import_bundle(value)
            except BaseException:
                # A rejected portable scene must not consume the asset budget.
                # Only new content-addressed images from this attempt retire;
                # prior assets and the active scene remain untouched.
                for path in self.assets.iterdir():
                    if path.name not in before and re.fullmatch(r"[0-9a-f]{64}\.image", path.name) and path.is_file() and not path.is_symlink():
                        path.unlink()
                raise

    def check_bundle(self, value):
        """Validate a portable scene completely without storing any texture."""
        if isinstance(value, dict) and set(value) == {"world", "assets"}:
            self._bundle_preview(migrate_world(value["world"]), value["assets"])
        else:
            validate_world(migrate_world(value))

    def _bundle_preview(self, world, assets):
        if not isinstance(world, dict) or not isinstance(world.get("objects"), list) or any(not isinstance(obj, dict) for obj in world["objects"]):
            raise ValueError("Invalid bundled world document")
        if not isinstance(assets, dict) or len(assets) > 32 or sum(len(v) for v in assets.values() if isinstance(v, str)) > 24 * 1024 * 1024:
            raise ValueError("World texture bundle exceeds its budget")
        # Asset references become their exact eventual private content paths.
        preview = copy.deepcopy(world)
        for obj in preview["objects"]:
            resource = obj.get("resource", "")
            if isinstance(resource, str) and resource.startswith("asset:"):
                name = resource.removeprefix("asset:")
                if name not in assets or not re.fullmatch(r"[0-9a-f]{64}\.image", name):
                    raise ValueError("World texture bundle is incomplete")
                obj["resource"] = str(self.assets / name)
        validate_world(preview)

    def _import_bundle(self, value):
        if isinstance(value, dict) and set(value) == {"world", "assets"}:
            world, assets = migrate_world(value["world"]), value["assets"]
            # Validate the complete scene before materializing any image.
            self._bundle_preview(world, assets)
            replacements = {}
            for name, encoded in assets.items():
                if not re.fullmatch(r"[0-9a-f]{64}\.image", name) or not isinstance(encoded, str):
                    raise ValueError("Invalid world texture identity")
                data = base64.b64decode(encoded, validate=True)
                if hashlib.sha256(data).hexdigest() != name.removesuffix(".image"):
                    raise ValueError("World texture digest mismatch")
                replacements["asset:" + name] = self.store_image(data)
            for obj in world.get("objects", []):
                if obj.get("resource", "").startswith("asset:"):
                    if obj["resource"] not in replacements:
                        raise ValueError("World texture bundle is incomplete")
                    obj["resource"] = replacements[obj["resource"]]
            return validate_world(world)
        return validate_world(migrate_world(value))

    def image_search(self, query, limit):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300 or type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("Image search requires bounded text and between one and 12 results")
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                  "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
                  "iiurlwidth": 1024, "iiextmetadatafilter": "LicenseShortName|Artist|Credit"}
        request = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params), headers={"User-Agent": "MLX2Desktop/0.1 (native agent workspace image search)"})
        with LANE.released(), urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(1024 * 1024 + 1)
        if len(data) > 1024 * 1024:
            raise ValueError("Image search response exceeds its budget")
        results = []
        for page in json.loads(data).get("query", {}).get("pages", {}).values():
            info = page.get("imageinfo", [{}])[0]
            url = info.get("thumburl", info.get("url", ""))
            parsed = urllib.parse.urlsplit(url)
            if parsed.scheme != "https" or parsed.hostname not in {"upload.wikimedia.org", "thumb.wikimedia.org"} or parsed.port not in {None, 443} or parsed.username or parsed.password:
                continue
            metadata = info.get("extmetadata", {})
            clean = lambda key, metadata=metadata: html.unescape(re.sub(r"<[^>]+>", "", metadata.get(key, {}).get("value", "")))[:1000]
            identity = uuid.uuid4().hex
            result = {"id": identity, "title": page.get("title", "")[:300], "url": url,
                      "description_url": info.get("descriptionurl", ""), "license": clean("LicenseShortName"), "credit": clean("Artist") or clean("Credit")}
            results.append(result)
        with self.lock:
            for result in results:
                self.search_results[result["id"]] = result
            if len(self.search_results) > 100:
                self.search_results = {r["id"]: r for r in results}
        return {"results": results, "source": "Wikimedia Commons", "query": query}

    def _saved(self, world_id):
        path = self.home / (identifier(world_id) + ".json")
        if world_id.casefold() in RESERVED_WORLD_IDS or not path.is_file() or path.is_symlink():
            raise ValueError("Saved world does not exist; read world_state for saved world IDs")
        value = validate_world(migrate_world(json.loads(path.read_text())))
        # On a case-insensitive volume ALPHA.json opens alpha.json; never load another world.
        if value["id"] != world_id:
            raise ValueError("Saved world does not exist; read world_state for saved world IDs")
        return value

    def state(self):
        """Compact model-facing view: every object's transform and appearance, text previews only."""
        snapshot = self.snapshot()
        return compact_state(snapshot)

    def check(self, request):
        """Validate a scene edit without applying it, so approval is never asked for a doomed call."""
        with self.lock:
            command = request["command"]
            if command in {"world_create", "world_save", "world_load", "world_export", "world_style",
                           "world_object_upsert", "world_object_remove", "world_get", "world_policy"}:
                self._proposal(request)
            elif command == "world_import":
                self._room_for(uuid.uuid4().hex)
                self.check_bundle(request["document"])

    def _room_for(self, identity):
        # World files live on a case-insensitive volume: an id spelled differently
        # from an existing saved world would overwrite that world's file.
        stems = {path.stem for path in self.home.glob("*.json") if path.name.casefold() != "state.json"}
        if identity not in stems and any(stem.casefold() == identity.casefold() for stem in stems):
            raise ValueError("A saved world already uses that identifier with different letter case; choose another identifier")
        if not (self.home / (identity + ".json")).exists() and sum(1 for _ in self.home.glob("*.json")) - 1 >= MAX_SAVED_WORLDS:
            raise ValueError(f"At most {MAX_SAVED_WORLDS} worlds can be saved")

    def _proposal(self, request):
        command = request["command"]
        if command == "world_create":
            value = validate_world(default_world(request["name"]))
            self._room_for(value["id"])
            return value
        if command == "world_save":
            value = validate_world(request["world"])
            self._room_for(value["id"])  # A save under a new id creates a file, so it counts against the cap.
            return value
        if command in {"world_load", "world_export"}:
            return self._saved(request["world_id"])
        if command == "world_get":
            obj = next((o for o in self.active["objects"] if o["id"] == request["object_id"]), None)
            if obj is None:
                raise ValueError("Scene object does not exist; read world_state for object IDs")
            return obj
        if command == "world_style":
            theme, transitions, name = request.get("theme"), request.get("transitions"), request.get("name")
            if theme is None and transitions is None and name is None:
                raise ValueError("Provide a name, theme or transitions change")
            if (theme is not None and not isinstance(theme, dict)) or (transitions is not None and not isinstance(transitions, dict)):
                raise ValueError("Theme and transitions must be objects")
            value = copy.deepcopy(self.active)
            value["theme"].update(theme or {})
            value["transitions"].update(transitions or {})
            if name is not None:
                value["name"] = name
            return validate_world(value)
        if command == "world_object_upsert":
            patch = request["object"]
            identity = patch.get("id") if isinstance(patch, dict) else None
            objects = self.active["objects"]
            index = next((i for i, o in enumerate(objects) if o["id"] == identity), None)
            obj = merge_object(None if index is None else objects[index], patch, self.active["theme"]["accent"])
            value = copy.deepcopy(self.active)
            if index is None:
                value["objects"].append(obj)
            else:
                value["objects"][index] = obj
            return validate_world(value)
        if command == "world_object_remove":
            if not any(o["id"] == request["object_id"] for o in self.active["objects"]):
                raise ValueError("Scene object does not exist; read world_state for object IDs")
            value = copy.deepcopy(self.active)
            value["objects"] = [o for o in value["objects"] if o["id"] != request["object_id"]]
            return validate_world(value)
        if command == "world_policy":
            changes = request["changes"]
            if not isinstance(changes, dict) or set(changes) != {"autonomous_scene"} or type(changes["autonomous_scene"]) is not bool:
                raise ValueError("Invalid autonomous scene policy")
            return dict(changes)
        raise ValueError("Unknown world operation")

    def dispatch(self, request):
        command = request["command"]
        # Network calls run outside the world lock and step out of the request
        # lane, so a slow Wikimedia request never blocks snapshots, scene edits
        # or other application requests.
        if command == "world_image_search":
            return self.image_search(request["query"], request["limit"])
        if command == "world_asset_fetch":
            with self.lock:
                result = self.search_results.get(request["result_id"])
            if result is None:
                raise ValueError("Select an image from a current image search result")
            class NoRedirect(urllib.request.HTTPRedirectHandler):
                def redirect_request(self, *args, **kwargs):
                    raise ValueError("World texture redirects are not permitted")
            opener = urllib.request.build_opener(NoRedirect)
            req = urllib.request.Request(result["url"], headers={"User-Agent": "MLX2Desktop/0.1"})
            with LANE.released(), opener.open(req, timeout=15) as response:
                data = response.read(8 * 1024 * 1024 + 1)
            with self.lock:
                return {"resource": self.store_image(data), "attribution": result}
        with self.lock:
            if command == "world_state":
                return self.state()
            if command == "world_get":
                return {"object": copy.deepcopy(self._proposal(request))}
            if command == "world_asset_import":
                path = Path(request["path"]).expanduser()
                if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Select a regular image file no larger than eight MiB")
                return {"resource": self.store_image(path.read_bytes())}
            if command == "world_export":
                return {"document": self.bundle(self._proposal(request))}
            if command == "world_import":
                identity = uuid.uuid4().hex
                self._room_for(identity)
                value = self.import_bundle(request["document"])
                value["id"] = identity
                self.active = value
            elif command == "world_policy":
                self.policy = self._proposal(request)
            else:
                value = self._proposal(request)
                self.active = value
                if command == "world_style":
                    self.persist()
                    return {"name": value["name"], "theme": copy.deepcopy(value["theme"]), "transitions": copy.deepcopy(value["transitions"])}
            self.persist()
            return self.snapshot()


def compact_state(snapshot):
    active = snapshot["active"]
    objects = []
    for obj in active["objects"]:
        row = {key: obj[key] for key in OBJECT_FIELDS if key not in {"text", "resource"}}
        row["text_preview"] = obj["text"][:160]
        row["has_resource"] = bool(obj["resource"])
        objects.append(row)
    return {"active": {"id": active["id"], "name": active["name"], "theme": active["theme"],
                       "transitions": active["transitions"], "objects": objects},
            "worlds": snapshot["worlds"], "policy": snapshot["policy"], "errors": snapshot.get("errors", [])}
