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
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

KINDS = {"chat", "memory", "knowledge", "web", "file", "folder", "avatar", "media"}


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise ValueError("Invalid world or object identifier")
    return value


def color(value):
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("Colors must be #RRGGBB")
    return value


def vector(value, low, high):
    if not isinstance(value, list) or len(value) != 3 or any(type(v) not in (int, float) or not math.isfinite(v) or not low <= v <= high for v in value):
        raise ValueError("Invalid bounded three dimensional transform")
    return value


def validate_object(value):
    required = {"id", "kind", "label", "position", "rotation", "scale", "color", "shape", "text", "resource", "animation"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Unexpected or missing scene object fields")
    identifier(value["id"])
    if not isinstance(value["kind"], str) or not isinstance(value["shape"], str) or value["kind"] not in KINDS or value["shape"] not in {"panel", "sphere", "cube", "ring"}:
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
    if not isinstance(animation, dict) or set(animation) != {"kind", "speed"} or not isinstance(animation["kind"], str) or animation["kind"] not in {"none", "pulse", "spin", "float"}:
        raise ValueError("Unsupported declarative animation")
    if type(animation["speed"]) not in (int, float) or not math.isfinite(animation["speed"]) or not 0.05 <= animation["speed"] <= 5:
        raise ValueError("Animation speed must be between 0.05 and 5")
    return copy.deepcopy(value)


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


def validate_world(value):
    if not isinstance(value, dict) or set(value) != {"schema_version", "id", "name", "theme", "transitions", "objects"} or value["schema_version"] != 1:
        raise ValueError("Unsupported world document schema")
    identifier(value["id"])
    if not isinstance(value["name"], str) or not 1 <= len(value["name"]) <= 200:
        raise ValueError("A world needs a name of at most 200 characters")
    theme = value["theme"]
    if not isinstance(theme, dict) or set(theme) != {"background", "accent", "exposure"}:
        raise ValueError("Invalid scene theme")
    color(theme["background"])
    color(theme["accent"])
    if type(theme["exposure"]) not in (int, float) or not math.isfinite(theme["exposure"]) or not 0.1 <= theme["exposure"] <= 4:
        raise ValueError("Exposure must be between 0.1 and 4")
    transitions = value["transitions"]
    if not isinstance(transitions, dict) or set(transitions) != {"duration", "style"} or not isinstance(transitions["style"], str) or transitions["style"] not in {"ease", "linear", "instant"}:
        raise ValueError("Invalid scene transitions")
    if type(transitions["duration"]) not in (int, float) or not math.isfinite(transitions["duration"]) or not 0 <= transitions["duration"] <= 5:
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
        if state.is_file():
            value = json.loads(state.read_text())
            self.policy = value["policy"]
            if not isinstance(self.policy, dict) or set(self.policy) != {"autonomous_scene"} or type(self.policy["autonomous_scene"]) is not bool:
                raise ValueError("Stored autonomous scene policy must be explicitly boolean")
            self.seen_artifacts = value.get("seen_artifacts", {})
            self.active = validate_world(value["active"])
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
            changed = False
            for job in jobs:
                if job.get("kind") not in {"media", "helper", "adapter", "document"} or job.get("state") not in {"completed", "succeeded", "partial"} or job["id"] in seen:
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
                    identity = "artifact-" + job["id"][:40] + "-" + str(index)
                    if any(o["id"] == identity for o in self.active["objects"]):
                        continue
                    resource = path
                    if capability == "image":
                        try:
                            file = Path(path)
                            resolved = file.resolve(strict=True)
                            if any(resolved.is_relative_to(root) for root in self.automatic_image_roots) and resolved.is_file() and not file.is_symlink() and resolved.stat().st_size <= 8 * 1024 * 1024:
                                resource = self.store_image(resolved.read_bytes())
                        except (OSError, ValueError):
                            pass
                    count = len(self.active["objects"])
                    self.active["objects"].append({"id": identity, "kind": "media" if capability in {"image", "video", "music", "voice"} else "file",
                        "label": Path(path).name[:200], "position": [((count % 5) - 2) * 2.5, 0, -2 - (count // 5) * 2],
                        "rotation": [0, 0, 0], "scale": [1, 1, 1], "color": "#38CBFF", "shape": "panel",
                        "text": (capability + " · " + job["state"] + "\nJob: " + job["id"] + "\n" + path)[:8192],
                        "resource": resource[:4096], "animation": {"kind": "none", "speed": 1}})
                    changed = True
                seen.append(job["id"])
                changed = True
            if changed:
                self.seen_artifacts[self.active["id"]] = seen[-4096:]
                self.persist()
            return changed

    def snapshot(self):
        with self.lock:
            worlds, errors = [], []
            for path in sorted(self.home.glob("*.json")):
                if path.name == "state.json":
                    continue
                try:
                    if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
                        raise ValueError("Invalid saved world file")
                    value = validate_world(json.loads(path.read_text()))
                    if value["id"] != path.stem:
                        raise ValueError("Saved world identity differs from its filename")
                    worlds.append({"id": value["id"], "name": value["name"]})
                except (ValueError, TypeError, OSError) as error:
                    errors.append(f"Could not read {path.name}: {error}")
            return {"active": copy.deepcopy(self.active), "worlds": worlds, "policy": dict(self.policy), "asset_root": str(self.assets), "errors": errors}

    def store_image(self, data):
        from PIL import Image
        if len(data) > 8 * 1024 * 1024:
            raise ValueError("World textures must fit within eight MiB")
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.width * image.height > 16_000_000 or image.format not in {"PNG", "JPEG", "WEBP"}:
                    raise ValueError("World textures must be bounded PNG, JPEG or WebP images")
                image.verify()
        except (OSError, Image.DecompressionBombError) as error:
            raise ValueError("World texture is not a valid bounded image") from error
        name = hashlib.sha256(data).hexdigest() + ".image"
        path = self.assets / name
        if path.is_symlink():
            raise ValueError("Stored world texture must not be a link")
        if path.exists():
            if not path.is_file() or path.stat().st_size != len(data) or path.read_bytes() != data:
                raise ValueError("Stored world texture digest mismatch")
            return str(path)
        if sum(p.stat().st_size for p in self.assets.iterdir() if not p.is_symlink() and p.is_file()) + len(data) > 64 * 1024 * 1024:
            raise ValueError("World texture storage exceeds its 64 MiB budget")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as stream:
                stream.write(data)
        return str(path)

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

    def _import_bundle(self, value):
        if isinstance(value, dict) and set(value) == {"world", "assets"}:
            world, assets = copy.deepcopy(value["world"]), value["assets"]
            if not isinstance(world, dict) or not isinstance(world.get("objects"), list) or any(not isinstance(obj, dict) for obj in world["objects"]):
                raise ValueError("Invalid bundled world document")
            if not isinstance(assets, dict) or len(assets) > 32 or sum(len(v) for v in assets.values() if isinstance(v, str)) > 24 * 1024 * 1024:
                raise ValueError("World texture bundle exceeds its budget")
            # Validate the complete scene before materializing any image. Asset
            # references become their exact eventual private content paths.
            preview = copy.deepcopy(world)
            for obj in preview["objects"]:
                resource = obj.get("resource", "")
                if isinstance(resource, str) and resource.startswith("asset:"):
                    name = resource.removeprefix("asset:")
                    if name not in assets or not re.fullmatch(r"[0-9a-f]{64}\.image", name):
                        raise ValueError("World texture bundle is incomplete")
                    obj["resource"] = str(self.assets / name)
            validate_world(preview)
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
        return validate_world(value)

    def image_search(self, query, limit):
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300 or type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("Image search requires bounded text and between one and 12 results")
        params = {"action": "query", "format": "json", "generator": "search", "gsrsearch": query,
                  "gsrnamespace": 6, "gsrlimit": limit, "prop": "imageinfo", "iiprop": "url|extmetadata|mime",
                  "iiurlwidth": 1024, "iiextmetadatafilter": "LicenseShortName|Artist|Credit"}
        request = urllib.request.Request("https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(params), headers={"User-Agent": "MLX2Desktop/0.1 (native agent workspace image search)"})
        with urllib.request.urlopen(request, timeout=15) as response:
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
            self.search_results[identity] = result
            results.append(result)
        if len(self.search_results) > 100:
            self.search_results = {r["id"]: r for r in results}
        return {"results": results, "source": "Wikimedia Commons", "query": query}

    def dispatch(self, request):
        with self.lock:
            command = request["command"]
            if command == "world_get":
                obj = next((o for o in self.active["objects"] if o["id"] == request["object_id"]), None)
                if obj is None:
                    raise ValueError("Scene object does not exist")
                return {"object": copy.deepcopy(obj)}
            if command == "world_style":
                value = copy.deepcopy(self.active)
                value.update(theme=request["theme"], transitions=request["transitions"])
                self.active = validate_world(value)
                self.persist()
                return {"theme": value["theme"], "transitions": value["transitions"]}
            if command == "world_image_search":
                return self.image_search(request["query"], request["limit"])
            if command == "world_asset_import":
                path = Path(request["path"]).expanduser()
                if not path.is_absolute() or path.is_symlink() or not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Select a regular image file no larger than eight MiB")
                return {"resource": self.store_image(path.read_bytes())}
            if command == "world_asset_fetch":
                result = self.search_results.get(request["result_id"])
                if result is None:
                    raise ValueError("Select an image from a current image search result")
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *args, **kwargs):
                        raise ValueError("World texture redirects are not permitted")
                opener = urllib.request.build_opener(NoRedirect)
                req = urllib.request.Request(result["url"], headers={"User-Agent": "MLX2Desktop/0.1"})
                with opener.open(req, timeout=15) as response:
                    data = response.read(8 * 1024 * 1024 + 1)
                return {"resource": self.store_image(data), "attribution": result}
            if command == "world_create":
                self.active = validate_world(default_world(request["name"]))
            elif command in {"world_save", "world_import"}:
                value = request["world"] if command == "world_save" else request["document"]
                value = self.import_bundle(value) if command == "world_import" else validate_world(value)
                if command == "world_import":
                    value["id"] = uuid.uuid4().hex
                self.active = value
            elif command in {"world_load", "world_export"}:
                path = self.home / (identifier(request["world_id"]) + ".json")
                value = validate_world(json.loads(path.read_text()))
                if command == "world_export":
                    return {"document": self.bundle(value)}
                self.active = value
            elif command == "world_object_upsert":
                obj = validate_object(request["object"])
                value = copy.deepcopy(self.active)
                value["objects"] = [o for o in value["objects"] if o["id"] != obj["id"]] + [obj]
                self.active = validate_world(value)
            elif command == "world_object_remove":
                value = copy.deepcopy(self.active)
                value["objects"] = [o for o in value["objects"] if o["id"] != request["object_id"]]
                self.active = validate_world(value)
            elif command == "world_policy":
                changes = request["changes"]
                if not isinstance(changes, dict) or set(changes) != {"autonomous_scene"} or type(changes["autonomous_scene"]) is not bool:
                    raise ValueError("Invalid autonomous scene policy")
                self.policy = dict(changes)
            else:
                raise ValueError("Unknown world operation")
            self.persist()
            return self.snapshot()
