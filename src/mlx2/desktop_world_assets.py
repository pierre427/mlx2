"""Poly Haven CC0 3D models and HDRI skies for the native world.

Search runs locally over the downloaded public catalogue, so the operator's
search text never leaves the machine. Downloads come only from Poly Haven's API
and file hosts, without redirects, and every file is checked against the size
and MD5 the catalogue publishes before it is admitted to the cache. Assets are
named by catalogue identity (``polyhaven:model/<id>``), so world documents stay
portable and a reference can never name a path outside the cache.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .desktop_lane import LANE

API = "https://api.polyhaven.com"
FILE_HOSTS = {"dl.polyhaven.org"}
USER_AGENT = "MLX2Desktop/0.1 (native world asset fetch)"
KINDS = {"model": "models", "hdri": "hdris"}
ASSET_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
TEXTURE_PATH = re.compile(r"textures/[A-Za-z0-9_.-]{1,128}\.(?:jpg|jpeg|png|exr)")
MAX_POLYCOUNT = 500_000          # SceneKit stays interactive; heavier trees run to millions.
MAX_MODEL_BYTES = 160 * 1024 * 1024
MAX_HDRI_BYTES = 16 * 1024 * 1024
CACHE_BUDGET = 2 * 1024 * 1024 * 1024
CATALOGUE_TTL = 3600
MODEL_RESOLUTION = "1k"
HDRI_RESOLUTION = "2k"           # A 1k sky looks blurred as a background.


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError("Poly Haven redirects are not permitted")


_OPENER = urllib.request.build_opener(_NoRedirect)


def _polycount(info):
    value = info.get("polycount")
    return value if type(value) is int else 0


def _authors(info):
    value = info.get("authors")
    return [str(name) for name in value][:16] if isinstance(value, (dict, list)) else []


def _allowed(url, hosts):
    parts = urllib.parse.urlsplit(url)
    return (parts.scheme == "https" and parts.hostname in hosts and parts.port in {None, 443}
            and not parts.username and not parts.password)


def _get(url, limit, hosts, timeout=30):
    if not _allowed(url, hosts):
        raise ValueError("Poly Haven returned a file outside its download hosts")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with LANE.released(), _OPENER.open(request, timeout=timeout) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Poly Haven response exceeds its budget")
    return data


class PolyHaven:
    def __init__(self, root: Path, references):
        """``references`` returns the text of every document that may name an asset."""
        self.root = root
        self.references = references
        self.lock = threading.RLock()          # Guards cache checks, stamps and reclaim.
        self.download_lock = threading.Lock()  # One download at a time keeps the budget check honest.
        self.fetching: dict[str, threading.Lock] = {}
        self.catalogues: dict[str, tuple[float, dict]] = {}

    # ---- paths -----------------------------------------------------------------
    def path(self, kind, asset_id):
        if kind not in KINDS or not ASSET_ID.fullmatch(asset_id or ""):
            raise ValueError("Poly Haven assets are named by kind (model or hdri) and catalogue id")
        if kind == "model":
            return self.root / "models" / asset_id / "model.usdc"
        return self.root / "hdris" / (asset_id + ".hdr")

    def cached(self, kind, asset_id):
        path = self.path(kind, asset_id)
        manifest = path.parent / "manifest.json" if kind == "model" else path.with_suffix(".json")
        return path.is_file() and not path.is_symlink() and manifest.is_file()

    # ---- search ----------------------------------------------------------------
    def catalogue(self, kind):
        with self.lock:
            entry = self.catalogues.get(kind)
            if entry and time.time() - entry[0] < CATALOGUE_TTL:
                return entry[1]
        data = json.loads(_get(f"{API}/assets?t={KINDS[kind]}", 8 * 1024 * 1024, {"api.polyhaven.com"}))
        if not isinstance(data, dict):
            raise ValueError("Unexpected Poly Haven catalogue")
        with self.lock:
            self.catalogues[kind] = (time.time(), data)
        return data

    def search(self, query, kind, limit):
        if kind not in KINDS:
            raise ValueError("Search kind must be model or hdri")
        if not isinstance(query, str) or not 1 <= len(query.strip()) <= 300 or type(limit) is not int or not 1 <= limit <= 12:
            raise ValueError("Model search requires bounded text and between one and 12 results")
        terms = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1] or [query.strip().lower()]
        scored = []
        for asset_id, info in self.catalogue(kind).items():
            if not ASSET_ID.fullmatch(asset_id) or not isinstance(info, dict):
                continue
            if kind == "model" and not 0 < _polycount(info) <= MAX_POLYCOUNT:
                continue
            words = " ".join([asset_id.replace("_", " "), str(info.get("name", "")),
                              " ".join(map(str, info.get("tags", []))), " ".join(map(str, info.get("categories", [])))]).lower()
            score = sum(3 if re.search(rf"\b{re.escape(t)}\b", words) else 1 if t in words else 0 for t in terms)
            if score:
                scored.append((-score, asset_id, info))
        scored.sort(key=lambda row: (row[0], row[1]))
        results = []
        for _, asset_id, info in scored[:limit]:
            row = {"asset_id": asset_id, "kind": kind, "name": str(info.get("name", asset_id))[:200],
                   "categories": [str(c)[:60] for c in info.get("categories", [])][:8],
                   "authors": [a[:120] for a in _authors(info)][:4],
                   "license": "CC0", "page": f"https://polyhaven.com/a/{asset_id}",
                   "downloaded": self.cached(kind, asset_id)}
            if kind == "model":
                row["polycount"] = _polycount(info)
                dimensions = info.get("dimensions")
                if isinstance(dimensions, list) and len(dimensions) == 3 and all(isinstance(v, (int, float)) for v in dimensions):
                    # The catalogue publishes millimetres; scenes use metres.
                    row["size_m"] = [round(v / 1000, 2) for v in dimensions]
            results.append(row)
        return {"results": results, "source": "Poly Haven (CC0)", "query": query, "kind": kind,
                "note": "Matched locally against the public catalogue; fetch an asset_id with world_model_fetch."}

    # ---- fetch -----------------------------------------------------------------
    def _cached_info(self, kind, asset_id):
        """Stamp a cached asset as used and return its manifest, or None. Holds the reclaim lock."""
        with self.lock:
            if not self.cached(kind, asset_id):
                return None
            target = self.path(kind, asset_id)
            manifest = target.parent / "manifest.json" if kind == "model" else target.with_suffix(".json")
            os.utime(target)
            try:
                value = json.loads(manifest.read_text())
            except (OSError, ValueError):
                value = {}
            return {"name": str(value.get("name", asset_id)), "authors": {str(a): "" for a in value.get("authors", []) if isinstance(a, str)}}

    def fetch(self, kind, asset_id):
        self.path(kind, asset_id)  # Validates kind and id before anything else.
        with self.lock:
            gate = self.fetching.setdefault(f"{kind}/{asset_id}", threading.Lock())
        with gate:
            # A cached asset needs no network, so this also works offline.
            info = self._cached_info(kind, asset_id)
            if info is None:
                info = self.catalogue(kind).get(asset_id)
                if not isinstance(info, dict):
                    raise ValueError("That asset id is not in the Poly Haven catalogue; search first")
                if kind == "model" and not 0 < _polycount(info) <= MAX_POLYCOUNT:
                    raise ValueError("That model is too detailed for the interactive world")
                files = json.loads(_get(f"{API}/files/{urllib.parse.quote(asset_id)}", 4 * 1024 * 1024, {"api.polyhaven.com"}))
                plan = self._plan(kind, files)
                with self.download_lock:
                    self._room(sum(size for _, _, size, _ in plan))
                    self._download(kind, asset_id, plan, info)
                with self.lock:
                    os.utime(self.path(kind, asset_id))
        return {"resource": f"polyhaven:{kind}/{asset_id}", "name": info.get("name", asset_id),
                "license": "CC0", "authors": _authors(info)[:4], "page": f"https://polyhaven.com/a/{asset_id}",
                "use": ("Set an object's shape to model and resource to this reference. Models are real-world size in metres "
                        "with their origin at the base; the floor is at y = -1.8.") if kind == "model" else
                       "Set theme.environment to this reference with world_style for the sky and image-based lighting."}

    def _plan(self, kind, files):
        """Return [(relative path, url, size, md5)] after checking every published entry."""
        try:
            if kind == "model":
                entry = files["usd"][MODEL_RESOLUTION]["usd"]
                plan = [("model.usdc", entry["url"], entry["size"], entry["md5"])]
                for relative, item in (entry.get("include") or {}).items():
                    if not isinstance(relative, str) or not TEXTURE_PATH.fullmatch(relative):
                        raise ValueError("Poly Haven model references an unexpected file")
                    plan.append((relative, item["url"], item["size"], item["md5"]))
                limit = MAX_MODEL_BYTES
            else:
                entry = files["hdri"][HDRI_RESOLUTION]["hdr"]
                plan = [("sky.hdr", entry["url"], entry["size"], entry["md5"])]
                limit = MAX_HDRI_BYTES
        except (KeyError, TypeError) as error:
            raise ValueError("Poly Haven has no suitable file for that asset") from error
        for _, url, size, md5 in plan:
            if not isinstance(url, str) or not _allowed(url, FILE_HOSTS) or type(size) is not int or size <= 0 \
                    or not isinstance(md5, str) or not re.fullmatch(r"[0-9a-f]{32}", md5):
                raise ValueError("Poly Haven published an unexpected file entry")
        if sum(size for _, _, size, _ in plan) > limit:
            raise ValueError("That asset exceeds the world download budget")
        return plan

    def _download(self, kind, asset_id, plan, info):
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        staging = Path(tempfile.mkdtemp(dir=self.root, prefix=".fetch-"))
        try:
            hashes = {}
            for relative, url, size, md5 in plan:
                data = _get(url, size, FILE_HOSTS, timeout=120)
                if len(data) != size or hashlib.md5(data).hexdigest() != md5:
                    raise ValueError("A Poly Haven download did not match its published size and checksum")
                if relative == "model.usdc" and not data.startswith(b"PXR-USDC"):
                    raise ValueError("Poly Haven model is not a binary USD file")
                if relative == "sky.hdr" and not data.startswith((b"#?RADIANCE", b"#?RGBE")):
                    raise ValueError("Poly Haven sky is not a Radiance HDR image")
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                hashes[relative] = hashlib.sha256(data).hexdigest()
            manifest = {"source": "polyhaven", "kind": kind, "asset_id": asset_id, "name": info.get("name", asset_id),
                        "license": "CC0", "authors": _authors(info), "files": hashes, "fetched": time.time()}
            (staging / "manifest.json").write_text(json.dumps(manifest, indent=1))
            if kind == "model":
                final = self.path(kind, asset_id).parent
                final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                if final.exists():
                    shutil.rmtree(final)
                os.replace(staging, final)
            else:
                final = self.path(kind, asset_id)
                final.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.replace(staging / "sky.hdr", final)
                os.replace(staging / "manifest.json", final.with_suffix(".json"))
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    # ---- budget ----------------------------------------------------------------
    def _bytes(self):
        if not self.root.is_dir():
            return 0
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file() and not p.is_symlink())

    def _room(self, incoming):
        if self._bytes() + incoming > CACHE_BUDGET:
            self.reclaim()
        if self._bytes() + incoming > CACHE_BUDGET:
            raise ValueError("The world model cache is full (2 GiB); remove unused models from worlds first")

    def reclaim(self, grace_seconds=24 * 3600):
        """Delete cached assets no document references and nothing fetched recently."""
        with self.lock:
            try:
                text = self.references()
            except OSError:
                return 0  # An unreadable document might reference anything; keep everything.
            now, removed = time.time(), 0
            # Staging left by an interrupted download; a live one is minutes old at most.
            if self.root.is_dir():
                for staging in self.root.glob(".fetch-*"):
                    if staging.is_dir() and not staging.is_symlink() and now - staging.stat().st_mtime > 3600:
                        shutil.rmtree(staging, ignore_errors=True)
            for kind, folder in KINDS.items():
                base = self.root / folder
                if not base.is_dir():
                    continue
                for entry in (base.iterdir() if kind == "model" else base.glob("*.hdr")):
                    asset_id = entry.name if kind == "model" else entry.stem
                    if entry.is_symlink() or not ASSET_ID.fullmatch(asset_id) or f"polyhaven:{kind}/{asset_id}" in text:
                        continue
                    stamp = (entry / "model.usdc") if kind == "model" else entry
                    if stamp.exists() and now - stamp.stat().st_mtime < grace_seconds:
                        continue
                    if entry.is_dir():
                        shutil.rmtree(entry)
                    else:
                        entry.unlink()
                        entry.with_suffix(".json").unlink(missing_ok=True)
                    removed += 1
            return removed
