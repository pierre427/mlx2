"""Neural ABI v0 package identity, registry and immutable admission (spec v0.4 sections 3.3 and 5).

* Canonical JSON: sorted keys, no insignificant whitespace, UTF-8, and only
  integers, strings, booleans, null, lists and objects. Floats are refused, so
  the payload never depends on float formatting; the strict RFC 8785 number
  rules are therefore not needed in v0. Duplicate keys are refused on parse.
* Package digest: sha256 over a domain-tagged canonical payload that contains
  the ordered artifact descriptors and their byte digests. The package digest,
  aliases, signatures and qualification receipts are excluded, so an external
  receipt never changes package identity and a weight byte always does.
* Contract digest: a separate hash of the typed contract (ABI majors, law,
  ports, state, composition, limits).
* Registry: maps an address to a package digest. It allocates no device
  tensor. Admission pins the resolved digests for a request against a named
  snapshot; a later catalog change cannot alter an admitted set.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "ens.module.v0"
EXCLUDED = ("package_digest", "aliases", "signatures", "receipts")
CONTRACT_KEYS = ("neural_abi_major", "execution_abi_major", "law", "ports", "state", "composition", "limits",
                 "core_artifact")
REQUIRED = ("schema", "address", "core_artifact", "neural_abi_major", "execution_abi_major", "law",
            "implementation_digest", "ports", "state", "composition", "limits", "artifacts")
KNOWN_FEATURES: frozenset = frozenset()          # v0 understands no optional required-features


class ManifestError(ValueError):
    pass


def _check_value(v, path="$"):
    if isinstance(v, bool) or v is None or isinstance(v, (int, str)):
        return
    if isinstance(v, float):
        raise ManifestError(f"{path}: floats are not canonical in ens.module.v0; use an integer or string")
    if isinstance(v, list):
        for i, x in enumerate(v):
            _check_value(x, f"{path}[{i}]")
        return
    if isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise ManifestError(f"{path}: non-string key")
            _check_value(x, f"{path}.{k}")
        return
    raise ManifestError(f"{path}: unsupported type {type(v).__name__}")


def canonical(obj) -> bytes:
    _check_value(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _no_dupes(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ManifestError(f"duplicate key {k!r}")
        out[k] = v
    return out


def parse(text: str) -> dict:
    return json.loads(text, object_pairs_hook=_no_dupes)


def validate(manifest: dict) -> None:
    missing = [k for k in REQUIRED if k not in manifest]
    if missing:
        raise ManifestError(f"missing required fields: {missing}")
    if manifest["schema"] != SCHEMA:
        raise ManifestError(f"unknown schema {manifest['schema']!r}")
    unknown = set(manifest.get("required_features", [])) - KNOWN_FEATURES
    if unknown:
        raise ManifestError(f"unknown required features: {sorted(unknown)}")
    names = [a["name"] for a in manifest["artifacts"]]
    if len(names) != len(set(names)):
        raise ManifestError("duplicate artifact names")
    for a in manifest["artifacts"]:
        n = a["name"]
        if "/" in n or "\\" in n or n.startswith(".") or not n:
            raise ManifestError(f"artifact name must be a plain file name: {n!r}")
    sites = [(p["direction"], p["site"]) for p in manifest["ports"]]
    if len(sites) != len(set(sites)):
        raise ManifestError("conflicting port bindings")
    for p in manifest["ports"]:
        if p["direction"] not in ("consume", "produce"):
            raise ManifestError(f"bad port direction {p['direction']!r}")
    _check_value(manifest)


def payload(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k not in EXCLUDED}


def package_digest(manifest: dict) -> str:
    validate(manifest)
    return hashlib.sha256(b"ens.package.v0\0" + canonical(payload(manifest))).hexdigest()


def contract_digest(manifest: dict) -> str:
    validate(manifest)
    contract = {k: manifest[k] for k in CONTRACT_KEYS}
    return hashlib.sha256(b"ens.contract.v0\0" + canonical(contract)).hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_descriptor(path: Path) -> dict:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Admission:
    snapshot: str
    digests: tuple            # (address, package digest), in request order

    @property
    def empty(self) -> bool:
        return not self.digests


@dataclass
class Registry:
    """Address -> package digest. Holds manifests only; never device tensors."""
    _entries: dict = field(default_factory=dict)
    _manifests: dict = field(default_factory=dict)
    _revision: int = 0

    def register(self, manifest: dict) -> str:
        digest = package_digest(manifest)
        addr = manifest["address"]
        prior = self._entries.get(addr)
        if prior is not None and prior != digest:
            raise ManifestError(f"alias collision at {addr}: an explicit alias change is required")
        self._entries[addr] = digest
        self._manifests[digest] = copy.deepcopy(manifest)
        self._revision += 1
        return digest

    def move_alias(self, address: str, digest: str) -> None:
        if digest not in self._manifests:
            raise KeyError(digest)
        self._entries[address] = digest
        self._revision += 1

    def snapshot(self) -> str:
        blob = canonical([list(kv) for kv in sorted(self._entries.items())])
        return f"r{self._revision}-" + hashlib.sha256(blob).hexdigest()[:16]

    def admit(self, addresses, *, core_artifact: str) -> Admission:
        pinned = []
        for a in addresses:
            if a not in self._entries:
                raise KeyError(f"unregistered address {a}")
            d = self._entries[a]
            m = self._manifests[d]
            if m["core_artifact"] != core_artifact:
                raise ManifestError(f"{a}: built for {m['core_artifact']}, not {core_artifact}")
            pinned.append((a, d))
        return Admission(self.snapshot(), tuple(pinned))

    def manifest(self, digest: str) -> dict:
        return copy.deepcopy(self._manifests[digest])

    def __len__(self):
        return len(self._entries)
