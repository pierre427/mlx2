"""Derive a live paged price identity without trusting the price receipt.

This is called only at explicit native route admission.  Every byte identity
comes from the active checkout, loaded model root, installed MLX binary and
loaded native extension; configured paths identify files but supply no hashes.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import threading
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_IDENTITY_LOCK = threading.RLock()
_IDENTITY_MEMO: dict[tuple[str, str, str, str], tuple[dict[str, str], tuple]] = {}
_SOURCE_MEMO: dict[tuple[str, str], dict] = {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    files = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).split(b"\0")
    digest = hashlib.sha256()
    for raw in sorted(filter(None, files)):
        path = ROOT / os.fsdecode(raw)
        if path.is_file():
            digest.update(raw + b"\0")
            digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _artifact_sha256(manifest_path: Path, adapter_root: Path) -> str:
    data = json.loads(manifest_path.read_text())
    root = Path(data["root"]).resolve()
    files = data["files"]
    if root != adapter_root or not root.is_dir() or not isinstance(files, dict):
        raise RuntimeError("artifact manifest root differs from loaded adapter")
    if not {"config.json", "tokenizer.json"} <= set(files):
        raise RuntimeError("artifact manifest lacks model configuration/tokenizer")
    if "model.safetensors" not in files:
        index_name = "model.safetensors.index.json"
        if index_name not in files:
            raise RuntimeError("artifact manifest lacks complete weight binding")
        index = json.loads((root / index_name).read_text())
        mapping = index.get("weight_map")
        if (type(mapping) is not dict or not mapping or
                any(not isinstance(name, str) or not name.endswith(".safetensors") or
                    name not in files for name in mapping.values())):
            raise RuntimeError("artifact manifest lacks every indexed weight shard")
    for name, expected in files.items():
        if (not isinstance(name, str) or Path(name).is_absolute() or
                not isinstance(expected, str) or len(expected) != 64 or
                any(char not in "0123456789abcdef" for char in expected)):
            raise RuntimeError("malformed artifact hash entry")
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"artifact bytes differ: {name}")
    return _sha256(manifest_path)


def _wheel_identity(wheel_path: Path) -> tuple[str, str]:
    import mlx.core as mx

    core_path = Path(mx.__file__).resolve()
    lib_path = core_path.parent / "lib/libmlx.dylib"
    if not core_path.is_file() or not lib_path.is_file():
        raise RuntimeError("installed MLX binary paths missing")
    with zipfile.ZipFile(wheel_path) as archive:
        for name, installed in ((f"mlx/{core_path.name}", core_path),
                                ("mlx/lib/libmlx.dylib", lib_path)):
            if hashlib.sha256(archive.read(name)).hexdigest() != _sha256(installed):
                raise RuntimeError(f"configured MLX wheel differs from loaded {name}")
    return mx.__version__, _sha256(wheel_path)


def compute_live_price_identity(
    artifact_manifest_path: str | Path,
    mlx_wheel_path: str | Path,
    kernel_path: str | Path,
    *,
    adapter_artifact_root: str | Path,
) -> dict[str, str]:
    """Return an exact identity from live bytes or fail closed.

    This may hash a multi-gigabyte artifact and should be cached only for the
    verified process lifetime.  It never reads the proposed price JSON.
    """
    import _paged_kv_native as native

    manifest_path = Path(artifact_manifest_path).resolve()
    wheel_path = Path(mlx_wheel_path).resolve()
    kernel_path = Path(kernel_path).resolve()
    adapter_root = Path(adapter_artifact_root).resolve()
    if Path(native.__file__).resolve() != kernel_path or not kernel_path.is_file():
        raise RuntimeError("configured kernel differs from loaded native extension")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise RuntimeError("dirty source tree cannot bind a live price")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    hardware = json.loads(subprocess.check_output(
        ["system_profiler", "SPHardwareDataType", "-json"],
        text=True, timeout=10)).get("SPHardwareDataType", [])
    if len(hardware) != 1 or not isinstance(hardware[0].get("chip_type"), str):
        raise RuntimeError("unique Apple hardware identity missing")
    version, wheel_digest = _wheel_identity(wheel_path)
    return {"host": platform.node(), "hardware": hardware[0]["chip_type"],
            "artifact_sha256": _artifact_sha256(manifest_path, adapter_root),
            "source_commit": commit, "source_tree_sha256": _source_tree_sha256(),
            "mlx_wheel_version": version, "mlx_wheel_sha256": wheel_digest,
            "kernel_sha256": _sha256(kernel_path)}


def _watched_paths(manifest_path: Path, wheel_path: Path, kernel_path: Path) -> tuple[Path, ...]:
    """Paths whose live bytes were hashed by the full attestation."""
    import mlx.core as mx

    artifact = json.loads(manifest_path.read_text())
    root = Path(artifact["root"]).resolve()
    files = artifact["files"]
    if not isinstance(files, dict):
        raise RuntimeError("artifact manifest files changed after attestation")
    core = Path(mx.__file__).resolve()
    return (manifest_path, wheel_path, kernel_path, core,
            core.parent / "lib/libmlx.dylib",
            *(root / name for name in sorted(files)))


def _file_signatures(paths: tuple[Path, ...]) -> tuple:
    signatures = []
    for path in paths:
        resolved = path.resolve(strict=True)
        signatures.append((str(resolved), _path_signature(path)))
    return tuple(signatures)


def _path_signature(path: str | Path):
    """lstat bytes-bearing names; bind symlink text and resolved target too."""
    try:
        st=os.lstat(path)
        result=(st.st_dev,st.st_ino,st.st_mode,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
        if stat.S_ISLNK(st.st_mode):
            target=Path(path).resolve(strict=True);ts=target.stat()
            result+=(os.readlink(path),str(target),ts.st_dev,ts.st_ino,ts.st_mode,
                     ts.st_size,ts.st_mtime_ns,ts.st_ctime_ns)
        return result
    except (FileNotFoundError,NotADirectoryError):return None


def _walk_error(error):raise error


def _source_directories(root: Path):
    paths=[]
    for directory,names,_ in os.walk(root,followlinks=False,onerror=_walk_error):
        names[:]=sorted(name for name in names if name!='.git')
        paths.extend((directory,str(Path(directory)/'.gitignore')))
        # os.walk never descends symlink directories; bind their names/targets.
        paths.extend(str(Path(directory)/name) for name in names
                     if os.path.islink(Path(directory)/name))
    return tuple(sorted(set(paths)))


def _git_metadata_paths(root: Path):
    gitdir=Path(subprocess.check_output(['git','rev-parse','--absolute-git-dir'],cwd=root,text=True).strip())
    common=Path(subprocess.check_output(['git','rev-parse','--git-common-dir'],cwd=root,text=True).strip())
    if not common.is_absolute():common=root/common
    common=common.resolve()
    # Git status creates/removes index.lock in gitdir: its directory timestamps
    # are not source drift. A shared common/refs directory changes when another
    # worktree creates a branch; only this checkout's HEAD chain affects its
    # clean source commit. A missing loose ref is watched too, since Git may
    # replace a packed ref with a loose one.
    paths=set()
    if not (root/'.git').is_dir() or (root/'.git').is_symlink():paths.add(str(root/'.git'))
    paths.update(str(gitdir/name) for name in
                 ('HEAD','index','index.lock','config','config.worktree','commondir','info/exclude'))
    paths.update(str(common/name) for name in ('config','packed-refs','info/exclude'))
    head=gitdir/'HEAD';content=head.read_text()
    seen=set()
    for _ in range(16):
        if not content.startswith('ref: '):break
        ref=content[5:].strip()
        if (not ref.startswith('refs/') or ref in seen or
                any(part in ('','.','..') for part in ref.split('/'))):
            raise RuntimeError('invalid current Git symbolic ref chain')
        seen.add(ref)
        refpath=Path(subprocess.check_output(
            ['git','rev-parse','--git-path',ref],cwd=root,text=True).strip())
        if not refpath.is_absolute():refpath=root/refpath
        paths.add(str(refpath))
        content=refpath.read_text() if refpath.is_file() else ''
    else:raise RuntimeError('current Git symbolic ref chain exceeds bound')
    # Git cleanliness also depends on global/include config and ignore files.
    home=Path(os.environ.get('HOME',str(Path.home())))
    xdg=Path(os.environ.get('XDG_CONFIG_HOME',str(home/'.config')))
    paths.update(str(p) for p in (home/'.gitconfig',xdg/'git/config',xdg/'git/ignore',Path('/etc/gitconfig')))
    origins=subprocess.check_output(['git','config','--show-origin','--null','--list'],cwd=root).split(b'\0')
    for origin,entry in zip(origins[::2],origins[1::2]):
        if origin.startswith(b'file:'):
            p=Path(os.fsdecode(origin[5:])).expanduser()
            p=p if p.is_absolute() else root/p;paths.add(str(p))
            key,_,value=entry.partition(b'\n')
            if key.startswith(b'include') and key.endswith(b'.path') and value:
                included=Path(os.fsdecode(value)).expanduser()
                paths.add(str(included if included.is_absolute() else p.parent/included))
    paths.update(os.environ[name] for name in ('GIT_CONFIG','GIT_CONFIG_GLOBAL','GIT_CONFIG_SYSTEM') if os.environ.get(name))
    excludes=subprocess.run(['git','config','--path','--get','core.excludesfile'],cwd=root,text=True,capture_output=True)
    if excludes.returncode not in (0,1):raise RuntimeError('Git exclude configuration unavailable')
    if excludes.returncode==0:
        p=Path(excludes.stdout.strip()).expanduser();paths.add(str(p if p.is_absolute() else root/p))
    return tuple(sorted(paths))


def _git_source_clean(root: Path,commit: str):
    return (subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()==commit
            and not subprocess.check_output(['git','--no-optional-locks','status','--porcelain'],cwd=root).strip())


def _signatures(paths):return tuple(_path_signature(path) for path in paths)


def _git_environment():
    return tuple(sorted((key,value) for key,value in os.environ.items()
                        if key.startswith('GIT_') or key in ('HOME','XDG_CONFIG_HOME')))


def _source_unchanged(commit: str) -> bool:
    """Verified source inventory fast path; any bytes drift poisons this memo.

    New names change a watched directory. HEAD/index/ref/config changes and
    directory changes require real Git cleanliness before refreshing metadata.
    All original tracked names, symlink targets and signatures remain fixed.
    """
    root=ROOT.resolve();key=(str(root),commit)
    with _IDENTITY_LOCK:
        memo=_SOURCE_MEMO.get(key)
        try:
            if memo is None:
                if not _git_source_clean(root,commit):return False
                names=tuple(os.fsdecode(name) for name in subprocess.check_output(['git','ls-files','-z'],cwd=root).split(b'\0') if name)
                paths=tuple(str(root/name) for name in names)
                files=_signatures(paths)
                # Gitlinks/linked directories have unbound descendant bytes in
                # this source hash schema; require a separate explicit contract.
                if any(signature is None or stat.S_ISDIR(signature[2]) or
                       (stat.S_ISLNK(signature[2]) and stat.S_ISDIR(signature[10]))
                       for signature in files):return False
                dirs=_source_directories(root);metadata=_git_metadata_paths(root)
                memo={'paths':paths,'files':files,'dirs':dirs,'directories':_signatures(dirs),
                      'metadata':metadata,'git':_signatures(metadata),'environment':_git_environment(),'invalid':False}
                # Inventory creation itself must not race source/Git mutation.
                if (not _git_source_clean(root,commit) or files!=_signatures(paths) or
                        memo['directories']!=_signatures(dirs) or memo['git']!=_signatures(metadata)):
                    return False
                _SOURCE_MEMO[key]=memo
                return True
            if memo['invalid']:return False
            if _git_environment()!=memo['environment']:
                memo['invalid']=True;return False
            if _signatures(memo['paths'])!=memo['files']:
                memo['invalid']=True;return False
            directories=_signatures(memo['dirs']);metadata=_signatures(memo['metadata'])
            if directories!=memo['directories'] or metadata!=memo['git']:
                # Includes new untracked names, index edits, moved refs and excludes.
                if not _git_source_clean(root,commit):
                    memo['invalid']=True;return False
                names=tuple(os.fsdecode(name) for name in subprocess.check_output(['git','ls-files','-z'],cwd=root).split(b'\0') if name)
                if tuple(str(root/name) for name in names)!=memo['paths'] or _signatures(memo['paths'])!=memo['files']:
                    memo['invalid']=True;return False
                dirs=_source_directories(root);gitpaths=_git_metadata_paths(root)
                newdirs=_signatures(dirs);newgit=_signatures(gitpaths)
                # Recheck after recapturing new directory/ref names. A concurrent
                # untracked creation or HEAD edit must not enter the fast memo.
                if (not _git_source_clean(root,commit) or _signatures(memo['paths'])!=memo['files'] or
                        newdirs!=_signatures(dirs) or newgit!=_signatures(gitpaths)):
                    memo['invalid']=True;return False
                memo.update(dirs=dirs,directories=newdirs,metadata=gitpaths,git=newgit)
            return True
        except (OSError,RuntimeError,subprocess.SubprocessError):
            if memo is not None:memo['invalid']=True
            return False


def cached_live_price_identity(
    artifact_manifest_path: str | Path,
    mlx_wheel_path: str | Path,
    kernel_path: str | Path,
    *,
    adapter_artifact_root: str | Path,
) -> dict[str, str]:
    """Reuse a process attestation only while source and byte-bearing files stay fixed.

    The first admission runs the full byte verification. Subsequent admissions
    check every tracked source/target and directory plus Git metadata; changes
    require actual Git cleanliness. They also check device, inode, size, mtime and ctime of
    every model, wheel, loaded MLX and kernel file. Any drift permanently
    refuses the memo until process restart; it never silently reattests a
    changed artifact. The price receipt itself is validated on every request.
    """
    manifest = Path(artifact_manifest_path).resolve(strict=True)
    wheel = Path(mlx_wheel_path).resolve(strict=True)
    kernel = Path(kernel_path).resolve(strict=True)
    model = Path(adapter_artifact_root).resolve(strict=True)
    key = (str(manifest), str(wheel), str(kernel), str(model))
    with _IDENTITY_LOCK:
        paths = _watched_paths(manifest, wheel, kernel)
        signatures = _file_signatures(paths)
        memo = _IDENTITY_MEMO.get(key)
        if memo is not None:
            identity, original = memo
            if original != signatures or not _source_unchanged(identity["source_commit"]):
                raise RuntimeError("live price identity changed after attestation; restart required")
            return dict(identity)
        # Seed an inventory before full hashing, then require the same names/bytes
        # after it. This binds the cached source hash to one stable checkout.
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        if not _source_unchanged(source_commit):
            raise RuntimeError('source inventory cannot bind initial attestation')
        identity = compute_live_price_identity(
            manifest, wheel, kernel, adapter_artifact_root=model)
        if (identity["source_commit"]!=source_commit or signatures != _file_signatures(paths) or
                not _source_unchanged(identity["source_commit"])):
            raise RuntimeError("live price identity changed during attestation")
        _IDENTITY_MEMO[key] = (dict(identity), signatures)
        return dict(identity)


__all__ = ["compute_live_price_identity", "cached_live_price_identity"]
