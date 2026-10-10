"""Conservative, import-free source dependency hashing.

All imports, including imports inside functions, contribute. Dynamic dispatch
may be narrowed only at a reviewed, byte-exact module. Unknown local dispatch
raises; callers retain their full-source identity rather than omit evidence.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path


class UnresolvedDependency(ValueError):
    pass


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def source_closure(root, roots, *, reviewed_dynamic=None):
    root = Path(root).resolve()
    package = root.name
    reviewed_dynamic = reviewed_dynamic or {}
    files = {}
    pending = list(roots)
    visited = set()

    def resolve(module):
        if module != package and not module.startswith(package + "."):
            return None
        parts = module.split(".")[1:]
        file = (
            root.joinpath(*parts).with_suffix(".py") if parts else root / "__init__.py"
        )
        directory = root.joinpath(*parts)
        init = directory / "__init__.py"
        if init.is_file():
            return init
        if file.is_file():
            return file
        if directory.is_dir():
            return directory  # namespace package, no initializer to hash
        return None

    def add(module, *, required=False):
        if module == package or module.startswith(package + "."):
            path = resolve(module)
            if path is not None:
                pending.append(module)
            elif required:
                raise UnresolvedDependency(f"missing local module: {module}")

    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = resolve(module)
        if path is None:
            raise UnresolvedDependency(f"missing root: {module}")
        for size in range(1, len(module.split("."))):
            add(".".join(module.split(".")[:size]))
        if path.is_dir():
            continue
        if not path.resolve().is_relative_to(root):
            raise UnresolvedDependency(f"source escaped package root: {module}")
        if (
            list(path.parent.glob(path.stem + "*.so"))
            or path.with_suffix(".pyc").is_file()
        ):
            raise UnresolvedDependency(f"non-source module shadows source: {module}")
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        files[relative] = hashlib.sha256(data).hexdigest()
        parent = module if path.name == "__init__.py" else module.rpartition(".")[0]
        try:
            tree = ast.parse(data, filename=str(path))
        except (SyntaxError, ValueError) as exc:
            raise UnresolvedDependency(f"unparseable source: {module}") from exc
        dynamic = []
        loader_names = {"__import__", "eval", "exec"}
        opaque_loaders = {"eval", "exec", "spec_from_file_location"}
        builtin_imports = {"__import__"}
        loader_modules = {"importlib", "builtins"}
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
        direct_loaders = {id(n.func) for n in calls}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {"importlib", "importlib.util", "builtins"}:
                        loader_modules.add(alias.asname or alias.name.split(".")[0])
            if isinstance(node, ast.ImportFrom) and node.module in (
                "importlib",
                "importlib.util",
                "builtins",
            ):
                for alias in node.names:
                    if alias.name in {
                        "import_module",
                        "spec_from_file_location",
                        "__import__",
                        "eval",
                        "exec",
                    }:
                        bound = alias.asname or alias.name
                        loader_names.add(bound)
                        if alias.name in opaque_loaders:
                            opaque_loaders.add(bound)
                        if alias.name == "__import__":
                            builtin_imports.add(bound)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name, required=True)
            elif isinstance(node, ast.ImportFrom):
                name = "." * node.level + (node.module or "")
                try:
                    base = (
                        importlib.util.resolve_name(name, parent)
                        if node.level
                        else name
                    )
                except (ImportError, ValueError) as exc:
                    raise UnresolvedDependency(
                        f"invalid relative import: {module}:{node.lineno}"
                    ) from exc
                add(base, required=True)
                for alias in node.names:
                    if alias.name != "*":
                        add(base + "." + alias.name)
            elif isinstance(node, ast.Call):
                name = ast.unparse(node.func)
                leaf = getattr(node.func, "attr", name)
                if (
                    name in loader_names
                    or leaf
                    in {
                        "import_module",
                        "spec_from_file_location",
                        "__import__",
                    }
                    or leaf in {"eval", "exec"}
                    and name.startswith("builtins.")
                ):
                    dynamic.append(node)
                elif (
                    name == "getattr"
                    and len(node.args) > 1
                    and (
                        ast.unparse(node.args[0]).split(".")[0] in loader_modules
                        or isinstance(node.args[1], ast.Constant)
                        and node.args[1].value
                        in {
                            "import_module",
                            "spec_from_file_location",
                            "__import__",
                        }
                    )
                ):
                    # Reflection can hide a loader reference from the import
                    # graph. Reflective loader access requires explicit review.
                    dynamic.append(node)
            elif (
                isinstance(node, ast.Attribute)
                and node.attr
                in {
                    "import_module",
                    "spec_from_file_location",
                    "__import__",
                }
                or isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in loader_names
            ) and id(node) not in direct_loaders:
                dynamic.append(node)
        if dynamic:
            review = reviewed_dynamic.get(module)
            if review is not None:
                if review.get("sha256") != files[relative]:
                    raise UnresolvedDependency(f"reviewed dispatcher changed: {module}")
                for target in review.get("targets", ()):
                    add(target, required=True)
                continue
            for node in dynamic:
                if not isinstance(node, ast.Call):
                    raise UnresolvedDependency(
                        f"aliased dynamic loader: {module}:{node.lineno}"
                    )
                name = ast.unparse(node.func)
                leaf = getattr(node.func, "attr", name)
                value = node.args[0] if node.args else None
                # Absolute external imports do not add local source. A literal
                # fromlist does not change their package; relative/opaque
                # __import__ arguments still require an exact-byte review.
                if (
                    (name in builtin_imports or leaf == "__import__")
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value != package
                    and not value.value.startswith(package + ".")
                    and len(node.args) == 1
                    and all(
                        kw.arg == "fromlist"
                        and isinstance(kw.value, (ast.List, ast.Tuple))
                        and all(
                            isinstance(x, ast.Constant) and isinstance(x.value, str)
                            for x in kw.value.elts
                        )
                        for kw in node.keywords
                    )
                ):
                    continue
                if (
                    name in opaque_loaders
                    or leaf in opaque_loaders
                    or name == "getattr"
                    or not isinstance(value, ast.Constant)
                    or not isinstance(value.value, str)
                    or (name in builtin_imports or leaf == "__import__")
                    and (len(node.args) > 1 or node.keywords)
                ):
                    raise UnresolvedDependency(
                        f"unresolved dynamic import: {module}:{node.lineno}"
                    )
                target = value.value
                if target.startswith("."):
                    # A nonconstant package argument cannot be inferred.
                    package_arg = (
                        node.args[1]
                        if len(node.args) > 1
                        else next(
                            (kw.value for kw in node.keywords if kw.arg == "package"),
                            None,
                        )
                    )
                    if (
                        isinstance(package_arg, ast.Name)
                        and package_arg.id == "__package__"
                    ):
                        target_package = parent
                    elif isinstance(package_arg, ast.Constant) and isinstance(
                        package_arg.value, str
                    ):
                        target_package = package_arg.value
                    else:
                        raise UnresolvedDependency(
                            f"unknown dispatch package: {module}:{node.lineno}"
                        )
                    target = importlib.util.resolve_name(target, target_package)
                add(target, required=True)
    return dict(sorted(files.items()))


def digest_files(files):
    """Bind names and content with unambiguous framing."""
    return canonical_digest(files)


def add_resources(root, files):
    """Bind adjacent files and declared package-data directories.

    Package data is not discoverable from Python import syntax. Documentation
    is excluded: source checkouts can carry READMEs absent from wheels. These owned
    data roots are included conservatively whenever their owner is reachable.
    """
    root = Path(root).resolve()
    directories = {(root / name).parent for name in files}
    nested = []
    if any(name.startswith("runtime/") for name in files):
        nested.append(root / "runtime/data")
    if any(name.startswith("adapters/") for name in files):
        nested.append(root / "adapters/assets")
    if "adapters/vlm_runtime.py" in files:
        nested.append(root / "adapters/vlm_contracts")
    paths = [path for directory in directories for path in directory.iterdir()]
    paths.extend(path for directory in nested for path in directory.rglob("*"))
    for path in paths:
        if (
            path.is_file()
            and path.suffix.lower() not in {".py", ".pyc", ".pyo", ".md", ".rst"}
            and not path.name.startswith(".")
        ):
            if not path.resolve().is_relative_to(root):
                raise UnresolvedDependency("resource escaped package root")
            files[path.relative_to(root).as_posix()] = file_digest(path)
    return files
