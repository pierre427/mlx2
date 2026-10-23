#!/usr/bin/env python3
"""Build the local-development MLX2 menu bar app; never starts a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import plistlib
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=root / ".venv/bin/python")
    parser.add_argument("--output", type=Path, default=root / "build/macos/MLX2.app")
    parser.add_argument(
        "--install-dir", type=Path, default=Path.home() / "Applications"
    )
    parser.add_argument(
        "--install", action="store_true", help="copy into ~/Applications/MLX2.app"
    )
    parser.add_argument(
        "--replace-installed",
        action="store_true",
        help="preserve the previous MLX2.app in Application Support and install this build",
    )
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Building this app requires macOS and Xcode command line tools")
    python = args.python.expanduser().absolute()  # Preserve virtualenv identity.
    subprocess.run(
        [str(python), "-c", "import psutil; import mlx2.desktop"],
        env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
        check=True,
    )
    runtime = {"python": str(python), "source": str(root)}
    if args.install:
        # A login-launched app must not execute its interpreter or source from
        # Desktop (TCC). Prepare the complete substrate in Application Support.
        setup = """
import json, threading
from pathlib import Path
from mlx2.desktop_runner import RunnerManager
home = Path.home() / 'Library/Application Support/MLX2'
home.mkdir(parents=True, exist_ok=True)
log = home / 'companion-setup.log'
log.touch(mode=0o600)
record = RunnerManager(home).ensure(home, 'desktop-companion', multimodal=True,
    cancel=threading.Event(), progress=lambda message: None, log_path=log, purpose='desktop-companion')
print(json.dumps({'python': record['python'], 'source': record['source']}))
"""
        print(
            "Preparing login-safe MLX2 runtime (CPU only); details: ~/Library/Application Support/MLX2/companion-setup.log",
            flush=True,
        )
        runtime = json.loads(
            subprocess.check_output(
                [str(python), "-c", setup],
                env={**__import__("os").environ, "PYTHONPATH": str(root / "src")},
                text=True,
            )
        )
    app = args.output.expanduser().absolute()
    contents = app / "Contents"
    binary = contents / "MacOS/MLX2"
    resources = contents / "Resources"
    binary.parent.mkdir(parents=True, exist_ok=True)
    resources.mkdir(parents=True, exist_ok=True)
    iconset = root / "build/macos/MLX2.iconset"
    subprocess.run(
        ["xcrun", "swift", str(root / "scripts/macos_app_icon.swift"), str(iconset)],
        check=True,
    )
    subprocess.run(
        ["iconutil", "-c", "icns", str(iconset), "-o", str(resources / "MLX2.icns")],
        check=True,
    )
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-swift-version",
            "5",
            "-O",
            "-parse-as-library",
            "-target",
            "arm64-apple-macosx15.0",
            "-module-name",
            "MLX2Desktop",
            *map(str, sorted((root / "native/macos").glob("*.swift"))),
            "-o",
            str(binary),
        ],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    (resources / "runtime.json").write_text(
        json.dumps(
            {
                **runtime,
                "build_revision": revision,
                "source_hashes": {
                    str(path.relative_to(root)): hashlib.sha256(
                        path.read_bytes()
                    ).hexdigest()
                    for path in [
                        *sorted((root / "src/mlx2").glob("desktop*.py")),
                        *sorted((root / "native/macos").glob("*.swift")),
                    ]
                },
            },
            indent=2,
        )
        + "\n"
    )
    with (contents / "Info.plist").open("wb") as stream:
        plistlib.dump(
            {
                "CFBundleExecutable": "MLX2",
                "CFBundleIdentifier": "com.mlx2.desktop",
                "CFBundleName": "MLX2",
                "CFBundleDisplayName": "MLX2",
                "NSCameraUsageDescription": "Use your camera only when you enable it in the MLX2 world workspace.",
                "NSMicrophoneUsageDescription": "Use your microphone for voice input when you enable it in MLX2.",
                "NSSpeechRecognitionUsageDescription": "Transcribe speech for your MLX2 conversation when you enable voice input.",
                "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": "0.1.0",
                "CFBundleVersion": "1",
                "CFBundleIconFile": "MLX2",
                "LSUIElement": True,
                "LSMinimumSystemVersion": "15.0",
                "NSHighResolutionCapable": True,
                "NSHumanReadableCopyright": "Copyright © 2026 MLX2 contributors",
            },
            stream,
        )
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
    if args.install:
        destination = args.install_dir.expanduser().resolve() / "MLX2.app"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink():
                parser.error("Refusing to replace a linked application bundle")
            if not args.replace_installed:
                parser.error(
                    f"{destination} already exists; pass --replace-installed to preserve and replace it"
                )
            with (destination / "Contents/Info.plist").open("rb") as stream:
                if (
                    plistlib.load(stream).get("CFBundleIdentifier")
                    != "com.mlx2.desktop"
                ):
                    parser.error("Existing destination is not the MLX2 desktop app")
            backup = (
                Path.home()
                / "Library/Application Support/MLX2/app-backups"
                / f"MLX2-{uuid.uuid4().hex}.app"
            )
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(destination), str(backup))
            try:
                shutil.copytree(app, destination)
            except BaseException:
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.move(str(backup), str(destination))
                raise
        else:
            shutil.copytree(app, destination)
        app = destination
    print(app)


if __name__ == "__main__":
    main()
