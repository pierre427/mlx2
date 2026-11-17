"""Device setup shared by the amoeba probes: run on the Apple GPU (MPS) or the CPU.

setup("mps") selects the GPU; place(), called right after the model loads, makes new
tensors default to it and maps torch.load onto it. Probes create
random generators with generator() (not torch.Generator(), which stays on the CPU); patching
torch.Generator itself breaks torch.distributed's type annotations at import.
Operations MPS lacks (e.g. linalg.eigh) need PYTORCH_ENABLE_MPS_FALLBACK=1 in the launch
environment. Receipts record DEVICE, never a hard-coded "cpu".
"""
from __future__ import annotations

import functools

import torch

DEVICE = "cpu"


def setup(device: str = "cpu") -> str:
    global DEVICE
    DEVICE = device
    if device == "cpu":
        return DEVICE
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS requested but not available")
    return DEVICE


def place() -> str:
    """Call after the model is loaded (transformers refuses a default device without accelerate):
    new tensors then default to DEVICE and torch.load maps onto it."""
    if DEVICE != "cpu":
        torch.set_default_device(DEVICE)
        if not isinstance(torch.load, functools.partial):
            torch.load = functools.partial(torch.load, map_location=DEVICE)
    return DEVICE


def generator(seed: int = 0) -> torch.Generator:
    """A seeded generator on the current device (matches tensors created by default)."""
    return torch.Generator(device=DEVICE).manual_seed(seed)
