from __future__ import annotations

from typing import Literal

from .hf import HFBackend
from .vllm import VLLMBackend

BackendName = Literal["hf", "vllm"]

def get_backend(name: BackendName):
    if name == "hf":
        return HFBackend()
    if name == "vllm":
        return VLLMBackend()
    raise ValueError(f"Unknown backend: {name}")