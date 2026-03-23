from __future__ import annotations

from .hf import HFBackend
from .vllm import VLLMBackend
from .triton_openai import TritonOpenAIBackend
# from .triton_http import TritonHTTPBackend
# from .trtllm import TRTLLMBackend
# from .triton_trtllm import TritonTRTLLMBackend

_BACKENDS = {
    "hf": HFBackend,
    "vllm": VLLMBackend,
    "triton_openai": TritonOpenAIBackend,
    # "triton_http": TritonHTTPBackend,
    # "trtllm": TRTLLMBackend,
    # "triton_trtllm": TritonTRTLLMBackend,
}

def get_backend(name: str):
    try:
        return _BACKENDS[name]()
    except KeyError as e:
        available = ", ".join(sorted(_BACKENDS))
        raise ValueError(f"Unknown backend: {name}. Available: {available}") from e