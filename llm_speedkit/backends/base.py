from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseBackend(ABC):
    name: str

    @abstractmethod
    def run(self, cfg) -> Dict[str, Any]:
        ...

    def healthcheck(self, cfg) -> Dict[str, Any]:
        return {"status": "ok"}

    def supports_bench(self) -> bool:
        return False
    
# Then make HFBackend and VLLMBackend inherit from it.

# This is small, but it matters because Triton and TRT-LLM are not going to behave like your current local backends.