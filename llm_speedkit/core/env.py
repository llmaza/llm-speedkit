from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any, Dict, List

def run_cmd(cmd: List[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out.strip()
    except Exception:
        return ""

def collect_env() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_sha": run_cmd(["git", "rev-parse", "HEAD"]),
    }
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            env["gpu_total_vram_mb"] = int(props.total_memory / (1024**2))
    except Exception:
        pass
    try:
        import transformers
        env["transformers"] = transformers.__version__
    except Exception:
        pass
    return env
