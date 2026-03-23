from __future__ import annotations
import json
import time
import urllib.request
from typing import Any, Dict, Optional

def post_json(url: str, payload: Dict[str, Any], timeout_sec: float, api_key: Optional[str] = None) -> tuple[Dict[str, Any], float]:
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        raw = resp.read().decode("utf-8")
    dt = time.perf_counter() - t0
    return json.loads(raw), dt