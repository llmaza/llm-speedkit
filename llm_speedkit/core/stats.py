from __future__ import annotations
from typing import List

def pctl(xs: List[float], q: float) -> float:
    """Nearest-rank percentile (matches old _pctl behavior)."""
    if not xs:
        return 0.0
    xs = sorted(xs)
    idx = int((len(xs) - 1) * q)
    return float(xs[idx])

def parse_csv_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_csv_strs(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]
