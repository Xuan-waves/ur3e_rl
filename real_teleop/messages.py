from __future__ import annotations

import json
import time
from typing import Any

import numpy as np


def now() -> float:
    return time.monotonic()


def dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"))


def loads(text: str) -> dict[str, Any]:
    return json.loads(text)


def as_vec(value: Any, size: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(value if value is not None else [default] * size, dtype=float)
    if arr.shape != (size,):
        raise ValueError(f"Expected vector of length {size}, got {arr.shape}")
    return arr

