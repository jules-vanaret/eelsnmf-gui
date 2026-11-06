from __future__ import annotations

import hashlib
import os
import sys
from typing import Tuple

import numpy as np


class HiddenPrints:
    """Context manager that suppresses stdout/stderr inside the block."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._null = open(os.devnull, "w")
        sys.stdout = self._null
        sys.stderr = self._null

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr
        self._null.close()


def array_signature(
    arr: np.ndarray | None,
) -> Tuple[str, Tuple[int, ...], str] | None:
    """Build a hash signature for numpy arrays used in cache keys."""
    if arr is None:
        return None
    digest = hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()
    return (digest, arr.shape, str(arr.dtype))
