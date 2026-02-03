"""Provide a fallback imghdr module for Streamlit on Python 3.13.

Python 3.13 removed the stdlib `imghdr` module, but Streamlit 1.19 still
imports it at startup. When this file is on `sys.path`, Python loads it
automatically after site initialization, so we can register a small shim
module in `sys.modules` before Streamlit imports occur.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from types import ModuleType
from typing import Callable, List, Tuple

from PIL import Image

Test = Tuple[str, Callable[[bytes], str | None]]


def _pillow_detector(header: bytes) -> str | None:
    try:
        with Image.open(BytesIO(header)) as img:
            return img.format.lower() if img.format else None
    except Exception:
        return None


tests: List[Test] = [
    ("pillow", _pillow_detector),
]


def what(file: str | bytes | Path | None, h: bytes | None = None) -> str | None:
    if h is None:
        if file is None:
            return None
        if isinstance(file, bytes):
            h = file[:32]
        else:
            path = Path(file)
            try:
                with path.open("rb") as handle:
                    h = handle.read(32)
            except OSError:
                return None

    for _name, test in tests:
        result = test(h)
        if result:
            return result
    return None


if "imghdr" not in sys.modules:
    module = ModuleType("imghdr")
    module.__all__ = ["what", "tests"]
    module.tests = tests
    module.what = what
    sys.modules["imghdr"] = module
