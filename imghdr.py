"""Backport shim for the Python stdlib imghdr module removed in Python 3.13.

Streamlit 1.19 imports `imghdr` for image type detection. Python 3.13
removed the stdlib module, so provide a minimal compatible API using
Pillow, which Streamlit already depends on.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable, List, Tuple

from PIL import Image

__all__ = ["what", "tests"]

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
    """Return the image type based on a filename or header.

    Mirrors the stdlib signature enough for Streamlit's use:
    - If `h` is provided, use it as the header buffer.
    - Otherwise read the first 32 bytes of `file`.
    """
    if h is None:
        if file is None:
            return None
        path = Path(file) if not isinstance(file, bytes) else None
        if path is None:
            h = file[:32]
        else:
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
