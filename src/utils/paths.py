"""Path helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_path(path: str) -> str:
    """Resolve a path to an absolute string path.

    Args:
        path: Input path.

    Returns:
        Absolute path string.
    """
    return str(Path(path).expanduser().resolve())
