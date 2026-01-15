"""I/O utilities for reproducibility and metadata."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def ensure_dir(path: str | Path) -> str:
    """Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The string path.
    """
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    """Save a JSON file.

    Args:
        data: JSON-serializable dict.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_yaml(data: Dict[str, Any], path: str | Path) -> None:
    """Save a YAML file.

    Args:
        data: YAML-serializable dict.
        path: Output file path.
    """
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def get_git_hash() -> str:
    """Return the current git commit hash if available."""
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "unknown"


def get_env_snapshot() -> Dict[str, Any]:
    """Capture environment information for reproducibility."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def get_hardware_info() -> Dict[str, Any]:
    """Capture hardware info (GPU/CPU) for reproducibility."""
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    gpu_count = torch.cuda.device_count()
    return {
        "gpu_name": gpu_name,
        "gpu_count": gpu_count,
    }


def get_software_info() -> Dict[str, Any]:
    """Capture software versions."""
    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
