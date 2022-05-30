"""Utilities."""

from pathlib import Path


def get_project_root_path() -> Path:
    """Get path of the project root.

    Returns:
        Path: Project root path
    """
    return Path(__file__).parents[2].absolute()
