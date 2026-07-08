from __future__ import annotations

import re
from pathlib import Path

import lance_ray

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _metadata_version(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE)
    assert match is not None, f"version field not found in {path.name}"
    return match.group(1)


def test_release_version_metadata_is_consistent() -> None:
    project_version = _metadata_version(
        _REPO_ROOT / "pyproject.toml",
        r'^version = "([^"]+)"$',
    )
    bump_version = _metadata_version(
        _REPO_ROOT / ".bumpversion.toml",
        r'^current_version = "([^"]+)"$',
    )

    assert project_version == bump_version == lance_ray.__version__
