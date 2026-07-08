from __future__ import annotations

import re
from pathlib import Path

import lance_ray
import tomllib
from packaging.requirements import Requirement
from packaging.version import Version

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


def test_gpu_extra_requires_rapids_2606() -> None:
    with (_REPO_ROOT / "pyproject.toml").open("rb") as stream:
        pyproject = tomllib.load(stream)
    requirements = pyproject["project"]["optional-dependencies"]["gpu"]
    assert len(requirements) == 1

    cudf = Requirement(requirements[0])
    assert cudf.name == "cudf-cu12"
    assert Version("26.6.0") in cudf.specifier
    assert Version("25.10.0") not in cudf.specifier
    assert Version("26.7.0") not in cudf.specifier
    assert str(cudf.marker) == (
        'python_version >= "3.11" and platform_system == "Linux" '
        'and platform_machine == "x86_64"'
    )
