# Verifier bundle generation-half module (self-contained; no external deps on the vendored source).
"""The single source of truth for the verifier BUNDLE layout, so every reader and
writer stays consistent.

PORT DELTA: upstream hard-coded the bundle under ``<uuid>/verification/verifiers/``.
Here every path takes a ``base_dir`` and is rooted at it, so taskgen can point the
bundle at ``solution/verifier/``. The upstream location stays derivable
(``default_bundle_dir(uuid_root)``) and the filenames are unchanged:

    <base_dir>/  TRUTH.md  coverage.json  target_manifest.json
                 pytest/ { test_truth_generated.py, predicates.json }
                 rubric/ { rubric.json }

The upstream ``results/`` tree (findings from evaluating agent runs) belongs to the
results/eval half and is deliberately not modelled here.
"""
from __future__ import annotations

from pathlib import Path


def bundle_dir(base_dir: str | Path) -> Path:
    return Path(base_dir)


def default_bundle_dir(uuid_root: str | Path) -> Path:
    """The upstream on-disk location, for callers that still root at a uuid dir."""
    return Path(uuid_root) / "verification" / "verifiers"


def truth_path(base_dir: str | Path) -> Path:
    return bundle_dir(base_dir) / "TRUTH.md"


def coverage_path(base_dir: str | Path) -> Path:
    return bundle_dir(base_dir) / "coverage.json"


def manifest_path(base_dir: str | Path) -> Path:
    return bundle_dir(base_dir) / "target_manifest.json"


def pytest_dir(base_dir: str | Path) -> Path:
    return bundle_dir(base_dir) / "pytest"


def pytest_code_path(base_dir: str | Path) -> Path:
    return pytest_dir(base_dir) / "test_truth_generated.py"


def predicates_path(base_dir: str | Path) -> Path:
    return pytest_dir(base_dir) / "predicates.json"


def rubric_dir(base_dir: str | Path) -> Path:
    return bundle_dir(base_dir) / "rubric"


def rubric_path(base_dir: str | Path) -> Path:
    return rubric_dir(base_dir) / "rubric.json"


def bundle_paths(base_dir: str | Path) -> dict[str, Path]:
    """Every artifact path of one bundle, keyed by artifact name."""
    return {
        "truth": truth_path(base_dir),
        "coverage": coverage_path(base_dir),
        "manifest": manifest_path(base_dir),
        "pytest_code": pytest_code_path(base_dir),
        "predicates": predicates_path(base_dir),
        "rubric": rubric_path(base_dir),
    }
