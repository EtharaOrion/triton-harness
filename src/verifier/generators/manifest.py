# Verifier bundle generation-half module (self-contained; no external deps on the vendored source).
"""#3a — the persisted target-module manifest.

The harness re-derives the target file set each run and never persists it, so
"did the agent address ALL N target modules?" can't be checked. The manifest is
derivable HOST-SIDE from the golden diff (the files the golden solution changed =
the files that were stubbed) — no run-time harness change needed. It is written
alongside the frozen bundle and consumed by an upgraded DRAFT_MODULES_ADDRESSED.
"""
from __future__ import annotations

import json
from pathlib import Path

# The stub-body markers each language's stubber plants (tools/*stubber*). Python's
# `all` mode is deliberately absent: it leaves bare `pass` bodies with no marker.
_STUB_MARKERS = (
    "STUB: not implemented",           # go (gostubber) + rust (ruststubber)
    'new Error("STUB")',               # ts/js (stub_ts.ts)
    "__builtin_trap",                  # c/cpp (cppstubber / tree-sitter fallback)
    "UnsupportedOperationException",   # java (javastubber)
    "NotImplementedError",             # python docstring-mode stubs
)


def target_files_from_diff(golden_diff: str) -> list[str]:
    out = []
    for line in golden_diff.splitlines():
        if line.startswith("+++ b/") and line[6:].strip() != "/dev/null":
            out.append(line[6:].strip())
        elif line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4:
                out.append(parts[-1][2:] if parts[-1].startswith("b/") else parts[-1])
    return sorted(set(out))


def required_target_files(golden_diff: str) -> list[str]:
    """The files the agent actually HAD to modify — the gating subset of
    ``target_files_from_diff``.

    The full golden-diff file list over-includes: the stub tree also strips
    docstrings/comments and the reference restores them, so files whose diff is
    ADDITION-ONLY (restored module docstring, license header, doc comment, a CI
    workflow) carry no stubbed code and an agent that passes every test without
    touching them did nothing wrong. Requiring them quarantined honest runs
    (ssh-audit: 100% tests passed, "missed" 13 doc-only files).

    Two signals, strongest first:
      * STUB MARKERS — every non-python stubber plants a recognizable marker in
        each stubbed body (go/rust `STUB: not implemented`, ts/js
        `new Error("STUB")`, c/cpp `__builtin_trap`, java
        `UnsupportedOperationException`). When ANY file's diff removes a marker,
        the marker files ARE the stubbed set and nothing else is required — a
        `doc.go` whose block doc-comment was stripped has real deletions but no
        marker, and must not gate.
      * NON-BLANK DELETIONS — python's `all` stub mode leaves marker-less `pass`
        bodies, so with no marker anywhere a file is required iff its diff
        removes at least one non-blank line (addition-only doc restorations drop
        out).
    Either way, scoring/build/config/test/CI paths are excluded — those are
    frozen by L0.TEST_FILES_UNTOUCHED, so requiring the agent to edit them would
    make the two gates contradictory.
    """
    removed_nonblank: dict[str, int] = {}
    marker_files: set[str] = set()
    current: str | None = None
    for line in golden_diff.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:].strip()
            current = None if path == "/dev/null" else path
            if current:
                removed_nonblank.setdefault(current, 0)
        elif line.startswith("+++ /dev/null"):
            current = None                      # deleted file: nothing to implement
        elif current and line.startswith("-") and not line.startswith("---"):
            if line[1:].strip():
                removed_nonblank[current] = removed_nonblank.get(current, 0) + 1
                if any(m in line for m in _STUB_MARKERS):
                    marker_files.add(current)
    from .deterministic import _is_cheatable_path

    def _keep(path: str) -> bool:
        return not (path.startswith(".github/") or _is_cheatable_path(path))

    if marker_files:
        return sorted(p for p in marker_files if _keep(p))
    return sorted(p for p, n in removed_nonblank.items() if n > 0 and _keep(p))


def write_manifest(base_dir: str | Path, stub_files: list[str],
                   required: list[str] | None = None) -> Path:
    from . import layout
    out = layout.manifest_path(base_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {"stage1": sorted(set(stub_files))}
    if required is not None:
        data["stage1_required"] = sorted(set(required))
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return out


def load_manifest(base_dir_or_child: str | Path) -> dict | None:
    from . import layout
    p = Path(base_dir_or_child)
    for anc in (p, *p.parents):
        cand = layout.manifest_path(anc)
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
    return None
