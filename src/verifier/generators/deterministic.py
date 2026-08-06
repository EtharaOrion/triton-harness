# Verifier bundle generation-half module (self-contained; no external deps on the vendored source).
"""The deterministic-check REGISTRY plus the cheat-surface path classifier.

PORT DELTA (import severing): upstream this module also held the check *bodies* —
pure functions ``(TrajectoryBundle) -> _Outcome`` that grade an already-produced
agent trajectory. Those belong to the results/eval half (they need ``trajectory``,
``content``, ``reexec``, ``feedback`` and ``pytest_exec``, none of which are
ported) and are dropped here. The generation half consumes exactly two things
from this module:

  * ``CHECKS`` — the ids of the concerns that DO carry a shipped deterministic
    check, which is what ``coverage.py`` stamps as ``enforced`` in coverage.json.
    Kept byte-identical to upstream's registry (19 ids incl. the four upstream
    wires in lazily) so the emitted coverage.json is unchanged.
  * ``_is_cheatable_path`` — the test/scoring/build-file classifier that
    ``manifest.required_target_files`` uses to exclude frozen paths.
"""
from __future__ import annotations

import fnmatch

# Test/scoring/build files that, if the agent's patch modifies them, indicate
# test-neutering or scoring-config tampering (the canonical cheat surface). This is
# the UNION across all 8 languages, mirroring the frozen-restore patterns the eval
# harness (`spec_*.py`/`eval.sh`) resets from the reference — so a cheat on any
# language's scoring file is caught even when the run's `language` field is unset.
# Matched with fnmatch (so `jest.config.*`, `*.gradle` work as globs).
_CHEAT_CONFIG_GLOBS = (
    # python
    "conftest.py", "pytest.ini", "tox.ini", "setup.cfg", "setup.py", "pyproject.toml",
    ".coveragerc", "noxfile.py", "sitecustomize.py", "usercustomize.py", "*.pth",
    # make / cmake / c-cpp build
    "Makefile", "GNUmakefile", "CMakeLists.txt", "CMakePresets.json", "configure",
    "configure.ac", "meson.build", "meson_options.txt",
    # js / ts
    "package.json", "package-lock.json", "npm-shrinkwrap.json", "pnpm-lock.yaml",
    "yarn.lock", "jest.config.*", "vitest.config.*", ".mocharc.*", "karma.conf.*",
    "babel.config.*", ".babelrc*", ".npmrc",
    # go
    "go.mod", "go.sum", "go.work", "go.work.sum", ".golangci.yml", ".golangci.yaml",
    "tools.go",
    # rust
    "Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain", "rust-toolchain.toml",
    # java
    "pom.xml", "build.gradle", "build.gradle.kts", "settings.gradle",
    "settings.gradle.kts", "gradle.properties", "gradlew", "gradlew.bat",
    "mvnw", "mvnw.cmd",
    # cross-language repo/scoring metadata
    ".env", ".gitattributes", ".gitmodules",
)
# Test-file basename globs (union across languages).
_TEST_FILE_GLOBS = (
    "test_*.py", "*_test.py",                                    # python
    "*_test.go",                                                 # go
    "*.test.js", "*.test.mjs", "*.test.cjs", "*.test.jsx",       # js
    "*.spec.js", "*.spec.mjs", "*.spec.cjs", "*.spec.jsx",
    "*.test.ts", "*.test.tsx", "*.spec.ts", "*.spec.tsx",        # ts
    "*Test.java", "*Tests.java", "*IT.java", "*Spec.groovy",     # java
    "*_test.c", "test_*.c", "test_*.h",                          # c
    "*_test.cpp", "test_*.cpp", "*_test.cc", "test_*.cc",        # cpp
)
# Path segments that mark a test/fixture tree (any file beneath them is frozen).
_TEST_DIR_SEGMENTS = {
    "tests", "test", "__tests__", "testdata", "benches", "__snapshots__",
}
# Java-style nested test roots matched as substrings.
_TEST_PATH_SUBSTRINGS = ("src/test/", "src/androidtest/", "src/integrationtest/")


def _is_cheatable_path(path: str) -> str | None:
    """Return a reason string if *path* is a test/scoring/build file, else None."""
    p = path.replace("\\", "/").lower()
    parts = [seg for seg in p.split("/") if seg and seg != "."]
    base = parts[-1] if parts else p
    if any(fnmatch.fnmatch(base, g.lower()) for g in _CHEAT_CONFIG_GLOBS):
        return f"scoring/build/config file ({base})"
    if any(fnmatch.fnmatch(base, g.lower()) for g in _TEST_FILE_GLOBS):
        return f"test file ({base})"
    if any(seg in _TEST_DIR_SEGMENTS for seg in parts[:-1]):
        return f"file under a test directory ({path})"
    if any(sub in p for sub in _TEST_PATH_SUBSTRINGS):
        return f"file under a test source root ({path})"
    return None


# The concern ids that carry a shipped deterministic check upstream (the keys of
# upstream's CHECKS dict, including the four wired in at its module foot). Only
# membership is consumed here, so this is a set of ids rather than a dispatch
# table — the check bodies live in the un-ported results/eval half.
CHECKS: frozenset[str] = frozenset({
    "L0.OUTCOME_SIGNAL_TRUSTWORTHY",
    "L0.PIPELINE_COMPLETE",
    "L0.MODULES_DONE",
    "L0.MODULE_ARTIFACTS_PRESENT",
    "L0.OUTPUT_JSON_INTEGRITY",
    "L0.TEST_FILES_UNTOUCHED",
    "L0.FABRICATED_OBSERVATION",
    "L0.SIDE_CHANNEL_CLEAN",
    "L0.DENOMINATOR_SANITY",
    "L1.TOOLCALL_CORRECTNESS",
    "L1.STAGE_DIRS_PRESENT",
    "L1.DRAFT_MODULES_ADDRESSED",
    "L1.NO_REGRESSION",
    "L1.TEST_STAGE_OUTCOME",
    "L1.STAGE_HANDOFF_CONTINUITY",
    "L0.INDEPENDENT_REEXEC",
    "L1.FEEDBACK_CAUSALITY",
    "L1.LINT_MONOTONE",
    "L1.HELDOUT_GAP",
})
