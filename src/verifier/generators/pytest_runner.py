# Verifier bundle generation-half module (self-contained; no external deps on the vendored source).
"""Execute a generated pytest module against a solution DIRECTORY and parse the
result.

The generated suite is run two ways for the soundness loop:
  * against the GOLDEN tree -> should PASS (the tests are sound), and
  * against the STUB tree   -> should FAIL (the tests actually discriminate).

PORT DELTA: upstream also shipped ``checkout_solution``/``remove_worktree``, which
materialise a tree with `git worktree add` at a commit. taskgen has both trees in
memory (carve.originals / carve.overlay) and no git seam, so only the
directory-based runner is ported; how a directory gets materialised is the
caller's business.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

_SUMMARY_RE = {
    "passed": re.compile(r"(\d+) passed"),
    "failed": re.compile(r"(\d+) failed"),
    "error": re.compile(r"(\d+) errors?"),
    "skipped": re.compile(r"(\d+) skipped"),
}
# `-rA` short-summary lines: "PASSED <nodeid>" / "FAILED <nodeid> - <reason>".
# pytest appends " - <reason>" to FAILED/ERROR lines when the message is short, so
# the nodeid must be captured with an OPTIONAL trailing reason (not anchored to EOL).
_PERTEST_RE = re.compile(
    r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+::\S+?)(?:\s+-\s.*)?\s*$", re.M)
_OUTCOME_MAP = {"PASSED": "pass", "XPASS": "pass", "FAILED": "fail", "ERROR": "error",
                "SKIPPED": "skip", "XFAIL": "skip"}
_TESTFILE = "_verifier_generated_test.py"


def test_key(nodeid: str) -> str:
    """Stable per-test key across golden/stub/solution runs (drop the filename)."""
    return nodeid.split("::", 1)[1] if "::" in nodeid else nodeid


def parse_per_test(text: str) -> dict[str, str]:
    """{test_key: 'pass'|'fail'|'error'|'skip'} from the -rA short summary."""
    out: dict[str, str] = {}
    for m in _PERTEST_RE.finditer(text):
        out[test_key(m.group(2))] = _OUTCOME_MAP.get(m.group(1), "fail")
    return out


@dataclass
class PytestResult:
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    collected: int = 0
    status: str = "OK"        # OK | COLLECTION_ERROR | NO_TESTS | INFRA
    detail: str = ""
    output: str = ""          # full pytest stdout+stderr (which tests failed + tracebacks)
    # PORT DELTA: upstream declared `per_test: dict = None` + a __post_init__
    # fixup; a default_factory is the same behaviour without the wrong annotation.
    per_test: dict[str, str] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.passed + self.failed + self.errors

    @property
    def pass_rate(self) -> float | None:
        return self.passed / self.total if self.total else None

    def to_dict(self, *, include_output: bool = False) -> dict:
        d = asdict(self)
        d["total"] = self.total
        d["pass_rate"] = self.pass_rate
        if not include_output:
            d.pop("output", None)   # kept in a .log file, not inlined by default
        return d


def _parse(text: str, returncode: int) -> PytestResult:
    def n(key):
        m = _SUMMARY_RE[key].search(text)
        return int(m.group(1)) if m else 0
    passed, failed, errors, skipped = n("passed"), n("failed"), n("error"), n("skipped")
    status = "OK"
    if returncode == 5 or (passed + failed + errors == 0):
        status = "NO_TESTS"
    if "errors during collection" in text or "ERROR collecting" in text or errors and not (passed + failed):
        status = "COLLECTION_ERROR"
    return PytestResult(passed=passed, failed=failed, errors=errors, skipped=skipped,
                        collected=passed + failed + errors + skipped, status=status,
                        detail=text.strip().splitlines()[-1] if text.strip() else "",
                        output=text, per_test=parse_per_test(text))


def run_pytest_in_dir(solution_dir: str | Path, test_code: str, *,
                      timeout: int = 180, python: str = "python") -> PytestResult:
    """Write *test_code* into *solution_dir* and run pytest on just that file, with
    the solution on sys.path. Isolated to that one file so we only run the generated
    tests, not the repo's own suite. Cleans up afterward."""
    solution_dir = Path(solution_dir)
    if not solution_dir.is_dir():
        return PytestResult(status="INFRA", detail=f"solution dir missing: {solution_dir}")
    test_path = solution_dir / _TESTFILE
    test_path.write_text(test_code, encoding="utf-8")
    try:
        r = subprocess.run(
            [python, "-m", "pytest", _TESTFILE, "-p", "no:cacheprovider",
             "--no-header", "--tb=short", "-rA", "-o", "addopts="],
            cwd=str(solution_dir), capture_output=True, text=True, timeout=timeout,
            env=_env_with_pythonpath(solution_dir))
        return _parse(r.stdout + "\n" + r.stderr, r.returncode)
    except subprocess.TimeoutExpired:
        return PytestResult(status="INFRA", detail="pytest timed out")
    except (OSError, subprocess.SubprocessError) as e:
        return PytestResult(status="INFRA", detail=str(e))
    finally:
        try:
            test_path.unlink()
        except OSError:
            pass


def _env_with_pythonpath(solution_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    src = solution_dir / "src"
    parts = [str(solution_dir)]
    if src.is_dir():
        parts.append(str(src))
    if env.get("PYTHONPATH"):
        parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env
