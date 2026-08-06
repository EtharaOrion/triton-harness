"""The v2 soundness executor: run a generated pytest module against BOTH trees.

`verifier_loop.sound_pytest_loop` keeps a generated test only when it PASSES on
the golden (oracle-restored) tree and FAILS on the stub tree. That is not a
text property — it has to be executed. This module is the execution seam:

    golden tree   a private copy of the checkout with `carve.originals`
                  written back over every carved path (the oracle, exactly as
                  `solve.sh` would restore it)
    stub tree     the same copy with `carve.overlay` written and
                  `carve.deleted_relpaths` removed (what the solver sees)

Both trees are materialised UNDER `work_dir`, never in the source checkout and
never in a staged build context: `pytest_runner.run_pytest_in_dir` drops a test
file into the directory it runs in, and neither the user's repo nor a docker
build context may be written to. The oracle therefore never leaves the host and
never reaches a layer.

WHICH INTERPRETER. The default is `sys.executable`, which under `uv run taskgen`
IS the project environment's interpreter — the same one `uv run pytest` would
select. Spawning a nested `uv run` inside the materialised tree would instead
resolve the CARVED REPO's own lockfile, i.e. hit the network at generate time
and install a second environment; `--verifier` is opt-in, not a licence to go
online behind the caller's back. Point `python=` (or `TASKGEN_VERIFIER_PYTHON`)
at a prepared interpreter when the target repo needs its own dependencies; when
the imports are unavailable the generated tests error on BOTH trees, no test is
sound, and the bundle aborts loudly rather than shipping unvalidated checks.

PYTHON ONLY, ON PURPOSE. The other five taskgen languages grade a whole suite
through a docker measure image, which is a different executor entirely; asking
for one here raises `NotImplementedError`. It must never degrade to "assume
sound" — an unexecuted check is exactly the unsound artifact v2 exists to
prevent (PLAN §3.1, §8b follow-up matrix).
"""
from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .generators.pytest_runner import run_pytest_in_dir

__all__ = [
    "ExecEnvError",
    "PythonExecEnv",
    "TREE_KINDS",
    "make_exec_env",
    "run_generated_pytest",
]

#: The two trees the soundness gate compares. `verifier_loop.RunPytest` passes
#: these exact strings.
TREE_KINDS = ("golden", "stub")

#: Never copied into a materialised tree: VCS history (a leak route AND huge),
#: prebuilt environments, and caches that would import stale bytecode.
_SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".tox", "node_modules",
})

#: Override the interpreter the generated tests run under.
PYTHON_ENV_VAR = "TASKGEN_VERIFIER_PYTHON"


class ExecEnvError(RuntimeError):
    """The soundness environment could not be materialised. Never downgraded."""


def _default_python() -> str:
    return os.environ.get(PYTHON_ENV_VAR) or sys.executable


def _ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in _SKIP_DIRS}


@dataclass
class PythonExecEnv:
    """Materialises the golden and stub trees once, then runs generated tests.

    The trees are built lazily and cached: the soundness loop runs up to three
    generations x two trees, and copying the checkout six times would dominate
    generate time for no additional proof.
    """

    repo: Path
    carve: object
    work_dir: Path
    python: str = field(default_factory=_default_python)
    timeout: int = 180
    _trees: dict[str, Path] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.repo = Path(self.repo).resolve()
        self.work_dir = Path(self.work_dir).resolve()
        if not self.repo.is_dir():
            raise ExecEnvError(f'repo is not a directory: {self.repo}')
        if self.work_dir == self.repo or self.repo in self.work_dir.parents:
            raise ExecEnvError(
                f'refusing to materialise the soundness trees inside the source '
                f'checkout ({self.work_dir} is under {self.repo}): the runner writes '
                'a test file into the tree it runs in'
            )

    # --- materialisation --------------------------------------------------

    def tree(self, tree_kind: str) -> Path:
        if tree_kind not in TREE_KINDS:
            raise ExecEnvError(
                f'unknown tree kind {tree_kind!r}; expected one of {TREE_KINDS}'
            )
        cached = self._trees.get(tree_kind)
        if cached is not None:
            return cached
        dest = self._materialise(tree_kind)
        self._trees[tree_kind] = dest
        return dest

    def _materialise(self, tree_kind: str) -> Path:
        dest = self.work_dir / tree_kind
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.repo, dest, ignore=_ignore, symlinks=True)

        originals = dict(getattr(self.carve, 'originals', {}) or {})
        overlay = dict(getattr(self.carve, 'overlay', {}) or {})
        deleted = tuple(getattr(self.carve, 'deleted_relpaths', ()) or ())

        if tree_kind == 'golden':
            # Restore the oracle explicitly rather than trusting the checkout to
            # still hold it: the golden tree must be the bytes solve.sh restores.
            for rel, text in sorted(originals.items()):
                self._write(dest, rel, text)
        else:
            for rel, text in sorted(overlay.items()):
                self._write(dest, rel, text)
            for rel in sorted(deleted):
                victim = dest / rel
                if victim.is_file():
                    victim.unlink()
        return dest

    @staticmethod
    def _write(root: Path, rel: str, text: str) -> None:
        target = root / Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding='utf-8', newline='\n')

    # --- execution --------------------------------------------------------

    def run(self, test_code: str, tree_kind: str) -> dict[str, str]:
        """`{test_name: 'pass'|'fail'|'error'|'skip'}` for one generated module."""
        result = run_pytest_in_dir(
            self.tree(tree_kind), test_code,
            timeout=self.timeout, python=self.python,
        )
        return dict(result.per_test)

    def as_run_pytest(self) -> Callable[[str, str], dict[str, str]]:
        """The `verifier_loop.RunPytest` callable: `(code, "golden"|"stub") -> outcomes`."""
        return self.run

    def cleanup(self) -> None:
        for path in self._trees.values():
            shutil.rmtree(path, ignore_errors=True)
        self._trees.clear()


def make_exec_env(language: str, *, repo: Path, carve: object, work_dir: Path,
                  python: str | None = None, timeout: int = 180) -> PythonExecEnv:
    """The soundness executor for `language`, or a loud refusal.

    Only python has one. go, rust, c, cpp and java grade a whole suite inside a
    docker measure image; wiring that executor is the PLAN §8b follow-up, and
    until it exists asking for it must fail rather than silently produce a
    bundle whose checks were never executed.
    """
    if language != 'python':
        raise NotImplementedError(
            f'no verifier soundness executor for {language!r}: only python is '
            'implemented. The whole-suite languages (go, rust, c, cpp, java) need '
            'the measure-image executor from PLAN §8b, which is a follow-up. '
            'Refusing to emit a bundle whose checks were never executed.'
        )
    return PythonExecEnv(
        repo=Path(repo), carve=carve, work_dir=Path(work_dir),
        python=python or _default_python(), timeout=timeout,
    )


def run_generated_pytest(test_code: str, tree_kind: str, work_dir: Path, *,
                         repo: Path, carve: object, language: str = 'python',
                         python: str | None = None,
                         timeout: int = 180) -> dict[str, str]:
    """One-shot form: materialise `tree_kind` under `work_dir` and run `test_code`.

    Convenient for a single probe; the loop should hold a `PythonExecEnv` so the
    trees are materialised once instead of once per attempt.
    """
    env = make_exec_env(language, repo=repo, carve=carve, work_dir=work_dir,
                        python=python, timeout=timeout)
    return env.run(test_code, tree_kind)
