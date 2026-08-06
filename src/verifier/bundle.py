"""The bundle driver: author, VALIDATE, then freeze — or write nothing at all.

This is taskgen's own driver, not a port of the upstream `orchestrate.build_bundle` /
`build_inputs.build_and_freeze_from_uuid`: those two are welded to a uuid dir, a
git repo and a trajectory reader, none of which exist here. What is reused is
every generator and both soundness loops, unchanged.

Order, and why:

    generate_truth        TRUTH.md, with its verbatim-golden leak-guard ACTIVE —
                          a draft that reproduces the answer is rejected, not
                          patched up (truth.leak_guard)
    sound_pytest_loop     authors a pytest module, EXECUTES it on the golden and
                          the stub tree, keeps only pass-golden ∧ fail-stub
    generate_predicates   decidable checks, with vacuous ones pruned
    sound_rubric_loop     authors criteria, JUDGES them against golden and stub
                          code, keeps only the discriminating ones
    coverage / manifest   100% deterministic, derived from the taxonomy and the
                          golden diff

ABORT-SAFE. Nothing is written until every gate has passed. If fewer than
`min_tests` sound tests or `min_criteria` sound criteria survive, the function
writes NO file, logs the reason and returns False — the task still ships, just
without a bundle (PLAN D5/D9: additive and non-blocking).

NON-BLOCKING IS NOT SILENT-ON-ANY-FAILURE. A model/proxy failure is NOT an
abort: `client.complete` raising propagates out of here so the caller can fail
loud (PLAN §V4). Only the soundness floor produces a quiet, logged skip.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .generators import layout
from .generators._support import VerdictLike
from .generators.coverage import emit_coverage
from .generators.manifest import required_target_files, write_manifest
from .generators.model_client import ModelClient
from .generators.predicates import generate_predicates
from .generators.rubric import Criterion, Rubric
from .generators.truth import TruthInputs, generate_truth
from .generators.verifier_loop import (
    MIN_SOUND_CRITERIA,
    MIN_SOUND_TESTS,
    sound_pytest_loop,
    sound_rubric_loop,
)

__all__ = [
    "CodeJudgeResult",
    "CodeVerdict",
    "build_verifier_bundle",
    "judge_code_factory",
]

#: `verifier_loop.RunPytest`: (generated module, "golden"|"stub") -> per-test outcomes.
ExecRunner = Callable[[str, str], "dict[str, str] | None"]


# --------------------------------------------------------------------------
# The code judge (judge.py is in the un-ported results/eval half)
# --------------------------------------------------------------------------


@dataclass
class CodeVerdict:
    """One criterion's verdict. Mutable attributes, to satisfy `VerdictLike`."""

    criterion_id: str
    passed: bool
    justification: str = ""


@dataclass
class CodeJudgeResult:
    """A judgement over ONE bare solution body, shaped for `_support.validate_criteria`.

    `ok` means the judge actually answered. A model that returned nothing must
    read as INCONCLUSIVE, never as "every criterion failed": `validate_criteria`
    keeps every criterion when the anchors are unavailable, so an unreachable
    judge cannot quietly prune the rubric down to nothing.
    """

    verdicts: list[CodeVerdict] = field(default_factory=list)
    raw_verdicts: int = 0
    response_chars: int = 0

    @property
    def ok(self) -> bool:
        return self.response_chars > 0 and self.raw_verdicts > 0

    def by_id(self) -> dict[str, VerdictLike]:
        return {v.criterion_id: v for v in self.verdicts}


_JUDGE_SYSTEM = (
    "You are a rigorous, skeptical code-review JUDGE. You score a candidate "
    "implementation against a rubric, using the reference behavioral contract below "
    "as the definition of what a correct solution must achieve.\n"
    "For EACH criterion return a binary verdict:\n"
    "- pass ONLY if the code clearly satisfies it, with concrete cited evidence.\n"
    "- fail if it is violated OR if the evidence is insufficient to be confident "
    "(default to fail when uncertain).\n"
    "The contract describes a DESTINATION; a different but valid route that reaches it "
    "must still pass. Never reference internal document names in your justification.\n"
    'Return ONLY a JSON array, one object per criterion: '
    '[{"criterion_id": "...", "verdict": "pass"|"fail", "justification": "<one sentence>"}].'
)


def _code_digest(code: str) -> str:
    return ("## Solution code\nThe following is the solution's implementation code. "
            "Judge ONLY the code-and-behavior criteria against it.\n\n"
            f"```\n{code[:16000]}\n```")


def _extract_json_array(text: str) -> list:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def judge_code_factory(truth_md: str, client: ModelClient, *, max_tokens: int = 16000
                       ) -> Callable[[Rubric, str], CodeJudgeResult]:
    """The `verifier_loop.JudgeCode` callable: `(rubric, code) -> CodeJudgeResult`."""

    def judge_code(rubric: Rubric, code: str) -> CodeJudgeResult:
        crit_lines = "\n".join(f"- {c.id}: {c.text}" for c in rubric.criteria)
        user = (f"# Reference behavioral contract\n\n{truth_md}\n\n"
                f"# Rubric criteria to score\n{crit_lines}\n\n"
                f"# Candidate implementation\n{_code_digest(code)}\n\n"
                "Score every criterion now. Return the JSON array.")
        text = client.complete(_JUDGE_SYSTEM, user, max_tokens=max_tokens).text or ""
        raw = {
            str(d["criterion_id"]): d
            for d in _extract_json_array(text)
            if isinstance(d, dict) and d.get("criterion_id")
        }
        verdicts = []
        for c in rubric.criteria:
            d = raw.get(c.id)
            if d is None:
                # Conservative: a criterion the judge skipped is not a pass.
                verdicts.append(CodeVerdict(
                    c.id, passed=False,
                    justification="not scored by judge (treated as fail)"))
                continue
            verdicts.append(CodeVerdict(
                c.id,
                passed=str(d.get("verdict") or "").strip().lower() == "pass",
                justification=str(d.get("justification") or "")[:500]))
        return CodeJudgeResult(verdicts=verdicts, raw_verdicts=len(raw),
                               response_chars=len(text))

    return judge_code


# --------------------------------------------------------------------------
# The driver
# --------------------------------------------------------------------------


def _anchorable_count(criteria: Sequence[Criterion]) -> int:
    return sum(1 for c in criteria if c.anchorable)


def build_verifier_bundle(
    inputs: TruthInputs,
    client: ModelClient,
    base_dir: str | Path,
    *,
    exec_runner: ExecRunner,
    golden_code: str,
    stub_code: str,
    min_criteria: int = MIN_SOUND_CRITERIA,
    min_tests: int = MIN_SOUND_TESTS,
    echo: Callable[[str], object] = print,
) -> bool:
    """Author + validate + freeze one bundle into `base_dir`. True iff it shipped.

    `golden_code`/`stub_code` are the two bare solution bodies the rubric loop
    judges (`inputs.golden_diff` is the DIFF between them and cannot stand in for
    either side). `inputs.py:solution_code` builds them from the same carve.

    The floors are enforced HERE, on the loops' own results, so
    `--verifier-min-criteria` can tighten the shipped bar above the ported
    `MIN_SOUND_CRITERIA` default without editing the ported loop.
    """
    base_dir = Path(base_dir)

    truth = generate_truth(inputs, client)
    if truth.leak_violations:
        echo(f'verifier: no bundle -- TRUTH.md failed its leak/section guard after '
             f'{truth.regenerations} regeneration(s): {"; ".join(truth.leak_violations)}')
        return False

    pytest_code, pt_loop = sound_pytest_loop(
        truth.text, list(inputs.stub_files), golden_code, exec_runner, client)
    if len(pt_loop.sound) < min_tests:
        echo(f'verifier: no bundle -- only {len(pt_loop.sound)} sound test(s) survived '
             f'pass-golden AND fail-stub, floor is {min_tests} '
             f'({pt_loop.detail or "regenerations exhausted"})')
        return False

    predicates = generate_predicates(truth.text, client)

    judged = sound_rubric_loop(
        truth.text, golden_code, stub_code,
        judge_code_factory(truth.text, client), client)
    if judged is None:
        echo('verifier: no bundle -- the rubric soundness loop produced no rubric at all')
        return False
    rubric, _golden_jr, _stub_jr, rb_loop = judged
    n_sound = _anchorable_count(rubric.criteria)
    if n_sound < min_criteria:
        echo(f'verifier: no bundle -- only {n_sound} sound criteri{"on" if n_sound == 1 else "a"} '
             f'survived golden-pass AND stub-fail, floor is {min_criteria} '
             f'({rb_loop.detail or "regenerations exhausted"})')
        return False

    # Every gate passed: only now does anything touch the disk.
    _freeze(base_dir, truth_text=truth.text, pytest_code=pytest_code,
            predicates=predicates, rubric=rubric, stub_files=list(inputs.stub_files),
            golden_diff=inputs.golden_diff)
    echo(f'verifier: bundle frozen -- {len(pt_loop.sound)} sound test(s), '
         f'{n_sound} sound criteri{"on" if n_sound == 1 else "a"}, '
         f'{len(predicates)} predicate(s)')
    return True


def _dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8', newline='\n')


def _freeze(base_dir: Path, *, truth_text: str, pytest_code: str, predicates,
            rubric: Rubric, stub_files: list[str], golden_diff: str) -> None:
    """Write the six named artifacts under `base_dir`. Called only on success."""
    truth_path = layout.truth_path(base_dir)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(truth_text, encoding='utf-8', newline='\n')

    code_path = layout.pytest_code_path(base_dir)
    code_path.parent.mkdir(parents=True, exist_ok=True)
    code_path.write_text(pytest_code, encoding='utf-8', newline='\n')

    _dump(layout.predicates_path(base_dir), [p.to_dict() for p in predicates])
    _dump(layout.rubric_path(base_dir), rubric.to_dict())

    emit_coverage(base_dir)
    write_manifest(base_dir, stub_files, required=required_target_files(golden_diff))
