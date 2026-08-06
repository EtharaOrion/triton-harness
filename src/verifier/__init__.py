"""The verifier-bundle GENERATION half, ported from the upstream verifier.

Public surface: the artifact authors (TRUTH.md, rubric, pytest, predicates,
coverage, manifest), the soundness loops that keep only pass-golden ∧ fail-stub
checks, the bundle layout, and the `.llm_config` wiring. The upstream results/eval
half is deliberately absent, and litellm stays lazily imported inside
``LiteLLMClient.complete`` so importing this package pulls no LLM dependency.

The taskgen glue sits on top of that port and is deliberately NOT re-exported
here: `inputs`, `exec_env` and `bundle` are imported by module
(`from verifier.bundle import build_verifier_bundle`) so that importing this
package stays as cheap as importing the generators alone.
"""
from __future__ import annotations

from .generators.coverage import coverage_manifest, emit_coverage, registry_violations
from .generators.manifest import (
    load_manifest,
    required_target_files,
    target_files_from_diff,
    write_manifest,
)
from .generators.model_client import (
    LiteLLMClient,
    MockClient,
    ModelClient,
    ModelResponse,
)
from .generators.predicates import (
    Predicate,
    evaluate_predicate,
    generate_predicates,
    prune_vacuous,
)
from .generators.pytest_gen import generate_pytest, prune_tests
from .generators.pytest_runner import PytestResult, run_pytest_in_dir
from .generators.rubric import Criterion, Rubric, generate_rubric
from .generators.truth import TruthDoc, TruthInputs, generate_truth, leak_guard
from .generators.verifier_loop import (
    MIN_SOUND_CRITERIA,
    MIN_SOUND_TESTS,
    LoopResult,
    sound_pytest_loop,
    sound_rubric_loop,
)
from .llm_config import (
    LLMConfigError,
    client_from_config,
    load_llm_config,
    resolve_config,
)

__all__ = [
    "LLMConfigError",
    "LiteLLMClient",
    "LoopResult",
    "MIN_SOUND_CRITERIA",
    "MIN_SOUND_TESTS",
    "MockClient",
    "ModelClient",
    "ModelResponse",
    "Criterion",
    "Predicate",
    "PytestResult",
    "Rubric",
    "TruthDoc",
    "TruthInputs",
    "client_from_config",
    "coverage_manifest",
    "emit_coverage",
    "evaluate_predicate",
    "generate_predicates",
    "generate_pytest",
    "generate_rubric",
    "generate_truth",
    "leak_guard",
    "load_llm_config",
    "load_manifest",
    "prune_tests",
    "prune_vacuous",
    "registry_violations",
    "required_target_files",
    "resolve_config",
    "run_pytest_in_dir",
    "sound_pytest_loop",
    "sound_rubric_loop",
    "target_files_from_diff",
    "write_manifest",
]
