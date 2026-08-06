"""The ported generation half (src/verifier) must be import-clean, offline, and
behave exactly like the upstream generators.

Every LLM step is driven by MockClient and every execution step by an injected
callable, so this module never touches the network, litellm, or Docker.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import verifier
from verifier.generators import (
    coverage,
    deterministic,
    layout,
    manifest,
    model_client,
    predicates,
    pytest_gen,
    pytest_runner,
    rubric,
    truth,
    verifier_loop,
)

SRC = Path(verifier.__file__).resolve().parents[1]

# The exact coverage.json this package emits
# (the upstream coverage.py, json.dumps(..., indent=2)). Pinned so
# the severed deterministic.CHECKS registry can never silently drift from it.
# Differs from the pristine upstream mirror only in one concern TITLE string;
# the concern ids, layers, owners and enforced set are byte-identical to it.
UPSTREAM_COVERAGE_SHA256 = '64c8a7b2b899018c4c09433cca8490c0a949d1f6af7003ae70cb9a7a72fc5177'

TRUTH_MD = '\n'.join(
    f'## {section}\n\nBody for {section.lower()}.\n' for section in truth.TRUTH_SECTIONS
)

GOLDEN_DIFF = """diff --git a/src/pkg/core.py b/src/pkg/core.py
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -1,3 +1,4 @@
-    raise NotImplementedError("STUB")
+    return sorted(item.strip() for item in items if item.strip())
diff --git a/docs/guide.md b/docs/guide.md
--- a/docs/guide.md
+++ b/docs/guide.md
@@ -1,0 +1,1 @@
+Restored documentation paragraph that carries no implementation at all.
diff --git a/tests/test_core.py b/tests/test_core.py
--- a/tests/test_core.py
+++ b/tests/test_core.py
@@ -1,2 +1,2 @@
-    raise NotImplementedError("STUB")
+    assert normalize([' a ']) == ['a']
"""


def _truth_inputs(**kw: Any) -> truth.TruthInputs:
    base: dict[str, Any] = dict(repo='demo', language='python',
                                spec_text='normalize a list', golden_diff=GOLDEN_DIFF,
                                stub_files=['src/pkg/core.py'])
    base.update(kw)
    return truth.TruthInputs(**base)


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #
def test_every_ported_submodule_imports_cleanly():
    mods = [p.stem for p in sorted((SRC / 'verifier' / 'generators').glob('*.py'))
            if p.stem != '__init__']
    out = subprocess.run(
        [sys.executable, '-c',
         'import verifier, verifier.llm_config;'
         + ''.join(f'import verifier.generators.{m};' for m in mods)
         + 'print("IMPORT OK")'],
        capture_output=True, text=True, cwd=str(SRC.parent), timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'IMPORT OK'


def test_importing_the_package_does_not_import_litellm():
    out = subprocess.run(
        [sys.executable, '-c',
         'import sys, verifier; print("litellm" in sys.modules)'],
        capture_output=True, text=True, cwd=str(SRC.parent), timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == 'False'


def test_the_results_eval_half_is_not_ported():
    present = {p.stem for p in (SRC / 'verifier' / 'generators').glob('*.py')}
    eval_half = {'orchestrate', 'judge', 'reexec', 'reexec_runner', 'mutation',
                 'mutation_runner', 'evaluator', 'differential', 'trajectory',
                 'bridges', 'autorun', 'content', 'solution_code', 'sandbox',
                 'rubric_layer', 'predicate_layer', 'rubric_anchor', 'tiers',
                 'digest', 'feedback', 'build_inputs', 'pytest_exec'}
    assert present & eval_half == set()


# --------------------------------------------------------------------------- #
# truth.py
# --------------------------------------------------------------------------- #
def test_generate_truth_accepts_a_complete_leak_free_draft():
    client = model_client.MockClient(responses=[TRUTH_MD])
    doc = truth.generate_truth(_truth_inputs(), client)

    assert doc.ok
    assert doc.regenerations == 0
    assert set(doc.sections) == set(truth.TRUTH_SECTIONS)
    assert len(client.calls) == 1


def test_generate_truth_regenerates_when_a_section_is_missing():
    partial = TRUTH_MD.replace('## Cheat surface', '## Something else')
    client = model_client.MockClient(responses=[partial, TRUTH_MD])
    doc = truth.generate_truth(_truth_inputs(), client)

    assert doc.ok
    assert doc.regenerations == 1
    assert 'fix them' in client.calls[1][1].lower()


def test_leak_guard_catches_verbatim_golden_lines():
    leaked = TRUTH_MD + '\n'.join([
        'return sorted(item.strip() for item in items if item.strip())',
        'assert normalize([\' a \']) == [\'a\']',
        'Restored documentation paragraph that carries no implementation at all.',
    ])
    violations = truth.leak_guard(leaked, GOLDEN_DIFF)

    assert any('verbatim route leak' in v for v in violations)


def test_leak_guard_ignores_short_golden_lines():
    assert truth.leak_guard(TRUTH_MD, '--- a/x\n+++ b/x\n+x = 1\n+y = 2\n+z = 3\n') == []


def test_generate_truth_reports_violations_when_the_bound_is_hit():
    bad = '## Problem\n\nnothing else\n'
    client = model_client.MockClient(responses=[bad] * 3)
    doc = truth.generate_truth(_truth_inputs(), client, max_regen=2)

    assert not doc.ok
    assert doc.regenerations == 2
    assert len(doc.leak_violations) == len(truth.TRUTH_SECTIONS) - 1


# --------------------------------------------------------------------------- #
# rubric.py
# --------------------------------------------------------------------------- #
def _rubric_response(n: int) -> str:
    return json.dumps([{'id': f'ts.c{i}', 'text': f'Does it satisfy sub-goal {i}?',
                        'contract_ref': 'Behavioral contract'} for i in range(n)])


def test_generate_rubric_is_backbone_plus_task_criteria():
    client = model_client.MockClient(responses=[_rubric_response(3)])
    result = rubric.generate_rubric(TRUTH_MD, client)

    assert [c.id for c in result.backbone()] == [c['id'] for c in rubric.BACKBONE_CRITERIA]
    assert [c.id for c in result.task_specific()] == ['ts.c0', 'ts.c1', 'ts.c2']
    assert all(c.contract_ref for c in result.criteria)


def test_generate_rubric_caps_and_dedupes_the_task_layer():
    dupes = json.dumps([{'id': f'ts.d{i}', 'text': 'Same criterion text'}
                        for i in range(4)])
    client = model_client.MockClient(responses=[dupes])
    assert len(rubric.generate_rubric(TRUTH_MD, client).task_specific()) == 1

    many = _rubric_response(rubric.MAX_TASK_CRITERIA + 5)
    client = model_client.MockClient(responses=[many])
    assert len(rubric.generate_rubric(TRUTH_MD, client).task_specific()) == \
        rubric.MAX_TASK_CRITERIA


def test_rubric_criteria_never_name_the_internal_answer_key():
    resp = json.dumps([{'id': 'ts.leak', 'text': 'Does it match TRUTH.md section 2?',
                        'contract_ref': 'TRUTH.md'}])
    client = model_client.MockClient(responses=[resp])
    crit = rubric.generate_rubric(TRUTH_MD, client).task_specific()[0]

    assert 'TRUTH.md' not in crit.text
    assert 'TRUTH.md' not in crit.contract_ref
    assert 'the behavioral contract' in crit.text


# --------------------------------------------------------------------------- #
# pytest_gen.py
# --------------------------------------------------------------------------- #
GENERATED_SUITE = '''```python
import pytest

from pkg.core import normalize


def test_alpha():
    assert normalize([" a "]) == ["a"]


def test_beta():
    assert normalize([]) == []


def test_gamma():
    assert normalize(["b", "a"]) == ["a", "b"]


def test_delta():
    assert normalize([" "]) == []


def test_epsilon():
    assert isinstance(normalize([]), list)


def test_zeta():
    assert normalize(["a"]) == ["A"]
```'''


def test_generate_pytest_extracts_the_fenced_module():
    client = model_client.MockClient(responses=[GENERATED_SUITE])
    code = pytest_gen.generate_pytest(TRUTH_MD, client, stub_files=['src/pkg/core.py'])

    assert code.startswith('import pytest')
    assert '```' not in code
    assert 'from pkg.core import normalize' in code


def test_generate_pytest_feeds_the_import_hint_and_feedback():
    client = model_client.MockClient(responses=[GENERATED_SUITE])
    pytest_gen.generate_pytest(TRUTH_MD, client, stub_files=['src/pkg/core.py'],
                               feedback='2 tests failed on the golden')
    _, user = client.calls[0]

    assert 'src/pkg/core.py' in user
    assert 'from pkg.core import *' in user
    assert 'FIX it' in user


def test_prune_tests_keeps_only_the_named_functions():
    code = pytest_gen.extract_code(GENERATED_SUITE)
    pruned = pytest_gen.prune_tests(code, {'test_alpha', 'test_gamma'})

    assert 'def test_alpha' in pruned and 'def test_gamma' in pruned
    assert 'def test_beta' not in pruned and 'def test_zeta' not in pruned
    assert 'from pkg.core import normalize' in pruned


# --------------------------------------------------------------------------- #
# predicates.py
# --------------------------------------------------------------------------- #
def test_generate_predicates_parses_and_prunes_vacuous_ones():
    resp = json.dumps([
        {'id': 'pt.sorted', 'type': 'symbol_present', 'target': 'sorted',
         'negative_fixture': 'def normalize(items): return items'},
        {'id': 'vacuous', 'type': 'symbol_present', 'target': 'sorted',
         'negative_fixture': 'sorted(x)'},
        {'id': 'pt.bogus', 'type': 'semantic_vibes', 'target': 'x'},
        {'id': 'pt.nofixture', 'type': 'literal_absent', 'target': '42'},
    ])
    client = model_client.MockClient(responses=[resp])
    preds = predicates.generate_predicates(TRUTH_MD, client)

    assert [p.id for p in preds] == ['pt.sorted', 'pt.nofixture']


def test_prune_vacuous_drops_predicates_their_own_fixture_passes():
    keeps = predicates.Predicate(id='pt.k', type='pattern_present', target=r'return\s+sorted',
                                 negative_fixture='return items')
    drops = predicates.Predicate(id='pt.d', type='pattern_present', target=r'return\s+sorted',
                                 negative_fixture='return sorted(items)')

    assert predicates.prune_vacuous([keeps, drops]) == [keeps]


def test_evaluate_predicate_is_deterministic_per_type():
    code = 'import os\nreturn sorted(items)\n'

    assert predicates.evaluate_predicate(
        predicates.Predicate('p', 'symbol_present', 'sorted'), code)[0]
    assert not predicates.evaluate_predicate(
        predicates.Predicate('p', 'literal_absent', 'sorted'), code)[0]
    assert predicates.evaluate_predicate(
        predicates.Predicate('p', 'import_present', 'import os'), code)[0]
    # a malformed generated predicate must never spuriously quarantine
    assert predicates.evaluate_predicate(
        predicates.Predicate('p', 'pattern_present', '('), code)[0]


# --------------------------------------------------------------------------- #
# coverage.py — 100% deterministic, no LLM
# --------------------------------------------------------------------------- #
def test_coverage_manifest_is_byte_identical_to_upstream():
    import hashlib

    blob = json.dumps(coverage.coverage_manifest(), indent=2).encode()

    assert hashlib.sha256(blob).hexdigest() == UPSTREAM_COVERAGE_SHA256


def test_coverage_manifest_is_stable_across_calls():
    assert coverage.coverage_manifest() == coverage.coverage_manifest()


def test_coverage_manifest_covers_every_taxonomy_concern():
    from verifier.generators.taxonomy import TAXONOMY

    m = coverage.coverage_manifest()

    assert m['n_concerns'] == len(TAXONOMY)
    assert m['n_enforced'] + m['n_pending'] == m['n_concerns']
    assert {c['id'] for c in m['concerns'] if c['enforced']} == set(deterministic.CHECKS)


def test_registry_lint_is_clean():
    assert coverage.registry_violations() == []


def test_emit_coverage_writes_under_the_given_base_dir(tmp_path):
    out = coverage.emit_coverage(tmp_path / 'solution' / 'verifier')

    assert out == tmp_path / 'solution' / 'verifier' / 'coverage.json'
    assert json.loads(out.read_text())['taxonomy_version'] == \
        coverage.coverage_manifest()['taxonomy_version']


# --------------------------------------------------------------------------- #
# manifest.py — 100% deterministic, from the golden diff
# --------------------------------------------------------------------------- #
def test_target_files_from_diff_lists_every_touched_path():
    assert manifest.target_files_from_diff(GOLDEN_DIFF) == [
        'docs/guide.md', 'src/pkg/core.py', 'tests/test_core.py']


def test_required_target_files_keeps_only_stubbed_non_frozen_files():
    assert manifest.required_target_files(GOLDEN_DIFF) == ['src/pkg/core.py']


def test_required_target_files_falls_back_to_non_blank_deletions():
    diff = ('--- a/src/pkg/a.py\n+++ b/src/pkg/a.py\n-    pass\n+    return 1\n'
            '--- a/docs/b.md\n+++ b/docs/b.md\n+a restored doc line\n')

    assert manifest.required_target_files(diff) == ['src/pkg/a.py']


def test_manifest_generation_is_deterministic():
    assert manifest.required_target_files(GOLDEN_DIFF) == \
        manifest.required_target_files(GOLDEN_DIFF)


def test_write_and_load_manifest_round_trip_under_a_base_dir(tmp_path):
    base = tmp_path / 'solution' / 'verifier'
    out = manifest.write_manifest(base, manifest.target_files_from_diff(GOLDEN_DIFF),
                                  manifest.required_target_files(GOLDEN_DIFF))

    assert out == base / 'target_manifest.json'
    assert manifest.load_manifest(base) == {
        'stage1': ['docs/guide.md', 'src/pkg/core.py', 'tests/test_core.py'],
        'stage1_required': ['src/pkg/core.py'],
    }
    assert manifest.load_manifest(base / 'pytest') is not None
    assert manifest.load_manifest(tmp_path / 'elsewhere') is None


# --------------------------------------------------------------------------- #
# layout.py — the base_dir port delta
# --------------------------------------------------------------------------- #
def test_layout_roots_the_bundle_filenames_at_any_base_dir(tmp_path):
    base = tmp_path / 'solution' / 'verifier'
    paths = layout.bundle_paths(base)

    assert paths['truth'] == base / 'TRUTH.md'
    assert paths['coverage'] == base / 'coverage.json'
    assert paths['manifest'] == base / 'target_manifest.json'
    assert paths['pytest_code'] == base / 'pytest' / 'test_truth_generated.py'
    assert paths['predicates'] == base / 'pytest' / 'predicates.json'
    assert paths['rubric'] == base / 'rubric' / 'rubric.json'


def test_layout_can_still_derive_the_upstream_uuid_location(tmp_path):
    assert layout.truth_path(layout.default_bundle_dir(tmp_path)) == \
        tmp_path / 'verification' / 'verifiers' / 'TRUTH.md'


# --------------------------------------------------------------------------- #
# verifier_loop.py — soundness with an injected executor
# --------------------------------------------------------------------------- #
SOUND = ('test_alpha', 'test_beta', 'test_gamma', 'test_delta')


def _run_pytest(sound: tuple[str, ...]):
    def run(code: str, tree: str) -> dict[str, str]:
        assert tree in ('golden', 'stub')
        assert 'def test_alpha' in code
        outcomes = {name: ('pass' if tree == 'golden' else 'fail') for name in sound}
        outcomes['test_epsilon'] = 'pass'                              # passes on both
        outcomes['test_zeta'] = 'fail' if tree == 'golden' else 'fail'  # fails on golden
        return outcomes
    return run


def test_min_floors_match_upstream():
    assert verifier_loop.MIN_SOUND_TESTS == 4
    assert verifier_loop.MIN_SOUND_CRITERIA == 6


def test_sound_pytest_loop_keeps_only_pass_golden_and_fail_stub():
    classification = json.dumps([{'id': 'test_zeta', 'cause': 'wrong_verifier',
                                  'why': 'asserts an incorrect expected value'}])
    client = model_client.MockClient(responses=[GENERATED_SUITE, classification])
    code, res = verifier_loop.sound_pytest_loop(
        TRUTH_MD, ['src/pkg/core.py'], 'def normalize(items): ...',
        _run_pytest(SOUND), client)

    assert res.ok
    assert res.sound == sorted(SOUND)
    assert res.dropped_wrong_oracle == ['test_zeta']          # failed on the golden
    assert res.dropped_non_discriminating == ['test_epsilon']  # passed on the stub
    assert res.golden_defects == []
    for name in SOUND:
        assert f'def {name}' in code
    assert 'def test_epsilon' not in code and 'def test_zeta' not in code


def test_sound_pytest_loop_flags_a_golden_defect_without_dropping_it_silently():
    classification = json.dumps([{'id': 'test_zeta', 'cause': 'golden_defect',
                                  'why': 'the golden really violates the contract'}])
    client = model_client.MockClient(responses=[GENERATED_SUITE, classification])
    _, res = verifier_loop.sound_pytest_loop(
        TRUTH_MD, ['src/pkg/core.py'], 'golden', _run_pytest(SOUND), client)

    assert res.golden_defects == ['test_zeta']


def test_sound_pytest_loop_aborts_below_the_min_floor():
    classification = json.dumps([{'id': 'test_zeta', 'cause': 'wrong_verifier'}])
    client = model_client.MockClient(
        responses=[GENERATED_SUITE, classification] * 2)
    code, res = verifier_loop.sound_pytest_loop(
        TRUTH_MD, ['src/pkg/core.py'], 'golden',
        _run_pytest(SOUND[:3]), client, max_regen=1)

    assert not res.ok
    assert res.regenerations == 1
    assert len(res.sound) == 3 < verifier_loop.MIN_SOUND_TESTS
    assert 'could not reach 4 sound tests' in res.detail
    assert 'def test_delta' not in code
    # the failure was fed back into the regeneration prompt
    assert 'non-discriminating' in client.calls[2][1]


@dataclass
class _Verdict:
    criterion_id: str
    passed: bool
    justification: str = ''


@dataclass
class _JudgeResult:
    verdicts: list[_Verdict]

    @property
    def ok(self) -> bool:
        return bool(self.verdicts)

    def by_id(self) -> dict[str, _Verdict]:
        return {v.criterion_id: v for v in self.verdicts}


def _judge_code(passing: bool):
    def judge(scored: rubric.Rubric, code: str) -> _JudgeResult:
        return _JudgeResult([_Verdict(c.id, passing) for c in scored.criteria])
    return judge


def test_sound_rubric_loop_keeps_golden_pass_and_stub_fail_criteria():
    client = model_client.MockClient(responses=[_rubric_response(6)])
    calls: list[str] = []

    def judge(scored: rubric.Rubric, code: str) -> _JudgeResult:
        calls.append(code)
        return _JudgeResult([_Verdict(c.id, code == 'golden') for c in scored.criteria])

    result = verifier_loop.sound_rubric_loop(TRUTH_MD, 'golden', 'stub', judge, client)

    assert result is not None
    kept, _golden, _stub, res = result
    assert res.ok
    assert calls == ['golden', 'stub']
    assert len(kept.criteria) == len(rubric.BACKBONE_CRITERIA) + 6
    # process criteria are unanchorable, so they are kept but never anchor-judged
    assert {c.id for c in kept.criteria if not c.anchorable} == {
        'bb.reasoning_faithfulness', 'bb.stage_legitimacy'}


def test_sound_rubric_loop_aborts_when_the_stub_passes_everything():
    client = model_client.MockClient(responses=[_rubric_response(6)] * 2)
    result = verifier_loop.sound_rubric_loop(
        TRUTH_MD, 'golden', 'stub', _judge_code(True), client, max_regen=1)

    assert result is not None
    kept, _golden, _stub, res = result
    assert not res.ok
    assert 'could not reach 6 sound criteria' in res.detail
    assert [c.id for c in kept.criteria if c.anchorable] == []


# --------------------------------------------------------------------------- #
# pytest_runner.py — the directory-based executor
# --------------------------------------------------------------------------- #
def test_parse_per_test_reads_the_short_summary():
    text = ('PASSED tests/test_x.py::test_ok\n'
            'FAILED tests/test_x.py::test_bad - AssertionError: nope\n'
            'ERROR tests/test_x.py::test_boom\n'
            'SKIPPED tests/test_x.py::test_skip\n')

    assert pytest_runner.parse_per_test(text) == {
        'test_ok': 'pass', 'test_bad': 'fail', 'test_boom': 'error', 'test_skip': 'skip'}


def test_pytest_result_defaults_to_an_empty_per_test_map():
    r = pytest_runner.PytestResult()

    assert r.per_test == {}
    assert r.pass_rate is None
    assert 'output' not in r.to_dict()


def test_run_pytest_in_dir_executes_against_that_directory(tmp_path):
    (tmp_path / 'mymod.py').write_text('def normalize(items):\n'
                                       '    return sorted(i.strip() for i in items)\n')
    code = ('from mymod import normalize\n\n\n'
            'def test_ok():\n    assert normalize([" b ", "a"]) == ["a", "b"]\n\n\n'
            'def test_bad():\n    assert normalize(["a"]) == ["A"]\n')

    res = pytest_runner.run_pytest_in_dir(tmp_path, code, python=sys.executable)

    assert res.status == 'OK'
    assert res.per_test == {'test_ok': 'pass', 'test_bad': 'fail'}
    assert (res.passed, res.failed) == (1, 1)
    assert not list(tmp_path.glob('_verifier_generated_test.py'))


def test_run_pytest_in_dir_reports_a_missing_directory(tmp_path):
    res = pytest_runner.run_pytest_in_dir(tmp_path / 'nope', 'def test_x(): pass')

    assert res.status == 'INFRA'


# --------------------------------------------------------------------------- #
# model_client.py — the .llm_config patch
# --------------------------------------------------------------------------- #
def test_explicit_api_base_bypasses_the_env_gate(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_BASE', raising=False)
    client = model_client.LiteLLMClient(model='anthropic/claude-opus-4-8',
                                        api_base='http://127.0.0.1:8765', api_key='k')

    assert client._bridge_kwargs() == {'api_base': 'http://127.0.0.1:8765', 'api_key': 'k'}


def test_env_driven_bridge_is_unchanged_without_an_explicit_config(monkeypatch):
    monkeypatch.delenv('ANTHROPIC_API_BASE', raising=False)

    assert model_client.LiteLLMClient(model='anthropic/claude-opus-4-8')._bridge_kwargs() == {}

    monkeypatch.setenv('ANTHROPIC_API_BASE', 'http://env:1')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'envkey')

    assert model_client.LiteLLMClient(model='anthropic/claude-opus-4-8')._bridge_kwargs() == {
        'api_base': 'http://env:1', 'api_key': 'envkey'}


def _fake_litellm(recorded: list[dict]) -> types.SimpleNamespace:
    """A stand-in bound by `import litellm` inside complete(), so the real
    dependency is never needed to prove the call shape."""
    def completion(**kwargs: Any) -> types.SimpleNamespace:
        recorded.append(kwargs)
        message = types.SimpleNamespace(content='hi', reasoning_content='why')
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=4))

    return types.SimpleNamespace(completion=completion,
                                 completion_cost=lambda **_: 0.5)


def test_complete_threads_the_configured_endpoint_timeout_and_retries(monkeypatch):
    recorded: list[dict] = []
    monkeypatch.setitem(sys.modules, 'litellm', _fake_litellm(recorded))
    client = model_client.LiteLLMClient(model='anthropic/claude-opus-4-8',
                                        api_base='http://127.0.0.1:8765', api_key='k',
                                        timeout=600.0, num_retries=2)

    resp = client.complete('sys', 'user', max_tokens=64)

    assert resp.text == 'hi'
    assert resp.reasoning == 'why'
    assert (resp.prompt_tokens, resp.completion_tokens, resp.cost) == (3, 4, 0.5)
    assert recorded[0]['api_base'] == 'http://127.0.0.1:8765'
    assert recorded[0]['api_key'] == 'k'
    assert recorded[0]['timeout'] == 600.0
    assert recorded[0]['num_retries'] == 2
    assert recorded[0]['max_tokens'] == 64


def test_complete_omits_timeout_and_retries_when_unset(monkeypatch):
    recorded: list[dict] = []
    monkeypatch.setitem(sys.modules, 'litellm', _fake_litellm(recorded))

    model_client.LiteLLMClient(model='anthropic/claude-opus-4-8').complete('s', 'u')

    assert 'timeout' not in recorded[0]
    assert 'num_retries' not in recorded[0]


def test_mock_client_is_still_a_scripted_test_double():
    client = model_client.MockClient(responses=['one'])

    assert client.complete('s', 'u').text == 'one'
    with pytest.raises(AssertionError):
        client.complete('s', 'u')

    responder = model_client.MockClient(responder=lambda system, user: f'{system}|{user}')
    assert responder.complete('s', 'u').text == 's|u'


# --------------------------------------------------------------------------- #
# llm_config.py
# --------------------------------------------------------------------------- #
CONFIG = {'model': 'anthropic/claude-opus-4-8', 'base_url': 'http://127.0.0.1:8765',
          'api_key': 'sk-ant-oauth-bridge-stub', 'timeout': 600, 'num_retries': 2}


def _write_config(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg), encoding='utf-8')
    return path


def test_load_llm_config_parses_the_dotfile(tmp_path):
    cfg = verifier.load_llm_config(_write_config(tmp_path / '.llm_config', CONFIG))

    assert cfg == CONFIG


def test_load_llm_config_fails_loudly_when_missing(tmp_path):
    with pytest.raises(verifier.LLMConfigError, match='requires an LLM config'):
        verifier.load_llm_config(tmp_path / '.llm_config')


def test_load_llm_config_rejects_malformed_content(tmp_path):
    bad = tmp_path / '.llm_config'
    bad.write_text('{not json', encoding='utf-8')
    with pytest.raises(verifier.LLMConfigError, match='not valid JSON'):
        verifier.load_llm_config(bad)

    bad.write_text('["a list"]', encoding='utf-8')
    with pytest.raises(verifier.LLMConfigError, match='expected a JSON object'):
        verifier.load_llm_config(bad)

    _write_config(bad, {**CONFIG, 'api_key': ''})
    with pytest.raises(verifier.LLMConfigError, match="missing/empty \\['api_key'\\]"):
        verifier.load_llm_config(bad)

    _write_config(bad, {**CONFIG, 'timeout': 'soon'})
    with pytest.raises(verifier.LLMConfigError, match='timeout must be a number'):
        verifier.load_llm_config(bad)


def test_client_from_config_maps_base_url_to_api_base():
    client = verifier.client_from_config(CONFIG)

    assert isinstance(client, model_client.LiteLLMClient)
    assert client.model == CONFIG['model']
    assert client.api_base == CONFIG['base_url']
    assert client.api_key == CONFIG['api_key']
    assert client.timeout == 600.0
    assert client.num_retries == 2
    assert client._bridge_kwargs() == {'api_base': CONFIG['base_url'],
                                       'api_key': CONFIG['api_key']}


def test_client_from_config_tolerates_absent_optional_fields():
    client = verifier.client_from_config({k: CONFIG[k] for k in ('model', 'base_url', 'api_key')})

    assert client.timeout is None
    assert client.num_retries is None


def test_resolve_config_prefers_the_explicit_path(tmp_path, monkeypatch):
    explicit = _write_config(tmp_path / 'explicit.json', {**CONFIG, 'api_key': 'explicit'})
    root = tmp_path / 'root'
    root.mkdir()
    _write_config(root / '.llm_config', {**CONFIG, 'api_key': 'root'})
    monkeypatch.setattr(verifier.llm_config, '_repo_root', lambda: root)

    assert verifier.resolve_config(explicit)['api_key'] == 'explicit'
    assert verifier.resolve_config(None)['api_key'] == 'root'


def test_resolve_config_reads_the_config_inside_a_llm_config_directory(tmp_path, monkeypatch):
    root = tmp_path / 'root'
    (root / '.llm_config').mkdir(parents=True)
    _write_config(root / '.llm_config' / 'claude-code-oauth.json', {**CONFIG, 'api_key': 'in-dir'})
    monkeypatch.setattr(verifier.llm_config, '_repo_root', lambda: root)

    assert verifier.resolve_config(None)['api_key'] == 'in-dir'


def test_resolve_config_directory_without_the_config_file_is_a_clear_error(tmp_path, monkeypatch):
    (tmp_path / '.llm_config').mkdir()
    monkeypatch.setattr(verifier.llm_config, '_repo_root', lambda: tmp_path)

    with pytest.raises(verifier.LLMConfigError, match='requires an LLM config/proxy'):
        verifier.resolve_config(None)


def test_resolve_config_without_any_config_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier.llm_config, '_repo_root', lambda: tmp_path)

    with pytest.raises(verifier.LLMConfigError, match='requires an LLM config/proxy'):
        verifier.resolve_config(None)


def test_repo_root_is_the_directory_holding_pyproject():
    assert (verifier.llm_config._repo_root() / 'pyproject.toml').is_file()
