"""The `pytest --collect-only` cross-check of the emitted allowlist.

Docker-free by construction: `crosscheck_allowlist` and the parse around it are
pure, and the container is only ever asked for a string.

The case that motivated all of this: `test_base64` is parametrized, so pytest
collects `test_base64[...]` and never the bare id the allowlist records.
harbor_filter matches node ids exactly, deselects it, and the equality floor
then reports "the denominator moved" -- naming neither the id nor the reason.
"""

from __future__ import annotations

import pytest

from taskgen import verify as V

ENCODING = 'tests/test_itsdangerous/test_encoding.py'
BASE64 = f'{ENCODING}::test_base64'
BASE64_BAD = f'{ENCODING}::test_base64_bad'


def _fenced(*ids: str) -> str:
    body = '\n'.join(ids)
    return f'some pytest chatter\n{V.COLLECT_BEGIN}\n{body}\n{V.COLLECT_END}\n2 tests\n'


# ------------------------------------------------------------ crosscheck ----


def test_exact_matches_are_not_a_problem():
    assert V.crosscheck_allowlist([BASE64_BAD], [BASE64_BAD, f'{ENCODING}::test_other']) == []


def test_a_parametrized_base_id_is_reported_with_its_real_expansions():
    problems = V.crosscheck_allowlist(
        [BASE64],
        [f'{BASE64}[value0]', f'{BASE64}[value1]', BASE64_BAD],
    )
    assert len(problems) == 1
    only = problems[0]
    assert BASE64 in only
    assert 'PARAMETRIZED' in only
    assert f'{BASE64}[value0]' in only and f'{BASE64}[value1]' in only
    assert 'regenerate' in only


def test_a_mistyped_id_is_reported_as_absent_not_as_parametrized():
    problems = V.crosscheck_allowlist([f'{ENCODING}::test_bas64'], [BASE64, BASE64_BAD])
    assert len(problems) == 1
    assert 'collects NO node' in problems[0]
    assert 'PARAMETRIZED' not in problems[0]


def test_every_offending_id_is_named_once():
    problems = V.crosscheck_allowlist(
        [BASE64, BASE64_BAD, f'{ENCODING}::test_gone'],
        [f'{BASE64}[value0]', BASE64_BAD],
    )
    assert len(problems) == 2
    assert any('PARAMETRIZED' in p for p in problems)
    assert any('test_gone' in p for p in problems)


def test_an_expansion_never_satisfies_a_different_base_id():
    """`test_base64[x]` must not be read as covering `test_base64_bad`."""
    assert V.crosscheck_allowlist([BASE64_BAD], [f'{BASE64}[value0]'])


def test_an_empty_allowlist_has_nothing_to_report():
    assert V.crosscheck_allowlist([], [BASE64]) == []


# ---------------------------------------------------------------- parsing ---


def test_collected_ids_are_read_from_between_the_fences():
    assert V.parse_collected_ids(_fenced(BASE64_BAD, f'{BASE64}[value0]')) == [
        BASE64_BAD, f'{BASE64}[value0]',
    ]


def test_chatter_outside_the_fences_is_not_a_node_id():
    noisy = f'{ENCODING}::leaked\n' + _fenced(BASE64_BAD)
    assert V.parse_collected_ids(noisy) == [BASE64_BAD]


def test_a_missing_fence_fails_closed():
    """No fence means collection never ran; an unproven allowlist is not a clean one."""
    with pytest.raises(V.VerifyError, match='no output fence'):
        V.parse_collected_ids('bash: uv: command not found\n')


# ----------------------------------------------------------------- wiring ---


def test_only_the_pytest_allowlist_kind_is_cross_checked():
    assert V.wants_allowlist_crosscheck('pytest-allowlist', [BASE64])
    assert not V.wants_allowlist_crosscheck('whole-suite', [])
    assert not V.wants_allowlist_crosscheck('go-run', ['./pkg -run TestX'])
    assert not V.wants_allowlist_crosscheck('pytest-allowlist', [])


def test_the_probed_files_are_the_allowlist_s_own():
    assert V.allowlist_test_files([BASE64, BASE64_BAD, 'tests/test_signer.py::test_a']) == (
        ENCODING, 'tests/test_signer.py',
    )


def test_the_collect_script_runs_without_the_filter_it_is_checking():
    script = V.collect_only_script((ENCODING,))
    assert '--collect-only' in script
    assert 'harbor_filter' not in script
    assert ENCODING in script
    assert V.COLLECT_BEGIN in script and V.COLLECT_END in script


def test_the_entry_cross_check_reads_the_image_and_reports_the_expansions(tmp_path):
    entry = tmp_path / 'entry'
    (entry / 'tests').mkdir(parents=True)
    (entry / 'tests' / 'allowlist.txt').write_text(f'{BASE64}\n{BASE64_BAD}\n')
    spec = V.EntrySpec(
        entry_dir=entry, entry_id='e', slug='s', condition='no_context',
        image='taskgen-x:local', expected=2, target_file='f.py', target_func='g',
        graded_kind='pytest-allowlist',
    )

    class Runner:
        def run(self, image, script, quiet=False, mounts=(), network=None):
            assert network == 'none', 'collection is offline; it must not get a network'
            return _fenced(f'{BASE64}[value0]', BASE64_BAD)

    result = V.crosscheck_entry_allowlist(spec, Runner(), echo=lambda *_: None)
    assert not result.ok
    assert len(result.reasons) == 1
    assert f'{BASE64}[value0]' in result.reasons[0]


def test_a_whole_suite_entry_never_reaches_the_probe(tmp_path):
    entry = tmp_path / 'entry'
    (entry / 'tests').mkdir(parents=True)
    spec = V.EntrySpec(
        entry_dir=entry, entry_id='e', slug='s', condition='no_context',
        image='taskgen-x:local', expected=92, target_file='-', target_func='-',
        lang='rust', graded_kind='whole-suite',
    )

    class Runner:
        def run(self, *a, **k):
            raise AssertionError('a whole-suite language must not be probed with pytest')

    assert V.crosscheck_entry_allowlist(spec, Runner(), echo=lambda *_: None).ok
