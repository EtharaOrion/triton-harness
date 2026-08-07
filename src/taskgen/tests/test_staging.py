"""Host-side staging: ship ONLY the carved tree, keep the receipt on the host.

`staging.py` is a WRAPPER: the copy/prune, the tripwire picker and the
content-clean pre-flight all come from harbor's already-proven
`shared/tooling/{stage_context,stage_carved,carve}.py`. These tests assert the
invariants of plan §(e).1/6/8 and the §(f) fail-closed tripwire ladder.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from taskgen.staging import (
    STRONG_TRIPWIRE_CHARS,
    NoLineTripwireError,
    StagedTree,
    StagingError,
    TripwireError,
    stage_carved_tree,
)

KEEP = '''\
"""Surviving module with plenty of long distinctive lines of its own."""

alpha = 1
SEPARATOR_ONE = 'a fairly long surviving line number one here'
beta = 2
SEPARATOR_TWO = 'a fairly long surviving line number two here'
gamma = 3


def surviving_helper(a, b):
    return a + b
'''

CARVED_A = '''\
def compute_weighted_average(values, weights):
    """Weighted mean of two equal-length sequences."""
    total = sum(value * weight for value, weight in zip(values, weights))
    divisor = sum(weights) or 1
    return total / divisor
'''

CARVED_A_STUB = '''\
def compute_weighted_average(values, weights):
    raise NotImplementedError
'''

CARVED_B = '''\
def normalise_connection_identifier(raw_identifier, gateway_prefix):
    return (gateway_prefix << 16) | (raw_identifier & 0xFFFF)
'''

#: Every line is short boilerplate that ALSO occurs in the surviving tree, so
#: the line rung cannot fire and only the content digest is distinctive.
BOILER = 'alpha = 1\nbeta = 2\ngamma = 3\n'

JUNK = {
    '.git/objects/ab/cdef': 'gitobject',
    '.git/config': '[core]\n',
    'pkg/__pycache__/keep.cpython-312.pyc': 'pyc',
    'build/artifact.o': 'obj',
    'node_modules/left-pad/index.js': 'module.exports = 1;\n',
    'target/debug/bin': 'elf',
    '.venv/lib/site.py': 'x = 1\n',
}


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / 'repo'
    _write(r, 'pkg/__init__.py', '')
    _write(r, 'pkg/keep.py', KEEP)
    _write(r, 'pkg/carved_a.py', CARVED_A)
    _write(r, 'pkg/carved_b.py', CARVED_B)
    _write(r, 'pkg/boiler.py', BOILER)
    _write(r, 'README.md', '# repo\n')
    for rel, text in JUNK.items():
        _write(r, rel, text)
    return r


CARVED = ('pkg/boiler.py', 'pkg/carved_a.py', 'pkg/carved_b.py')


@pytest.fixture()
def staged(repo: Path, tmp_path: Path) -> StagedTree:
    return stage_carved_tree(
        repo,
        CARVED,
        {'pkg/carved_a.py': CARVED_A_STUB},
        ('pkg/boiler.py', 'pkg/carved_b.py'),
        tmp_path / 'staging',
    )


# --------------------------------------------------------------------------
# the shipped tree
# --------------------------------------------------------------------------


def _staged_relpaths(staged: StagedTree) -> set[str]:
    return {
        p.relative_to(staged.repo_dir).as_posix()
        for p in staged.repo_dir.rglob('*')
        if p.is_file()
    }


def test_deleted_files_are_absent(staged):
    rels = _staged_relpaths(staged)
    assert 'pkg/carved_b.py' not in rels
    assert 'pkg/boiler.py' not in rels


def test_skeleton_is_overlaid(staged):
    assert (staged.repo_dir / 'pkg/carved_a.py').read_text() == CARVED_A_STUB


def test_survivors_are_untouched(staged):
    assert (staged.repo_dir / 'pkg/keep.py').read_text() == KEEP
    assert (staged.repo_dir / 'README.md').read_text() == '# repo\n'


@pytest.mark.parametrize('junk', sorted(JUNK))
def test_vcs_and_build_detritus_never_reaches_the_context(staged, junk):
    assert not (staged.repo_dir / junk).exists()


@pytest.mark.parametrize(
    'pruned', ['.git', '.venv', '__pycache__', 'node_modules', 'build', 'target']
)
def test_pruned_directory_names_appear_nowhere_in_the_context(staged, pruned):
    assert not [p for p in staged.ctx_dir.rglob('*') if pruned in p.parts]


def test_no_carve_receipt_anywhere_in_the_shipped_context(staged):
    assert list(staged.ctx_dir.rglob('carve_receipt.json')) == []


def test_no_tripwire_file_in_the_shipped_context(staged):
    assert list(staged.ctx_dir.rglob('*tripwire*')) == []


def test_no_oracle_originals_in_the_shipped_context(staged):
    blob = '\n'.join(
        p.read_text(encoding='utf-8', errors='ignore')
        for p in staged.ctx_dir.rglob('*')
        if p.is_file()
    )
    assert 'total = sum(value * weight' not in blob
    assert 'gateway_prefix << 16' not in blob


# --------------------------------------------------------------------------
# the host-only receipt
# --------------------------------------------------------------------------


def test_receipt_exists_host_side_beside_the_tree(staged):
    assert staged.receipt_path.is_file()
    assert staged.ctx_dir not in staged.receipt_path.parents
    on_disk = json.loads(staged.receipt_path.read_text())
    assert on_disk == staged.receipt


def test_receipt_enumerates_the_carve(staged):
    assert tuple(staged.receipt['carved_relpaths']) == CARVED
    assert staged.receipt['removed_count'] == 2
    assert staged.receipt['overlaid_relpaths'] == ['pkg/carved_a.py']
    assert staged.receipt['original_sha256']['pkg/carved_a.py'] == hashlib.sha256(
        CARVED_A.encode('utf-8')
    ).hexdigest()


# --------------------------------------------------------------------------
# tripwires: coverage, fallback ladder, fail-closed
# --------------------------------------------------------------------------


def test_tripwire_coverage_equals_carved_file_count(staged):
    assert len(staged.tripwires) == len(CARVED)
    assert {t.relpath for t in staged.tripwires} == set(CARVED)


def test_distinctive_files_use_the_line_rung(staged):
    by_rel = {t.relpath: t for t in staged.tripwires}
    assert by_rel['pkg/carved_a.py'].kind == 'line'
    assert by_rel['pkg/carved_a.py'].rung == 1
    assert 'total = sum(value * weight' in by_rel['pkg/carved_a.py'].pattern


def test_boilerplate_file_falls_back_down_the_ladder(staged):
    tw = {t.relpath: t for t in staged.tripwires}['pkg/boiler.py']
    assert tw.rung >= 3, f'expected a fallback rung, got {tw!r}'
    assert tw.kind == 'sha256'


def test_tripwires_are_exposed_for_the_leak_scan(staged):
    assert staged.tripwire_path.is_file()
    assert staged.tripwire_path.read_text().strip()
    assert len(staged.tripwire_patterns) == len(
        [t for t in staged.tripwires if t.kind != 'sha256']
    )


def test_no_tripwire_occurs_in_the_staged_tree(staged):
    blob = '\n'.join(
        p.read_text(encoding='utf-8', errors='ignore')
        for p in staged.repo_dir.rglob('*')
        if p.is_file()
    )
    for pattern in staged.tripwire_patterns:
        assert pattern not in blob


def test_undistinguishable_file_fails_closed(repo, tmp_path):
    """A carved file byte-identical to a survivor exhausts all four rungs."""
    _write(repo, 'pkg/dup.py', KEEP)
    with pytest.raises(TripwireError):
        stage_carved_tree(
            repo,
            ('pkg/dup.py',),
            {},
            ('pkg/dup.py',),
            tmp_path / 'staging-dup',
        )


def test_a_leaking_overlay_is_refused(repo, tmp_path):
    """Negative control: the harbor content-clean pre-flight must actually fire."""
    with pytest.raises((TripwireError, StagingError)):
        stage_carved_tree(
            repo,
            ('pkg/carved_a.py',),
            {'pkg/carved_a.py': CARVED_A},  # "stub" still holds the answer
            (),
            tmp_path / 'staging-leak',
        )


# --------------------------------------------------------------------------
# safety
# --------------------------------------------------------------------------


def test_the_pristine_repo_is_never_mutated(repo, staged):
    assert (repo / 'pkg/carved_a.py').read_text() == CARVED_A
    assert (repo / 'pkg/carved_b.py').is_file()
    assert (repo / 'pkg/boiler.py').is_file()
    assert not (repo / 'carve_receipt.json').exists()


def test_staging_into_repos_src_is_refused(repo, tmp_path):
    protected = tmp_path / 'repos-src' / 'staging'
    with pytest.raises(StagingError):
        stage_carved_tree(repo, ('pkg/carved_b.py',), {}, ('pkg/carved_b.py',), protected)


def test_carved_set_must_be_fully_accounted_for(repo, tmp_path):
    with pytest.raises(StagingError):
        stage_carved_tree(
            repo, ('pkg/carved_a.py', 'pkg/carved_b.py'), {}, ('pkg/carved_b.py',),
            tmp_path / 'staging-partial',
        )


def test_a_file_cannot_be_both_deleted_and_overlaid(repo, tmp_path):
    with pytest.raises(StagingError):
        stage_carved_tree(
            repo, ('pkg/carved_a.py',), {'pkg/carved_a.py': CARVED_A_STUB},
            ('pkg/carved_a.py',), tmp_path / 'staging-both',
        )


def test_empty_carve_set_fails_closed(repo, tmp_path):
    with pytest.raises(StagingError):
        stage_carved_tree(repo, (), {}, (), tmp_path / 'staging-empty')


def test_staging_is_rerunnable_and_deterministic(repo, tmp_path):
    def run(name):
        return stage_carved_tree(
            repo, CARVED, {'pkg/carved_a.py': CARVED_A_STUB},
            ('pkg/boiler.py', 'pkg/carved_b.py'), tmp_path / name,
        )

    a, b = run('s1'), run('s2')
    assert [t.pattern for t in a.tripwires] == [t.pattern for t in b.tripwires]
    assert a.staged_relpaths == b.staged_relpaths
    assert a.receipt['original_sha256'] == b.receipt['original_sha256']


# --------------------------------------------------------------------------
# what the LINE-BASED scanner can actually consume (found by real docker)
# --------------------------------------------------------------------------

#: The skeleton stub removes only function BODIES, so a carved file's whole
#: candidate pool can be a single line of language boilerplate. Real docker
#: found `super().__init__(self.message)` -- 30 chars, over harbor's 24 floor,
#: and present in a dozen unrelated site-packages of the base image.
GENERIC_BODY = '''\
class ThingError(Exception):
    def __init__(self, message):
        super().__init__(self.message)
'''
GENERIC_STUB = '''\
class ThingError(Exception):
    def __init__(self, message):
        raise NotImplementedError
'''


@pytest.fixture()
def generic(repo: Path, tmp_path: Path) -> StagedTree:
    """A generic-only file carved ALONGSIDE one that does carry a strong line.

    Carved alone it is refused (`test_a_carve_with_no_grep_pattern_at_all_is_
    refused`), because an all-digest set writes an empty `tripwires.txt` and
    leakscan fails closed. Pairing it keeps the subject of these tests -- what
    the line rung will and will not accept from `pkg/errors.py` -- while the
    carve as a whole stays one a task could actually build.
    """
    _write(repo, 'pkg/errors.py', GENERIC_BODY)
    return stage_carved_tree(
        repo, ('pkg/errors.py', 'pkg/carved_a.py'),
        {'pkg/errors.py': GENERIC_STUB, 'pkg/carved_a.py': CARVED_A_STUB},
        (), tmp_path / 'staging-generic',
    )


def test_a_short_generic_line_is_never_used_as_a_grep_tripwire(generic):
    """A sub-strong line is boilerplate, and boilerplate is not evidence.

    leakscan.sh greps EVERY file of a multi-GiB image, so a tripwire that also
    occurs in urllib3 reports a leak that did not happen. The build then fails
    on a true statement about the wrong file.
    """
    tw = {t.relpath: t for t in generic.tripwires}['pkg/errors.py']
    assert tw.kind != 'line' or len(tw.pattern) >= 40, (
        f'accepted {tw.pattern!r} ({len(tw.pattern)} chars) as a grep tripwire'
    )
    assert 'super().__init__(self.message)' not in generic.tripwire_patterns


def test_every_grep_pattern_clears_the_strong_floor(generic, staged):
    for tree in (generic, staged):
        for pattern in tree.tripwire_patterns:
            assert len(pattern) >= 40, f'{pattern!r} is too short to be distinctive'


def test_a_file_with_only_generic_lines_still_gets_covered(generic):
    """Coverage is never dropped -- it degrades to the content digest."""
    by_rel = {t.relpath: t for t in generic.tripwires}
    assert set(by_rel) == {'pkg/errors.py', 'pkg/carved_a.py'}
    assert by_rel['pkg/errors.py'].kind == 'sha256'
    assert generic.tripwire_digests


def test_no_grep_pattern_spans_more_than_one_line(staged, generic):
    """`grep -F -f` reads the file as ONE PATTERN PER LINE.

    A multi-line pattern is therefore not matched as a unit: it silently
    becomes N independent patterns, each weaker than the one that was
    rejected, while the coverage count still reads as full.
    """
    for tree in (staged, generic):
        for pattern in tree.tripwire_patterns:
            assert '\n' not in pattern, f'{pattern!r} degrades into separate patterns'
        written = tree.tripwire_path.read_text().splitlines()
        assert [ln for ln in written if ln.strip()] == list(tree.tripwire_patterns)


# --------------------------------------------------------------------------
# an all-digest carve can never build, so generate refuses instead
# --------------------------------------------------------------------------


def test_a_carve_with_no_grep_pattern_at_all_is_refused(repo, tmp_path):
    """Zero LINE tripwires == an empty tripwires.txt == a task that cannot build.

    leakscan.sh runs INSIDE the Dockerfile and fails closed on an empty pattern
    file ("LEAKSCAN FATAL: no tripwires", exit 2). Emitting the entries anyway
    is the silent degrade the project forbids, so staging refuses first.
    """
    _write(repo, 'pkg/errors.py', GENERIC_BODY)
    with pytest.raises(TripwireError) as exc:
        stage_carved_tree(
            repo, ('pkg/errors.py',), {'pkg/errors.py': GENERIC_STUB},
            (), tmp_path / 'staging-all-digest', target_label='pkg/errors.py::__init__',
        )
    assert isinstance(exc.value, NoLineTripwireError)
    reason = exc.value.reason
    assert 'no usable tripwire for the carved target pkg/errors.py::__init__' in reason
    assert 'leak-absence could not be proven' in reason
    assert f'shorter than {STRONG_TRIPWIRE_CHARS} characters' in reason
    assert 'also present in a surviving file' in reason
    assert '--carve-scope' in reason


def test_the_refusal_leaves_no_tripwire_file_to_be_built_from(repo, tmp_path):
    """The refusal is EARLIER than the artefact it is refusing to write."""
    _write(repo, 'pkg/errors.py', GENERIC_BODY)
    out = tmp_path / 'staging-no-artefact'
    with pytest.raises(NoLineTripwireError):
        stage_carved_tree(
            repo, ('pkg/errors.py',), {'pkg/errors.py': GENERIC_STUB}, (), out,
        )
    assert not (out / 'trip').exists()
    assert list(out.rglob('tripwires.txt')) == []
    assert list(out.rglob('tripwire-digests.txt')) == []


def test_one_strong_line_anywhere_in_the_carve_is_enough_to_proceed(repo, tmp_path):
    """The bar is the GREP SET, not the per-file rung. Digests stay legal."""
    _write(repo, 'pkg/errors.py', GENERIC_BODY)
    tree = stage_carved_tree(
        repo, ('pkg/errors.py', 'pkg/carved_a.py'),
        {'pkg/errors.py': GENERIC_STUB, 'pkg/carved_a.py': CARVED_A_STUB},
        (), tmp_path / 'staging-mixed',
    )
    assert len(tree.tripwire_patterns) == 1
    assert {t.kind for t in tree.tripwires} == {'sha256', 'line'}
    assert tree.tripwire_path.read_text().strip() == tree.tripwire_patterns[0]


def test_a_normal_carve_is_untouched_by_the_new_refusal(repo, tmp_path):
    """The unchanged path: strong lines present, every file covered, no refusal."""
    tree = stage_carved_tree(
        repo, CARVED, {'pkg/carved_a.py': CARVED_A_STUB},
        ('pkg/boiler.py', 'pkg/carved_b.py'), tmp_path / 'staging-normal',
    )
    assert len(tree.tripwires) == len(CARVED)
    assert tree.tripwire_patterns
    for pattern in tree.tripwire_patterns:
        assert len(pattern) >= STRONG_TRIPWIRE_CHARS
