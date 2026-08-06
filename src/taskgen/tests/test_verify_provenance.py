"""T-clone/verify: an entry that records a pinned source verifies without a pre-clone.

`verify` re-hashes every carved file against a live checkout (oracle-integrity)
before it builds anything, so it needs the upstream tree on disk. Historically
that tree had to be put there by hand under `harbor-tasks/repos-src/`. When
task.toml carries a PINNED `[provenance]` block, verify can now rebuild it
itself.

The boundary being defended: a self-clone happens ONLY when there is nothing
else to use AND the entry names an exact commit. `--repo`, an existing
checkout, a floating entry and a provenance-free entry all keep the old
behaviour exactly -- an unpinned clone would silently re-point oracle-integrity
at whatever upstream looks like today, and report drift as a tampered oracle.

Docker is never involved: everything here stops at repo resolution.
"""

from __future__ import annotations

import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

from taskgen import verify as V

pytestmark = pytest.mark.skipif(
    shutil.which('git') is None, reason='git is not installed on this host'
)


def _git(cwd: Path, *argv: str) -> str:
    proc = subprocess.run(
        ['git', '-c', 'user.name=taskgen', '-c', 'user.email=taskgen@example.invalid',
         '-c', 'commit.gpgsign=false', *argv],
        cwd=str(cwd), capture_output=True, text=True, check=True,
    )
    return proc.stdout.strip()


PROVENANCE = '''
[provenance]
repo_url = "{repo_url}"
commit = "{commit}"
clone_kind = "{clone_kind}"
'''


def _entry(root: Path, *, name: str = 'sample-repo', provenance: str = '') -> Path:
    entry_id = '11111111-1111-5111-8111-111111111111'
    d = root / entry_id
    (d / 'tests').mkdir(parents=True)
    (d / 'task.toml').write_text(
        textwrap.dedent(f"""
        schema_version = "1.4"
        [metadata]
        id = "{entry_id}"
        name = "{name}__mod__no_context"
        condition = "no_context"
        target_file = "src/pkg/mod.py"
        target_func = "mod"
        graded_tests = ["t.py::a"]
        [environment]
        docker_image = "taskgen-{name}:local"
        """).strip()
        + '\n' + provenance
    )
    (d / 'tests' / 'graded.json').write_text(
        '{"expected": 1, "kind": "pytest-allowlist", "selectors": ["t.py::a"]}\n'
    )
    return d


@pytest.fixture()
def origin(tmp_path: Path):
    repo = tmp_path / 'origin' / 'sample-repo'
    repo.mkdir(parents=True)
    _git(repo, 'init', '-q', '-b', 'main')
    (repo / 'file.txt').write_text('one\n', encoding='utf-8')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', 'one')
    return repo, _git(repo, 'rev-parse', 'HEAD')


# ------------------------------------------------------------- load_entry ---


def test_an_entry_without_provenance_records_none(tmp_path):
    spec = V.load_entry(_entry(tmp_path))
    assert spec.repo_url is None
    assert spec.commit == ''
    assert not spec.pinned_source


def test_a_pinned_entry_exposes_its_source_and_commit(tmp_path):
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url='https://github.com/owner/sample-repo', commit='a' * 40,
        clone_kind='pinned',
    )))
    assert spec.repo_url == 'https://github.com/owner/sample-repo'
    assert spec.commit == 'a' * 40
    assert spec.pinned_source


def test_a_floating_entry_is_not_a_pinned_source(tmp_path):
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url='https://github.com/owner/sample-repo', commit='',
        clone_kind='floating',
    )))
    assert spec.repo_url == 'https://github.com/owner/sample-repo'
    assert not spec.pinned_source


# ---------------------------------------------------------- resolving repo --


def test_an_explicit_repo_wins_and_never_clones(tmp_path, origin, monkeypatch):
    repo, commit = origin
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url=str(repo), commit=commit, clone_kind='pinned',
    )))
    monkeypatch.setattr(V, '_self_clone', _must_not_run)
    given = tmp_path / 'given'
    given.mkdir()

    assert V._resolve_repo(spec, given) == given.resolve()


def test_a_missing_checkout_with_no_provenance_still_asks_for_repo(tmp_path):
    spec = V.load_entry(_entry(tmp_path, name='no-such-repo-anywhere'))
    with pytest.raises(V.VerifyError, match='Pass --repo explicitly'):
        V._resolve_repo(spec, None)


def test_a_missing_checkout_with_a_floating_entry_does_not_clone(tmp_path, origin,
                                                                monkeypatch):
    repo, _commit = origin
    spec = V.load_entry(_entry(tmp_path, name='no-such-repo-anywhere',
                               provenance=PROVENANCE.format(
                                   repo_url=str(repo), commit='',
                                   clone_kind='floating')))
    monkeypatch.setattr(V, '_self_clone', _must_not_run)

    with pytest.raises(V.VerifyError, match='Pass --repo explicitly'):
        V._resolve_repo(spec, None)


def test_a_missing_checkout_with_a_pinned_entry_self_clones(tmp_path, origin,
                                                            monkeypatch):
    repo, commit = origin
    cache = tmp_path / 'repos-src'
    cache.mkdir()
    monkeypatch.setattr(V, '_repos_cache_dir', lambda: cache)
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url=str(repo), commit=commit, clone_kind='pinned',
    )))

    resolved = V._resolve_repo(spec, None, echo=lambda *a: None)

    assert resolved == cache / 'sample-repo'
    assert (resolved / 'file.txt').read_text() == 'one\n'


def test_a_failed_self_clone_is_reported_as_a_verify_error(tmp_path, monkeypatch):
    cache = tmp_path / 'repos-src'
    cache.mkdir()
    monkeypatch.setattr(V, '_repos_cache_dir', lambda: cache)
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url=f'file://{tmp_path}/gone', commit='b' * 40, clone_kind='pinned',
    )))

    with pytest.raises(V.VerifyError, match='self-clone failed'):
        V._resolve_repo(spec, None, echo=lambda *a: None)


def test_a_provenance_path_that_is_no_longer_a_git_repo_is_reported_by_the_caller(
    tmp_path, monkeypatch,
):
    """A recorded plain directory has nothing to clone, so it is handed back as-is.

    `verify_entry` then fails on its own `is_dir()` check with the path in the
    message -- which is the honest diagnosis: the source is gone, not private.
    """
    cache = tmp_path / 'repos-src'
    cache.mkdir()
    monkeypatch.setattr(V, '_repos_cache_dir', lambda: cache)
    gone = tmp_path / 'gone'
    spec = V.load_entry(_entry(tmp_path, provenance=PROVENANCE.format(
        repo_url=str(gone), commit='b' * 40, clone_kind='pinned',
    )))

    assert V._resolve_repo(spec, None, echo=lambda *a: None) == gone.resolve()
    assert not gone.exists()


def _must_not_run(*args, **kwargs):
    raise AssertionError('verify must not clone when it has a checkout to use')
