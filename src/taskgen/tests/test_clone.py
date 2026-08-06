"""T-clone: `generate` can obtain the checkout itself, without losing a pin.

Every test here is OFFLINE. The "remote" is a real git repository created in a
tmp dir, reached over `file://` or as a plain local path, so the suite proves
the clone contract without ever touching a network.

The four properties that matter, and why:

    PINNED       the emitted task is locked to exact upstream bytes by two
                 sha256 gates downstream, so a clone that lands anywhere but
                 the requested commit is a task that verifies today and fails
                 tomorrow
    IDEMPOTENT   a second `generate` (and `diff -r`) must not need the network,
                 which is proved here by DELETING the remote before rerunning
    REFUSE       an unpinned clone is refused unless the caller asks for it
    PASSTHROUGH  a plain directory snapshot has no `.git` to clone -- `git
                 clone` fails outright on one -- so it is used directly, which
                 is what every existing taskgen test and the repos-src samples
                 depend on
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from taskgen import clone

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


def _commit(repo: Path, relpath: str, text: str, message: str) -> str:
    path = repo / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    _git(repo, 'add', '-A')
    _git(repo, 'commit', '-q', '-m', message)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture()
def origin(tmp_path: Path):
    """A real two-commit git repo standing in for a public remote."""
    repo = tmp_path / 'origin' / 'sample-repo'
    repo.mkdir(parents=True)
    _git(repo, 'init', '-q', '-b', 'main')
    first = _commit(repo, 'src/pkg/mod.py', 'VALUE = 1\n', 'first')
    second = _commit(repo, 'src/pkg/mod.py', 'VALUE = 2\n', 'second')
    return repo, first, second


@pytest.fixture()
def cache(tmp_path: Path) -> Path:
    return tmp_path / 'repos-src'


# ------------------------------------------------------------- classifying --


def test_derive_name_strips_the_git_suffix_and_any_trailing_slash():
    assert clone.derive_name('https://github.com/owner/name.git') == 'name'
    assert clone.derive_name('https://github.com/owner/name/') == 'name'
    assert clone.derive_name('git@github.com:owner/name.git') == 'name'
    assert clone.derive_name('file:///tmp/a/python-a2a-python') == 'python-a2a-python'
    assert clone.derive_name('../../harbor-tasks/repos-src/go-multigres') == 'go-multigres'


def test_derive_name_refuses_a_source_with_no_basename():
    with pytest.raises(SystemExit) as exc:
        clone.derive_name('https://github.com/')
    assert '--repo-name' in str(exc.value)


def test_a_plain_directory_is_not_a_clone_source(tmp_path: Path):
    snapshot = tmp_path / 'python-a2a-python'
    (snapshot / 'src').mkdir(parents=True)
    assert clone.classify_source(str(snapshot)) == 'plain-dir'
    assert not clone.is_clone_source(str(snapshot))


def test_a_local_git_repo_and_every_url_scheme_are_clone_sources(origin):
    repo, _first, _second = origin
    assert clone.classify_source(str(repo)) == 'git-dir'
    assert clone.is_clone_source(str(repo))
    for url in ('https://h/o/n', 'http://h/o/n', 'ssh://h/o/n', 'git://h/o/n',
                'file:///tmp/n', 'git@h:o/n'):
        assert clone.classify_source(url) == 'url', url


# ---------------------------------------------------------------- pinned ----


def test_a_pinned_clone_lands_at_the_requested_commit(origin, cache):
    repo, first, _second = origin

    out = clone.ensure_repo(
        source=f'file://{repo}', commit=first, cache_dir=cache,
        name='sample-repo', allow_floating=False,
    )

    assert out == cache / 'sample-repo'
    assert clone.head_commit(out) == first
    assert (out / 'src/pkg/mod.py').read_text() == 'VALUE = 1\n'


def test_the_clone_basename_is_the_name_argument_not_the_source_basename(origin, cache):
    """Identity contract: the basename seeds the uuid5 ids, slug and image tag."""
    repo, _first, second = origin

    out = clone.ensure_repo(
        source=str(repo), commit=second, cache_dir=cache,
        name='python-a2a-python', allow_floating=False,
    )

    assert out.name == 'python-a2a-python'
    assert clone.head_commit(out) == second


def test_a_local_git_repo_source_clones_rather_than_being_used_in_place(origin, cache):
    repo, first, _second = origin

    out = clone.ensure_repo(
        source=str(repo), commit=first, cache_dir=cache,
        name='sample-repo', allow_floating=False,
    )

    assert out != repo
    assert clone.head_commit(repo) != first, 'the source must not be checked out'


# ------------------------------------------------------------- idempotent ---


def test_reuse_is_offline_the_remote_can_be_gone(origin, cache):
    repo, _first, second = origin
    src = f'file://{repo}'
    once = clone.ensure_repo(source=src, commit=second, cache_dir=cache,
                             name='sample-repo', allow_floating=False)
    shutil.rmtree(repo)

    twice = clone.ensure_repo(source=src, commit=second, cache_dir=cache,
                              name='sample-repo', allow_floating=False)

    assert twice == once
    assert clone.head_commit(twice) == second


def test_reuse_moves_an_existing_checkout_to_the_requested_commit(origin, cache):
    repo, first, second = origin
    src = f'file://{repo}'
    clone.ensure_repo(source=src, commit=second, cache_dir=cache,
                      name='sample-repo', allow_floating=False)
    shutil.rmtree(repo)

    out = clone.ensure_repo(source=src, commit=first, cache_dir=cache,
                            name='sample-repo', allow_floating=False)

    assert clone.head_commit(out) == first
    assert (out / 'src/pkg/mod.py').read_text() == 'VALUE = 1\n'


def test_a_dirty_reused_checkout_is_refused(origin, cache):
    repo, _first, second = origin
    out = clone.ensure_repo(source=f'file://{repo}', commit=second, cache_dir=cache,
                            name='sample-repo', allow_floating=False)
    (out / 'src/pkg/mod.py').write_text('VALUE = 999\n', encoding='utf-8')

    with pytest.raises(SystemExit) as exc:
        clone.ensure_repo(source=f'file://{repo}', commit=second, cache_dir=cache,
                          name='sample-repo', allow_floating=False)
    assert 'uncommitted' in str(exc.value)


def test_a_non_git_directory_in_the_cache_is_never_clobbered(origin, cache):
    repo, _first, second = origin
    squatter = cache / 'sample-repo'
    squatter.mkdir(parents=True)
    (squatter / 'keep.txt').write_text('mine\n', encoding='utf-8')

    with pytest.raises(SystemExit) as exc:
        clone.ensure_repo(source=f'file://{repo}', commit=second, cache_dir=cache,
                          name='sample-repo', allow_floating=False)

    assert 'refusing to clobber' in str(exc.value)
    assert (squatter / 'keep.txt').read_text() == 'mine\n'


# ----------------------------------------------------------------- refuse ---


def test_a_clone_source_with_no_commit_is_refused(origin, cache):
    repo, _first, _second = origin

    with pytest.raises(SystemExit) as exc:
        clone.ensure_repo(source=f'file://{repo}', commit=None, cache_dir=cache,
                          name='sample-repo', allow_floating=False)

    message = str(exc.value)
    assert '--commit' in message
    assert 'verify' in message
    assert not cache.exists(), 'a refused clone must write nothing'


def test_allow_floating_clones_head_without_a_commit(origin, cache):
    repo, _first, second = origin

    out = clone.ensure_repo(source=f'file://{repo}', commit=None, cache_dir=cache,
                            name='sample-repo', allow_floating=True)

    assert clone.head_commit(out) == second


def test_a_credential_bearing_url_is_refused_before_git_ever_runs(cache):
    with pytest.raises(SystemExit) as exc:
        clone.ensure_repo(
            source='https://token:x@github.com/owner/name.git', commit='0' * 40,
            cache_dir=cache, name='name', allow_floating=False,
        )
    assert 'no token' in str(exc.value)
    assert not cache.exists()


def test_an_unreachable_source_fails_loudly_and_leaves_nothing_behind(tmp_path, cache):
    missing = tmp_path / 'no-such-repo'
    missing.mkdir()
    (missing / '.git').mkdir()

    with pytest.raises(SystemExit) as exc:
        clone.ensure_repo(source=str(missing), commit='0' * 40, cache_dir=cache,
                          name='no-such-repo', allow_floating=False)

    assert 'git clone' in str(exc.value)
    assert not (cache / 'no-such-repo').exists()


# ------------------------------------------------------------ passthrough ---


def test_a_plain_directory_is_used_directly_and_nothing_is_cloned(tmp_path, cache):
    snapshot = tmp_path / 'python-a2a-python'
    (snapshot / 'src').mkdir(parents=True)
    (snapshot / 'src/mod.py').write_text('VALUE = 1\n', encoding='utf-8')

    out = clone.ensure_repo(source=str(snapshot), commit=None, cache_dir=cache,
                            name='python-a2a-python', allow_floating=False)

    assert out == snapshot.resolve()
    assert not cache.exists()


def test_a_plain_directory_ignores_a_commit_instead_of_pretending_to_pin(tmp_path, cache):
    snapshot = tmp_path / 'snapshot'
    snapshot.mkdir()

    out = clone.ensure_repo(source=str(snapshot), commit='0' * 40, cache_dir=cache,
                            name='snapshot', allow_floating=False)

    assert out == snapshot.resolve()
    assert clone.head_commit(out) is None


def test_head_commit_is_none_for_a_tree_with_no_history(tmp_path):
    plain = tmp_path / 'plain'
    plain.mkdir()
    assert clone.head_commit(plain) is None
