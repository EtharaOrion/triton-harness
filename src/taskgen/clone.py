"""Obtain the repository checkout `generate` and `verify` carve from.

taskgen historically took ONE input: a directory that already existed on the
host (`--repo`). That is still supported verbatim -- many source trees are plain
snapshots with no `.git` at all, and `git clone` cannot read those. This module
adds the second input: a clone *source* (a URL, or a local directory that really
is a git repository), pinned to a commit.

THE THREE RULES, and why each exists:

1. NO TOKEN, EVER. Only public sources. Credential helpers are switched off for
   every git invocation (`-c credential.helper=`) and terminal prompts are
   disabled (`GIT_TERMINAL_PROMPT=0`), so a private URL fails LOUDLY within
   seconds instead of quietly succeeding on one developer's keychain and
   nowhere else. A URL that embeds credentials is refused outright.

2. PINNED BY DEFAULT. Two independent sha256 gates downstream (verify's
   oracle-integrity, measure's tree provenance) pin an emitted task to exact
   upstream bytes. A clone of a moving branch therefore generates today and
   fails `verify` tomorrow, so a commit is MANDATORY unless the caller passes
   `allow_floating` and accepts a non-reproducible task.

3. IDEMPOTENT AND OFFLINE ON REUSE. If the target directory already exists at
   the requested commit, no git process touches the network. Only the FIRST
   generate needs connectivity; reruns (and `diff -r` determinism checks) do
   not.

BASENAME IDENTITY. The clone lands at `<cache_dir>/<name>`, and `name` is what
`emit.plan_carve` reads as `repo_name` -- it seeds the uuid5 entry ids, the slug
and the image tag. The caller derives it from the source basename
(`derive_name`) so a clone is indistinguishable from a hand-placed checkout.

HOST SIDE ONLY. Nothing here ever enters a docker build context: `.git` is the
carved code's full history, which is why staging prunes it and why
`langs/base.py` bans the pristine-tree contexts outright.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

__all__ = [
    'CLONE_SCHEMES',
    'classify_source',
    'derive_name',
    'ensure_repo',
    'head_commit',
    'is_clone_source',
]

#: Schemes `git clone` can fetch and that we are willing to hand it.
CLONE_SCHEMES = frozenset({'http', 'https', 'ssh', 'git', 'file'})

#: `user@host:path/to/repo.git` -- git's scp-like syntax, which is ssh.
_SCP_LIKE = re.compile(r'^[A-Za-z0-9_.+-]+@[A-Za-z0-9_.\-]+:(?!/)')

#: Substrings git puts on stderr when a source needs credentials we refuse to
#: supply. Matched case-insensitively against the failed command's output.
_AUTH_MARKERS = (
    'authentication failed',
    'could not read username',
    'could not read password',
    'terminal prompts disabled',
    'permission denied',
    'access denied',
    'repository not found',
    '403 forbidden',
    'returned error: 403',
    'invalid username or password',
)

_SOURCE_URL = 'url'
_SOURCE_GIT_DIR = 'git-dir'
_SOURCE_PLAIN_DIR = 'plain-dir'


# --------------------------------------------------------------------------
# classifying the source
# --------------------------------------------------------------------------


def _split_scheme(source: str) -> str:
    """The URL scheme of `source`, or '' when it is a plain filesystem path."""
    if _SCP_LIKE.match(source):
        return 'ssh'
    scheme = urlsplit(source).scheme.lower()
    # A bare Windows drive letter parses as a one-character scheme; a POSIX path
    # never has one at all. Only multi-character known schemes count.
    return scheme if len(scheme) > 1 else ''


def classify_source(source: str) -> str:
    """`'url'`, `'git-dir'` or `'plain-dir'`.

    `plain-dir` is the historical `--repo` input: a directory snapshot with no
    `.git`. `git clone` cannot read one ("repository ... does not exist"), so it
    is used DIRECTLY, exactly as taskgen has always used `--repo`.
    """
    source = str(source)
    if _split_scheme(source) in CLONE_SCHEMES:
        return _SOURCE_URL
    path = Path(source).expanduser()
    if (path / '.git').exists():
        return _SOURCE_GIT_DIR
    return _SOURCE_PLAIN_DIR


def is_clone_source(source: str) -> bool:
    """True when `source` is something `git clone` can actually fetch."""
    return classify_source(source) != _SOURCE_PLAIN_DIR


def derive_name(source: str) -> str:
    """The checkout basename implied by `source`, `.git` suffix stripped.

    This is the identity contract: `emit` uses the directory basename as
    `repo_name`, which seeds the uuid5 entry ids, the slug and the image tag.
    """
    source = str(source).strip()
    if _SCP_LIKE.match(source):
        tail = source.split(':', 1)[1]
    elif _split_scheme(source) in CLONE_SCHEMES:
        tail = urlsplit(source).path
    else:
        tail = source
    tail = tail.rstrip('/')
    name = tail.rsplit('/', 1)[-1]
    if name.endswith('.git'):
        name = name[: -len('.git')]
    if not name or name in ('.', '..'):
        raise SystemExit(
            f'cannot derive a repository name from {source!r}; pass --repo-name '
            'explicitly. The name is the identity: it seeds the entry ids, the '
            'slug and the image tag.'
        )
    return name


# --------------------------------------------------------------------------
# git
# --------------------------------------------------------------------------


def _git(argv: list[str], *, what: str, source: str | None = None) -> str:
    """Run git with credentials and prompts disabled. Returns stdout.

    `-c credential.helper=` is not paranoia: an osxkeychain/manager helper would
    silently supply a token for a private URL, and the task would generate on
    exactly one machine.
    """
    proc = subprocess.run(
        ['git', '-c', 'credential.helper=', *argv],
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if proc.returncode == 0:
        return proc.stdout
    detail = (proc.stderr or proc.stdout).strip()
    if source is not None and _looks_like_auth_failure(detail):
        raise SystemExit(
            f'{what} failed for {source}: private repos unsupported; provide a '
            'local checkout path. taskgen never uses an access token, so a repo '
            f'that needs one cannot be generated from a URL.\ngit said: {detail}'
        )
    raise SystemExit(f'{what} failed (git exit {proc.returncode}): {detail}')


def _git_env() -> dict[str, str]:
    """The host environment, minus every way git could ask for a secret.

    Inherited rather than wiped so a proxy configuration still works; the two
    overrides below are what turn "hangs waiting for a password" and "works only
    on the machine with the keychain entry" into an immediate, explainable
    failure.
    """
    env = dict(os.environ)
    env['GIT_TERMINAL_PROMPT'] = '0'
    env.pop('GIT_ASKPASS', None)
    env.pop('SSH_ASKPASS', None)
    return env


def _looks_like_auth_failure(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _AUTH_MARKERS)


def head_commit(repo: Path) -> str | None:
    """The 40-char HEAD sha of a checkout, or None when it is not a git repo."""
    repo = Path(repo)
    if not (repo / '.git').exists():
        return None
    return _git(['-C', str(repo), 'rev-parse', 'HEAD'], what='git rev-parse').strip()


def _has_commit(repo: Path, commit: str) -> bool:
    """True when `commit` is already an object in this checkout (no network)."""
    proc = subprocess.run(
        ['git', '-C', str(repo), 'cat-file', '-e', f'{commit}^{{commit}}'],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def _assert_clean(repo: Path) -> None:
    dirty = _git(
        ['-C', str(repo), 'status', '--porcelain', '--untracked-files=no'],
        what='git status',
    ).strip()
    if dirty:
        raise SystemExit(
            f'the checkout at {repo} has uncommitted changes, so the commit pin '
            'does not describe the bytes that would be carved:\n' + dirty
        )


def _checkout(repo: Path, commit: str) -> None:
    _git(
        ['-C', str(repo), 'checkout', '--quiet', '--detach', commit],
        what=f'git checkout {commit}',
    )


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------


def ensure_repo(source: str, commit: str | None, cache_dir: Path, name: str,
                allow_floating: bool) -> Path:
    """Return a local checkout of `source` named `name`, at `commit`.

    * plain local directory (no `.git`)   -> returned as-is; `commit` is moot,
      there is no history to pin to. This is the historical `--repo` behaviour.
    * URL, or a local directory that IS a git repo -> cloned to
      `<cache_dir>/<name>` and checked out at `commit`.

    Reuse is idempotent and offline: an existing `<cache_dir>/<name>` already at
    `commit` is returned without running a single network operation.
    """
    source = str(source)
    kind = classify_source(source)
    if kind == _SOURCE_PLAIN_DIR:
        # No history exists, so nothing can be pinned and nothing can drift
        # under us either: the directory IS the provenance.
        return Path(source).expanduser().resolve()

    if commit is None and not allow_floating:
        raise SystemExit(
            f'--repo-url {source} needs --commit: an emitted task is pinned to '
            'exact upstream bytes by two sha256 gates (verify oracle-integrity, '
            'measure tree provenance), so a clone of a moving branch generates '
            'today and fails verify the moment upstream advances. Pass --commit '
            '<sha> for a reproducible task, or --allow-floating to accept one '
            'that is not.'
        )
    if kind == _SOURCE_URL:
        _refuse_credentials(source)

    cache_dir = Path(cache_dir).expanduser().resolve()
    target = cache_dir / name

    if target.exists():
        return _reuse(target, source, commit)

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        _git(['clone', '--quiet', source, str(target)], what='git clone',
             source=source)
        if commit is not None:
            _checkout(target, commit)
    except BaseException:
        # A half-written clone would be "reused" by the next run and carve the
        # wrong bytes. Leave nothing behind.
        shutil.rmtree(target, ignore_errors=True)
        raise
    _assert_clean(target)
    return target


def _reuse(target: Path, source: str, commit: str | None) -> Path:
    if not (target / '.git').exists():
        raise SystemExit(
            f'{target} already exists and is not a git checkout; refusing to '
            'clobber it. Remove it, or point --repos-cache somewhere else.'
        )
    head = head_commit(target)
    if commit is None or head == commit:
        _assert_clean(target)
        return target
    if not _has_commit(target, commit):
        _git(['-C', str(target), 'fetch', '--quiet', 'origin', commit],
             what=f'git fetch {commit}', source=source)
    _checkout(target, commit)
    _assert_clean(target)
    return target


def _refuse_credentials(source: str) -> None:
    parts = urlsplit(source)
    if parts.scheme.lower() in ('http', 'https') and '@' in parts.netloc:
        raise SystemExit(
            'refusing a credential-bearing URL: taskgen clones public sources '
            'with no token, and a secret in --repo-url would be recorded in '
            "task.toml's [provenance] block. Provide a local checkout path for "
            'a private repository.'
        )
