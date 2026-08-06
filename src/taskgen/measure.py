"""Phase 1 of the two-phase build: measure the floor, then throw the image away.

THE CIRCULARITY. A pinned denominator (`EXPECTED`) is the defence against a
shrunken graded set, and it can only be measured against the INTACT tree. But
the script that enforces the floor is the same `test.sh` that would measure it,
and the only image that can run the intact suite contains the answer by
construction. A floor cannot be enforced by the run that discovers it.

THE RESOLUTION (plan A2).

    phase 1  MEASURE   an intact `repoctx`, a floor-FREE test.sh that only
                       counts, and `graded.lock.json` as the output
    phase 2  GRADED    the shipped image, built from the CARVED `repoctx`,
                       whose test.sh CONSUMES the lock and enforces the floor

`generate` then consumes the lock offline and never re-measures, so
regeneration stays byte-identical.

THE MEASURE IMAGE IS NEVER-SHIP. It is tagged `harbor-<repo>-measure:local`,
never `docker save`d, never entered into the dry-run matrix, and deleted in a
`finally` block. It contains the intact tree by construction: an escaped measure
image is not a partial leak, it is the whole answer. `assert_never_shipped()`
exists so that a future caller who wires it into a save path fails loudly.

DETERMINISM. The lock carries no timestamp and no host path -- only content:
the repo tree digest, the intact image digest, the measured count, the sorted
selectors and the fingerprints. Two hosts measuring the same repo against the
same intact image must produce byte-identical locks, or `diff -r` stops being a
regeneration proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from .depplan import (
    DepPlan,
    dep_plan_digest,
    env_lock_key,
    env_lock_key_from_canonical_json,
    to_canonical_json,
)
from .langs import base as langs_base

__all__ = [
    'LOCK_ENV_BLOCK',
    'LOCK_FILENAME',
    'MeasureError',
    'MeasureRequest',
    'assert_never_shipped',
    'check_env_lock_key',
    'check_provenance',
    'dep_plan_from_lock',
    'graded_set_from_lock',
    'is_measure_image',
    'load_lock',
    'measure',
    'measure_image_tag',
    'parse_measure_json',
    'render_lock',
    'repo_tree_sha256',
    'scrub_stderr',
    'split_measure_output',
    'write_lock',
]

LOCK_FILENAME = 'graded.lock.json'
LOCK_SCHEMA = 1

#: The ONE key a resolved `DepPlan` adds to the lock, and only when a plan was
#: actually resolved. A lock measured without one carries no `env` block at all
#: -- not an empty one -- because the lock is a regeneration proof compared byte
#: for byte, and an empty block would rewrite every lock ever emitted.
LOCK_ENV_BLOCK = 'env'
TOOL_VERSION = 'taskgen/1'

MEASURE_IMAGE_PREFIX = 'harbor-'
MEASURE_IMAGE_SUFFIX = '-measure:local'

#: Where the floor-FREE script lands in the measure image, and what it writes.
MEASURE_SCRIPT_PATH = '/opt/harbor/tests/measure.sh'
MEASURE_LOG_DIR = '/logs/verifier'
MEASURE_JSON_PATH = f'{MEASURE_LOG_DIR}/measure.json'

#: stderr is CAPTURED, not discarded: a build or collect error is the only
#: evidence of WHY the floor could not be measured. It is replayed AFTER a marker
#: rather than interleaved, so the bytes the parser reads are produced exactly as
#: before -- stdout still to /dev/null, `cat` still the sole source of the json.
MEASURE_STDERR_PATH = f'{MEASURE_LOG_DIR}/measure.stderr'
MEASURE_STDERR_MARKER = '---STDERR---'
MEASURE_STDERR_TAIL_BYTES = 4096
MEASURE_RUN_SCRIPT = (
    f'mkdir -p {MEASURE_LOG_DIR}; '
    f'bash {MEASURE_SCRIPT_PATH} >/dev/null 2>{MEASURE_STDERR_PATH}; '
    f'cat {MEASURE_JSON_PATH} 2>/dev/null; '
    f"echo '{MEASURE_STDERR_MARKER}'; "
    f'tail -c {MEASURE_STDERR_TAIL_BYTES} {MEASURE_STDERR_PATH} 2>/dev/null'
)

#: A build log is mostly progress noise; these substrings carry the reason.
STDERR_SIGNAL_MARKERS = (
    'error', 'fatal', 'cannot find', 'undefined reference', 'no such file',
    'not found', 'could not', 'unresolved', 'missing',
)

#: The INTACT tree's mount point: a compiler quoting `/opt/harbor/repo/thing.c:41`
#: names a carved path, so the whole line is dropped rather than surfaced.
STDERR_REDACT_PATHS = ('/opt/harbor/repo',)

#: Mirrors harbor `stage_context.PRUNE_DIRS` / `stage_carved.PRUNE`. `.git` is the
#: load-bearing one: it carries the carved files' full history, so letting it
#: into the digest would make provenance depend on the checkout rather than the
#: content.
PRUNE_DIRS = frozenset({
    '.git', '.hg', '.svn', '__pycache__', 'node_modules', 'target', 'build',
    'Build', 'Bin', '.gradle', '.venv', '.pytest_cache', '.ruff_cache', '.mypy_cache',
})


class MeasureError(RuntimeError):
    """Measurement could not be carried out, or its result cannot be pinned.

    `stderr` carries the scrubbed, bounded tail of the failed run so a caller can
    react to the reason. It is empty for every failure raised before a container
    ever ran, and for every error raised by the pure helpers.
    """

    def __init__(self, *args: object, stderr: str = '') -> None:
        super().__init__(*args)
        self.stderr = stderr


# --------------------------------------------------------------------------
# never-ship
# --------------------------------------------------------------------------


def measure_image_tag(repo_name: str) -> str:
    return f'{MEASURE_IMAGE_PREFIX}{repo_name.lower()}{MEASURE_IMAGE_SUFFIX}'


def is_measure_image(image: str) -> bool:
    return str(image).endswith(MEASURE_IMAGE_SUFFIX)


def assert_never_shipped(image: str) -> None:
    """Refuse any path that would let a measure image escape this module."""
    if is_measure_image(image):
        raise MeasureError(
            f'{image} is a MEASURE image and must never be saved, pushed, shipped or '
            'entered into the dry-run matrix: it is built from the INTACT tree, so '
            'every carved answer is in its layers by construction'
        )


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def _tree_files(repo: Path):
    for path in sorted(repo.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(repo)
        if PRUNE_DIRS.intersection(rel.parts[:-1]):
            continue
        yield rel.as_posix(), path


def repo_tree_sha256(repo) -> str:
    """A content digest of the repo: relpath and bytes, in one total order.

    Deliberately not `git rev-parse`: the provenance must describe the tree the
    measurement actually saw, including uncommitted edits, and must hold for a
    checkout with no `.git` at all.
    """
    repo = Path(repo)
    if not repo.is_dir():
        raise MeasureError(f'repo is not a directory: {repo}')
    hasher = hashlib.sha256()
    for rel, path in _tree_files(repo):
        hasher.update(rel.encode('utf-8'))
        hasher.update(b'\0')
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return hasher.hexdigest()


def check_env_lock_key(
    lock: Mapping, repo_sha256: str, intact_image_digest: str,
) -> list[str]:
    """Reasons a lock's PINNED environment does not key to this repo/image.

    A lock with no pinned plan returns no reasons at all. That is deliberate
    back-compat, not an oversight: every lock measured before a resolver existed
    describes a hardcoded gap that no key was ever taken over, and invalidating
    those would re-measure every repo in the corpus to learn nothing.

    When a plan IS pinned the key is confirmatory, since `check_provenance`
    already compared the tree and the image directly. It catches the case those
    two cannot: a lock whose plan was edited after it was written.
    """
    canonical = dep_plan_from_lock(lock)
    if not canonical:
        return []
    env = dict(lock.get(LOCK_ENV_BLOCK, {}))
    expected = env_lock_key_from_canonical_json(
        repo_sha256, intact_image_digest, canonical,
    )
    pinned = str(env.get('env_lock_key', ''))
    if pinned == expected:
        return []
    reason = (
        f'lock pins env_lock_key {pinned[:12] or "?"} but its pinned DepPlan '
        f'keys to {expected[:12]} against this repo and base image -- the '
        'pinned environment is not the one that was measured'
    )
    return [reason]


def check_provenance(lock: Mapping, repo_sha256: str, intact_image_digest: str) -> list[str]:
    """Reasons this lock does not describe the repo/image being generated from."""
    prov = dict(lock.get('provenance', {}))
    reasons = []
    if prov.get('repo_sha256') != repo_sha256:
        reasons.append(
            f'lock was measured against repo {prov.get("repo_sha256", "?")[:12]} but '
            f'generation is running against {repo_sha256[:12]} -- the pinned floor '
            'describes a different tree'
        )
    if prov.get('intact_image_digest') != intact_image_digest:
        reasons.append(
            f'lock was measured against intact image '
            f'{prov.get("intact_image_digest", "?")} but the image now resolves to '
            f'{intact_image_digest} -- the toolchain moved under the floor'
        )
    reasons.extend(check_env_lock_key(lock, repo_sha256, intact_image_digest))
    return reasons


# --------------------------------------------------------------------------
# the measured output
# --------------------------------------------------------------------------


def split_measure_output(payload: str) -> tuple[str, str]:
    """Cut the run's output into the measure.json part and the stderr part.

    Output with no marker -- an older script, a stubbed runner, a container that
    died before the `echo` -- is ALL measure.json part, byte-identical to what
    `parse_measure_json` was handed before capture existed.
    """
    lines = payload.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == MEASURE_STDERR_MARKER:
            return '\n'.join(lines[:index]), '\n'.join(lines[index + 1:])
    return payload, ''


def scrub_stderr(
    text: str,
    *,
    redact_paths: tuple[str, ...] = STDERR_REDACT_PATHS,
    extra_relpaths: tuple[str, ...] = (),
    cap_bytes: int = 2048,
) -> str:
    """A bounded, leak-scrubbed, deterministic view of a failed run's stderr.

    Pure: no I/O, no clock, no host path introduced. The tail is taken in BYTES
    because the cap is about what a caller can be handed, not about lines; the
    first line of a truncated tail is dropped because a half-path would slip past
    the redaction below.
    """
    encoded = text.encode('utf-8', errors='replace')
    truncated = len(encoded) > cap_bytes
    lines = encoded[-cap_bytes:].decode('utf-8', errors='replace').splitlines()
    if truncated and lines:
        lines = lines[1:]

    needles = tuple(n for n in (*redact_paths, *extra_relpaths) if n)
    kept = [ln for ln in lines
            if ln.strip() and not any(needle in ln for needle in needles)]

    signal: list[str] = []
    seen: set[str] = set()
    for line in kept:
        lowered = line.lower()
        if any(marker in lowered for marker in STDERR_SIGNAL_MARKERS) and line not in seen:
            seen.add(line)
            signal.append(line)
    return '\n'.join(signal or kept)


def parse_measure_json(payload: str) -> tuple[int, tuple[str, ...]]:
    """Read the floor-FREE run's output. Every failure is an error, never a zero.

    SCHEMA CONTRACT for Wave 2 plugins: `graded` is the ENUMERATION of the
    denominator, so `len(graded) == tests_total` whenever it is non-empty. A
    language whose selectors are coarser than its tests -- gradle `--tests
    pkg.Class` selects a class but counts methods -- must emit `graded: []` and
    let `tests_total` stand alone, rather than emit a list that silently means
    something else.
    """
    for line in reversed([ln.strip() for ln in payload.splitlines() if ln.strip()]):
        if not (line.startswith('{') and line.endswith('}')):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or 'tests_total' not in obj:
            continue
        total = obj['tests_total']
        if isinstance(total, bool) or not isinstance(total, int):
            raise MeasureError(f'tests_total is not an integer: {line}')
        if total < 1:
            raise MeasureError(
                f'the intact suite measured zero tests: {line}. A floor of zero is '
                'not a floor -- every carve would grade green for free'
            )
        graded = tuple(sorted(str(s) for s in obj.get('graded', []) or ()))
        if graded and len(graded) != total:
            raise MeasureError(
                f'the count and the selector list disagree ({total} vs {len(graded)}): '
                f'{line}. Neither can be trusted as a denominator'
            )
        return total, graded
    raise MeasureError(
        'no measure json in the container output -- '
        f'{MEASURE_JSON_PATH} was missing or unreadable. Output was:\n'
        f'{payload.strip() or "(empty)"}'
    )


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------


def build_lock(
    *,
    lang: str,
    scope: str,
    expected: int,
    graded: tuple[str, ...],
    floor_mode: str,
    fingerprint_sha256: Mapping[str, str],
    repo_sha256: str,
    intact_image: str,
    intact_image_digest: str,
    carve_set_sha256: str = '',
    tool_version: str = TOOL_VERSION,
    repo_url: str | None = None,
    commit: str | None = None,
    dep_plan: DepPlan | None = None,
) -> dict:
    if floor_mode not in langs_base.FLOOR_MODES:
        raise MeasureError(f'unknown floor_mode {floor_mode!r}')
    provenance = {
        'intact_image': intact_image,
        'intact_image_digest': intact_image_digest,
        'repo_sha256': repo_sha256,
    }
    # A lock from a plain checkout carries these keys nowhere: the lock is a
    # regeneration proof compared byte for byte, and `check_provenance` reads
    # neither key, so their absence invalidates nothing.
    if repo_url is not None:
        provenance['repo_url'] = str(repo_url)
        provenance['commit'] = str(commit or '')
    lock = {
        'carve_set_sha256': carve_set_sha256,
        'expected': int(expected),
        'fingerprint_sha256': dict(sorted(dict(fingerprint_sha256).items())),
        'floor_mode': floor_mode,
        'graded': list(graded),
        'lang': lang,
        'provenance': provenance,
        'schema': LOCK_SCHEMA,
        'scope': scope,
        'tool_version': tool_version,
    }
    if dep_plan is not None:
        lock[LOCK_ENV_BLOCK] = _env_block(
            dep_plan, repo_sha256=repo_sha256, intact_image_digest=intact_image_digest,
        )
    return lock


def _env_block(
    dep_plan: DepPlan, *, repo_sha256: str, intact_image_digest: str,
) -> dict[str, str]:
    """The resolved environment, pinned as the exact bytes its key was taken over.

    `graded` above is already the pinned enumeration of the denominator, so the
    ids are NOT repeated here: two copies of one collected set is two answers to
    `what does this grade`, and the second one drifts.
    """
    canonical = to_canonical_json(dep_plan)
    return {
        'dep_plan': canonical,
        'dep_plan_digest': dep_plan_digest(dep_plan),
        'env_lock_key': env_lock_key(repo_sha256, intact_image_digest, dep_plan),
    }


def dep_plan_from_lock(lock: Mapping) -> str:
    """The canonical plan json a lock pins, or `''` for a lock with no plan."""
    env = lock.get(LOCK_ENV_BLOCK)
    return str(env.get('dep_plan', '')) if isinstance(env, Mapping) else ''


def render_lock(lock: Mapping) -> str:
    return json.dumps(lock, indent=2, sort_keys=True) + '\n'


def write_lock(path, lock: Mapping) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lock(lock), encoding='utf-8')
    return path


def load_lock(path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise MeasureError(f'no lock at {path}; run the measure phase first')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as exc:
        raise MeasureError(f'{path} is not valid json: {exc}') from exc


def graded_set_from_lock(lock: Mapping) -> langs_base.GradedSet:
    """Phase 2's entry point: the floor the graded image must enforce."""
    return langs_base.GradedSet(
        expected=int(lock['expected']),
        floor_mode=str(lock['floor_mode']),
        fingerprint_sha256=dict(lock.get('fingerprint_sha256', {})),
        selectors=tuple(lock.get('graded', ())),
    )


# --------------------------------------------------------------------------
# phase 1
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MeasureRequest:
    repo: Path
    repo_name: str
    lang: str
    intact_image: str
    dockerfile: Path
    context: Path
    scope: str = 'function'
    floor_mode: str = 'equality'
    fingerprint_sha256: Mapping[str, str] = field(default_factory=dict)
    carve_set_sha256: str = ''
    tool_version: str = TOOL_VERSION
    repo_url: str | None = None
    commit: str | None = None
    dep_plan: DepPlan | None = None

    @property
    def image(self) -> str:
        return measure_image_tag(self.repo_name)


def _measure_failed(
    exc: BaseException, stderr_tail: str, request: MeasureRequest
) -> MeasureError:
    """Re-raise a failed measurement with a diagnostic a refine loop can read.

    A build that never reached the run has no captured stderr, so the runner's
    own message is scrubbed instead: it is the only evidence there is.
    """
    scrubbed = scrub_stderr(
        stderr_tail or str(exc), extra_relpaths=(str(request.repo),),
    )
    reason = f'{type(exc).__name__}: {exc}' if not isinstance(exc, MeasureError) else str(exc)
    if scrubbed:
        reason = f'{reason}\nmeasure stderr (scrubbed tail):\n{scrubbed}'
    return MeasureError(reason, stderr=scrubbed)


def measure(request: MeasureRequest, plugin, runner, echo=print) -> dict:
    """Build the measure image, count the intact suite, emit the lock, delete the image.

    The image is removed in a `finally`: a failure anywhere between the build
    and the lock must not be able to leave an intact-tree image on the host.
    """
    if plugin.floor_mode != request.floor_mode:
        raise MeasureError(
            f'the {plugin.name!r} plugin declares floor_mode={plugin.floor_mode!r} but '
            f'the request asks for {request.floor_mode!r}; the lock records ONE floor '
            'mode and two sources for it is one too many'
        )
    image = request.image
    if not is_measure_image(image):
        raise MeasureError(
            f'{image} is not tagged as a measure image, so nothing downstream can '
            f'recognise it as never-ship; it must end in {MEASURE_IMAGE_SUFFIX}'
        )

    repo_sha = repo_tree_sha256(request.repo)
    echo(f'== MEASURE (phase 1, floor-FREE) {image} ==')
    echo(f'repo       {request.repo} ({repo_sha[:12]})')
    echo(f'intact     {request.intact_image}')

    stderr_tail = ''
    try:
        intact_digest = runner.image_digest(request.intact_image)
        runner.build_with_contexts(
            image=image,
            dockerfile=request.dockerfile,
            context=request.context,
            contexts={langs_base.REPO_CONTEXT: str(request.repo)},
        )
        payload, stderr_tail = split_measure_output(runner.run(image, MEASURE_RUN_SCRIPT))
        expected, graded = parse_measure_json(payload)
    except Exception as exc:
        raise _measure_failed(exc, stderr_tail, request) from exc
    finally:
        echo(f'deleting the NEVER-SHIP measure image {image}')
        runner.remove_image(image)

    echo(f'measured   expected={expected} ({len(graded)} selector(s))')
    return build_lock(
        lang=request.lang,
        scope=request.scope,
        expected=expected,
        graded=graded,
        floor_mode=request.floor_mode,
        fingerprint_sha256=request.fingerprint_sha256,
        repo_sha256=repo_sha,
        intact_image=request.intact_image,
        intact_image_digest=intact_digest,
        carve_set_sha256=request.carve_set_sha256,
        tool_version=request.tool_version,
        repo_url=request.repo_url,
        commit=request.commit,
        dep_plan=request.dep_plan,
    )
