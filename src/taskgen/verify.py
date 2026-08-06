"""Docker oracle verifier: prove an emitted entry is RED before and GREEN after.

A generated task is only worth shipping if ALL FOUR hold for the real image:

    IMAGE  the shipped image contains no oracle at all
           (harbor gives the agent a shell; a baked-in solution is the answer
           key left in the exam room, and the task then measures nothing)
    LAYER  no LAYER of the image contains the carved answer
           (`docker save` unpacks every layer, and a file overwritten in layer
           N is still readable in layer N-1 -- see the LAYER AUDIT below)
    RED    the carved tree, untouched, scores reward 0.0
           (a task that is already green grades nothing)
    GREEN  solve.sh makes it score reward 1.0 AND binary 1.0
           (a task nobody can solve is unshippable)

THE LAYER AUDIT exists because IMAGE alone is not enough, and this module
shipped believing it was. The probe below asks the FINAL filesystem whether
/opt/harbor/solution is present. An image built by COPYing the intact tree and
then overwriting one file with a stub passes that probe and still hands the
answer to anyone who runs `docker save`. So the audit saves the image, unpacks
every layer blob, and greps every member of every layer for the carved content
and for the intact files' digests. A hit anywhere fails the verdict.

FRACTIONAL REWARD. reward.json now carries five keys for every language
(reward, tests_passed, tests_total, binary, compiled). The oracle bar is
reward == 1.0 AND binary == 1.0; a missing `binary` is a FAILURE, not an
assumed pass. How the denominator is defended depends on the language:

    equality             observed total must equal EXPECTED (harbor's shipped
                         defence against a shrunken denominator)
    pinned-denominator   observed total may be SMALLER -- a Go panic aborts the
                         whole test binary, so only the first selected test
                         emits an event (spike G1) -- but never LARGER, which
                         would be scope growth under the floor

WRAPPED, NOT REIMPLEMENTED. The content scan and the RED/GREEN gate already
exist and are already proven, so they are invoked as subprocesses:

    harbor-tasks/shared/tooling/leakscan.sh          (exit 0/1/2, fails closed)
    harbor-tasks/shared/tooling/three_state_gate.sh  (RED=0 / GREEN=1 legs)

IMAGE and GREEN are only jointly satisfiable because the oracle is BIND-MOUNTED
read-only for the GREEN run instead of being built into the image. The entry
keeps solution/ on disk -- harbor's own solvability gate needs to discover
solution/solve.sh there -- but nothing copies it into a layer.

Nothing here computes a reward. The number is read out of
/logs/verifier/reward.json, which only tests/test.sh ever writes, inside a
container built from the entry's own portable Dockerfile. If the JSON is
missing or unparseable that is an error, never a zero -- a swallowed failure
that reads as RED would let a broken task pass the first half of the gate.

    python -m taskgen.cli verify --entry <entry-dir> [--keep-image]
    python -m taskgen.cli verify --all  <out-dir>    [--keep-image]

`--all` exploits the fact that the context conditions of one target share
ONE image, ONE carve and ONE oracle: the condition only ever changes
instruction.md's pre-loaded context block. So it builds and grades once, then
proves the claim by comparing every image-bound asset across all of them
byte-for-byte. If any of them drifted, the single proof would not cover the
rest and verification fails instead of quietly over-claiming.

HOST-SIDE ORACLE INTEGRITY. solve.sh used to pin the intact sha256 and refuse
to restore anything else. It cannot any more: solve.sh ships inside the entry
and is readable from the container, so that digest would BE the answer. The
check therefore runs here, on the host, comparing `solution/carved/<relpath>`
against the real file in `repos-src` -- which proves the oracle payload is the
true upstream answer rather than a hand-written stand-in that happens to make
the graded tests pass. Without it, GREEN=1.0 says only "these tests pass", not
"the answer restores".

The build uses BuildKit NAMED CONTEXTS, which is what keeps the emitted
Dockerfile free of host paths. WHICH contexts, and which stage, are read from
the entry's own `environment/contexts.json` -- never assumed. The pre-W3
verifier hard-coded `--build-context repo-src=<repo>`, and after the move to a
host-side carve that name would have fed the build the INTACT tree: the exact
leak the carve exists to close.

    DOCKER_BUILDKIT=1 docker build \
        --build-context repoctx=<staged carved tree> \
        --build-context trip=<tripwire lines> \
        --build-context tooling=<harbor leakscan.sh> \
        --build-context entry=<entry> \
        -f <entry>/environment/Dockerfile --target <stage from contexts.json> \
        -t <image from contexts.json> <entry>/environment
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from . import _tooling_path
from . import measure
from .langs import base as langs_base

REPOS_SRC_RELPATH = ('harbor-tasks', 'repos-src')

#: Written by emit.py next to the Dockerfile it describes. Its relpaths are
#: resolved against that directory, so an entry stays relocatable.
CONTEXTS_JSON_RELPATH = ('environment', 'contexts.json')

#: The staged CARVED tree. Its absence is fatal rather than defaulted: a build
#: that silently fell back to the repo checkout would bake in the intact answer.
REQUIRED_BUILD_CONTEXT = 'repoctx'
#: Named context holding tripwires.txt, which the layer audit reuses as patterns.
TRIPWIRE_CONTEXT = 'trip'
TRIPWIRE_FILENAME = 'tripwires.txt'

#: Everything that reaches the graded container -- baked in by the Dockerfile, or
#: mounted for the oracle run. Anything outside this list (instruction.md,
#: task.toml) may differ between conditions without invalidating one shared build.
#: Not all are present in every language (allowlist/harbor_filter are pytest's
#: selector format; go has neither), so absent members are skipped, not fatal.
FIXED_IMAGE_ASSETS = (
    'tests/allowlist.txt',
    'tests/graded.json',
    'tests/harbor_filter.py',
    'tests/test.sh',
    'solution/solve.sh',
    'environment/Dockerfile',
)
#: Directory trees whose every file is baked in, discovered rather than listed.
IMAGE_ASSET_TREES = ('solution/carved', 'environment/carve')

#: Only images this module built are ever deleted. harbor-* took ~an hour and
#: several GB to produce and is shared with the hand-authored dataset.
REMOVABLE_IMAGE_PREFIX = 'taskgen-'

#: Where solve.sh's HARBOR_SOLUTION defaults to. The image does not contain this
#: path -- the GREEN run bind-mounts the entry's solution/ onto it read-only, so
#: the oracle exists only for the seconds the gate needs it and never inside the
#: artifact the agent is given a shell in.
SOLUTION_MOUNTPOINT = '/opt/harbor/solution'

RED_SCRIPT = 'bash /opt/harbor/tests/test.sh >/dev/null 2>&1; cat /logs/verifier/reward.json'
GREEN_SCRIPT = (
    f'bash {SOLUTION_MOUNTPOINT}/solve.sh && bash /opt/harbor/tests/test.sh >/dev/null 2>&1; '
    'cat /logs/verifier/reward.json'
)

CLEAN_MARKER = 'NO_SOLUTION'
LEAK_MARKER = 'SOLUTION_LEAK'
#: Run WITHOUT the mount: proves the shipped image has no oracle of its own.
NO_SOLUTION_SCRIPT = (
    f'test ! -e {SOLUTION_MOUNTPOINT} && echo {CLEAN_MARKER} '
    f'|| (echo {LEAK_MARKER}; ls -R {SOLUTION_MOUNTPOINT}; exit 1)'
)
#: Re-run of a failing state with the output kept, so a FAIL says why.
DIAGNOSTIC_SCRIPTS = {
    'RED': 'bash /opt/harbor/tests/test.sh; echo "--- pytest.log ---"; '
           'cat /logs/verifier/pytest.log 2>&1 | tail -40',
    'GREEN': 'bash /opt/harbor/solution/solve.sh; bash /opt/harbor/tests/test.sh; '
             'echo "--- pytest.log ---"; cat /logs/verifier/pytest.log 2>&1 | tail -40',
}


class VerifyError(RuntimeError):
    """Verification could not be carried out, or its result was not the bar."""


# ------------------------------------------------------------ reward.json ---


@dataclass(frozen=True)
class Reward:
    reward: float
    tests_passed: int
    tests_total: int
    raw: str
    #: None = key absent = FAILURE. Never read as 0.0, never as an assumed pass.
    binary: float | None = None
    compiled: float | None = None


def _optional_float(obj: dict, key: str, raw: str) -> float | None:
    if key not in obj:
        return None
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerifyError(f'{key} is not a number: {raw}')
    return float(value)


def parse_reward_json(payload: str) -> Reward:
    """Pull the reward object out of a container's stdout.

    The GREEN command prints solve.sh's chatter before `cat`, so the object is
    found by scanning lines from the end rather than by assuming the whole of
    stdout is JSON.
    """
    for line in reversed([ln.strip() for ln in payload.splitlines() if ln.strip()]):
        if not (line.startswith('{') and line.endswith('}')):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or 'reward' not in obj:
            continue
        reward = obj['reward']
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise VerifyError(f'reward is not a number: {line}')
        return Reward(
            reward=float(reward),
            tests_passed=int(obj.get('tests_passed', -1)),
            tests_total=int(obj.get('tests_total', -1)),
            raw=line,
            binary=_optional_float(obj, 'binary', line),
            compiled=_optional_float(obj, 'compiled', line),
        )
    raise VerifyError(
        'no reward json in the container output -- /logs/verifier/reward.json was '
        f'missing or unreadable. Output was:\n{payload.strip() or "(empty)"}'
    )


def check_red(r: Reward) -> list[str]:
    """Reasons the untouched carved tree failed to score zero."""
    reasons = []
    if r.reward != 0.0:
        reasons.append(
            f'reward on the untouched stub is {r.reward}, expected 0.0 -- the task '
            'grades green before any work is done and is therefore worthless'
        )
    if r.binary == 1.0:
        reasons.append(
            f'binary is 1.0 while reward is {r.reward} -- the two keys contradict each '
            "other, so the language's reward parser is wrong and neither is trustworthy"
        )
    return reasons


def check_floor(observed_total: int, expected: int, floor_mode: str) -> list[str]:
    """Reasons the graded denominator is not the one that was measured and pinned."""
    if floor_mode not in langs_base.FLOOR_MODES:
        raise VerifyError(
            f'unknown floor_mode {floor_mode!r}; expected one of '
            f'{", ".join(langs_base.FLOOR_MODES)}'
        )
    if floor_mode == 'equality':
        if observed_total != expected:
            return [
                f'tests_total is {observed_total}, expected {expected} -- the graded '
                'suite shrank or grew, and an equality floor is what refuses to reward '
                'a denominator that moved'
            ]
        return []
    if observed_total > expected:
        return [
            f'scope-growth: tests_total is {observed_total} but the pinned denominator '
            f'is {expected} -- the graded set grew under the floor'
        ]
    return []


def check_green(r: Reward, expected: int, floor_mode: str = 'equality') -> list[str]:
    """Reasons the oracle failed to score a full pass.

    The oracle restores the intact original byte for byte, so anything short of
    a total pass means the task is unsolvable and must not ship.
    """
    reasons = []
    if r.reward != 1.0:
        reasons.append(f'oracle reward is {r.reward}, expected 1.0 -- the task is unsolvable')
    if r.binary is None:
        reasons.append(
            'reward.json has no `binary` key, so the all-or-nothing bar is unproven; '
            'an unspoken bar is a failed one'
        )
    elif r.binary != 1.0:
        reasons.append(
            f'binary is {r.binary}, expected 1.0 -- the oracle left at least one graded '
            'test failing'
        )
    if r.tests_passed != expected:
        reasons.append(f'tests_passed is {r.tests_passed}, expected {expected}')
    reasons += check_floor(r.tests_total, expected, floor_mode)
    return reasons


def resolve_floor_mode(lang: str, declared: str | None = None) -> str:
    """The floor the language's plugin declares, or what the entry itself declares.

    Wave 1 ships no plugins, so `declared` (from task.toml) and the harbor
    default are the working path until Wave 2 registers python/go.
    """
    try:
        return langs_base.get(lang).floor_mode
    except langs_base.LangError:
        pass
    if declared:
        if declared not in langs_base.FLOOR_MODES:
            raise VerifyError(
                f'unknown floor_mode {declared!r} in task.toml; expected one of '
                f'{", ".join(langs_base.FLOOR_MODES)}'
            )
        return declared
    return 'equality'


# -------------------------------------------------------- layer archaeology --


@dataclass(frozen=True)
class LayerHit:
    layer: str
    member: str
    kind: str
    detail: str


#: 1 MiB. Layer blobs run to gigabytes, so members are streamed rather than read.
_CHUNK = 1 << 20


def _scan_stream(fh, patterns: list[bytes]) -> tuple[set[bytes], str]:
    """Fixed-string search plus sha256, in one pass, without holding the member.

    The tail overlap is what makes a chunked search correct: a pattern straddling
    a chunk boundary is invisible to a per-chunk search, and that is precisely
    the case an attacker would land in by accident.
    """
    matched: set[bytes] = set()
    hasher = hashlib.sha256()
    overlap = max((len(p) for p in patterns), default=1) - 1
    tail = b''
    while True:
        chunk = fh.read(_CHUNK)
        if not chunk:
            break
        hasher.update(chunk)
        window = tail + chunk
        for pattern in patterns:
            if pattern not in matched and pattern in window:
                matched.add(pattern)
        tail = window[-overlap:] if overlap > 0 else b''
    return matched, hasher.hexdigest()


def _member_hits(layer: str, member: str, matched, digest, wanted_digests) -> list[LayerHit]:
    hits = [
        LayerHit(layer=layer, member=member, kind='tripwire',
                 detail=pattern.decode('utf-8', 'replace'))
        for pattern in sorted(matched)
    ]
    if digest in wanted_digests:
        hits.append(LayerHit(layer=layer, member=member, kind='digest', detail=digest))
    return hits


def scan_extracted_image(root, patterns=(), digests=()) -> tuple[LayerHit, ...]:
    """Every member of every layer of an unpacked `docker save`, content-scanned.

    Both layouts `docker save` produces are covered without special-casing
    either: layer blobs are found by asking whether a file IS a tar, not by
    parsing manifest.json, and loose files (config json, the manifest itself)
    are scanned in place. A leak in any of them is a leak.
    """
    root = Path(root)
    pats = [p.encode('utf-8') if isinstance(p, str) else p for p in patterns if p]
    wanted = {str(d).lower() for d in digests if d}
    if not pats and not wanted:
        return ()

    hits: list[LayerHit] = []
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if tarfile.is_tarfile(path):
            with tarfile.open(path) as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    fh = tf.extractfile(member)
                    if fh is None:
                        continue
                    with fh:
                        matched, digest = _scan_stream(fh, pats)
                    hits += _member_hits(rel, member.name, matched, digest, wanted)
        else:
            with path.open('rb') as fh:
                matched, digest = _scan_stream(fh, pats)
            hits += _member_hits(rel, rel, matched, digest, wanted)
    return tuple(hits)


def check_layers(hits) -> list[str]:
    hits = tuple(hits)
    if not hits:
        return []
    reasons = [
        f'the carved answer is recoverable from {len({h.layer for h in hits})} image '
        f'layer(s) via `docker save` ({len(hits)} hit(s)); overwriting a file in a '
        'later layer does not remove it from the earlier one'
    ]
    for hit in hits[:10]:
        reasons.append(f'  layer {hit.layer} :: {hit.member} [{hit.kind}] {hit.detail[:100]}')
    return reasons


def oracle_signals(entry_dir: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What to hunt for in the layers, taken from the entry's own oracle.

    `solution/carved/` holds the intact originals, so their digests ARE the
    answer bytes -- no derivation and no guessing. Tripwire lines are used too
    when staging.py left a file, since a digest only catches a verbatim copy
    while a line also catches a re-indented or partially-embedded one.
    """
    entry_dir = Path(entry_dir)
    digests = []
    carved_root = entry_dir / 'solution' / 'carved'
    if carved_root.is_dir():
        digests = sorted(
            hashlib.sha256(p.read_bytes()).hexdigest()
            for p in carved_root.rglob('*') if p.is_file()
        )
    candidates = [entry_dir / 'environment' / TRIPWIRE_FILENAME,
                  entry_dir / 'tests' / TRIPWIRE_FILENAME]
    try:
        trip = load_build_plan(entry_dir).context(TRIPWIRE_CONTEXT)
    except VerifyError:
        trip = None
    if trip is not None:
        candidates.insert(0, trip / TRIPWIRE_FILENAME)

    patterns: list[str] = []
    for candidate in candidates:
        if candidate.is_file():
            patterns = [ln for ln in candidate.read_text().splitlines() if ln.strip()]
            break
    return tuple(patterns), tuple(digests)


def audit_image_layers(image, runner, patterns=(), digests=(), tmp_root=None,
                       echo=print) -> CheckResult:
    """`docker save` the image, unpack it, and scan every layer. Never keeps the dir."""
    measure.assert_never_shipped(image)
    pats = tuple(p for p in patterns if p)
    digs = tuple(d for d in digests if d)
    if not pats and not digs:
        return CheckResult('LAYER', [
            'no tripwire lines and no oracle digests were available, so the layer '
            'audit could not run -- an unproven image is not a clean one'
        ])

    workdir = Path(tempfile.mkdtemp(
        prefix='taskgen-save-', dir=str(tmp_root) if tmp_root else None
    ))
    try:
        echo(f'$ docker save {image} | tar -x -C {workdir}')
        runner.save(image, workdir)
        hits = scan_extracted_image(workdir, pats, digs)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return CheckResult('LAYER', check_layers(hits))


# ------------------------------------------- wrapped harbor gate scripts ----


def harbor_tooling_dir() -> Path:
    """The vendored harbor tooling directory, inside this worktree.

    leakscan.sh and three_state_gate.sh are wrapped, not reimplemented. They
    ship in src/harbor_tooling/ so the harness resolves them from its own tree
    rather than from a sibling checkout.
    """
    tooling = _tooling_path.TOOLING
    if not tooling.is_dir():
        raise VerifyError(
            f'vendored harbor tooling not found at {tooling}; leakscan.sh and '
            'three_state_gate.sh are wrapped, not reimplemented'
        )
    return tooling


def harbor_script(name: str) -> Path:
    path = harbor_tooling_dir() / name
    if not path.is_file():
        raise VerifyError(
            f'harbor tooling is missing {name} (looked in {path.parent}); this module '
            'wraps that script rather than reimplementing its scan'
        )
    return path


def leakscan_verdict(status: int, output: str) -> list[str]:
    """Map `leakscan.sh`'s documented exit codes (:15) to reasons. Fails closed."""
    tail = '\n'.join(output.strip().splitlines()[-4:])
    if status == 0:
        if 'LEAKSCAN PASS' in output:
            return []
        return [
            'leakscan.sh exited 0 but never printed LEAKSCAN PASS, so the scan could '
            f'not run and proved nothing. Output:\n{tail or "(empty)"}'
        ]
    if status == 1:
        return [f'leakscan.sh found carved source content inside the image:\n{tail}']
    if status == 2:
        return [
            'leakscan.sh could not run (exit 2 -- no tripwires, or an empty tripwire '
            f'file). It fails closed and so does this.\n{tail}'
        ]
    return [
        f'leakscan.sh exited {status}, which is not one of its documented codes, so '
        f'the scan could not run:\n{tail or "(empty)"}'
    ]


def three_state_verdict(status: int, output: str) -> list[str]:
    """Map `three_state_gate.sh`'s RED/GREEN legs (:53-54, :70-72) to reasons."""
    lines = [ln.strip() for ln in output.splitlines()]
    failed_legs = [ln for ln in lines if ln.startswith('FAIL:')]
    if status == 0 and 'GATE PASS' in output and not failed_legs:
        return []
    reasons = [f'three_state_gate.sh: {leg}' for leg in failed_legs]
    if 'GATE PASS' not in output and 'GATE FAIL' not in output:
        reasons.append(
            f'three_state_gate.sh exited {status} without printing a verdict, so the '
            'RED/GREEN legs never ran to completion'
        )
    if not reasons:
        reasons.append(f'three_state_gate.sh exited {status} after printing GATE PASS')
    return reasons


def _docker_bin(runner) -> str:
    return getattr(runner, 'binary', 'docker')


def run_leakscan(image, tripwire_file, runner, label: str = 'image') -> CheckResult:
    """harbor's own whole-filesystem content scan, run against the built image."""
    script = harbor_script('leakscan.sh')
    argv = [
        _docker_bin(runner), 'run', '--rm', '--network=none',
        '-v', f'{script}:/usr/local/bin/leakscan.sh:ro',
        '-v', f'{tripwire_file}:/tmp/.harbor-tripwires:ro',
        '-e', 'TRIPWIRE_FILE=/tmp/.harbor-tripwires',
        '-e', f'LEAKSCAN_LABEL={label}',
        image, 'bash', '/usr/local/bin/leakscan.sh',
    ]
    status, out = runner.script(argv)
    return CheckResult('LEAKSCAN', leakscan_verdict(status, out))


def run_three_state_gate(image, repo_in_image, oracle_dir, verifier_cmd, runner,
                         host_test=None) -> CheckResult:
    """harbor's RED/GREEN gate, which restores the oracle from OUTSIDE the image."""
    argv = [
        str(harbor_script('three_state_gate.sh')),
        str(image), str(repo_in_image), str(oracle_dir), str(verifier_cmd),
    ]
    if host_test:
        argv.append(str(host_test))
    status, out = runner.script(argv)
    return CheckResult('GATE', three_state_verdict(status, out))


# ----------------------------------------------------------------- entries --


@dataclass(frozen=True)
class EntrySpec:
    entry_dir: Path
    entry_id: str
    slug: str
    condition: str
    image: str
    expected: int
    target_file: str
    target_func: str
    lang: str = 'python'
    carve_scope: str = 'function'
    floor_mode: str | None = None
    repo_url: str | None = None
    commit: str = ''
    clone_kind: str = ''

    @property
    def repo_dirname(self) -> str:
        return self.slug.split('__')[0]

    @property
    def pinned_source(self) -> bool:
        """True when the entry names a clone source AND the commit to take."""
        return bool(self.repo_url) and bool(self.commit) and self.clone_kind == 'pinned'


def load_entry(entry_dir: Path) -> EntrySpec:
    entry_dir = Path(entry_dir).resolve()
    toml_path = entry_dir / 'task.toml'
    if not toml_path.is_file():
        raise VerifyError(f'not a task entry (no task.toml): {entry_dir}')
    with toml_path.open('rb') as fh:
        doc = tomllib.load(fh)

    meta = doc.get('metadata', {})
    env = doc.get('environment', {})
    prov = doc.get('provenance', {})
    image = env.get('docker_image')
    if not image:
        raise VerifyError(f'{toml_path}: [environment].docker_image is missing')

    graded_path = entry_dir / 'tests' / 'graded.json'
    if not graded_path.is_file():
        raise VerifyError(
            f'missing {graded_path}; it is the language-agnostic graded set (selectors '
            'plus the PINNED denominator) and every plugin writes it'
        )
    try:
        graded_doc = json.loads(graded_path.read_text())
    except json.JSONDecodeError as exc:
        raise VerifyError(f'{graded_path}: not valid json ({exc})') from exc

    graded = meta.get('graded_tests', [])
    selectors = list(graded_doc.get('selectors') or ())
    is_whole_suite = graded_doc.get('kind') == 'whole-suite'
    if not is_whole_suite and not graded:
        raise VerifyError(f'{toml_path}: [metadata].graded_tests is missing or empty')
    if selectors != list(graded):
        raise VerifyError(
            f'{entry_dir.name}: tests/graded.json ({len(selectors)} selectors) disagrees '
            f'with [metadata].graded_tests ({len(graded)}); the graded set is ambiguous'
        )

    # pytest selects through an allowlist file; go selects with `go test -run`
    # and ships none. Cross-check it only where the language emits it.
    allowlist = entry_dir / 'tests' / 'allowlist.txt'
    if allowlist.is_file():
        ids = [ln.strip() for ln in allowlist.read_text().splitlines() if ln.strip()]
        if ids != list(graded):
            raise VerifyError(
                f'{entry_dir.name}: tests/allowlist.txt ({len(ids)} ids) disagrees with '
                f'[metadata].graded_tests ({len(graded)} ids); the graded set is ambiguous'
            )

    # The PINNED denominator, not len(selectors): a pinned-denominator language
    # may legitimately observe fewer events than it selected (go spike G1).
    expected = graded_doc.get('expected')
    if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
        raise VerifyError(
            f'{graded_path}: `expected` must be a positive integer, got {expected!r}'
        )

    return EntrySpec(
        entry_dir=entry_dir,
        entry_id=meta.get('id', entry_dir.name),
        slug=meta.get('name', entry_dir.name),
        condition=meta.get('condition', '?'),
        image=image,
        expected=expected,
        target_file=meta.get('target_file', '?'),
        target_func=meta.get('target_func', '?'),
        lang=meta.get('language', 'python'),
        carve_scope=meta.get('carve_scope', 'function'),
        floor_mode=meta.get('floor_mode'),
        repo_url=prov.get('repo_url') or None,
        commit=str(prov.get('commit') or ''),
        clone_kind=str(prov.get('clone_kind') or ''),
    )


def discover_entries(out_dir: Path) -> list[EntrySpec]:
    """Every entry under `out_dir`, sorted by id, all sharing one image."""
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_dir():
        raise VerifyError(f'not a directory: {out_dir}')
    specs = [
        load_entry(child)
        for child in sorted(out_dir.iterdir())
        if child.is_dir() and (child / 'task.toml').is_file()
    ]
    if not specs:
        raise VerifyError(f'no entries (directories with a task.toml) under {out_dir}')
    images = sorted({s.image for s in specs})
    if len(images) > 1:
        raise VerifyError(
            'entries do not share one docker_image, so one build cannot cover them: '
            + ', '.join(images)
        )
    return specs


# ------------------------------------------------------- build contexts -----


@dataclass(frozen=True)
class BuildPlan:
    """How the entry says its own image is built. Nothing here is inferred."""

    image: str
    stage: str
    dockerfile: Path
    context_dir: Path
    #: (name, absolute host path), sorted -- so the command is deterministic.
    contexts: tuple[tuple[str, Path], ...] = ()

    def context(self, name: str) -> Path | None:
        return dict(self.contexts).get(name)


def load_build_plan(entry_dir) -> BuildPlan:
    """Read `environment/contexts.json` and resolve every named context.

    Fails closed on every axis. A missing file, a missing `repoctx`, or a
    context pointing at a directory that is not there are all refusals, because
    the alternative -- guessing -- is how the verifier ends up proving an image
    nobody ships.
    """
    entry_dir = Path(entry_dir).resolve()
    path = entry_dir.joinpath(*CONTEXTS_JSON_RELPATH)
    if not path.is_file():
        raise VerifyError(
            f'{entry_dir.name}: no environment/contexts.json, so which BuildKit named '
            'contexts the Dockerfile expects is unknown. It is written by `generate`; '
            'regenerate the entry rather than guessing the old repo-src/entry shape'
        )
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise VerifyError(f'{path}: not valid json ({exc})') from exc

    declared = doc.get('build_contexts') or {}
    if not isinstance(declared, dict):
        raise VerifyError(f'{path}: build_contexts must be an object')
    if REQUIRED_BUILD_CONTEXT not in declared:
        raise VerifyError(
            f'{path}: no {REQUIRED_BUILD_CONTEXT!r} build context. That context IS the '
            'host-carved tree; without it the build has no repo to grade, and any '
            'fallback would be the intact one'
        )

    base = path.parent
    resolved = []
    for name in sorted(declared):
        target = (base / str(declared[name])).resolve()
        if not target.is_dir():
            raise VerifyError(
                f'{path}: build context {name!r} points at {target}, which does not '
                'exist. The staged contexts live beside the entries and are produced '
                'by `generate`; re-run it if they were cleaned away'
            )
        resolved.append((name, target))

    image = doc.get('image')
    if not image:
        raise VerifyError(f'{path}: no image name')
    return BuildPlan(
        image=str(image),
        stage=str(doc.get('stage') or 'graded'),
        dockerfile=entry_dir / 'environment' / 'Dockerfile',
        context_dir=entry_dir / 'environment',
        contexts=tuple(resolved),
    )


def build_argv(binary: str, plan: BuildPlan) -> list[str]:
    """The exact `docker build` command, assembled from the plan and nothing else."""
    argv = [binary, 'build']
    for name, target in plan.contexts:
        argv += ['--build-context', f'{name}={target}']
    argv += [
        '-f', str(plan.dockerfile),
        '--target', plan.stage,
        '-t', plan.image,
        str(plan.context_dir),
    ]
    return argv


# ------------------------------------------------ host-side oracle integrity --


def check_oracle_integrity(entry_dir, repo, runner=None) -> CheckResult:
    """Prove `solution/carved/` is the upstream answer, byte for byte.

    Computed on the HOST against the real checkout. It deliberately never
    enters a container: the digest of the intact file is the answer, and
    anything the container can read, the agent can read.

    A tampered oracle is not a cosmetic problem. The GREEN leg only shows that
    the graded tests pass after solve.sh runs; if the payload were a
    hand-written function tuned to those tests, GREEN would still be 1.0 while
    the task's stated answer was a fake, and every solver would be graded
    against something that is not the upstream code.
    """
    del runner  # host-only by construction; accepted so callers can pass one
    entry_dir = Path(entry_dir).resolve()
    repo = Path(repo).resolve()
    carved_root = entry_dir / 'solution' / 'carved'
    files = sorted(p for p in carved_root.rglob('*') if p.is_file()) \
        if carved_root.is_dir() else []
    if not files:
        return CheckResult('INTEGRITY', [
            f'no oracle payload under {carved_root} -- solution/carved is empty, so '
            'solve.sh restores nothing and the GREEN leg would be proving that the '
            'carved tree already passes'
        ])

    reasons = []
    for path in files:
        rel = path.relative_to(carved_root).as_posix()
        upstream = repo / rel
        want = hashlib.sha256(path.read_bytes()).hexdigest()
        if not upstream.is_file():
            reasons.append(
                f'{rel}: missing from the source checkout {repo}, so the oracle '
                'claims to restore a file that upstream does not have'
            )
            continue
        got = hashlib.sha256(upstream.read_bytes()).hexdigest()
        if got != want:
            reasons.append(
                f'{rel}: the oracle payload is not the upstream file '
                f'(oracle {want[:12]}, {repo.name} {got[:12]}) -- solution/carved has '
                'been tampered with and is not the true answer'
            )
    return CheckResult('INTEGRITY', reasons)


# ------------------------------------------- one image covers every entry --


@dataclass(frozen=True)
class SharedAssets:
    reference: str
    covered: tuple[str, ...]
    identical: tuple[str, ...] = ()
    comment_only: tuple[str, ...] = ()
    mismatched: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def equivalent(self) -> bool:
        return not self.mismatched and not self.missing


def _strip_comments(blob: bytes) -> bytes:
    """Drop whole-line `#` comments, so a slug in a header is not a difference."""
    keep = [
        ln for ln in blob.splitlines()
        if ln.strip() and not ln.lstrip().startswith(b'#')
    ]
    return b'\n'.join(keep)


def _asset_relpaths(entry_dir: Path) -> list[str]:
    paths = [rel for rel in FIXED_IMAGE_ASSETS if (entry_dir / rel).is_file()]
    for tree in IMAGE_ASSET_TREES:
        root = entry_dir / tree
        if root.is_dir():
            paths += sorted(
                p.relative_to(entry_dir).as_posix()
                for p in root.rglob('*') if p.is_file()
            )
    return sorted(set(paths))


def compare_entries(entry_dirs) -> SharedAssets:
    """Prove one build covers every entry by comparing all image-bound assets.

    Files that differ ONLY in comment lines are reported separately: the
    emitted Dockerfile and test.sh carry the condition in a header comment,
    which cannot change a single byte of the built image.
    """
    dirs = [Path(d).resolve() for d in entry_dirs]
    if len(dirs) < 2:
        return SharedAssets(reference=dirs[0].name if dirs else '', covered=tuple(d.name for d in dirs))

    ref, rest = dirs[0], dirs[1:]
    identical, comment_only, mismatched, missing = [], [], [], []

    for rel in _asset_relpaths(ref):
        ref_bytes = (ref / rel).read_bytes()
        verdicts = set()
        for other in rest:
            candidate = other / rel
            if not candidate.is_file():
                missing.append(f'{other.name}:{rel}')
                verdicts.add('missing')
                continue
            other_bytes = candidate.read_bytes()
            if other_bytes == ref_bytes:
                verdicts.add('identical')
            elif _strip_comments(other_bytes) == _strip_comments(ref_bytes):
                verdicts.add('comment_only')
            else:
                verdicts.add('mismatched')
        if 'mismatched' in verdicts:
            mismatched.append(rel)
        elif 'missing' in verdicts:
            continue
        elif 'comment_only' in verdicts:
            comment_only.append(rel)
        else:
            identical.append(rel)

    return SharedAssets(
        reference=ref.name,
        covered=tuple(d.name for d in dirs),
        identical=tuple(identical),
        comment_only=tuple(comment_only),
        mismatched=tuple(mismatched),
        missing=tuple(missing),
    )


# ------------------------------------------------------------------ docker --


def docker_binary() -> str:
    """The docker CLI, including Docker Desktop's non-PATH install location."""
    found = shutil.which('docker')
    if found:
        return found
    fallback = Path.home() / '.docker' / 'bin' / 'docker'
    if fallback.is_file() and os.access(fallback, os.X_OK):
        return str(fallback)
    raise VerifyError(
        'docker CLI not found on PATH nor at ~/.docker/bin/docker. '
        'Try: export PATH="$HOME/.docker/bin:$PATH"'
    )


class DockerRunner:
    """The only place this module shells out. Streams output as it arrives."""

    def __init__(self, binary: str | None = None, echo=print):
        self._binary = binary or docker_binary()
        self._echo = echo

    def _stream(self, argv: list[str], env=None, quiet: bool = False) -> tuple[int, str]:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        lines = []
        # Popen with stdout=PIPE always gives a readable stdout; the guard is
        # only here because the type checker cannot know that.
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if not quiet:
                self._echo(line.rstrip('\n'))
        return proc.wait(), ''.join(lines)

    def build(self, plan: BuildPlan) -> str:
        env = dict(os.environ, DOCKER_BUILDKIT='1', BUILDKIT_PROGRESS='plain')
        argv = build_argv(self._binary, plan)
        self._echo(f'$ DOCKER_BUILDKIT=1 {" ".join(argv[1:])}')
        status, out = self._stream(argv, env=env)
        if status != 0:
            raise VerifyError(f'docker build failed (exit {status}) for image {plan.image}')
        return out

    def run(self, image, script, quiet=False, mounts=()) -> str:
        binds = []
        for source, target in mounts:
            binds += ['-v', f'{source}:{target}:ro']
        argv = [self._binary, 'run', '--rm', *binds, image, 'bash', '-lc', script]
        self._echo(
            f'$ docker run --rm {" ".join(binds)} {image} bash -lc {script!r}'.replace(
                '  ', ' '
            )
        )
        status, out = self._stream(argv, quiet=quiet)
        # A non-zero exit is NOT fatal on its own: test.sh always exits 0 and
        # the reward file is the source of truth, but `cat` failing is how a
        # missing reward file surfaces, and parse_reward_json turns that into
        # an error rather than a zero.
        if status != 0:
            self._echo(f'  (container exited {status})')
        return out

    def remove_image(self, image) -> str:
        status, out = self._stream([self._binary, 'image', 'rm', image], quiet=True)
        if status != 0:
            self._echo(f'  warning: could not remove image {image} (exit {status})')
        return out

    def image_digest(self, image) -> str:
        """The content-address of a locally-loaded image, for measure provenance."""
        argv = [self._binary, 'inspect', '--format', '{{.Id}}', image]
        status, out = self._stream(argv, quiet=True)
        if status != 0:
            raise VerifyError(
                f'docker inspect --format Id failed for {image} (exit {status}); '
                'is the image loaded locally?'
            )
        return out.strip()

    def build_with_contexts(self, *, image, dockerfile, context, contexts) -> str:
        """Build with a caller-supplied dict of named build contexts (measure.py)."""
        env = dict(os.environ, DOCKER_BUILDKIT='1', BUILDKIT_PROGRESS='plain')
        argv = [self._binary, 'build']
        for name, target in sorted(contexts.items()):
            argv += ['--build-context', f'{name}={target}']
        argv += [
            '-f', str(dockerfile),
            '-t', image,
            str(context),
        ]
        self._echo(f'$ DOCKER_BUILDKIT=1 {" ".join(argv[1:])}')
        status, out = self._stream(argv, env=env)
        if status != 0:
            raise VerifyError(f'docker build failed (exit {status}) for image {image}')
        return out

    @property
    def binary(self) -> str:
        return self._binary

    def script(self, argv, env=None) -> tuple[int, str]:
        """Run a host command -- the wrapped harbor scripts -- and keep its output."""
        self._echo(f'$ {" ".join(str(a) for a in argv)}')
        return self._stream([str(a) for a in argv], env=env)

    def save(self, image, dest) -> None:
        """`docker save <image> | tar -x -C <dest>`, streamed rather than buffered.

        An image tarball is gigabytes; materialising it in memory or as a second
        file on disk is what makes people skip this audit.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        saver = subprocess.Popen(
            [self._binary, 'save', image], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert saver.stdout is not None
        untar = subprocess.Popen(
            ['tar', '-x', '-C', str(dest)], stdin=saver.stdout, stderr=subprocess.PIPE
        )
        saver.stdout.close()
        untar_status = untar.wait()
        save_status = saver.wait()
        if save_status != 0 or untar_status != 0:
            err = (saver.stderr.read() if saver.stderr else b'').decode('utf-8', 'replace')
            raise VerifyError(
                f'docker save {image} | tar -x failed (save exit {save_status}, '
                f'tar exit {untar_status}): {err.strip()[:400]}'
            )


def is_removable_image(image: str) -> bool:
    return image.startswith(REMOVABLE_IMAGE_PREFIX)


def cleanup_image(image: str, runner) -> None:
    if not is_removable_image(image):
        raise VerifyError(
            f'refusing to delete {image}: only {REMOVABLE_IMAGE_PREFIX}* images are '
            'built by this verifier'
        )
    runner.remove_image(image)


# ------------------------------------------------------------ orchestration --


@dataclass
class StateResult:
    name: str
    reward: Reward
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


@dataclass
class CheckResult:
    name: str
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def check_image_has_no_solution(output: str) -> list[str]:
    """Reasons the shipped image is not free of the oracle.

    Read line-exact rather than by substring: `ls -R` of a leaked tree could
    print anything, and a marker matched inside that listing would invert the
    verdict. Silence is a failure too -- if neither marker arrives the probe did
    not run, and an unproven image must not pass as a clean one.
    """
    lines = {ln.strip() for ln in output.splitlines()}
    if LEAK_MARKER in lines:
        return [
            f'the built image contains {SOLUTION_MOUNTPOINT} -- harbor gives the agent '
            'a shell, so the oracle solution is readable and the task grades nothing'
        ]
    if CLEAN_MARKER not in lines:
        return [
            f'neither {CLEAN_MARKER} nor {LEAK_MARKER} came back from the probe, so the '
            f'absence of {SOLUTION_MOUNTPOINT} in the image is unproven'
        ]
    return []


@dataclass
class VerifyResult:
    image: str
    expected: int
    red: StateResult
    green: StateResult
    build_seconds: float = 0.0
    shared: SharedAssets | None = None
    image_clean: CheckResult | None = None
    layer_clean: CheckResult | None = None
    integrity: CheckResult | None = None
    floor_mode: str = 'equality'
    lang: str = 'python'
    carve_scope: str = 'function'

    @property
    def red_reasons(self) -> list[str]:
        return self.red.reasons

    @property
    def green_reasons(self) -> list[str]:
        return self.green.reasons

    @property
    def runtime_ok(self) -> bool:
        """RED, GREEN and the runtime oracle probe -- everything one container shows."""
        shared_ok = self.shared is None or self.shared.equivalent
        clean_ok = self.image_clean is None or self.image_clean.ok
        return self.red.ok and self.green.ok and shared_ok and clean_ok

    @property
    def passed(self) -> bool:
        """The full bar. Unrun audits (`None`) never pass.

        The layer audit is not optional and not defaulted: this module once
        reported PASS for an image whose first layer held the intact answer,
        because only the final filesystem had been probed. The oracle integrity
        check earns the same treatment for the same reason -- solve.sh stopped
        pinning the intact digest, so if nothing on the host compares the
        payload to upstream, nothing does at all.
        """
        return (
            self.runtime_ok
            and self.layer_clean is not None and self.layer_clean.ok
            and self.integrity is not None and self.integrity.ok
        )


def run_states(image: str, runner, expected: int, solution_dir=None, echo=print,
               floor_mode: str = 'equality') -> VerifyResult:
    """Grade the image twice: as shipped, then after the oracle has run.

    The oracle reaches the container through a read-only bind mount of
    `solution_dir`, never through the image, so the GREEN half proves
    solvability without contradicting the clean-image half.
    """
    echo('')
    echo(f'== IMAGE HYGIENE: no {SOLUTION_MOUNTPOINT} in the shipped image ==')
    clean = CheckResult('IMAGE', check_image_has_no_solution(runner.run(image, NO_SOLUTION_SCRIPT)))
    echo(f'image-solution-absent: {"PASS" if clean.ok else "FAIL"}')

    echo('')
    echo('== RED: the carved tree as the agent receives it ==')
    red_out = runner.run(image, RED_SCRIPT)
    red = StateResult('RED', parse_reward_json(red_out))
    red.reasons = check_red(red.reward)

    echo('')
    echo(f'== GREEN: after solve.sh, bind-mounted from {solution_dir} ==')
    mounts = ((str(solution_dir), SOLUTION_MOUNTPOINT),) if solution_dir else ()
    green_out = runner.run(image, GREEN_SCRIPT, mounts=mounts)
    green = StateResult('GREEN', parse_reward_json(green_out))
    green.reasons = check_green(green.reward, expected, floor_mode)

    result = VerifyResult(
        image=image, expected=expected, red=red, green=green, image_clean=clean,
        floor_mode=floor_mode,
    )
    for state in (red, green):
        if not state.ok:
            echo('')
            echo(f'== DIAGNOSTIC re-run of the failing {state.name} state ==')
            runner.run(
                image,
                DIAGNOSTIC_SCRIPTS[state.name],
                mounts=mounts if state is green else (),
            )
    return result


def _default_repo(spec: EntrySpec) -> Path:
    """`<checkout root>/harbor-tasks/repos-src/<repo>`, found by walking up."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.joinpath(*REPOS_SRC_RELPATH, spec.repo_dirname)
        if candidate.is_dir():
            return candidate
    raise VerifyError(
        f'could not locate the source checkout for {spec.repo_dirname!r} under any '
        f'ancestor of {here}. Pass --repo explicitly.'
    )


def _repos_cache_dir() -> Path | None:
    """The `repos-src/` a self-clone would land in, whether or not it holds this repo."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent.joinpath(*REPOS_SRC_RELPATH)
        if candidate.is_dir():
            return candidate
    return None


def _self_clone(spec: EntrySpec, echo=print) -> Path:
    """Re-materialise the pinned upstream checkout the entry was carved from.

    Only reachable when the entry records a PINNED source: oracle-integrity
    re-hashes every carved file against this tree, so cloning a moving branch
    here would turn an upstream commit into a verify failure that looks like a
    tampered oracle.
    """
    from . import clone

    source = spec.repo_url
    if not source or not spec.commit:
        raise VerifyError(
            f'{spec.entry_id} has no pinned source to clone from; its task.toml '
            'records no [provenance].repo_url and commit pair'
        )
    cache = _repos_cache_dir()
    if cache is None:
        raise VerifyError(
            f'{spec.entry_id} records a pinned source ({spec.repo_url} @ '
            f'{spec.commit}) but no harbor-tasks/repos-src/ directory exists to '
            'clone it into. Pass --repo explicitly.'
        )
    echo(f'self-clone {source} @ {spec.commit} -> {cache / spec.repo_dirname}')
    try:
        return clone.ensure_repo(
            source=source,
            commit=spec.commit,
            cache_dir=cache,
            name=spec.repo_dirname,
            allow_floating=False,
        )
    except SystemExit as exc:
        raise VerifyError(f'self-clone failed: {exc}') from exc


def _resolve_repo(spec: EntrySpec, repo, echo=print) -> Path:
    if repo:
        return Path(repo).resolve()
    try:
        return _default_repo(spec)
    except VerifyError:
        if not spec.pinned_source:
            raise
    return _self_clone(spec, echo=echo)


def verify_entry(
    entry_dir: Path,
    repo: Path | None = None,
    keep_image: bool = False,
    runner=None,
    also_cover=(),
    echo=print,
    lang: str | None = None,
    carve_scope: str | None = None,
) -> VerifyResult:
    spec = load_entry(entry_dir)
    lang = lang or spec.lang
    carve_scope = carve_scope or spec.carve_scope
    floor_mode = resolve_floor_mode(lang, spec.floor_mode)
    repo_path = _resolve_repo(spec, repo, echo=echo)
    if not repo_path.is_dir():
        raise VerifyError(f'repo checkout does not exist: {repo_path}')
    runner = runner or DockerRunner(echo=echo)

    echo(f'entry      {spec.entry_id}  ({spec.condition})')
    echo(f'target     {spec.target_file}::{spec.target_func}')
    echo(f'graded     {spec.expected} test ids')
    echo(f'lang       {lang}  scope={carve_scope}  floor={floor_mode}')
    echo(f'repo       {repo_path}')
    echo(f'image      {spec.image}')

    shared = None
    if also_cover:
        shared = compare_entries([spec.entry_dir, *also_cover])
        _report_shared(shared, echo)
        if not shared.equivalent:
            raise VerifyError(
                'the entries are NOT image-equivalent, so one build cannot prove all '
                'of them: ' + ', '.join(shared.mismatched + shared.missing)
            )

    echo('')
    echo(f'== HOST-SIDE ORACLE INTEGRITY: solution/carved vs {repo_path.name} ==')
    integrity = check_oracle_integrity(spec.entry_dir, repo_path)
    echo(f'oracle-integrity: {"PASS" if integrity.ok else "FAIL"}')
    for reason in integrity.reasons:
        echo(f'  {reason}')

    plan = load_build_plan(spec.entry_dir)
    if plan.image != spec.image:
        raise VerifyError(
            f'{spec.entry_dir.name}: task.toml names image {spec.image} but '
            f'contexts.json builds {plan.image}; the graded image is ambiguous'
        )
    echo('')
    echo(f'== BUILD (stage {plan.stage}) ==')
    for name, target in plan.contexts:
        echo(f'  --build-context {name}={target}')
    started = time.monotonic()
    runner.build(plan)
    elapsed = time.monotonic() - started
    echo(f'build finished in {elapsed:.1f}s')

    try:
        result = run_states(
            spec.image,
            runner,
            spec.expected,
            solution_dir=spec.entry_dir / 'solution',
            echo=echo,
            floor_mode=floor_mode,
        )
        echo('')
        echo('== LAYER ARCHAEOLOGY: docker save, then scan EVERY layer ==')
        patterns, digests = oracle_signals(spec.entry_dir)
        result.layer_clean = audit_image_layers(
            spec.image, runner, patterns=patterns, digests=digests, echo=echo
        )
        echo(f'layer-leak-absent: {"PASS" if result.layer_clean.ok else "FAIL"} '
             f'({len(patterns)} tripwire line(s), {len(digests)} oracle digest(s))')
    finally:
        if not keep_image:
            echo('')
            echo(f'cleaning up {spec.image} (pass --keep-image to keep it)')
            cleanup_image(spec.image, runner)

    result.build_seconds = elapsed
    result.shared = shared
    result.integrity = integrity
    result.lang = lang
    result.carve_scope = carve_scope
    return result


def _report_shared(shared: SharedAssets, echo=print) -> None:
    echo('')
    echo(f'== SHARED-IMAGE PROOF: {len(shared.covered)} entries, reference {shared.reference} ==')
    for rel in shared.identical:
        echo(f'  byte-identical   {rel}')
    for rel in shared.comment_only:
        echo(f'  identical*       {rel}   (* differs only in comment lines: the slug header)')
    for rel in shared.mismatched:
        echo(f'  MISMATCH         {rel}')
    for rel in shared.missing:
        echo(f'  MISSING          {rel}')


def report(result: VerifyResult, specs=(), echo=print) -> None:
    echo('')
    echo('=' * 72)
    for state in (result.red, result.green):
        echo(f'{state.name:<6} reward.json  {state.reward.raw}')
    echo(f'expected     {result.expected} graded tests ({result.floor_mode} floor)')
    echo(f'lang/scope   {result.lang} / {result.carve_scope}')
    if result.build_seconds:
        echo(f'build time   {result.build_seconds:.1f}s')
    if result.image_clean is not None:
        echo(f'image-solution-absent: {"PASS" if result.image_clean.ok else "FAIL"}')
        for reason in result.image_clean.reasons:
            echo(f'FAIL [IMAGE] {reason}')
    echo('layer-leak-absent: '
         + ('PASS' if result.layer_clean and result.layer_clean.ok else 'FAIL'))
    for reason in (result.layer_clean.reasons if result.layer_clean else
                   ['the layer audit never ran, so the image is unproven']):
        echo(f'FAIL [LAYER] {reason}')
    echo('oracle-integrity: '
         + ('PASS' if result.integrity and result.integrity.ok else 'FAIL'))
    for reason in (result.integrity.reasons if result.integrity else
                   ['the oracle was never compared to upstream, so it is unproven']):
        echo(f'FAIL [INTEGRITY] {reason}')
    for state in (result.red, result.green):
        for reason in state.reasons:
            echo(f'FAIL [{state.name}] {reason}')
    if result.shared is not None and not result.shared.equivalent:
        echo('FAIL [SHARED] entries are not image-equivalent')

    if result.passed:
        echo(
            'VERDICT: PASS -- no oracle in the image, no carved bytes in any layer, '
            'the oracle payload is the upstream file, stub scores 0.0, oracle scores '
            f'reward=1.0 binary={result.green.reward.binary} with '
            f'{result.green.reward.tests_passed}/{result.expected} graded tests'
        )
    else:
        echo('VERDICT: FAIL')
    if specs:
        echo('')
        echo(f'covered by this single image/carve/oracle ({len(specs)} entries):')
        for spec in specs:
            echo(f'  {spec.entry_id}  {spec.condition}')
    echo('=' * 72)


def main(entry=None, out=None, repo=None, keep_image=False, echo=print,
         lang=None, carve_scope=None) -> int:
    """CLI body. Returns a process exit code; raises VerifyError on setup faults."""
    if bool(entry) == bool(out):
        raise VerifyError('pass exactly one of --entry or --all')

    specs = ()
    if out:
        specs = discover_entries(out)
        entry = specs[0].entry_dir
        others = [s.entry_dir for s in specs[1:]]
        echo(f'verifying {len(specs)} entries through one shared image')
    else:
        others = []

    result = verify_entry(
        entry_dir=entry,
        repo=repo,
        keep_image=keep_image,
        also_cover=others,
        echo=echo,
        lang=lang,
        carve_scope=carve_scope,
    )
    report(result, specs=specs, echo=echo)
    return 0 if result.passed else 1


if __name__ == '__main__':
    sys.exit(main(entry=Path(sys.argv[1])))
