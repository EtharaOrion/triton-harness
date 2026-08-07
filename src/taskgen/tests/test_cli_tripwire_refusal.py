"""generate REFUSES a carve the leak gate cannot cover, instead of shipping it.

The live defect: a small idiomatic target (go `trimZeroDecimal`) whose every
body line is either under the strong floor or duplicated in a surviving file
leaves the tripwire ladder with nothing but content digests. `trip/tripwires.txt`
is then written EMPTY, `generate` reports success and writes all eleven entries,
and every one of them dies at docker build:

    LEAKSCAN FATAL: no tripwires at /tmp/.harbor-tripwires   (exit 2)

An emitted task that can never build is the silent degrade SHIP-or-REFUSE
forbids, so the outcome is a refusal at generate time. These tests pin the
refusal to the SAME convention `ResolveRefused` uses -- `REFUSE(reason)` on
stderr, exit 3, nothing shipped -- and pin the normal carve as unchanged.

The subject is `--lang go`: a parser-backed language never reaches the resolver
or a DepPlan, so the refusal cannot live in the refine machinery.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taskgen import cli
from taskgen.staging import NoLineTripwireError

MODULE = 'github.com/acme/gadget'
SRC = 'pkg/trim.go'
FUNC = 'TrimZeroDecimal'

#: Every body line is short, idiomatic go -- `return s`, `default:`,
#: `foundZero = true`. This is the shape of the live failure.
GENERIC_BODY = (
    '\tvar foundZero bool\n'
    '\tfor i := len(s); i > 0; i-- {\n'
    '\t\tswitch s[i-1] {\n'
    '\t\tcase 0x2E:\n'
    '\t\t\tif foundZero {\n'
    '\t\t\t\treturn s[:i-1]\n'
    '\t\t\t}\n'
    '\t\tcase 0x30:\n'
    '\t\t\tfoundZero = true\n'
    '\t\tdefault:\n'
    '\t\t\treturn s\n'
    '\t\t}\n'
    '\t}\n'
    '\treturn s\n'
)

#: The same function with ONE line long enough and repo-specific enough to be
#: handed to `grep -F`. Nothing else about the carve differs.
DISTINCTIVE_BODY = GENERIC_BODY.replace(
    '\tvar foundZero bool\n',
    '\tvar foundZero bool // gadget-specific trailing zero decimal trimmer\n',
)


def _write_module(root: Path, body: str) -> Path:
    (root / 'pkg').mkdir(parents=True)
    (root / 'go.mod').write_text(f'module {MODULE}\n\ngo 1.26\n')
    (root / 'go.sum').write_text('')
    (root / SRC).write_text(
        'package pkg\n'
        '\n'
        '// TrimZeroDecimal drops a trailing zero-only decimal part.\n'
        f'func {FUNC}(s string) string {{\n'
        f'{body}'
        '}\n'
    )
    (root / 'pkg' / 'trim_test.go').write_text(
        'package pkg\n'
        '\n'
        'import "testing"\n'
        '\n'
        '// TestTrimOne exercises TrimZeroDecimal.\n'
        'func TestTrimOne(t *testing.T) {\n'
        f'\tif {FUNC}("1.00") != "1" {{\n\t\tt.Fatal("no")\n\t}}\n'
        '}\n'
        '\n'
        '// TestTrimTwo exercises TrimZeroDecimal again.\n'
        'func TestTrimTwo(t *testing.T) {\n'
        f'\tif {FUNC}("2.0") != "2" {{\n\t\tt.Fatal("no")\n\t}}\n'
        '}\n'
    )
    return root


@pytest.fixture
def generic_repo(tmp_path: Path) -> Path:
    return _write_module(tmp_path / 'gadget-generic', GENERIC_BODY)


@pytest.fixture
def normal_repo(tmp_path: Path) -> Path:
    return _write_module(tmp_path / 'gadget-normal', DISTINCTIVE_BODY)


def _argv(repo: Path, out: Path) -> list[str]:
    return [
        'generate', '--repo', str(repo), '--out', str(out),
        '--lang', 'go', '--file', SRC, '--func', FUNC,
    ]


def _entries(out: Path) -> list[Path]:
    return sorted(out.rglob('task.toml'))


# --------------------------------------------------------------------------
# the refusal
# --------------------------------------------------------------------------


def test_a_carve_with_no_greppable_tripwire_is_refused_not_shipped(
        generic_repo, tmp_path, capsys):
    out = tmp_path / 'out'
    code = cli.main(_argv(generic_repo, out))

    assert code == 3, 'a refusal must exit non-zero, on the ResolveRefused code'
    err = capsys.readouterr().err
    assert 'REFUSE(' in err, err[-400:]
    assert _entries(out) == [], 'a refused carve must ship no entry at all'


def test_the_refusal_says_what_was_carved_why_and_what_to_do(
        generic_repo, tmp_path, capsys):
    """An unactionable refusal just moves the dead end one step earlier."""
    cli.main(_argv(generic_repo, tmp_path / 'out'))
    err = capsys.readouterr().err

    assert f'{SRC}::{FUNC}' in err, 'the refusal must name the carved target'
    assert 'leak-absence could not be proven' in err
    assert 'shorter than 40 characters' in err
    assert 'also present in a surviving file' in err
    assert 'LEAKSCAN FATAL: no tripwires' in err
    for remedy in ('carve a larger target', 'a different function', '--carve-scope'):
        assert remedy in err, remedy


def test_the_refused_run_leaves_no_buildable_looking_output_behind(
        generic_repo, tmp_path):
    """Not one artefact a later step could mistake for a finished task."""
    out = tmp_path / 'out'
    assert cli.main(_argv(generic_repo, out)) == 3

    assert list(out.rglob('task.toml')) == []
    assert list(out.rglob('instruction.md')) == []
    assert list(out.rglob('Dockerfile')) == []
    assert list(out.rglob('tripwires.txt')) == [], (
        'the empty tripwire file is the thing that could never build'
    )


def test_the_refusal_never_touches_the_resolver_or_a_dep_plan(
        generic_repo, tmp_path, monkeypatch):
    """go is parser-backed: it has no DepPlan, so this cannot be refine's job."""
    from taskgen import refine

    def explode(*args, **kwargs):
        raise AssertionError('a parser-backed refusal reached the refine loop')

    monkeypatch.setattr(refine, 'refine_dep_plan', explode, raising=False)
    assert cli.main(_argv(generic_repo, tmp_path / 'out')) == 3


def test_the_refusal_is_raised_before_anything_is_written(generic_repo, tmp_path):
    """Straight out of emit, as a typed refusal -- not a print-and-continue."""
    from taskgen.emit import emit_all

    out = tmp_path / 'out'
    with pytest.raises(NoLineTripwireError) as exc:
        emit_all(repo=generic_repo, out=out, lang='go', file=SRC, func=FUNC,
                 echo=lambda *_: None)
    assert exc.value.reason
    assert _entries(out) == []


# --------------------------------------------------------------------------
# and the carve that was always fine is still fine
# --------------------------------------------------------------------------


def test_a_normal_carve_still_ships_all_eleven_entries(normal_repo, tmp_path, capsys):
    out = tmp_path / 'out'
    code = cli.main(_argv(normal_repo, out))
    capsys.readouterr()

    assert code == 0
    assert len(_entries(out)) == 11


def test_a_normal_carve_ships_a_tripwire_file_leakscan_can_actually_read(
        normal_repo, tmp_path, capsys):
    """`[ ! -s ]` is the first thing leakscan.sh checks, and it fails closed."""
    out = tmp_path / 'out'
    assert cli.main(_argv(normal_repo, out)) == 0
    capsys.readouterr()

    written = next(out.rglob('tripwires.txt')).read_text()
    assert written.strip(), 'an empty tripwire file is LEAKSCAN FATAL at build'
    for line in written.splitlines():
        assert len(line) >= 40, f'{line!r} is too weak to be evidence'


def test_a_thin_but_workable_carve_warns_and_ships_anyway(
        normal_repo, tmp_path, capsys):
    """One tripwire builds and gates, so it SHIPS -- the warning is log-only.

    Two real passing runs sat on exactly one line, one short-or-duplicated line
    from the refusal above, and that margin should be visible.
    """
    out = tmp_path / 'out'
    assert cli.main(_argv(normal_repo, out)) == 0
    logged = capsys.readouterr().out

    assert 'only 1 grep tripwire line(s) cover this carve' in logged
    assert f'{SRC}::{FUNC}' in logged
    assert len(_entries(out)) == 11
