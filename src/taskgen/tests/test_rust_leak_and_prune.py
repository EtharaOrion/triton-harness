r"""The rust leak gate's PRECISION, and the prune's internal consistency.

Two live-diagnosed defects, one file: both are about a rust plan or a rust
image refusing a repository it has no business refusing.

The strings-target leak assert greps `target/` for the carved crate. It used to
grep for the BARE crate name, unanchored, which is a substring search: for
`rust-lang/log` the standard library's own `core::num::int_log10` and
`core::slice::sort::stable::drift::logical_merge` matched and the gate reported
a LEAK on an image that leaked nothing. The assert now matches by rust MANGLING
STRUCTURE, and the corpora below pin BOTH directions -- a std symbol that merely
contains the crate name is not a hit, a symbol the crate DEFINES is.

The tests run the real `grep -E` on the real pattern: `leak_symbol_ere` is
rendered here with literal values and inside the Dockerfile with the shell
expansions that hold the same three values, so what is proved and what runs in
the image are the same string. Making the gate permissive again -- reverting to
`grep "${CRATE}"`, or loosening any anchor -- fails `test_a_std_symbol_that_only
_contains_the_crate_name_is_not_a_leak`, which is what that test is for.

Pure: no docker, no network, no LLM. `grep` only.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from taskgen import langs
from taskgen.depplan import DepPlan, canonicalize
from taskgen.langs import base as B
from taskgen.langs import rust as R

from .test_rust_harness_golden import plan_with

WORKDIR = B.EnvSpec(repo_name='r').workdir

#: Real rustc output shapes that merely CONTAIN "log". Every one of these was a
#: hit under the old bare-substring gate; the first two are quoted verbatim from
#: the live failure on `rust-lang/log`.
STD_SYMBOLS_FOR_LOG: tuple[str, ...] = (
    '_ZN4core3num9int_log103u3217h1a2b3c4d5e6f7a8bE',
    '_ZN4core5slice4sort6stable5drift13logical_merge17hdeadbeefcafef00dE',
    '_ZN4core3ptr9alignment9Alignment4log217h0011223344556677E',
    '_ZN4core3f6413_$LT$impl$u20$f64$GT$3log17h9988776655443322E',
    '_ZN4core3f3213_$LT$impl$u20$f32$GT$5log1017hccddeeff00112233E',
    '_ZN5tokio7logging6logger17habcdef0123456789E',
    '_ZN3std2io5stdio6_print17h00ff00ff00ff00ffE',
    f'{WORKDIR}/vendor/serde/src/de/mod.rs',
    f'{WORKDIR}/vendor/log-mdc/src/lib.rs',
    '/rustc/9b00956e56009bab2aa15d7bff10916599e3d6d6/library/core/src/num/mod.rs',
    'catalog logic logging dialogue prologue',
)

#: Symbols and paths that only exist if the CARVED crate reached a layer.
LEAKED_SYMBOLS_FOR_LOG: tuple[str, ...] = (
    '_ZN3log5Level6as_str17h0123456789abcdefE',
    '_ZN3log6kv3key3Key6as_str17hfedcba9876543210E',
    '_ZN3log11__private_api3log17h4444555566667777E',
    '_ZN72_$LT$$RF$T$u20$as$u20$log..Log$GT$3log17h1122334455667788E',
    '_ZN4core3ptr13drop_in_place$LT$log..kv..Value$GT$17h8888999900001111E',
    '_RNvNtCsbGf3fpMk9wM_3log2kv5Value',
    '_RNvNtC3log5level5Level',
    f'{WORKDIR}/src/lib.rs',
    f'{WORKDIR}/src/kv/value.rs',
)


def ere(crate: str) -> str:
    """The production pattern with the three shell holes filled literally."""
    return R.leak_symbol_ere(crate.replace('-', '_'), str(len(crate)), crate, WORKDIR)


def hits(pattern: str, lines: tuple[str, ...], tmp_path) -> tuple[str, ...]:
    """`grep -E` over the lines, exactly as the image greps `strings` output."""
    corpus = tmp_path / 'strings.txt'
    corpus.write_text('\n'.join(lines) + '\n')
    done = subprocess.run(
        ['grep', '-E', pattern, str(corpus)],
        capture_output=True, text=True, check=False,
    )
    assert done.returncode in (0, 1), done.stderr
    return tuple(done.stdout.splitlines())


@pytest.mark.parametrize('symbol', STD_SYMBOLS_FOR_LOG)
def test_a_std_symbol_that_only_contains_the_crate_name_is_not_a_leak(
    symbol: str, tmp_path,
) -> None:
    """THE negative test: this is the false positive that blocked `rust-lang/log`.

    It also fails the moment anyone makes the gate permissive again -- a bare
    `grep "${CRATE}"`, or dropping the length prefix, or dropping the trailing
    digit that pins the crate to the FIRST mangled path component -- so it is
    the guard on the fix as well as the proof of it.
    """
    assert hits(ere('log'), (symbol,), tmp_path) == ()


@pytest.mark.parametrize('symbol', LEAKED_SYMBOLS_FOR_LOG)
def test_a_symbol_the_carved_crate_defines_is_still_a_leak(
    symbol: str, tmp_path,
) -> None:
    """The direction that matters: precision must not cost detection."""
    assert hits(ere('log'), (symbol,), tmp_path) == (symbol,)


def test_the_bare_substring_gate_is_what_these_corpora_reject(tmp_path) -> None:
    """The regression itself, spelled out: the OLD pattern fails the new corpus.

    Without this, a future "simplification" back to `grep "${CRATE}"` would look
    like it still passed the leak tests -- it would catch every leaked symbol,
    because it catches everything.
    """
    old = f'log|{WORKDIR}/src'
    assert len(hits(old, STD_SYMBOLS_FOR_LOG, tmp_path)) >= 6
    assert hits(ere('log'), STD_SYMBOLS_FOR_LOG, tmp_path) == ()
    assert len(hits(ere('log'), LEAKED_SYMBOLS_FOR_LOG, tmp_path)) == len(
        LEAKED_SYMBOLS_FOR_LOG
    )


@pytest.mark.parametrize(
    ('crate', 'innocent', 'leaked'),
    [
        ('sort', '_ZN4core5slice4sort6stable5drift13logical_merge17hab12cd34ef56ab78E',
         '_ZN4sort9quicksort4heap17h1234abcd5678ef90E'),
        ('num', '_ZN4core3num7flt2dec8strategy5grisu17hffeeddccbbaa9988E',
         '_ZN3num4sign6Signum6signum17h5566778899aabbccE'),
        ('str', '_ZN4core3str7pattern11StrSearcher3new17h0f1e2d3c4b5a6978E',
         '_ZN3str6encode6encode17h1a1b1c1d1e1f2a2bE'),
        ('value', '_ZN4core3fmt5value9formatter17h2b3c4d5e6f7a8b9cE',
         '_ZN5value3ast5parse17h3c4d5e6f7a8b9c0dE'),
        ('uuid-macros', '_ZN4uuid6macros5build17h4d5e6f7a8b9c0d1eE',
         '_ZN11uuid_macros7derive17h5e6f7a8b9c0d1e2fE'),
    ],
)
def test_the_anchoring_generalises_past_the_one_crate_that_exposed_it(
    crate: str, innocent: str, leaked: str, tmp_path,
) -> None:
    """No ignore list: the rule is the mangling grammar, so it holds for any name.

    `sort`, `num`, `str` and `value` are all substrings of real std symbols the
    way `log` is; `uuid-macros` additionally proves the `-` -> `_` rewrite rustc
    applies to a package name before it becomes a symbol component.
    """
    assert hits(ere(crate), (innocent,), tmp_path) == ()
    assert hits(ere(crate), (leaked,), tmp_path) == (leaked,)


def test_the_shipped_assert_greps_exactly_the_pattern_under_test() -> None:
    """The seam: the image's grep line IS `leak_symbol_ere`, holes and all."""
    df = langs.get('rust').render_dockerfile(B.EnvSpec(repo_name='rust-spacewasm'))
    assert 'SYM=$(printf \'%s\' "${CRATE}" | sed \'s/-/_/g\')' in df
    assert 'N=${#SYM}' in df
    assert R.leak_symbol_ere('${SYM}', '${N}', '${CRATE}', WORKDIR) in df
    assert 'grep -E "${CRATE}' not in df
    assert 'LEAK: rustc artifacts under target/' in df


def test_the_measure_image_still_carries_no_strings_assert() -> None:
    """The precision fix must not have grown the assert a second home."""
    instructions = langs.get('rust').render_measure_dockerfile(
        B.EnvSpec(repo_name='rust-spacewasm')
    )
    assert 'xargs -0 -r strings' not in instructions
    assert 'LEAK' not in instructions


PLUGIN = langs.get('rust')


def test_a_pruned_path_left_in_the_workspace_members_is_refused() -> None:
    """DEFECT 2, exactly as `uuid-rs/uuid` hit it live.

    `prune_paths` deleted examples/ while `prune_manifest_entries` left it in
    `[workspace] members`; `cargo vendor` then refused to load a member manifest
    that was no longer on disk and the run REFUSEd after a docker build. The
    contradiction is visible in the PLAN, so it is caught here instead.
    """
    plan = plan_with(
        prune_paths=('examples',),
        prune_manifest_keys=(),
        prune_manifest_entries=(),
    )
    with pytest.raises(B.LangError) as caught:
        PLUGIN.validate_dep_plan(plan)
    message = str(caught.value)
    assert 'examples' in message
    assert 'prune_manifest_entries' in message
    assert 'prune_paths' in message


def test_a_de_listed_member_left_on_disk_is_refused() -> None:
    """The same contradiction from the other side, so neither slot can drift."""
    plan = plan_with(
        prune_paths=(),
        prune_manifest_keys=('members',),
        prune_manifest_entries=('examples',),
    )
    with pytest.raises(B.LangError) as caught:
        PLUGIN.validate_dep_plan(plan)
    assert 'examples' in str(caught.value)
    assert 'prune_paths' in str(caught.value)


def test_the_refusal_names_the_offending_path_and_both_repairs() -> None:
    """One repair round: the model is told the path, the slot and the two exits."""
    plan = plan_with(
        prune_paths=('crates', 'benches'),
        prune_manifest_keys=('members',),
        prune_manifest_entries=('crates/*',),
    )
    with pytest.raises(B.LangError) as caught:
        PLUGIN.validate_dep_plan(plan)
    message = str(caught.value)
    assert "prunes 'benches' from disk" in message
    assert '"benches/*"' in message
    assert 'prune_manifest_keys' in message


@pytest.mark.parametrize(
    ('paths', 'keys', 'entries'),
    [
        ((), (), ()),
        (('crates', 'fuzz'), ('members', 'exclude'), ('crates/*', 'fuzz')),
        (('crates',), ('members',), ('crates/*',)),
        (('crates/inner',), ('members',), ('crates/*',)),
        (('examples',), ('members',), ('examples',)),
        (('examples', 'macros'), ('members', 'members'), ('examples', 'macros')),
    ],
)
def test_a_prune_whose_halves_agree_is_accepted(
    paths: tuple[str, ...], keys: tuple[str, ...], entries: tuple[str, ...],
) -> None:
    """Precision here too: a glob entry covers the directory it globs."""
    PLUGIN.validate_dep_plan(
        plan_with(
            prune_paths=paths,
            prune_manifest_keys=keys,
            prune_manifest_entries=entries,
        )
    )


SPACEWASM_WORKSPACE = '''[workspace]

members = ["crates/*"]
exclude = ["fuzz"]

[workspace.package]
edition = "2024"
'''

MULTI_MEMBER_WORKSPACE = '''[workspace]
members = [
    "rng",
    "examples",
    "tests/smoke-test",
    "tests/wasm32-getrandom-test",
]

[package]
name = "uuid"
'''


def run_rendered_prune(manifest: str, plan: DepPlan, tmp_path) -> tuple[int, str, str]:
    """Execute the prune script the image runs, against a real Cargo.toml.

    The script is lifted out of the rendered heredoc rather than restated, so
    this exercises the bytes that ship. Only the manifest path is repointed.
    """
    env = B.EnvSpec(repo_name='r')
    blocks = '\n'.join(PLUGIN.pre_leakgate_blocks_for(env, plan))
    body = blocks.split("<<'PY'\n", 1)[1].split('\nPY', 1)[0]
    target = tmp_path / 'Cargo.toml'
    target.write_text(manifest)
    body = body.replace(f'"{env.workdir}/Cargo.toml"', f'"{target}"')
    done = subprocess.run(
        [sys.executable, '-c', body], capture_output=True, text=True, check=False,
    )
    return done.returncode, target.read_text(), done.stderr


def test_the_prune_drops_one_member_and_leaves_the_others(tmp_path) -> None:
    """DEFECT 2's other half: the edit that used to no-op on a real workspace.

    `uuid-rs/uuid` declares four members. The old edit replaced the whole array
    literal, matched nothing, and left `examples` listed while the `rm -rf`
    beside it deleted the directory -- which is the exact state `cargo vendor`
    refuses. The element-wise edit removes what the plan named, and only that.
    """
    plan = plan_with(
        prune_paths=('examples', 'tests/smoke-test', 'tests/wasm32-getrandom-test'),
        prune_manifest_keys=('members', 'members', 'members'),
        prune_manifest_entries=(
            'examples', 'tests/smoke-test', 'tests/wasm32-getrandom-test',
        ),
    )
    code, manifest, stderr = run_rendered_prune(MULTI_MEMBER_WORKSPACE, plan, tmp_path)
    assert code == 0, stderr
    assert 'members = ["rng"]' in manifest
    assert 'examples' not in manifest
    assert 'smoke-test' not in manifest
    assert 'name = "uuid"' in manifest


def test_the_prune_still_empties_a_sole_member_array(tmp_path) -> None:
    """rust-spacewasm's manifest comes out of the new edit unchanged in effect.

    The Dockerfile bytes moved; what the image ends up holding did not.
    """
    plan = plan_with(
        prune_paths=('crates', 'fuzz'),
        prune_manifest_keys=('members', 'exclude'),
        prune_manifest_entries=('crates/*', 'fuzz'),
    )
    code, manifest, stderr = run_rendered_prune(SPACEWASM_WORKSPACE, plan, tmp_path)
    assert code == 0, stderr
    assert 'members = []' in manifest
    assert 'exclude = []' in manifest
    assert '[workspace.package]\nedition = "2024"' in manifest


def test_a_prune_the_manifest_does_not_list_fails_loud(tmp_path) -> None:
    """A prune that matches nothing must not pass for a prune that worked.

    Silence is what made the live failure so hard to read: cargo blamed a
    missing member manifest three RUN blocks later. The plan can only be
    repaired against a message that names the array and the entry.
    """
    plan = plan_with(
        prune_paths=('benches',),
        prune_manifest_keys=('members',),
        prune_manifest_entries=('benches',),
    )
    code, manifest, stderr = run_rendered_prune(MULTI_MEMBER_WORKSPACE, plan, tmp_path)
    assert code != 0
    assert 'PRUNE FAILED' in stderr
    assert 'benches' in stderr
    assert manifest == MULTI_MEMBER_WORKSPACE


def test_a_prune_of_an_array_the_manifest_lacks_fails_loud(tmp_path) -> None:
    """No `exclude` array at all is the same mismatch, reported the same way."""
    plan = plan_with(
        prune_paths=('fuzz',),
        prune_manifest_keys=('exclude',),
        prune_manifest_entries=('fuzz',),
    )
    code, _manifest, stderr = run_rendered_prune(MULTI_MEMBER_WORKSPACE, plan, tmp_path)
    assert code != 0
    assert 'PRUNE FAILED' in stderr
    assert 'exclude' in stderr


def test_the_shipped_spacewasm_plan_still_passes_the_new_cross_check() -> None:
    """The fixed point: rust's own harness pairs `crates`/`fuzz` with its arrays."""
    PLUGIN.validate_dep_plan(
        canonicalize(
            DepPlan(
                lang='rust',
                toolchain_version=R.RUST_VERSION,
                package_manager='cargo',
                build_flags=(
                    ('shims_path', R.SHIMS_PATH),
                    ('toolchain_source', R.TOOLCHAIN_SOURCE),
                ),
                test_invocation=(
                    ('command', R.RUST_SPACEWASM_HARNESS_SPEC.graded_argv),
                ),
                harness=R.RUST_SPACEWASM_HARNESS,
            )
        )
    )
