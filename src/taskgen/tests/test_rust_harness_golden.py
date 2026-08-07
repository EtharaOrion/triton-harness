r"""The rust harness seam: a plan renders the scripts the plugin used to hardcode.

The last of the six languages, and the same discipline as
`test_c_harness_golden`, `test_cpp_harness_golden` and `test_java_harness_golden`.
`render_gap` proved a DepPlan could reproduce the measure Dockerfile's toolchain
bytes; this module proves a DepPlan can reproduce the two grading scripts, the
oracle and every pre-leakgate block -- which is where the rust-spacewasm-shaped
constants lived:

  * `HARNESSES`, the four named spacewasm integration binaries, woven through
    `TEST_COMMAND`, the test.sh prose and -- structurally -- `SUMMARIES -eq 4`
  * `INTEGRITY_HARNESS_FILES`, spacewasm's three `tests/util/*.rs` oracle files
  * `MIN_WAST = 90` and the `find tests -name '*.wast'` grade-time gate
  * `--skip memory_max`, a named test exclusion (the c-xs `test_http_client` shape)
  * wabt 1.0.41 / `wast2json`, a vendored-asset assumption keyed to spectest.rs
  * the workspace prune's literal `rm -rf crates fuzz` and Cargo.toml edits

The digests below were measured against the commit BEFORE the harness became
plan-driven. They are the safety net for an already-proven task: as long as they
hold, rust-spacewasm's shipped bytes did not move, and c/cpp/java/python/go were
not in this slice at all.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import hashlib

import pytest

from taskgen.depplan import DepPlan, DepPlanError, canonicalize, validate
from taskgen.langs import base as B
from taskgen.langs import rust as R

#: sha256 of `render_test_sh` / `measure_test_sh` / `render_solve_sh` for
#: rust-spacewasm as rendered by the commit before `DepPlan.harness` reached
#: rust. `(digest, length)`; the length is carried too so a collision claim has
#: to explain two numbers, not one.
PRE_HARNESS_TEST_SH = (
    '5b004d8d7d4997fa72321dd2764e72cb3b98f3ed9b8330363c9e43a982f2c75a', 4678,
)
PRE_HARNESS_MEASURE_SH = (
    'f1e8168e4d4d69f007b20930e96223280fb6246e911f318f4d047a3c0745d5db', 1165,
)
PRE_HARNESS_SOLVE_SH = (
    '330ded1bf130b0701fee28fad0dee8d7596c4d574c252b1a4bcbaf5872d218fe', 1308,
)

#: The four pre-leakgate RUN blocks joined -- wabt assert, workspace prune,
#: empty-lib-stub vendor, strings leak-assert -- plus the two Dockerfiles they
#: compose into. These are the leak-critical bytes: the strings assert and the
#: vendor stub are what prove no carved symbol reaches a layer.
#:
#: MOVED, deliberately, for two live-diagnosed defects. Both moves are in RUN
#: blocks whose OLD text was wrong for every repository but rust-spacewasm:
#:
#: 1. the strings leak-assert used to grep target/ for the BARE crate name, so
#:    the gate fired on any crate whose name is a substring of a std symbol
#:    (`log` collides with `core::num::int_log10`). It now matches the crate by
#:    rust MANGLING STRUCTURE -- see `leak_symbol_ere` for why that is strictly
#:    more precise rather than more permissive. Shipped render only; the
#:    measure image has never carried the assert.
#: 2. the workspace prune replaced the whole array literal (`members = ["x"]`
#:    -> `members = []`), which matches only a SOLE member. On a multi-member
#:    workspace it silently did nothing while the `rm -rf` beside it still
#:    deleted the directory, and `cargo vendor` died. It now drops the named
#:    element and leaves the others. This one is in BOTH images, which is why
#:    the measure digest moves too.
#:
#: For rust-spacewasm both edits are behaviour-preserving: its `members` and
#: `exclude` arrays hold one entry each, so the element-wise prune writes the
#: same `members = []` / `exclude = []` the replacement did, and its crate name
#: (`spacewasm`) was never a std substring. The rust row of
#: `test_render_dockerfile_golden` moves with these. Nothing else was touched.
PRE_HARNESS_PRELEAK = (
    '59007566c85bb2cf9726a851a3b3432ae4c9e20414efd370895f181fd1d1bdf0', 5281,
)
PRE_HARNESS_MEASURE_DOCKERFILE = (
    '284e9c74ef97db74f2bc0661218332ceaa93b4fe3dc55b68006fe51c00e0bea9', 3956,
)
PRE_HARNESS_SHIPPED_DOCKERFILE = (
    'f54fb16d07a3efd52000bc68419cfd1c2dda73cbdd47a54806a89abf530477ef', 7645,
)

#: The empirically-measured intact denominator for rust-spacewasm, and the four
#: integration targets `tests/*.rs` declares. `92` is what `measure.py` phase 1
#: counts; `4` is what the host-side target scan finds and what the plan must
#: agree with.
EXPECTED = 92
HARNESS_TARGETS = 4


@pytest.fixture
def plugin():
    return R.RustPlugin()


@pytest.fixture
def graded():
    """A graded set that mirrors what `emit._measure_and_pin` would produce.

    rust declares no `grader_fingerprint_globs`, so `fingerprint_sha256` is
    empty and the fingerprint gate renders its no-op comment -- exactly as the
    shipped task does.
    """
    return B.GradedSet(
        expected=EXPECTED, floor_mode='equality', kind='whole-suite',
        test_command=R.TEST_COMMAND,
    )


def digest(text: str) -> tuple[str, int]:
    return hashlib.sha256(text.encode('utf-8')).hexdigest(), len(text)


# ------------------------------------------------------------- byte lock ----


def test_the_spacewasm_test_sh_is_byte_identical(plugin, graded):
    """THE lock. rust-spacewasm ships today; this refactor may not move a byte.

    Rendered with no `dep_plan`, which is the pre-resolution path every caller
    written before the harness was plan-driven takes.
    """
    assert digest(plugin.render_test_sh(graded)) == PRE_HARNESS_TEST_SH


def test_the_spacewasm_measure_script_is_byte_identical(plugin, graded):
    assert digest(plugin.measure_test_sh(graded=graded)) == PRE_HARNESS_MEASURE_SH


def test_the_spacewasm_solve_sh_is_byte_identical(plugin):
    solve = plugin.render_solve_sh(('src/lib.rs', 'src/runtime/mod.rs'))
    assert digest(solve) == PRE_HARNESS_SOLVE_SH


def test_the_spacewasm_preleakgate_blocks_are_byte_identical(plugin):
    """The leak-critical half: vendor stub, strings assert, prune, wabt."""
    env = B.EnvSpec(repo_name='rust-spacewasm')
    blocks = plugin.pre_leakgate_blocks(env)
    assert len(blocks) == 7
    assert digest('\n'.join(blocks)) == PRE_HARNESS_PRELEAK


def test_the_spacewasm_dockerfiles_are_byte_identical(plugin):
    env = B.EnvSpec(repo_name='rust-spacewasm')
    assert digest(plugin.render_measure_dockerfile(env)) == PRE_HARNESS_MEASURE_DOCKERFILE
    assert digest(plugin.render_dockerfile(env)) == PRE_HARNESS_SHIPPED_DOCKERFILE


def test_the_spacewasm_plan_round_trips_through_the_schema():
    """The fixed point: spacewasm's harness is a plan a resolver could return."""
    validate(R.RUST_MEASURE_DEP_PLAN)
    assert canonicalize(R.RUST_MEASURE_DEP_PLAN) == R.RUST_MEASURE_DEP_PLAN
    assert R.HARNESSES == (
        'core_integration', 'regression_integration',
        'statistics_integration', 'custom_page_sizes_integration',
    )
    assert R.MIN_WAST == 90
    assert R.TEST_COMMAND.endswith('-- --skip memory_max')


# ------------------------------------------------------------ the seam ------


def plan_with(**harness) -> DepPlan:
    """rust-spacewasm's plan with named harness slots overridden.

    The argv follows the overridden harness when that harness can be read, and
    falls back to spacewasm's when it cannot -- so a deliberately BROKEN slot is
    still a constructible plan and gets rejected by the gate under test rather
    than by this helper.
    """
    merged = dict(R.RUST_SPACEWASM_HARNESS) | harness
    try:
        argv = R.read_harness(
            DepPlan(
                lang='rust', toolchain_version=R.RUST_VERSION,
                package_manager='cargo', harness=tuple(merged.items()),
            )
        ).graded_argv
    except B.LangError:
        argv = R.RUST_SPACEWASM_HARNESS_SPEC.graded_argv
    return canonicalize(
        DepPlan(
            lang='rust',
            toolchain_version=R.RUST_VERSION,
            package_manager='cargo',
            manifest_files=('Cargo.lock', 'Cargo.toml'),
            build_flags=(
                ('shims_path', R.SHIMS_PATH),
                ('toolchain_source', R.TOOLCHAIN_SOURCE),
            ),
            test_invocation=(('command', argv),),
            harness=tuple(merged.items()),
        )
    )


def test_a_different_harness_moves_the_test_sh_in_exactly_the_right_places(
    plugin, graded,
):
    """Two targets, a different corpus and a different tool -- and nothing else."""
    other = plan_with(
        harness_names=('alpha', 'beta'),
        harness_files=('tests/alpha.rs', 'tests/beta.rs'),
        support_files=('tests/common/mod.rs',),
        exclude_names=(),
        corpus_dir='fixtures',
        corpus_label='golden',
        corpus_min=7,
        corpus_pattern='*.json',
        tool_names=('jq',),
        tool_versions=('1.7',),
        tool_packages=('jq',),
        tool_consumers=('tests/common/mod.rs',),
    )
    plugin.validate_dep_plan(other)
    sh = plugin.render_test_sh(graded, dep_plan=other, integration_targets=2)

    assert 'cargo test --offline --test alpha --test beta' in sh
    assert '--skip' not in sh
    assert "GOLDEN_COUNT=$(find fixtures -name '*.json' | wc -l | tr -d ' ')" in sh
    assert '"${GOLDEN_COUNT}" -ge 7' in sh
    assert 'command -v jq >/dev/null 2>&1 || fail "jq missing from image"' in sh
    assert '[ "${SUMMARIES}" -eq 2 ]' in sh
    assert 'across 2 harnesses' in sh
    for gone in ('core_integration', 'memory_max', 'wast', 'wast2json'):
        assert gone not in sh
    # The language-level grammar is NOT plan-fed and must survive untouched.
    assert "grep -c '^test result:'" in sh
    assert "PASSED=$(sum_field 'passed;')" in sh
    assert '${IGNORED}" -eq 0' in sh


def test_a_repo_with_no_corpus_and_no_tool_gets_neither_block(plugin, graded):
    """THE unseen-repo shape: a plain crate with two integration tests.

    rust-spacewasm's `.wast` corpus and its wabt dependency were the two
    assumptions most specific to it. A crate with neither must render a test.sh
    that mentions neither -- not one carrying a vacuous gate over an empty
    directory.
    """
    bare = plan_with(
        harness_names=('smoke',),
        harness_files=('tests/smoke.rs',),
        support_files=(),
        exclude_names=(),
        corpus_dir='',
        corpus_label='',
        corpus_min=0,
        corpus_pattern='',
        tool_names=(),
        tool_versions=(),
        tool_packages=(),
        tool_consumers=(),
        prune_paths=(),
        prune_manifest_keys=(),
        prune_manifest_entries=(),
    )
    plugin.validate_dep_plan(bare)
    sh = plugin.render_test_sh(graded, dep_plan=bare, integration_targets=1)

    for gone in ('wast', 'wabt', 'find tests -name', 'command -v', 'corpus truncated'):
        assert gone not in sh
    assert '[ "${SUMMARIES}" -eq 1 ]' in sh
    assert 'cargo test --offline --test smoke' in sh

    env = B.EnvSpec(repo_name='bare-crate')
    blocks = plugin.pre_leakgate_blocks_for(env, bare)
    joined = '\n'.join(blocks)
    assert 'wast2json' not in joined
    assert '# Prune out-of-scope' not in joined
    assert 'p.write_text(s)' not in joined
    # The leak proof is NEVER plan-scoped: both survive a harness that asked
    # for nothing at all.
    assert 'cargo vendor' in joined
    assert 'LEAK: rustc artifacts under target/' in joined

    dockerfile = plugin.render_dockerfile(env, dep_plan=bare)
    assert 'wabt' not in dockerfile
    assert 'prune out-of-scope' not in dockerfile
    assert 'LEAK: rustc artifacts under target/' in dockerfile


def test_a_different_prune_renders_a_different_manifest_edit(plugin):
    pruned = plan_with(
        prune_paths=('tools', 'bench'),
        prune_manifest_keys=('members', 'members'),
        prune_manifest_entries=('tools/*', 'bench'),
    )
    plugin.validate_dep_plan(pruned)
    blocks = '\n'.join(
        plugin.pre_leakgate_blocks_for(B.EnvSpec(repo_name='r'), pruned)
    )
    assert 'RUN rm -rf tools bench \\' in blocks
    assert 'for key, entry in [("members", "tools/*"), ("members", "bench")]:' in blocks
    assert 'rm -rf crates' not in blocks
    assert 'crates/*' not in blocks
    assert '"exclude"' not in blocks


@pytest.mark.parametrize(
    'harness, needle',
    [
        ({'harness_names': ()}, 'harness_names'),
        ({'harness_files': ()}, 'harness_files'),
        ({'harness_files': ('tests/a.rs',)}, 'same length'),
        ({'harness_names': 4}, 'not a scalar'),
        ({'harness_files': ('/etc/passwd',) * 4}, 'climbs out of'),
        ({'corpus_min': 'ninety'}, 'a whole number'),
        ({'corpus_label': 'a b'}, 'shell variable stem'),
        ({'tool_versions': ()}, 'same length'),
        ({'prune_manifest_keys': ('deps', 'exclude')}, 'one of exclude, members'),
        ({'prune_manifest_entries': ('crates/*',)}, 'same length'),
    ],
)
def test_a_broken_slot_is_an_actionable_langerror_never_a_crash(
    plugin, graded, harness, needle,
):
    """Every failure names its slot, so the refine loop can repair it."""
    broken = plan_with(**harness)
    with pytest.raises(B.LangError) as caught:
        plugin.validate_dep_plan(broken)
    assert needle in str(caught.value)
    with pytest.raises(B.LangError):
        plugin.render_test_sh(graded, dep_plan=broken)


def test_a_corpus_floor_with_no_corpus_declared_is_refused():
    """A floor over a corpus nobody described is a gate nobody can evaluate."""
    harness = dict(R.RUST_SPACEWASM_HARNESS) | {
        'corpus_dir': '', 'corpus_pattern': '', 'corpus_label': '', 'corpus_min': 90,
    }
    with pytest.raises(B.LangError) as caught:
        R.read_harness(
            DepPlan(
                lang='rust', toolchain_version=R.RUST_VERSION,
                package_manager='cargo', harness=tuple(harness.items()),
            )
        )
    assert 'A floor over nothing' in str(caught.value)


def test_a_metacharacter_in_a_harness_token_never_reaches_the_renderer():
    """The schema gate, not the plugin, is what stops a shell fragment."""
    with pytest.raises(DepPlanError):
        validate(
            DepPlan(
                lang='rust', toolchain_version=R.RUST_VERSION, package_manager='cargo',
                harness=(('harness_names', ('a; rm -rf /',)),),
            )
        )


def test_the_harness_reaches_the_digest_and_the_lock_key():
    """A plan that grades a different suite must not share a digest."""
    from taskgen import depplan

    other = plan_with(
        harness_names=('alpha',), harness_files=('tests/alpha.rs',),
    )
    assert depplan.dep_plan_digest(other) != depplan.dep_plan_digest(
        R.RUST_MEASURE_DEP_PLAN
    )
    assert depplan.env_lock_key('t', 'i', other) != depplan.env_lock_key(
        't', 'i', R.RUST_MEASURE_DEP_PLAN
    )
    assert '"harness_names":["alpha"]' in depplan.to_canonical_json(other)


# -------------------------------------------- the under-enumeration guard ---


def test_the_guard_refuses_a_plan_that_drops_a_target(plugin):
    """THE bug this exists for: a plan describing 3 of 4 targets.

    A shrunken denominator is self-consistent at the wrong number, so measure,
    the equality floor and RED/GREEN would all pass while a quarter of the
    suite went ungraded.
    """
    short = plan_with(
        harness_names=('core_integration', 'regression_integration'),
        harness_files=('tests/core_integration.rs', 'tests/regression_integration.rs'),
    )
    plugin.validate_dep_plan(short)
    with pytest.raises(B.LangError) as caught:
        plugin.assert_repo_agrees(
            short,
            integration_targets=R.HARNESSES,
            corpus_count=92,
        )
    message = str(caught.value)
    assert 'statistics_integration' in message
    assert 'custom_page_sizes_integration' in message
    assert 'self-consistent at the wrong number' in message


def test_the_guard_refuses_a_target_the_repo_does_not_declare(plugin):
    invented = plan_with(
        harness_names=(*R.HARNESSES, 'invented_integration'),
        harness_files=(
            *R.RUST_SPACEWASM_HARNESS_SPEC.files, 'tests/invented_integration.rs',
        ),
    )
    with pytest.raises(B.LangError) as caught:
        plugin.assert_repo_agrees(
            invented, integration_targets=R.HARNESSES, corpus_count=92,
        )
    assert 'invented_integration' in str(caught.value)


def test_the_guard_accepts_the_real_spacewasm_plan(plugin):
    plugin.assert_repo_agrees(
        R.RUST_MEASURE_DEP_PLAN, integration_targets=R.HARNESSES, corpus_count=92,
    )


def test_a_declared_corpus_that_is_not_there_is_refused(plugin):
    with pytest.raises(B.LangError) as caught:
        plugin.assert_repo_agrees(
            R.RUST_MEASURE_DEP_PLAN, integration_targets=R.HARNESSES, corpus_count=0,
        )
    assert 'holds no such file' in str(caught.value)


def test_a_corpus_floor_above_the_intact_size_is_refused(plugin):
    with pytest.raises(B.LangError) as caught:
        plugin.assert_repo_agrees(
            R.RUST_MEASURE_DEP_PLAN, integration_targets=R.HARNESSES, corpus_count=12,
        )
    assert 'fails every run' in str(caught.value)


def test_a_command_that_skips_a_declared_target_is_refused(plugin):
    """measure and grade must execute the same set, or the floor is unreachable."""
    mismatched = canonicalize(
        DepPlan(
            lang='rust', toolchain_version=R.RUST_VERSION, package_manager='cargo',
            build_flags=(
                ('shims_path', R.SHIMS_PATH),
                ('toolchain_source', R.TOOLCHAIN_SOURCE),
            ),
            test_invocation=(('command', ('cargo', 'test', '--test', 'core_integration')),),
            harness=R.RUST_SPACEWASM_HARNESS,
        )
    )
    with pytest.raises(B.LangError) as caught:
        plugin.validate_dep_plan(mismatched)
    assert 'regression_integration' in str(caught.value)


# ------------------------------------------- the host-side target scanner ---


def write_crate(tmp_path, manifest: str, *test_files: str):
    (tmp_path / 'Cargo.toml').write_text(manifest)
    for rel in test_files:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
    return tmp_path


BARE = '[package]\nname = "demo"\nversion = "0.1.0"\n'


def test_the_scanner_finds_files_and_main_rs_dirs_but_not_mod_rs(tmp_path):
    """`tests/util/mod.rs` is a helper module; only `main.rs` makes a directory a target."""
    from taskgen.emit import _cargo_integration_targets

    repo = write_crate(
        tmp_path, BARE,
        'tests/alpha.rs', 'tests/beta.rs',
        'tests/suite/main.rs', 'tests/suite/helper.rs',
        'tests/util/mod.rs', 'tests/data/fixture.wast',
    )
    assert _cargo_integration_targets(repo) == ('alpha', 'beta', 'suite')


def test_the_scanner_honours_autotests_false(tmp_path):
    from taskgen.emit import _cargo_integration_targets

    repo = write_crate(
        tmp_path,
        '[package]\nname = "demo"\nversion = "0.1.0"\nautotests = false\n\n'
        '[[test]]\nname = "only"\npath = "tests/only.rs"\n',
        'tests/only.rs', 'tests/ignored.rs',
    )
    assert _cargo_integration_targets(repo) == ('only',)


def test_an_explicit_test_with_a_path_replaces_rather_than_doubles(tmp_path):
    from taskgen.emit import _cargo_integration_targets

    repo = write_crate(
        tmp_path,
        BARE + '\n[[test]]\nname = "renamed"\npath = "tests/alpha.rs"\n',
        'tests/alpha.rs', 'tests/beta.rs',
    )
    assert _cargo_integration_targets(repo) == ('beta', 'renamed')


@pytest.mark.parametrize(
    'manifest, needle',
    [
        ('[workspace]\nmembers = []\n', 'bare workspace manifest'),
        (BARE + 'autotests = "yes"\n', 'not a boolean'),
        (
            BARE + '\n[[test]]\nname = "custom"\nharness = false\n',
            'prints no libtest',
        ),
        (
            BARE + '\n[[test]]\nname = "gated"\nrequired-features = ["extra"]\n',
            'required-features',
        ),
        (BARE + '\n[[test]]\npath = "tests/x.rs"\n', 'declares no string `name`'),
    ],
)
def test_the_scanner_refuses_rather_than_answering_low(tmp_path, manifest, needle):
    """Every construct whose count is not written in the manifest REFUSES.

    Answering low is the failure mode the whole guard exists to prevent, so an
    unknowable count must never be silently rounded down.
    """
    from taskgen.emit import _cargo_integration_targets

    repo = write_crate(tmp_path, manifest, 'tests/x.rs')
    with pytest.raises(B.LangError) as caught:
        _cargo_integration_targets(repo)
    assert needle in str(caught.value)


def test_the_scanner_matches_the_real_spacewasm_checkout():
    """The number that used to be the literal 4, derived from the tree instead."""
    from pathlib import Path

    from taskgen.emit import _cargo_integration_targets

    repo = Path(__file__).resolve().parents[3].parent.parent / (
        'harbor-tasks/repos-src/rust-spacewasm'
    )
    if not (repo / 'Cargo.toml').is_file():
        pytest.skip('rust-spacewasm checkout not present')
    assert set(_cargo_integration_targets(repo)) == set(R.HARNESSES)


# ------------------------------------ measure/graded selection agreement ----
#
# THE REGRESSION GUARD. rust-spacewasm regenerated to a pinned denominator of
# 92 while the graded run counted 93: the measure script preferred the stale
# `graded.test_command` (the plugin class default, which still carried
# rust-spacewasm's `-- --skip memory_max`) while `test.sh` rendered the
# RESOLVED harness, which excludes nothing. Both sides must select units from
# the same plan slots, so the tests below render both scripts from ONE plan and
# compare what they run -- a divergence fails here rather than only at live
# verify, six gates and one Docker build later.


#: A plan whose harness disagrees with the rust-spacewasm default in every slot
#: that affects COUNTING: different targets, and an exclusion the default lacks.
def selection_plan(**overrides):
    base = dict(
        harness_names=('alpha', 'beta'),
        harness_files=('tests/alpha.rs', 'tests/beta.rs'),
        support_files=(),
        exclude_names=('slow_case',),
        corpus_dir='', corpus_label='', corpus_min=0, corpus_pattern='',
        tool_names=(), tool_versions=(), tool_packages=(), tool_consumers=(),
    )
    base.update(overrides)
    return plan_with(**base)


def cargo_line(script: str) -> str:
    """The single `cargo test ...` invocation a rendered script actually runs."""
    lines = [
        line.strip() for line in script.splitlines()
        if line.strip().startswith('cargo test')
    ]
    assert len(lines) == 1, f'expected exactly one cargo invocation, got {lines}'
    return lines[0]


def graded_cargo(plugin, plan, graded) -> str:
    line = cargo_line(plugin.render_test_sh(graded, dep_plan=plan, integration_targets=2))
    return line.split(' > ')[0].strip()


def measure_cargo(plugin, plan, graded) -> str:
    line = cargo_line(plugin.measure_test_sh(graded=graded, dep_plan=plan))
    return line.split(' > ')[0].strip()


def test_measure_and_test_sh_select_units_from_the_same_plan_slots(plugin, graded):
    """THE bug, as an offline assertion: one plan, one selection, both scripts.

    `graded` deliberately still carries `R.TEST_COMMAND` -- rust-spacewasm's
    default, with `--skip memory_max` -- because that is exactly the stale value
    `emit` holds while the measure image runs. Neither script may read it.
    """
    plan = selection_plan()
    plugin.validate_dep_plan(plan)

    assert graded.test_command == R.TEST_COMMAND
    assert measure_cargo(plugin, plan, graded) == graded_cargo(plugin, plan, graded)
    assert measure_cargo(plugin, plan, graded) == (
        'cargo test --offline --test alpha --test beta -- --skip slow_case'
    )
    for stale in ('core_integration', 'memory_max'):
        assert stale not in plugin.measure_test_sh(graded=graded, dep_plan=plan)


@pytest.mark.parametrize(
    'exclusions',
    [(), ('slow_case',), ('slow_case', 'flaky_case')],
)
def test_every_exclusion_reaches_both_scripts_identically(plugin, graded, exclusions):
    """The exclusion list is the slot that broke; pin it across its whole range.

    An exclusion applied to one path and not the other moves the two
    denominators by exactly the number of tests it names.
    """
    plan = selection_plan(exclude_names=exclusions)
    plugin.validate_dep_plan(plan)

    measured, shipped = (
        measure_cargo(plugin, plan, graded), graded_cargo(plugin, plan, graded),
    )
    assert measured == shipped
    for name in exclusions:
        assert f'--skip {name}' in measured
    assert measured.count('--skip') == len(exclusions)


def test_the_harness_target_list_reaches_both_scripts_identically(plugin, graded):
    """The other counting slot: a target named in one script and not the other."""
    plan = selection_plan(
        harness_names=('alpha', 'beta', 'gamma'),
        harness_files=('tests/alpha.rs', 'tests/beta.rs', 'tests/gamma.rs'),
        exclude_names=(),
    )
    plugin.validate_dep_plan(plan)

    measured = measure_cargo(plugin, plan, graded)
    shipped = cargo_line(
        plugin.render_test_sh(graded, dep_plan=plan, integration_targets=3)
    ).split(' > ')[0].strip()
    assert measured == shipped
    for name in ('alpha', 'beta', 'gamma'):
        assert f'--test {name}' in measured


def test_the_spacewasm_default_still_agrees_across_both_scripts(plugin, graded):
    """The shipped repo itself, on the pre-resolution path: 92 must equal 92."""
    assert measure_cargo(plugin, R.RUST_MEASURE_DEP_PLAN, graded) == cargo_line(
        plugin.render_test_sh(
            graded, dep_plan=R.RUST_MEASURE_DEP_PLAN, integration_targets=4,
        )
    ).split(' > ')[0].strip() == R.TEST_COMMAND


# ------------------------------------------------- the wast corpus floor ----


def test_the_corpus_floor_is_the_host_count_not_the_plans_claim(plugin, graded):
    """A plan understating its own corpus must not ship a slack gate.

    spacewasm's resolved plan claimed `corpus_min=70` over a directory holding
    75 files, so five could go missing unnoticed. The host count wins.
    """
    plan = selection_plan(
        corpus_dir='tests/core', corpus_label='wast',
        corpus_min=70, corpus_pattern='*.wast',
    )
    plugin.validate_dep_plan(plan)

    sh = plugin.render_test_sh(
        graded, dep_plan=plan, integration_targets=2, corpus_count=75,
    )
    assert '"${WAST_COUNT}" -ge 75' in sh
    assert '-ge 70' not in sh


def test_the_corpus_floor_falls_back_to_the_plan_when_no_host_count(plugin, graded):
    """The pre-resolution path keeps rust-spacewasm's rendered bytes exactly."""
    sh = plugin.render_test_sh(graded, dep_plan=R.RUST_MEASURE_DEP_PLAN)
    assert f'"${{WAST_COUNT}}" -ge {R.MIN_WAST}' in sh
