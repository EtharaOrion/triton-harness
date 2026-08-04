"""The c plugin: whole-suite grading, measured floor, no-parser carve path.

Docker is MOCKED here -- nothing below builds an image. What is asserted is
the TEXT the plugin renders, plus a docker-less exercise of the fingerprint,
tls-count and sub-count arithmetic against a synthesised repo. The c-xs
task's docker matrix runs separately (see the plan's ACCEPTANCE GATE).

Key c-xs properties this file locks:

  C1  whole-suite, no parser. `SELECTOR_KIND['c']='whole-suite'` and the
      plugin declares `parser_backed=False`. `emit.plan_carve` takes the
      no-parser branch and produces a graded set built by
      `whole_suite_selection`, not by `derive_graded_set`.
  C2  equality floor. Every ./xs program and every unit _test binary is
      its own process, so a crash in one cannot abort the others (unlike
      Go, spike G1). observed==EXPECTED is a real assertion.
  C3  fingerprint plumbing. The plugin declares
      `grader_fingerprint_globs=('tests/**', 'Makefile')`; emit.plan_carve
      expands those against the intact repo, `_graded_spec` sha256s the
      resulting paths, and `fingerprint_gate_block` bakes the checks inline.
      rust behaviour is preserved: empty globs -> empty fingerprint -> the
      "# no fingerprinted files" placeholder in the shared helper.
  C4  no magic numbers in the plugin source. Sub-counts and tls_count are
      threaded by `emit._c_grader_metadata` from the intact tree.
  C5  DOCKERFILE INVARIANTS. `_assert_dockerfile_invariants` runs inside
      render_dockerfile and rejects `repos-src`, `repo-src`, `FROM warm`,
      `git init` (c does NOT synthesize git), and any second reference to
      the solution_mount.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from taskgen.langs import base as B
from taskgen.langs import c as C


@pytest.fixture
def plugin():
    return C.CPlugin()


@pytest.fixture
def graded():
    """A graded set that mirrors what emit._measure_and_pin would produce for c-xs.

    expected=91 is the pinned intact denominator (18+60+13). The fingerprint
    dict here is a stand-in for the 227-entry lock emit builds host-side; it
    is enough to exercise `fingerprint_gate_block` without needing 227 real
    files.
    """
    return B.GradedSet(
        expected=91,
        floor_mode='equality',
        kind='whole-suite',
        test_command=C.TEST_COMMAND,
        fingerprint_sha256={
            'Makefile': 'a' * 64,
            'tests/conformance/hello.xs': 'b' * 64,
        },
    )


# ------------------------------------------------------------------ axes ----


def test_plugin_declares_its_three_axes(plugin):
    assert plugin.name == 'c'
    assert plugin.toml_family == 'A'
    assert plugin.floor_mode == 'equality'
    assert plugin.parser_backed is False
    assert plugin.synthesizes_git is False


def test_c_uses_equality_floor_not_pinned_denominator(plugin):
    """Every graded unit is its own process; observed==EXPECTED is real."""
    assert C.CPlugin.floor_mode == 'equality'


def test_plugin_is_registered_under_its_name():
    from taskgen import langs
    assert langs.get('c') is not None
    assert langs.get('c').name == 'c'


def test_test_command_is_the_three_hermetic_targets(plugin):
    """Deliberately NOT `make test`: that sweeps in test_http_client.xs."""
    assert plugin.test_command == 'make test-conformance test-regression test-unit'
    assert 'http_client' not in plugin.test_command


def test_unit_names_match_reference_grader(plugin):
    """The 13 names must match the reference test.sh, in order."""
    assert plugin.unit_names == (
        'lexer', 'parser', 'sema', 'value', 'gc', 'utf8', 'bigint',
        'regex', 'msgpack', 'strbuf', 'limits', 'bytecode_buf', 'self',
    )
    assert len(plugin.unit_names) == 13


# --------------------------------------------------------- gradedset wire --


def test_selector_kind_registers_c_as_whole_suite():
    from taskgen import gradedset
    assert gradedset.SELECTOR_KIND['c'] == 'whole-suite'


def test_whole_suite_selection_accepts_fingerprint_relpaths():
    from taskgen.gradedset import whole_suite_selection
    sel = whole_suite_selection(
        'c', expected=91, test_command=C.TEST_COMMAND,
        fingerprint_relpaths=('Makefile', 'tests/conformance/x.xs'),
    )
    assert sel.kind == 'whole-suite'
    assert sel.expected == 91
    assert sel.fingerprint_relpaths == ('Makefile', 'tests/conformance/x.xs')


def test_rust_whole_suite_selection_stays_byte_identical_with_empty_fingerprints():
    """Rust supplies no fingerprint globs; the property must still return ()."""
    from taskgen.gradedset import whole_suite_selection
    sel = whole_suite_selection(
        'rust', expected=92, test_command='cargo test --offline',
    )
    assert sel.fingerprint_relpaths == ()


# ------------------------------------------------------------ toolchain ----


def test_toolchain_uses_harbor_base(plugin):
    tc = plugin.toolchain_spec()
    assert tc.base_image == 'harbor-base:local'
    assert tc.workdir == B.WORKDIR
    assert tc.env['LC_ALL'] == 'C.UTF-8'
    assert tc.env['MAKEFLAGS'] == '-j4'


def test_toolchain_does_not_install_gcc_or_make(plugin):
    """harbor-base already ships them; a duplicate install would slow every build."""
    tc = plugin.toolchain_spec()
    assert 'apt' not in tc.install_block
    assert 'gcc --version' in tc.install_block
    assert 'make --version' in tc.install_block


def test_dep_warm_is_empty(plugin):
    """Every C dep is stdlib (-lm/-lpthread/-ldl), all in libc6-dev."""
    dw = plugin.dep_warm_spec()
    assert dw.copy_paths == ()
    assert dw.stage_block == ''


def test_no_extra_ctx_assets(plugin):
    """Unlike rust (wabt tarball), c ships nothing extra."""
    assert plugin.extra_ctx_assets() == ()


def test_no_pre_leakgate_blocks(plugin):
    """No vendor step, no strings-target scan: nothing carved is ever compiled."""
    env = B.EnvSpec(repo_name='c-xs')
    assert plugin.pre_leakgate_blocks(env) == ()


# --------------------------------------------------------- render_test_sh --


def test_render_test_sh_requires_metadata(plugin, graded):
    """No magic numbers in the plugin source: without threaded metadata it refuses."""
    with pytest.raises(B.LangError, match='tls_count and corpus_counts'):
        plugin.render_test_sh(graded)


def test_render_test_sh_refuses_inconsistent_sub_counts(plugin, graded):
    with pytest.raises(B.LangError, match='do not sum to expected'):
        plugin.render_test_sh(
            graded,
            tls_count=306,
            corpus_counts={'conformance': 10, 'regression': 10, 'unit': 10},
        )


def test_render_test_sh_refuses_wrong_unit_count():
    """Plugin's unit_names has 13 entries; expected sub-count for unit must match."""
    plugin = C.CPlugin()
    graded_92 = B.GradedSet(
        expected=92, floor_mode='equality', kind='whole-suite',
        test_command=C.TEST_COMMAND,
    )
    with pytest.raises(B.LangError, match='unit_names'):
        plugin.render_test_sh(
            graded_92,
            tls_count=306,
            corpus_counts={'conformance': 18, 'regression': 60, 'unit': 14},
        )


def _render_ok(plugin, graded):
    return plugin.render_test_sh(
        graded,
        tls_count=306,
        corpus_counts={'conformance': 18, 'regression': 60, 'unit': 13},
    )


def test_render_test_sh_bakes_expected_and_sub_counts(plugin, graded):
    script = _render_ok(plugin, graded)
    assert 'EXPECT_TOTAL=91' in script
    assert 'EXPECT_CONF=18' in script
    assert 'EXPECT_REG=60' in script
    assert 'EXPECT_UNIT=13' in script


def test_render_test_sh_bakes_tls_count_and_asserts_it(plugin, graded):
    script = _render_ok(plugin, graded)
    assert '"${TLS_NOW}" = "306"' in script
    assert 'src/tls file count changed' in script


def test_render_test_sh_bakes_fingerprint_checks(plugin, graded):
    script = _render_ok(plugin, graded)
    assert 'check_sha256' in script
    assert "check_sha256 'Makefile' '" + ('a' * 64) + "'" in script
    assert "check_sha256 'tests/conformance/hello.xs' '" + ('b' * 64) + "'" in script


def test_render_test_sh_uses_make_clean_and_full_rebuild(plugin, graded):
    """`make clean` + `rm ./xs` + `rm tests/unit/*_test` is the anti-stale gate."""
    script = _render_ok(plugin, graded)
    assert 'make clean' in script
    assert 'rm -f ./xs' in script
    assert 'rm -f tests/unit/*_test' in script
    assert './xs survived make clean' in script


def test_render_test_sh_grades_the_three_corpora_by_program(plugin, graded):
    """The Makefile recipes exit-on-first-fail; the grader re-runs per program."""
    script = _render_ok(plugin, graded)
    assert 'run_corpus() {' in script
    assert './xs "${f}"' in script
    assert 'tests/conformance' in script
    assert 'tests/regression' in script
    assert 'tests/unit/${t}_test' in script


def test_render_test_sh_asserts_exact_ran_counts(plugin, graded):
    """A sweep that visited fewer programs scores 0, not a fraction."""
    script = _render_ok(plugin, graded)
    assert '"${CONF_RAN}" -eq "${EXPECT_CONF}"' in script
    assert '"${REG_RAN}" -eq "${EXPECT_REG}"' in script
    assert '"${UNIT_RAN}" -eq "${EXPECT_UNIT}"' in script


def test_render_test_sh_emits_five_key_schema(plugin, graded):
    """The 5 canonical keys must be present and in a shape verify.py can read."""
    script = _render_ok(plugin, graded)
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in script


def test_render_test_sh_extends_reward_json_with_sub_counts(plugin, graded):
    """The reference grader carries sub-counts; verify.py ignores extras."""
    script = _render_ok(plugin, graded)
    assert 'conformance_passed' in script
    assert 'regression_passed' in script
    assert 'unit_passed' in script


def test_render_test_sh_starts_with_zero_and_ends_with_exit_zero(plugin, graded):
    """Fail-closed: the zero lands first and the script always exits 0."""
    script = _render_ok(plugin, graded)
    assert 'emit 0.0 0 "${EXPECTED}" 0.0' in script
    assert script.rstrip().endswith('exit 0')


def test_render_test_sh_forbids_http_client_leaking_into_graded_globs(plugin, graded):
    """Refuses to grade a corpus that dragged in the network-dependent test."""
    script = _render_ok(plugin, graded)
    assert 'test_http_client' in script


# ---------------------------------------------------- measure_test_sh -----


def test_measure_test_sh_builds_and_counts(plugin):
    """Phase 1 must exercise the same three corpora against the intact tree."""
    script = plugin.measure_test_sh()
    assert 'make -j' in script
    assert 'tests/conformance' in script
    assert 'tests/regression' in script
    assert 'tests/unit/${t}_test' in script
    assert 'measure "${RAN}"' in script


def test_measure_test_sh_is_floor_free(plugin):
    """Phase 1 asserts nothing about the count -- it just reports it."""
    script = plugin.measure_test_sh()
    non_comment = '\n'.join(
        ln for ln in script.splitlines() if not ln.lstrip().startswith('#')
    )
    assert 'EXPECTED' not in non_comment
    assert 'floor_gate_block' not in non_comment
    assert 'fail "' not in non_comment


# ------------------------------------------------------- solve.sh ---------


def test_solve_sh_restores_carved_files(plugin):
    """Oracle mounts the carved tree at run time and restores byte-for-byte."""
    rels = ('src/runtime/interp.c', 'src/runtime/gc.c')
    script = plugin.render_solve_sh(rels)
    assert "restore 'src/runtime/interp.c'" in script
    assert "restore 'src/runtime/gc.c'" in script
    assert '/opt/harbor/solution' in script


# ----------------------------------------------- shipped Dockerfile -------


def test_render_dockerfile_asserts_invariants():
    """A shipped Dockerfile that undoes the leak fix must not render at all.

    Matches `_assert_dockerfile_invariants` semantics: the banned tokens are
    forbidden in INSTRUCTIONS, not in comments (the file necessarily quotes
    the very anti-patterns it warns about).
    """
    plugin = C.CPlugin()
    env = B.EnvSpec(repo_name='c-xs')
    text = plugin.render_dockerfile(env)
    instructions = '\n'.join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith('#')
    )
    assert 'FROM harbor-base:local AS graded' in text
    assert 'repos-src' not in instructions
    assert 'repo-src' not in instructions
    assert 'FROM warm' not in instructions
    assert 'git init' not in instructions
    assert text.count('/opt/harbor/solution') == 1


def test_render_measure_dockerfile_uses_intact_tree(plugin):
    """Measure image copies the INTACT tree directly (no repo/ prefix)."""
    env = B.EnvSpec(repo_name='c-xs')
    text = plugin.render_measure_dockerfile(env)
    assert 'NEVER SHIP' in text
    assert 'COPY --from=repoctx . /opt/harbor/repo/' in text
    assert 'leakscan' not in text
    assert 'carve_receipt' not in text


# ---------------------------------------- emit._c_grader_metadata --------


def test_emit_c_grader_metadata_computes_host_side(tmp_path):
    """No magic numbers: sub-counts and tls_count come from the intact tree."""
    from taskgen.emit import _c_grader_metadata, CarvePlan
    from taskgen.scope import CarveScope

    repo = tmp_path / 'repo'
    (repo / 'tests' / 'conformance').mkdir(parents=True)
    (repo / 'tests' / 'regression').mkdir(parents=True)
    (repo / 'tests' / 'unit').mkdir(parents=True)
    (repo / 'src' / 'tls').mkdir(parents=True)

    for i in range(3):
        (repo / 'tests' / 'conformance' / f'c{i}.xs').write_text('')
    for i in range(5):
        (repo / 'tests' / 'regression' / f'r{i}.xs').write_text('')
    for name in ('lexer', 'parser'):
        (repo / 'tests' / 'unit' / f'{name}_test.c').write_text('')
    for i in range(7):
        (repo / 'src' / 'tls' / f'x{i}.c').write_text('')

    plan = CarvePlan(
        lang='c', scope=CarveScope.FOLDER, repo=repo, repo_name='fake',
        target=None, carve=None, graded=None, staged=None, staging_key='',
    )
    meta = _c_grader_metadata(plan)
    assert meta['tls_count'] == 7
    assert meta['corpus_counts'] == {'conformance': 3, 'regression': 5, 'unit': 2}


# ---------------------------------------- fingerprint globs in emit -------


def test_plan_carve_expands_grader_fingerprint_globs(tmp_path):
    """A whole-suite plugin's globs are host-expanded against the intact repo."""
    from taskgen.emit import plan_carve

    repo = tmp_path / 'repo'
    (repo / 'src' / 'runtime').mkdir(parents=True)
    (repo / 'tests' / 'conformance').mkdir(parents=True)
    (repo / 'tests' / 'unit').mkdir(parents=True)

    (repo / 'Makefile').write_text('all:\n\techo hi\n')
    (repo / 'src' / 'runtime' / 'a.c').write_text('int a;\n')
    (repo / 'src' / 'runtime' / 'b.c').write_text('int b;\n')
    (repo / 'tests' / 'conformance' / 'hello.xs').write_text('hi\n')
    (repo / 'tests' / 'unit' / 'lexer_test.c').write_text('int main(void) { return 0; }\n')

    out = tmp_path / 'out'
    plan = plan_carve(
        repo, out, lang='c', carve_scope='folder',
        include=('src/runtime/**',), delete_whole_file=True,
    )
    fps = plan.graded.fingerprint_relpaths
    assert 'Makefile' in fps
    assert 'tests/conformance/hello.xs' in fps
    assert 'tests/unit/lexer_test.c' in fps
    assert not any(fp.startswith('src/runtime/') for fp in fps)
