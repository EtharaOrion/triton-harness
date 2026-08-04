"""The cpp plugin: whole-suite grading, measured floor, cmake+ninja+g++-14+doctest.

Docker is MOCKED here -- nothing below builds an image. What is asserted is
the TEXT the plugin renders, plus a docker-less exercise of the fingerprint
plumbing. The cpp-Rux task's full docker matrix runs separately (see the
plan's ACCEPTANCE GATE in Section 7 of the delivery brief).

Key cpp-Rux properties this file locks:

  CPP1  whole-suite, no parser. `SELECTOR_KIND['cpp']='whole-suite'` and the
        plugin declares `parser_backed=False`. `emit.plan_carve` takes the
        no-parser branch and produces a graded set built by
        `whole_suite_selection`, not by `derive_graded_set`.
  CPP2  equality floor. doctest reports per-case pass/fail counts in one
        summary line, and a crash mid-suite prints either a truncated summary
        or none at all (both caught ahead of the floor). observed==EXPECTED
        is a real assertion.
  CPP3  fingerprint plumbing. The plugin declares
        `grader_fingerprint_globs=('Tests/**',)`; emit.plan_carve expands
        those against the intact repo, `_graded_spec` sha256s the resulting
        paths, and `fingerprint_gate_block` bakes the per-file checks inline.
  CPP4  no magic repo-numbers in the plugin source. 170 (doctest TEST_CASEs)
        comes from `graded.expected`; ~2900 (assertions) is deliberately not
        threaded (see the plugin docstring's Route (b) rationale).
  CPP5  parallel-4 build. Reference uses unbounded `--parallel`; this plugin
        hardcodes `--parallel 4` to keep the delivered task gradeable at its
        declared 8 GB memory_mb on a big-core aarch64 host.
  CPP6  DOCKERFILE INVARIANTS. `_assert_dockerfile_invariants` runs inside
        render_dockerfile and rejects `repos-src`, `repo-src`, `FROM warm`,
        `git init` (cpp does NOT synthesize git), and any second reference to
        the solution_mount.
"""

from __future__ import annotations

import pytest

from taskgen.langs import base as B
from taskgen.langs import cpp as CPP


@pytest.fixture
def plugin():
    return CPP.CppPlugin()


@pytest.fixture
def graded():
    """A graded set mirroring what emit._measure_and_pin would produce for cpp-Rux.

    expected=170 is the pinned intact denominator (doctest TEST_CASEs). The
    fingerprint dict here is a stand-in for the 312-entry lock emit builds
    host-side; it is enough to exercise `fingerprint_gate_block` without
    needing 312 real files.
    """
    return B.GradedSet(
        expected=170,
        floor_mode='equality',
        kind='whole-suite',
        test_command=CPP.TEST_COMMAND,
        fingerprint_sha256={
            'Tests/Unit/CMakeLists.txt': 'a' * 64,
            'Tests/Unit/Main.cpp': 'b' * 64,
            'Tests/Unit/SemanticTests.cpp': 'c' * 64,
        },
    )


def test_plugin_declares_its_five_axes(plugin):
    assert plugin.name == 'cpp'
    assert plugin.toml_family == 'A'
    assert plugin.floor_mode == 'equality'
    assert plugin.parser_backed is False
    assert plugin.synthesizes_git is False


def test_cpp_uses_equality_floor_not_pinned_denominator(plugin):
    """doctest cases are their own units in one binary; equality is real (not G1)."""
    assert CPP.CppPlugin.floor_mode == 'equality'


def test_plugin_is_registered_under_its_name():
    from taskgen import langs
    assert langs.get('cpp') is not None
    assert langs.get('cpp').name == 'cpp'


def test_test_command_uses_parallel_4_and_not_unbounded_parallel(plugin):
    """The reference `--parallel` (unbounded) OOMs on big-core aarch64 hosts.

    64 concurrent cc1plus at ~1.5 GB each far exceeds any 8 GB cgroup, and
    the shipped task.toml declares 8 GB memory_mb. `--parallel 4` sits at
    ~6 GB and grades correctly.
    """
    assert '--parallel 4' in plugin.test_command
    assert '--parallel\n' not in plugin.test_command


def test_harness_files_are_the_seven_named_reference_files(plugin):
    """The 7 must match the reference test.sh's belt-and-braces existence check."""
    assert plugin.harness_files == (
        'Tests/Unit/CMakeLists.txt',
        'Tests/Unit/Main.cpp',
        'Tests/Unit/SemanticTests.cpp',
        'Tests/Unit/CodegenTests.cpp',
        'Tests/Unit/OptimizerTests.cpp',
        'Tests/Unit/GoldenDiagnosticsTests.cpp',
        'Tests/Unit/ThirdParty/doctest.h',
    )


def test_selector_kind_registers_cpp_as_whole_suite():
    from taskgen import gradedset
    assert gradedset.SELECTOR_KIND['cpp'] == 'whole-suite'


def test_whole_suite_selection_accepts_fingerprint_relpaths():
    from taskgen.gradedset import whole_suite_selection
    sel = whole_suite_selection(
        'cpp', expected=170, test_command=CPP.TEST_COMMAND,
        fingerprint_relpaths=('Tests/Unit/Main.cpp', 'Tests/Unit/SemanticTests.cpp'),
    )
    assert sel.kind == 'whole-suite'
    assert sel.expected == 170
    assert sel.fingerprint_relpaths == (
        'Tests/Unit/Main.cpp', 'Tests/Unit/SemanticTests.cpp',
    )


def test_default_test_globs_covers_capital_tests_dir():
    """cpp-Rux uses Tests/ (capital T); the guard must refuse to carve it."""
    from taskgen.scope import DEFAULT_TEST_GLOBS
    assert DEFAULT_TEST_GLOBS['cpp'] == ('Tests/**',)


def test_stub_phrase_for_cpp_matches_delete_semantics():
    """cpp always carves --delete-whole-file, matching rust/c prose."""
    from taskgen.contexts import stub_phrase, fence
    assert stub_phrase('cpp') == 'the file being deleted outright'
    assert fence('cpp') == 'cpp'


def test_default_stub_body_resolves_for_cpp():
    """cpp is --delete-whole-file always, so the stub is never spliced -- but
    default_stub_body('cpp') must resolve rather than raise (mirrors rust/c)."""
    from taskgen.carve_file import default_stub_body
    body = default_stub_body('cpp')
    assert body.startswith('{')
    assert body.endswith('}')


def test_toolchain_uses_harbor_base(plugin):
    tc = plugin.toolchain_spec()
    assert tc.base_image == 'harbor-base:local'
    assert tc.workdir == B.WORKDIR
    assert tc.env['CC'] == 'gcc-14'
    assert tc.env['CXX'] == 'g++-14'


def test_toolchain_installs_ninja_g14_and_clang_from_apt(plugin):
    """cpp is the first plugin whose install actually apt-installs at build time.

    harbor-base lacks cmake 4.4, g++-14, ninja and clang; all four have to be
    present before the leak gate runs, so they land in a graded-image layer.
    """
    tc = plugin.toolchain_spec()
    block = tc.install_block
    assert 'apt-get install' in block
    assert 'ninja-build' in block
    assert 'g++-14' in block
    assert 'clang' in block
    assert 'apt.kitware.com' in block
    assert 'cmake' in block


def test_toolchain_uses_update_alternatives_for_c14_default(plugin):
    """A plain `g++`/`cc` must pick 14 (not the shipped 13) for anything that
    probes the compiler via the generic name."""
    block = plugin.toolchain_spec().install_block
    assert 'update-alternatives' in block
    assert '/usr/bin/gcc-14' in block


def test_dep_warm_is_empty(plugin):
    """Deps arrive via apt at build time; no separate warm stage."""
    dw = plugin.dep_warm_spec()
    assert dw.copy_paths == ()
    assert dw.stage_block == ''


def test_no_extra_ctx_assets(plugin):
    """Unlike rust (wabt tarball), cpp ships nothing extra."""
    assert plugin.extra_ctx_assets() == ()


def test_no_pre_leakgate_blocks(plugin):
    """Everything cpp needs lands in the toolchain install (before the gate),
    so no additional pre-leakgate blocks are required."""
    env = B.EnvSpec(repo_name='cpp-Rux')
    assert plugin.pre_leakgate_blocks(env) == ()


def test_render_test_sh_bakes_expected_from_graded(plugin, graded):
    """170 is threaded from graded.expected -- NOT a literal in the plugin source."""
    script = plugin.render_test_sh(graded)
    assert 'EXPECTED=170' in script


def test_render_test_sh_uses_parallel_4_not_unbounded(plugin, graded):
    """The OOM-safe build parallelism must appear in the rendered grader."""
    script = plugin.render_test_sh(graded)
    assert 'cmake --build Build --config Release --parallel 4' in script
    assert 'cmake --build Build --config Release --parallel\n' not in script


def test_render_test_sh_bakes_fingerprint_checks(plugin, graded):
    script = plugin.render_test_sh(graded)
    assert 'check_sha256' in script
    assert "check_sha256 'Tests/Unit/CMakeLists.txt' '" + ('a' * 64) + "'" in script
    assert "check_sha256 'Tests/Unit/Main.cpp' '" + ('b' * 64) + "'" in script


def test_render_test_sh_asserts_the_seven_harness_files_exist(plugin, graded):
    script = plugin.render_test_sh(graded)
    for rel in plugin.harness_files:
        assert rel in script
    assert 'graded harness file missing' in script


def test_render_test_sh_removes_stale_build_and_bin_before_configure(plugin, graded):
    """A stale rux-tests binary would let ctest report green with 0 regenerated code."""
    script = plugin.render_test_sh(graded)
    assert 'rm -rf Build Bin' in script


def test_render_test_sh_runs_configure_build_ctest_and_doctest_directly(plugin, graded):
    script = plugin.render_test_sh(graded)
    assert 'cmake -S . -B Build -G Ninja' in script
    assert '-DRUX_WERROR=ON' in script
    assert '-DRUX_BUILD_TESTS=ON' in script
    assert 'ctest --test-dir Build --output-on-failure -C Release' in script
    assert '"${TEST_BIN}" > "${DOCTEST_LOG}"' in script


def test_render_test_sh_gates_ctest_registered_exactly_one(plugin, graded):
    """Rux registers exactly ONE ctest (`UnitTests`); anything else is tampering."""
    script = plugin.render_test_sh(graded)
    assert '"${CTEST_RAN}" -eq 1' in script


def test_render_test_sh_gates_ctest_pct_at_100(plugin, graded):
    """The one registered ctest must be green -- 100% is the honest gate."""
    script = plugin.render_test_sh(graded)
    assert '"${CTEST_PCT:-0}" -eq 100' in script


def test_render_test_sh_gates_nassert_strictly_positive(plugin, graded):
    """Structural anti-gaming: a stripped-out REQUIRE suite is caught here even
    before the fingerprint (though the fingerprint catches it too)."""
    script = plugin.render_test_sh(graded)
    assert '"${NASSERT}" -gt 0' in script


def test_render_test_sh_does_not_hardcode_repo_magic_numbers(plugin, graded):
    """170/2900/312 must NOT appear as bash-substantive literals in the source.

    170 flows in via `EXPECTED=170` from graded.expected (see fail_closed_preamble),
    and may appear in doc comments describing that value. What must NOT happen is
    a second bash literal like `[ "$CASES" -eq 170 ]` -- that would be a magic
    number in the plugin source. This test filters out comment lines before
    checking. 2900 (assertions) and 312 (fingerprint count) must not appear at all.
    """
    script = plugin.render_test_sh(graded)
    non_comment = '\n'.join(
        ln for ln in script.splitlines() if not ln.lstrip().startswith('#')
    )
    assert non_comment.count('170') == 1, (
        f'expected exactly one bash-literal 170 (EXPECTED= from graded.expected); '
        f'got {non_comment.count("170")}'
    )
    assert 'EXPECTED=170' in non_comment
    assert '2900' not in script
    assert '312' not in script


def test_render_test_sh_emits_five_key_schema(plugin, graded):
    """The 5 canonical keys must be present and in a shape verify.py can read."""
    script = plugin.render_test_sh(graded)
    for key in B.REWARD_KEYS:
        assert f'"{key}"' in script


def test_render_test_sh_extends_reward_json_with_doctest_extras(plugin, graded):
    """Reference grader reports assertion tallies + failure counts; verify.py
    ignores extras but downstream telemetry uses them."""
    script = plugin.render_test_sh(graded)
    assert 'assertions_total' in script
    assert 'assertions_passed' in script
    assert 'assertions_failed' in script
    assert 'cases_failed' in script
    assert 'ctest_pct' in script
    assert 'doctest_exit' in script


def test_render_test_sh_starts_with_zero_and_ends_with_exit_zero(plugin, graded):
    """Fail-closed: the zero lands first and the script always exits 0."""
    script = plugin.render_test_sh(graded)
    assert 'emit 0.0 0 "${EXPECTED}" 0.0' in script
    assert script.rstrip().endswith('exit 0')


def test_render_test_sh_binary_requires_doctest_exit_zero(plugin, graded):
    """A case that aborts a REQUIRE mid-way lowers the assertion count without
    lowering the case count; only doctest exit==0 proves the whole binary ran
    to a natural end."""
    script = plugin.render_test_sh(graded)
    assert '"${DOCTEST_STATUS}" -ne 0' in script


def test_measure_test_sh_builds_and_reports_case_count(plugin):
    """Phase 1 must configure, build --parallel 4, run rux-tests, and grep the
    doctest `test cases: N` summary (grep pattern uses escaped `\\[doctest\\]`)."""
    script = plugin.measure_test_sh()
    assert 'cmake -S . -B Build -G Ninja' in script
    assert 'cmake --build Build --config Release --parallel 4' in script
    assert 'Bin/Tests/Unit/rux-tests' in script
    assert 'test cases:' in script
    assert '\\[doctest\\]' in script
    assert 'measure "${CASES}"' in script


def test_measure_test_sh_is_floor_free(plugin):
    """Phase 1 asserts nothing about the count -- it just reports it."""
    script = plugin.measure_test_sh()
    non_comment = '\n'.join(
        ln for ln in script.splitlines() if not ln.lstrip().startswith('#')
    )
    assert 'EXPECTED' not in non_comment
    assert 'floor_gate_block' not in non_comment
    assert 'fail "' not in non_comment


def test_measure_test_sh_uses_parallel_4(plugin):
    """Measure runs on the SAME host; unbounded --parallel would OOM there too."""
    script = plugin.measure_test_sh()
    assert '--parallel 4' in script


def test_solve_sh_restores_carved_files_and_wipes_build(plugin):
    """Oracle mounts the carved tree at run time, restores byte-for-byte, and
    wipes Build/ + Bin/ so GREEN rebuilds honestly (mirrors reference solve.sh)."""
    rels = (
        'Compiler/Semantic/SemanticAnalyzer.cpp',
        'Compiler/Ir/Hir/Hir.h',
        'Compiler/CodeGen/X86_64/Encoder.cpp',
    )
    script = plugin.render_solve_sh(rels)
    for rel in rels:
        assert f"restore '{rel}'" in script
    assert '/opt/harbor/solution' in script
    assert 'rm -rf "${REPO}/Build" "${REPO}/Bin"' in script


def test_render_dockerfile_asserts_invariants():
    """A shipped Dockerfile that undoes the leak fix must not render at all.

    Matches `_assert_dockerfile_invariants` semantics: the banned tokens are
    forbidden in INSTRUCTIONS, not in comments.
    """
    plugin = CPP.CppPlugin()
    env = B.EnvSpec(repo_name='cpp-Rux')
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


def test_render_dockerfile_installs_toolchain_before_leak_gate():
    """The apt install lives in a graded-image layer BEFORE leakscan runs, so
    the leak gate scans the installed layers too."""
    plugin = CPP.CppPlugin()
    env = B.EnvSpec(repo_name='cpp-Rux')
    text = plugin.render_dockerfile(env)
    assert text.index('apt-get install') < text.index('leakscan.sh')


def test_render_measure_dockerfile_uses_intact_tree(plugin):
    """Measure image copies the INTACT tree directly (no repo/ prefix) and
    has no leak gate / no tripwire / no carve-receipt assert."""
    env = B.EnvSpec(repo_name='cpp-Rux')
    text = plugin.render_measure_dockerfile(env)
    assert 'NEVER SHIP' in text
    assert 'COPY --from=repoctx . /opt/harbor/repo/' in text
    assert 'leakscan' not in text
    assert 'carve_receipt' not in text


def test_render_measure_dockerfile_includes_toolchain_install(plugin):
    """Same apt install as the shipped image -- measure needs cmake/ninja/g++-14 too."""
    env = B.EnvSpec(repo_name='cpp-Rux')
    text = plugin.render_measure_dockerfile(env)
    assert 'ninja-build' in text
    assert 'g++-14' in text
    assert 'apt.kitware.com' in text


def test_plan_carve_expands_grader_fingerprint_globs(tmp_path):
    """A whole-suite plugin's globs are host-expanded against the intact repo.

    For cpp-Rux this expands to every file under Tests/** (312 in the real repo).
    """
    from taskgen.emit import plan_carve

    repo = tmp_path / 'repo'
    (repo / 'Compiler' / 'Semantic').mkdir(parents=True)
    (repo / 'Tests' / 'Unit' / 'ThirdParty').mkdir(parents=True)

    (repo / 'CMakeLists.txt').write_text('project(fake)\n')
    (repo / 'Compiler' / 'Semantic' / 'A.cpp').write_text('int a;\n')
    (repo / 'Compiler' / 'Semantic' / 'A.h').write_text('#pragma once\n')
    (repo / 'Tests' / 'Unit' / 'Main.cpp').write_text('int main(void) { return 0; }\n')
    (repo / 'Tests' / 'Unit' / 'SemanticTests.cpp').write_text('// TEST_CASE\n')
    (repo / 'Tests' / 'Unit' / 'ThirdParty' / 'doctest.h').write_text('// doctest\n')

    out = tmp_path / 'out'
    plan = plan_carve(
        repo, out, lang='cpp', carve_scope='folder',
        include=('Compiler/**',), delete_whole_file=True,
    )
    fps = plan.graded.fingerprint_relpaths
    assert 'Tests/Unit/Main.cpp' in fps
    assert 'Tests/Unit/SemanticTests.cpp' in fps
    assert 'Tests/Unit/ThirdParty/doctest.h' in fps
    assert not any(fp.startswith('Compiler/') for fp in fps)
    assert 'CMakeLists.txt' not in fps
