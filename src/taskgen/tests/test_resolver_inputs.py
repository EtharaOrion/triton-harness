"""`resolver_inputs`: what the model may see, and the proof of what it may not.

The interesting assertions here are NEGATIVE. A gatherer that returns the right
manifests is easy; a gatherer that provably never opens a test file is the
benchmark's integrity, so most of this module is built around a sentinel string
planted inside sources and tests and then hunted for in everything the gatherer
produced. `test_no_source_or_test_body_reaches_the_output` is the one that would
make the whole `--resolve-env` feature unshippable if it broke.

Every test builds its own checkout in `tmp_path`, except the two that read the
real c-xs tree and skip when it is not on this machine -- the suite's fixtures
live inside the worktree by policy, so an external checkout may only ever ADD
coverage, never gate it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taskgen import resolver_inputs as RI
from taskgen.env_resolver import build_user_prompt
from taskgen.langs import base as B
from taskgen.langs import c as C

#: Planted in every source and test body a fixture writes. If this string is
#: reachable from a `ResolverInputs`, a body leaked.
SENTINEL = 'XS_SENTINEL_BODY_e3b0c44298fc1c149afbf4c8996fb924'

#: The external c checkout the worked example in the README carves. Optional.
C_XS = Path(__file__).resolve().parents[3] / '..' / '..' / 'harbor-tasks' / 'repos-src' / 'c-xs'


def write(root: Path, rel: str, text: str) -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding='utf-8')
    return p


def blob(inputs: RI.ResolverInputs) -> str:
    """Everything a `ResolverInputs` could possibly carry into a prompt.

    Serialised through json so a leak hiding in a key, a value, a path or the
    framework string is caught by one search rather than four.
    """
    return json.dumps(
        {
            'manifest_files': dict(inputs.manifest_files),
            'test_paths': list(inputs.test_paths),
            'framework': inputs.framework,
            'capabilities': list(inputs.capabilities),
            'omitted': list(inputs.omitted_manifests),
        },
        sort_keys=True,
    )


@pytest.fixture
def c_repo(tmp_path: Path) -> Path:
    """A c checkout shaped like c-xs: one root Makefile, a big tests/ tree, and
    two decoys -- a vendored foreign manifest and a Makefile inside tests/."""
    root = tmp_path / 'c-repo'
    write(root, 'Makefile', 'CC=gcc\ntest-unit:\n\t./run\n')
    write(root, '.github/workflows/ci.yml', 'jobs:\n  build:\n    runs-on: ubuntu\n')
    write(root, 'src/runtime/gc.c', f'void gc_mark(void) {{ /* {SENTINEL} */ }}\n')
    write(root, 'src/runtime/gc_test.c', f'void t(void) {{ {SENTINEL} }}\n')
    write(root, 'tests/unit/test_lexer.c', f'int main(void) {{ {SENTINEL} }}\n')
    write(root, 'tests/conformance/closures.xs', f'print("{SENTINEL}")\n')
    write(root, 'tests/Makefile', f'# {SENTINEL}\nall:\n\t true\n')
    write(root, 'editors/zed/Cargo.toml', '[package]\nname = "zed-xs"\n')
    write(root, 'vendor/sub/Makefile', f'# vendored {SENTINEL}\n')
    return root


def gather(root: Path, lang: str = 'c') -> RI.ResolverInputs:
    return RI.gather_resolver_inputs(repo=root, lang=lang)


# ------------------------------------------------------- the leak boundary ---


def test_no_source_or_test_body_reaches_the_output(c_repo):
    """THE test. Five files carry the sentinel; none of them may be readable."""
    assert SENTINEL not in blob(gather(c_repo))


def test_the_sentinel_hunt_can_actually_fail(c_repo):
    """A negative assertion is worthless if the needle was never findable.

    Planting the sentinel in a RECOGNISED manifest must make the same search
    succeed -- otherwise the test above would pass on a gatherer that returned
    nothing at all.
    """
    write(c_repo, 'Makefile', f'# {SENTINEL}\nall:\n')
    assert SENTINEL in blob(gather(c_repo))


def test_test_bodies_contribute_a_path_and_nothing_else(c_repo):
    inputs = gather(c_repo)
    assert 'tests/unit/test_lexer.c' in inputs.test_paths
    assert 'tests/conformance/closures.xs' in inputs.test_paths
    assert 'src/runtime/gc_test.c' in inputs.test_paths
    assert not any(p in inputs.manifest_files for p in inputs.test_paths)


def test_a_makefile_inside_tests_is_a_path_not_a_manifest(c_repo):
    """The ordering rule: when 'build config' and 'lives under tests/' disagree,
    the rule that does not open the file wins."""
    inputs = gather(c_repo)
    assert 'tests/Makefile' in inputs.test_paths
    assert 'tests/Makefile' not in inputs.manifest_files


def test_a_foreign_nested_manifest_is_not_collected(c_repo):
    """`editors/zed/Cargo.toml` is a real manifest for someone else's build;
    feeding it to a c resolution invites a plan for the wrong toolchain."""
    assert 'editors/zed/Cargo.toml' not in gather(c_repo).manifest_files


def test_vendored_directories_are_never_descended(c_repo):
    inputs = gather(c_repo)
    assert not any(p.startswith('vendor/') for p in inputs.manifest_files)
    assert not any(p.startswith('vendor/') for p in inputs.test_paths)


def test_source_files_are_neither_read_nor_listed(c_repo):
    inputs = gather(c_repo)
    assert 'src/runtime/gc.c' not in inputs.manifest_files
    assert 'src/runtime/gc.c' not in inputs.test_paths


# --------------------------------------------------------- what IS gathered ---


def test_the_expected_manifest_set_and_framework(c_repo):
    inputs = gather(c_repo)
    assert sorted(inputs.manifest_files) == ['.github/workflows/ci.yml', 'Makefile']
    assert inputs.framework == 'make'
    assert 'CC=gcc' in inputs.manifest_files['Makefile']


def test_a_root_manifest_of_another_language_is_still_collected(tmp_path):
    """At the ROOT the whole catalog applies: a `.python-version` in a c repo is
    a fact about how its harness runs, and hiding it would be an opinion."""
    root = tmp_path / 'r'
    write(root, 'Makefile', 'all:\n')
    write(root, '.python-version', '3.12\n')
    assert sorted(gather(root).manifest_files) == ['.python-version', 'Makefile']


def test_nested_manifests_of_the_asked_language_are_collected(tmp_path):
    root = tmp_path / 'r'
    write(root, 'pyproject.toml', '[project]\nname="a"\n')
    write(root, 'packages/b/pyproject.toml', '[project]\nname="b"\n')
    assert sorted(gather(root, 'python').manifest_files) == [
        'packages/b/pyproject.toml', 'pyproject.toml',
    ]


def test_manifests_below_the_depth_cap_are_dropped(tmp_path):
    root = tmp_path / 'r'
    write(root, 'pyproject.toml', '[project]\n')
    deep = '/'.join(['a'] * RI.MAX_MANIFEST_DEPTH) + '/pyproject.toml'
    write(root, deep, '[project]\n')
    assert list(gather(root, 'python').manifest_files) == ['pyproject.toml']


def test_requirements_globs_are_manifests(tmp_path):
    root = tmp_path / 'r'
    write(root, 'requirements.txt', 'a==1\n')
    write(root, 'requirements-dev.txt', 'b==2\n')
    assert sorted(gather(root, 'python').manifest_files) == [
        'requirements-dev.txt', 'requirements.txt',
    ]


# -------------------------------------------------------------- determinism ---


def test_two_gathers_of_one_checkout_are_identical(c_repo):
    first, second = gather(c_repo), gather(c_repo)
    assert blob(first) == blob(second)
    assert list(first.manifest_files) == sorted(first.manifest_files)
    assert list(first.test_paths) == sorted(first.test_paths)


def test_crlf_is_normalised_so_a_checkout_setting_cannot_move_the_prompt(tmp_path):
    root = tmp_path / 'r'
    (root).mkdir()
    (root / 'Makefile').write_bytes(b'all:\r\n\ttrue\r\n')
    assert gather(root).manifest_files['Makefile'] == 'all:\n\ttrue\n'


def test_an_oversized_manifest_is_truncated_and_says_so(tmp_path):
    root = tmp_path / 'r'
    write(root, 'Makefile', 'x' * (RI.MAX_MANIFEST_BYTES + 500))
    text = gather(root).manifest_files['Makefile']
    assert 'truncated by taskgen' in text
    assert text.count('x') == RI.MAX_MANIFEST_BYTES


def test_manifests_over_the_file_cap_are_reported_not_silently_dropped(tmp_path):
    root = tmp_path / 'r'
    for i in range(RI.MAX_MANIFEST_FILES + 3):
        write(root, f'pkg{i:03d}/pyproject.toml', f'[project]\nname="p{i}"\n')
    inputs = gather(root, 'python')
    assert len(inputs.manifest_files) == RI.MAX_MANIFEST_FILES
    assert len(inputs.omitted_manifests) == 3
    assert set(inputs.omitted_manifests).isdisjoint(inputs.manifest_files)


def test_a_missing_checkout_is_a_loud_error(tmp_path):
    with pytest.raises(RI.ResolverInputsError, match='not a repository checkout'):
        gather(tmp_path / 'nope')


# ---------------------------------------------------- framework + capabilities ---


@pytest.mark.parametrize(('lang', 'files', 'expected'), [
    ('c', {'Makefile': 'all:\n'}, 'make'),
    ('c', {'CMakeLists.txt': 'project(x)\n'}, 'ctest'),
    ('c', {}, 'unknown'),
    ('cpp', {'Makefile': 'all:\n'}, 'make'),
    ('go', {'go.mod': 'module x\n'}, 'go test'),
    ('rust', {'Cargo.toml': '[package]\n'}, 'cargo test'),
    ('python', {'pyproject.toml': '[tool.pytest.ini_options]\n'}, 'pytest'),
    ('python', {'setup.py': 'setup()\n'}, 'unittest'),
    ('java', {'pom.xml': '<project/>\n'}, 'maven-surefire'),
    ('java', {'build.gradle': 'plugins {}\n'}, 'gradle'),
])
def test_framework_is_read_off_the_manifests(lang, files, expected):
    assert RI.detect_framework(lang, files, ()) == expected


def test_a_conftest_path_alone_implies_pytest():
    assert RI.detect_framework('python', {}, ('tests/conftest.py',)) == 'pytest'


def test_base_capabilities_come_from_the_plugin_not_the_repo():
    caps = RI.base_capabilities(B.get('c'))
    assert set(C.BAKED_CAPABILITIES) <= set(caps)
    assert 'ENV LC_ALL=C.UTF-8' in caps
    assert list(caps) == sorted(caps)


def test_gathered_inputs_are_accepted_by_the_prompt_builder(c_repo):
    """The gatherer's output must be a legal ARGUMENT set, not merely correct:
    `build_user_prompt` rejects a multi-line 'path' as a smuggled body."""
    inputs = gather(c_repo)
    prompt = build_user_prompt(
        lang='c',
        toolchain_capabilities=list(RI.base_capabilities(B.get('c'))),
        manifest_files=dict(inputs.manifest_files),
        test_paths=list(inputs.test_paths),
        framework=inputs.framework,
    )
    assert SENTINEL not in prompt
    assert 'tests/unit/test_lexer.c' in prompt


# ------------------------------------------------------- the real c-xs tree ---


@pytest.fixture(scope='session')
def c_xs() -> Path:
    if not (C_XS / 'Makefile').is_file():
        pytest.skip(f'the external c-xs checkout is not present: {C_XS}')
    return C_XS


def test_c_xs_yields_its_makefile_and_no_foreign_manifest(c_xs):
    inputs = RI.gather_resolver_inputs(repo=c_xs, lang='c')
    assert sorted(inputs.manifest_files) == ['.github/workflows/ci.yml', 'Makefile']
    assert inputs.framework == 'make'
    assert inputs.omitted_manifests == ()
    assert 'editors/zed/Cargo.toml' not in inputs.manifest_files
    assert len(inputs.test_paths) > 200
    assert 'tests/conformance' in '\n'.join(inputs.test_paths)


def test_c_xs_leaks_no_line_of_any_source_or_test_file(c_xs):
    """The real-tree version of the sentinel hunt.

    Every `.c`/`.h`/`.xs` file in the checkout donates its longest substantial
    line; none of them may appear anywhere in what the gatherer produced. This
    catches a leak the synthetic fixture cannot, because these are the actual
    bodies the benchmark is hiding.
    """
    inputs = RI.gather_resolver_inputs(repo=c_xs, lang='c')
    haystack = blob(inputs)

    needles = 0
    for path in sorted(c_xs.rglob('*')):
        if path.suffix not in ('.c', '.h', '.xs') or not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 60]
        if not lines:
            continue
        needle = max(lines, key=len)
        assert needle not in haystack, f'{path} leaked into the resolver inputs'
        needles += 1

    assert needles > 100, f'only {needles} bodies were checked; the hunt was vacuous'


def test_a_single_file_c_suite_is_seen_as_a_test_path_not_a_manifest(tmp_path):
    """A root `test.c` is the whole suite of a single-file C project.

    Two things at once: the resolver gets a path it can build a harness from
    instead of an empty list it can only refuse on, and the file is guaranteed
    never to be READ -- the test rule runs before the manifest rule, so its
    body cannot reach the model.
    """
    repo = tmp_path / 'repo'
    repo.mkdir()
    (repo / 'Makefile').write_text('all:\n\tgcc test.c\n')
    (repo / 'test.c').write_text('int main(void) { return SECRET_ANSWER; }\n')
    (repo / 'aes.c').write_text('int aes;\n')

    inputs = RI.gather_resolver_inputs(repo=repo, lang='c')
    assert 'test.c' in inputs.test_paths
    assert 'test.c' not in inputs.manifest_files
    assert 'SECRET_ANSWER' not in '\n'.join(inputs.manifest_files.values())


def test_the_new_c_test_names_do_not_leak_into_cpp(tmp_path):
    """cpp was not in this slice; its pattern set must be unchanged."""
    assert 'test.c' not in RI.TEST_NAME_PATTERNS['cpp']
    assert 'tests.c' not in RI.TEST_NAME_PATTERNS['cpp']
