"""The DepPlan data layer: closed enums, canonical form, digest, lock key.

Purely in-memory. No docker, no network, no LLM, no filesystem -- if any test
here needs one of those, the module has grown a dependency it must not have.

Two properties carry the weight:

  * CANONICAL FORM IS A FIXED POINT. `canonicalize` is idempotent, so a plan
    that has been through it once is stable, and the digest is a function of the
    environment rather than of how the resolver happened to spell it.
  * THE ALLOWLIST IS CLOSED. Tools come from a per-language set and every token
    is metacharacter-free, so a resolver cannot hand a downstream renderer
    something that changes the shape of a command.
"""

from __future__ import annotations

import json

import pytest

from taskgen import depplan as D


# --------------------------------------------------------------- fixtures ---


def python_plan(**overrides) -> D.DepPlan:
    """A realistic `uv sync` python environment."""
    fields = {
        'lang': 'python',
        'toolchain_version': '3.12.7',
        'package_manager': 'uv',
        'manifest_files': ('pyproject.toml', 'uv.lock'),
        'apt_packages': (),
        'install_commands': (D.InstallCommand('uv', ('sync', '--frozen')),),
        'build_flags': (('editable', False),),
        'test_invocation': (
            ('framework', 'pytest'),
            ('collect_args', ('--collect-only', '-q')),
            ('run_args', ('-q', '--junitxml=junit.xml')),
        ),
        'needs_git_metadata': False,
    }
    fields.update(overrides)
    return D.DepPlan(**fields)


def cpp_plan(**overrides) -> D.DepPlan:
    """A realistic cmake+ninja cpp environment, deps via apt."""
    fields = {
        'lang': 'cpp',
        'toolchain_version': '14.2.0',
        'package_manager': 'cmake',
        'manifest_files': ('CMakeLists.txt',),
        'apt_packages': ('cmake=4.4.2', 'ninja-build', 'g++-14'),
        'install_commands': (
            D.InstallCommand('apt-get', ('install', '-y', '--no-install-recommends',
                                         'ninja-build')),
            D.InstallCommand('cmake', ('-S', '.', '-B', 'Build', '-G', 'Ninja')),
        ),
        'build_flags': (('CMAKE_BUILD_TYPE', 'Release'), ('parallel', 4)),
        'test_invocation': (
            ('framework', 'doctest'),
            ('collect_args', ('--list-test-cases',)),
            ('run_args', ('--reporters=console',)),
        ),
        'needs_git_metadata': True,
    }
    fields.update(overrides)
    return D.DepPlan(**fields)


# ------------------------------------------------------------- (a) fixpoint ---


@pytest.mark.parametrize('make', [python_plan, cpp_plan])
def test_canonicalize_is_idempotent(make):
    once = D.canonicalize(make())
    assert D.canonicalize(once) == once


def test_canonicalize_is_idempotent_on_messy_input():
    messy = cpp_plan(
        manifest_files=('  CMakeLists.txt ', 'CMakeLists.txt', 'src/CMakeLists.txt'),
        apt_packages=('CMake==4.4.2', 'ninja-build', 'cmake=4.4.2'),
        install_commands=(D.InstallCommand('cmake', (' -S ', '.')),),
        build_flags=(('parallel', 4), ('CMAKE_BUILD_TYPE', ' Release ')),
    )
    once = D.canonicalize(messy)
    assert D.canonicalize(once) == once
    assert D.canonicalize(D.canonicalize(once)) == once


# ------------------------------------------------ (b) sorted/deduped/pinned ---


def test_manifest_files_sorted_deduped_and_stripped():
    plan = D.canonicalize(python_plan(
        manifest_files=('uv.lock', ' pyproject.toml', 'pyproject.toml', 'uv.lock'),
    ))
    assert plan.manifest_files == ('pyproject.toml', 'uv.lock')


def test_apt_packages_sorted_deduped_lowercased_and_pin_normalised():
    plan = D.canonicalize(cpp_plan(
        apt_packages=('Ninja-Build', 'cmake==4.4.2', 'cmake=4.4.2', 'G++-14',
                      'ninja-build'),
    ))
    assert plan.apt_packages == ('cmake=4.4.2', 'g++-14', 'ninja-build')


def test_unpinned_apt_package_keeps_its_bare_name():
    plan = D.canonicalize(cpp_plan(apt_packages=('ninja-build',)))
    assert plan.apt_packages == ('ninja-build',)


def test_build_flags_sorted_by_key_and_test_invocation_keyed():
    plan = D.canonicalize(cpp_plan(
        build_flags=(('parallel', 4), ('CMAKE_BUILD_TYPE', 'Release'), ('lto', True)),
    ))
    assert [k for k, _ in plan.build_flags] == ['CMAKE_BUILD_TYPE', 'lto', 'parallel']
    assert [k for k, _ in plan.test_invocation] == [
        'collect_args', 'framework', 'run_args'
    ]


def test_install_command_order_is_preserved_because_it_is_semantic():
    plan = D.canonicalize(cpp_plan())
    assert [c.tool for c in plan.install_commands] == ['apt-get', 'cmake']


# ------------------------------------------------------------ (c) validate ---


def test_valid_plans_pass_validate():
    for plan in (python_plan(), cpp_plan()):
        D.validate(plan)
        D.validate(D.canonicalize(plan))


@pytest.mark.parametrize('tool', ['bash', 'sh', 'curl', 'wget', 'sudo', 'git', 'cargo'])
def test_tool_outside_the_allowlist_is_refused(tool):
    plan = python_plan(install_commands=(D.InstallCommand(tool, ('x',)),))
    with pytest.raises(D.DepPlanError, match='allowlisted tool'):
        D.validate(plan)


@pytest.mark.parametrize('arg', [
    'a|b', 'a;b', 'a&b', '$(curl evil)', '`id`', 'a>b', 'a<b', 'a\\b', 'a\nb',
    '{a}', '(a)',
])
def test_metacharacter_in_an_install_arg_is_refused(arg):
    plan = python_plan(install_commands=(D.InstallCommand('uv', ('sync', arg)),))
    with pytest.raises(D.DepPlanError, match='shell metacharacter'):
        D.validate(plan)


@pytest.mark.parametrize('package', ['cmake;rm -rf /', 'cmake|tee', '$(id)', '`id`'])
def test_metacharacter_in_an_apt_package_is_refused(package):
    with pytest.raises(D.DepPlanError, match='shell metacharacter'):
        D.validate(cpp_plan(apt_packages=(package,)))


def test_metacharacter_in_a_build_flag_value_is_refused():
    with pytest.raises(D.DepPlanError, match='shell metacharacter'):
        D.validate(cpp_plan(build_flags=(('CMAKE_CXX_FLAGS', '-O2 $(id)'),)))


def test_package_manager_outside_the_language_enum_is_refused():
    with pytest.raises(D.DepPlanError, match='cannot be provisioned'):
        D.validate(python_plan(package_manager='cargo'))
    with pytest.raises(D.DepPlanError, match='cannot be provisioned'):
        D.validate(cpp_plan(package_manager='pip'))


def test_unknown_lang_is_refused():
    with pytest.raises(D.DepPlanError, match='unknown lang'):
        D.validate(python_plan(lang='haskell'))
    with pytest.raises(D.DepPlanError, match='unknown lang'):
        D.validate(python_plan(lang='csharp'))


def test_unknown_schema_is_refused():
    with pytest.raises(D.DepPlanError, match='schema'):
        D.validate(python_plan(schema=2))


@pytest.mark.parametrize('version', ['', '   '])
def test_empty_toolchain_version_is_refused(version):
    with pytest.raises(D.DepPlanError, match='toolchain_version is empty'):
        D.validate(python_plan(toolchain_version=version))


def test_empty_package_and_empty_arg_are_refused():
    with pytest.raises(D.DepPlanError, match='apt package is empty'):
        D.validate(cpp_plan(apt_packages=('',)))
    with pytest.raises(D.DepPlanError, match='argument'):
        D.validate(python_plan(install_commands=(D.InstallCommand('uv', ('sync', '')),)))


def test_malformed_apt_pin_is_refused():
    with pytest.raises(D.DepPlanError, match='name=version'):
        D.validate(cpp_plan(apt_packages=('cmake=',)))
    with pytest.raises(D.DepPlanError, match='name=version'):
        D.validate(cpp_plan(apt_packages=('=4.4.2',)))


def test_duplicate_mapping_keys_are_refused_because_json_would_drop_one():
    with pytest.raises(D.DepPlanError, match='more than once'):
        D.validate(cpp_plan(build_flags=(('lto', True), ('lto', False))))
    with pytest.raises(D.DepPlanError, match='more than once'):
        D.validate(cpp_plan(test_invocation=(('framework', 'a'), ('framework', 'b'))))


def test_apt_get_is_allowed_for_every_language():
    for lang in D.LANGS:
        assert D.APT_TOOL in D.TOOL_ALLOWLIST[lang]


# -------------------------------------------------------------- (d) digest ---


def test_digest_ignores_input_ordering_of_canonicalisable_lists():
    a = cpp_plan(
        manifest_files=('CMakeLists.txt', 'src/CMakeLists.txt'),
        apt_packages=('cmake==4.4.2', 'ninja-build', 'g++-14'),
        build_flags=(('CMAKE_BUILD_TYPE', 'Release'), ('parallel', 4)),
        test_invocation=(('framework', 'doctest'), ('run_args', ('-q',))),
    )
    b = cpp_plan(
        manifest_files=('src/CMakeLists.txt', 'CMakeLists.txt', 'CMakeLists.txt'),
        apt_packages=('G++-14', 'ninja-build', 'cmake=4.4.2'),
        build_flags=(('parallel', 4), ('CMAKE_BUILD_TYPE', 'Release')),
        test_invocation=(('run_args', ('-q',)), ('framework', 'doctest')),
    )
    assert D.dep_plan_digest(a) == D.dep_plan_digest(b)
    assert D.to_canonical_json(a) == D.to_canonical_json(b)


def test_digest_changes_when_the_environment_changes():
    base = python_plan()
    assert D.dep_plan_digest(base) != D.dep_plan_digest(
        python_plan(toolchain_version='3.13.0')
    )
    assert D.dep_plan_digest(base) != D.dep_plan_digest(
        python_plan(install_commands=(D.InstallCommand('uv', ('sync',)),))
    )
    assert D.dep_plan_digest(base) != D.dep_plan_digest(
        python_plan(needs_git_metadata=True)
    )


def test_install_command_order_changes_the_digest():
    forward = cpp_plan()
    reversed_plan = cpp_plan(install_commands=tuple(reversed(cpp_plan().install_commands)))
    assert D.dep_plan_digest(forward) != D.dep_plan_digest(reversed_plan)


def test_canonical_json_is_compact_sorted_and_reparsable():
    text = D.to_canonical_json(cpp_plan())
    assert ', ' not in text and '": ' not in text
    payload = json.loads(text)
    assert list(payload) == sorted(payload)
    assert payload['schema'] == D.SCHEMA
    assert payload['build_flags'] == {'CMAKE_BUILD_TYPE': 'Release', 'parallel': 4}
    assert payload['test_invocation']['collect_args'] == ['--list-test-cases']
    assert payload['install_commands'][0]['tool'] == 'apt-get'


def test_digest_of_a_plan_equals_digest_of_its_canonical_form():
    plan = cpp_plan(apt_packages=('CMake==4.4.2', 'ninja-build'))
    assert D.dep_plan_digest(plan) == D.dep_plan_digest(D.canonicalize(plan))


# ------------------------------------------------------------ (e) lock key ---


REPO_SHA = 'a' * 64
IMAGE_DIGEST = 'sha256:' + 'b' * 64


def test_env_lock_key_is_stable_for_identical_inputs():
    plan = python_plan()
    assert (D.env_lock_key(REPO_SHA, IMAGE_DIGEST, plan)
            == D.env_lock_key(REPO_SHA, IMAGE_DIGEST, D.canonicalize(plan)))


@pytest.mark.parametrize('repo,image,plan', [
    ('c' * 64, IMAGE_DIGEST, python_plan()),
    (REPO_SHA, 'sha256:' + 'd' * 64, python_plan()),
    (REPO_SHA, IMAGE_DIGEST, python_plan(toolchain_version='3.13.0')),
])
def test_env_lock_key_changes_when_any_input_changes(repo, image, plan):
    assert (D.env_lock_key(repo, image, plan)
            != D.env_lock_key(REPO_SHA, IMAGE_DIGEST, python_plan()))


def test_env_lock_key_is_a_sha256_hex_digest():
    key = D.env_lock_key(REPO_SHA, IMAGE_DIGEST, python_plan())
    assert len(key) == 64 and set(key) <= set('0123456789abcdef')


def test_lock_key_separates_its_three_inputs():
    """Concatenation without a separator would let a boundary shift go unseen."""
    assert (D.env_lock_key('ab', 'c', python_plan())
            != D.env_lock_key('a', 'bc', python_plan()))
