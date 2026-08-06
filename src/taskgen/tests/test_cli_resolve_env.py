"""Env resolution at the cli boundary: who builds the client, and who is told.

Four claims, all provable without a bridge:

  1. `--llm-config` is DISCOVERED, and a run that cannot discover one fails LOUD
     -- naming both remedies -- at the moment it actually needs a model, which is
     the first ask. It never degrades to the hardcoded environment.
  2. a whole-suite generate hands `emit_all` a NON-None resolver with NO flag at
     all, and that resolver builds itself from the config. Asserted against a
     patched `emit_all`, so the wiring is proved without a docker build, a model
     or a network socket anywhere in the call.
  3. a `ResolveRefused` out of the loop prints `REFUSE(reason)`, exits non-zero,
     and leaves NO lock and NO entry behind. That last clause is the one that
     matters: a task shipped with an unvouched-for environment is a task with a
     silent floor of zero, which is exactly what SHIP-or-REFUSE forbids.
  4. an unreachable bridge exits with an operational message, NOT a REFUSE.

Nothing here constructs a real `LiteLLMClient` except the tests that prove the
config is actually read, and even those only inspect the object.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from taskgen import cli
from taskgen.env_resolver import ResolveRefused
from taskgen.llm_resolver import LlmEnvResolver, ResolverTransportError
from verifier.generators.model_client import LiteLLMClient

CONFIG = {
    'model': 'anthropic/claude-opus-4-8',
    'base_url': 'http://127.0.0.1:8765',
    'api_key': 'sk-bridge-stub-not-a-secret',
}


@pytest.fixture
def llm_config(tmp_path: Path) -> Path:
    p = tmp_path / 'llm.json'
    p.write_text(json.dumps(CONFIG), encoding='utf-8')
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / 'c-repo'
    (root / 'tests').mkdir(parents=True)
    (root / 'Makefile').write_text('all:\n\ttrue\n', encoding='utf-8')
    (root / 'tests' / 'test_a.c').write_text('int main(void){return 0;}\n', encoding='utf-8')
    return root


def argv(repo: Path, out: Path, *extra: str) -> list[str]:
    return [
        'generate', '--repo', str(repo), '--out', str(out),
        '--lang', 'c', '--carve-scope', 'folder',
        '--include', 'src/**', '--delete-whole-file', *extra,
    ]


@pytest.fixture
def spy_emit(monkeypatch):
    """Replaces `emit_all` so the wiring is observable with nothing running."""
    seen: dict = {}

    def fake_emit_all(**kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, 'emit_all', fake_emit_all)
    return seen


# ---------------------------------------- 1. discovery, and the loud refusal ---


@pytest.fixture
def undiscoverable(monkeypatch):
    """No config anywhere: `resolve_config` raises the way a bare repo makes it."""
    import verifier.llm_config as llm_config_mod

    def missing(path=None):
        raise llm_config_mod.LLMConfigError(
            'no config given and no usable .llm_config at /nowhere/.llm_config'
        )

    monkeypatch.setattr(llm_config_mod, 'resolve_config', missing)


def ask(resolver, repo: Path):
    return resolver(lang='c', repo=repo, base_image='harbor-base:local', repair=None)


def test_no_discoverable_config_fails_loud_naming_both_remedies(
        repo, tmp_path, spy_emit, undiscoverable):
    """The governing rule, at the boundary: a cold whole-suite run that cannot
    find a model REFUSES, and the message says what to do about it -- both ways.

    Nothing here degrades: the run never reaches the hardcoded environment, and
    the resolver still went to `emit_all` so the decision belongs to the cold
    path rather than to argument parsing.
    """
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(repo, tmp_path / 'out'))
    assert spy_emit['resolver'] is not None
    assert spy_emit['resolve_env'] is None

    with pytest.raises(SystemExit) as exc:
        ask(spy_emit['resolver'], repo)

    message = str(exc.value)
    assert '--llm-config' in message
    assert '.llm_config/claude-code-oauth.json' in message
    assert '--no-resolve-env' in message
    assert 'refuses rather than silently degrading' in message


def test_the_loud_refusal_is_not_reported_as_a_refuse(
        repo, tmp_path, spy_emit, undiscoverable, capsys):
    """A missing config is an operational failure, not the model declining.

    `ResolveRefused` is an ANSWER and prints `REFUSE(...)`; this is neither, so
    it must not borrow that vocabulary and must not be catchable as one.
    """
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out'))

    with pytest.raises(SystemExit) as exc:
        ask(spy_emit['resolver'], repo)

    assert not isinstance(exc.value, ResolveRefused)
    assert 'REFUSE' not in str(exc.value)
    assert 'REFUSE' not in capsys.readouterr().err


def test_a_malformed_llm_config_is_named_not_swallowed(repo, tmp_path, spy_emit):
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--llm-config', str(bad)))

    with pytest.raises(SystemExit) as exc:
        ask(spy_emit['resolver'], repo)
    assert str(bad) in str(exc.value)


def test_an_unsupported_language_refuses_before_any_client_is_built(repo, tmp_path,
                                                                    llm_config, spy_emit):
    """`--resolve-env --lang python` must fail as a flag error, not by silently
    doing nothing -- and before a config is loaded, so the message is about the
    real problem."""
    with pytest.raises(SystemExit) as exc:
        cli.main([
            'generate', '--repo', str(repo), '--out', str(tmp_path / 'out'),
            '--lang', 'python', '--resolve-env', '--llm-config', str(llm_config),
        ])
    assert 'only supported for c, cpp, java, rust' in str(exc.value)
    assert spy_emit == {}


# --------------------------------------------- 2. a real resolver goes through ---


def test_emit_all_receives_a_non_none_resolver_with_no_flag_at_all(
        repo, tmp_path, llm_config, spy_emit, capsys):
    """Resolution is a step, not a request: a plain whole-suite generate wires a
    real resolver through, built from the discovered config on the first ask."""
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--llm-config', str(llm_config)))

    assert spy_emit['resolve_env'] is None
    deferred = spy_emit['resolver']
    assert deferred is not None

    resolver = deferred.built()
    assert isinstance(resolver, LlmEnvResolver)
    assert resolver.endpoint == CONFIG['base_url']
    assert 'gcc 13.3.0 (aarch64)' in resolver.capabilities

    client = resolver.client
    assert isinstance(client, LiteLLMClient)
    assert client.model == CONFIG['model']
    assert client.api_base == CONFIG['base_url']
    assert CONFIG['api_key'] not in capsys.readouterr().out


def test_the_client_is_not_built_until_something_actually_asks(
        repo, tmp_path, llm_config, spy_emit, monkeypatch):
    """The determinism contract's cli half: a run that never reaches a cold
    resolve -- a warm lock -- builds no client, so it needs no reachable bridge.

    Counted rather than inspected, and counted TWICE on the second half, because
    "built lazily" and "built once per run" are different promises and a refine
    loop asks up to three times.
    """
    built: list[object] = []
    real = cli.build_env_resolver
    monkeypatch.setattr(
        cli, 'build_env_resolver',
        lambda args, **kw: built.append(real(args, **kw)) or built[-1],
    )

    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--llm-config', str(llm_config)))
    deferred = spy_emit['resolver']
    assert isinstance(deferred, cli.DeferredEnvResolver)
    assert built == [], 'wiring a resolver must not build a client'

    assert deferred.built() is deferred.built()
    assert len(built) == 1


def test_a_parser_backed_language_gets_no_resolver_and_no_error(
        repo, tmp_path, spy_emit):
    """python runs no measure phase, so resolution is inapplicable, not off."""
    with pytest.raises(SystemExit):
        cli.main([
            'generate', '--repo', str(repo), '--out', str(tmp_path / 'out'),
            '--lang', 'python',
        ])
    assert spy_emit['resolver'] is None
    assert spy_emit['resolve_env'] is None


def test_the_opt_out_passes_no_resolver_at_all(repo, tmp_path, spy_emit):
    """Not one byte of the `--no-resolve-env` path moves: `resolve_env` is False
    and the resolver is None, so nothing can be constructed or contacted."""
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--no-resolve-env'))
    assert spy_emit['resolve_env'] is False
    assert spy_emit['resolver'] is None


def test_asking_for_both_at_once_is_rejected(repo, tmp_path, spy_emit):
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--resolve-env', '--no-resolve-env'))
    assert spy_emit == {}


def test_no_path_imports_a_model_sdk_before_something_asks(repo, tmp_path, spy_emit):
    """Neither the opt-out nor the DEFAULT path may pull litellm in: the default
    only reaches a model on a cold resolve, and no cold resolve happens here."""
    import sys

    for extra in ((), ('--no-resolve-env',)):
        with pytest.raises(SystemExit):
            cli.main(argv(repo, tmp_path / 'out', *extra))
        assert 'litellm' not in sys.modules


# --------------------------------------------------- 3. REFUSE ships nothing ---


def _out_is_empty(out: Path) -> bool:
    return not out.exists() or not any(out.rglob('*'))


def test_a_refuse_from_the_loop_prints_refuse_and_exits_nonzero(
        repo, tmp_path, llm_config, monkeypatch, capsys):
    reason = 'tests need a live postgres, impossible under --network=none'
    out = tmp_path / 'out'

    def refusing_emit_all(**kwargs):
        raise ResolveRefused(reason)

    monkeypatch.setattr(cli, 'emit_all', refusing_emit_all)
    code = cli.main(argv(repo, out, '--resolve-env', '--llm-config', str(llm_config)))

    assert code != 0
    assert f'REFUSE({reason})' in capsys.readouterr().err
    assert _out_is_empty(out), 'a REFUSE must leave no entry and no lock behind'


def test_a_refuse_writes_no_lock_even_when_the_output_dir_exists(
        repo, tmp_path, llm_config, monkeypatch):
    out = tmp_path / 'out'
    out.mkdir()

    monkeypatch.setattr(
        cli, 'emit_all',
        lambda **kw: (_ for _ in ()).throw(ResolveRefused('unresolvable')),
    )
    assert cli.main(argv(repo, out, '--resolve-env', '--llm-config', str(llm_config))) != 0
    assert list(out.rglob('graded.lock.json')) == []
    assert list(out.rglob('task.toml')) == []


# ------------------------------------------- 4. a dead bridge is not a REFUSE ---


def test_an_unreachable_bridge_exits_loud_and_is_not_reported_as_a_refuse(
        repo, tmp_path, llm_config, monkeypatch, capsys):
    def dead(**kwargs):
        raise ResolverTransportError(
            'the environment resolver could not reach its model. '
            'curl -sS http://127.0.0.1:8765/healthz'
        )

    monkeypatch.setattr(cli, 'emit_all', dead)
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(repo, tmp_path / 'out', '--llm-config', str(llm_config)))

    message = str(exc.value)
    assert 'environment resolution' in message
    assert 'healthz' in message
    assert 'REFUSE' not in capsys.readouterr().err


def test_build_env_resolver_is_the_only_place_a_client_is_made(llm_config):
    """The seam itself: given a config, this returns a callable resolver and
    reaches nothing. Constructing it must not contact the endpoint."""
    @dataclass
    class Args:
        llm_config: str
        lang: str = 'c'

    resolver = cli.build_env_resolver(Args(str(llm_config)), echo=lambda _: None)
    assert callable(resolver)
    assert isinstance(resolver, LlmEnvResolver)


def test_the_config_is_handed_over_to_be_announced_not_announced_here(
        llm_config, capsys):
    """Building a resolver is not asking one. A `--resolve-env` run that reuses
    a pinned lock builds this object and never calls it, so the cli hands the
    model down and lets the first actual ask say so."""
    @dataclass
    class Args:
        llm_config: str
        lang: str = 'c'

    resolver = cli.build_env_resolver(Args(str(llm_config)))

    assert isinstance(resolver, LlmEnvResolver)
    assert resolver.model == CONFIG['model']
    assert 'asking' not in capsys.readouterr().out
