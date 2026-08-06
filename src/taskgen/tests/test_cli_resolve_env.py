"""`--resolve-env` at the cli boundary: who builds the client, and who is told.

Four claims, all provable without a bridge:

  1. the flag REQUIRES `--llm-config`. There is no default proxy for this one.
  2. with it, `emit_all` receives a NON-None resolver built from that config.
     Asserted against a patched `emit_all`, so the wiring is proved without a
     docker build, a model or a network socket anywhere in the call.
  3. a `ResolveRefused` out of the loop prints `REFUSE(reason)`, exits non-zero,
     and leaves NO lock and NO entry behind. That last clause is the one that
     matters: a task shipped with an unvouched-for environment is a task with a
     silent floor of zero, which is exactly what SHIP-or-REFUSE forbids.
  4. an unreachable bridge exits with an operational message, NOT a REFUSE.

Nothing here constructs a real `LiteLLMClient` except the one test that proves
the config is actually read, and even that one only inspects the object.
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


# ------------------------------------------------------- 1. the flag's demand ---


def test_resolve_env_without_llm_config_is_a_clear_cli_error(repo, tmp_path, spy_emit):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv(repo, tmp_path / 'out', '--resolve-env'))
    message = str(exc.value)
    assert '--resolve-env requires --llm-config' in message
    assert '.llm_config' in message
    assert spy_emit == {}, 'emit must not be reached without a resolver'


def test_a_malformed_llm_config_is_named_not_swallowed(repo, tmp_path, spy_emit):
    bad = tmp_path / 'bad.json'
    bad.write_text('{not json', encoding='utf-8')
    with pytest.raises(SystemExit, match='--resolve-env'):
        cli.main(argv(repo, tmp_path / 'out', '--resolve-env', '--llm-config', str(bad)))
    assert spy_emit == {}


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


def test_emit_all_receives_a_non_none_resolver_built_from_the_config(
        repo, tmp_path, llm_config, spy_emit, capsys):
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out', '--resolve-env',
                      '--llm-config', str(llm_config)))

    assert spy_emit['resolve_env'] is True
    resolver = spy_emit['resolver']
    assert resolver is not None
    assert isinstance(resolver, LlmEnvResolver)
    assert resolver.endpoint == CONFIG['base_url']
    assert 'gcc 13.3.0 (aarch64)' in resolver.capabilities

    client = resolver.client
    assert isinstance(client, LiteLLMClient)
    assert client.model == CONFIG['model']
    assert client.api_base == CONFIG['base_url']
    assert CONFIG['api_key'] not in capsys.readouterr().out


def test_the_default_path_passes_no_resolver_at_all(repo, tmp_path, spy_emit):
    """Not one byte of the no-flag path moves: `resolve_env` stays False and the
    resolver stays None, so nothing can be constructed or contacted."""
    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out'))
    assert spy_emit['resolve_env'] is False
    assert spy_emit['resolver'] is None


def test_the_default_path_never_imports_a_model_sdk(repo, tmp_path, spy_emit):
    import sys

    with pytest.raises(SystemExit):
        cli.main(argv(repo, tmp_path / 'out'))
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
        cli.main(argv(repo, tmp_path / 'out', '--resolve-env',
                      '--llm-config', str(llm_config)))

    message = str(exc.value)
    assert '--resolve-env' in message
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
