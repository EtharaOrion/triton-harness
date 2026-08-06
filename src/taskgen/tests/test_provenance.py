"""T-clone/emit: `[provenance]` records the pin, and is absent when there is none.

`.git` is dropped from the staged tree and from the measure digest on purpose,
so a clone's commit cannot be recovered from the shipped artifact. It has to be
written into taskgen's own metadata or the task is not reproducible from what
was emitted.

THE ABSENCE IS THE OTHER HALF OF THE CONTRACT. A task generated from a plain
`--repo` checkout must be byte-identical to what taskgen emitted before cloning
existed -- `diff -r` between an old and a new run is the regeneration proof, and
a block that appeared unconditionally would break it for every existing task.
"""

from __future__ import annotations

import tomllib

import pytest

from taskgen import templates
from taskgen.emit import emit_all, make_provenance

from .conftest import FROZEN_FILE, FROZEN_FUNC

TOML_KW = dict(
    slug='r__f__no_context', n_conditions=9, func='f', upstream='o/r',
    condition='no_context', language='python', entry_id='id', license='MIT',
    budget=100, tokenizer='chars4', seed=0, target_file='src/m.py',
    target_func='f', graded_tests='[]', carve_scope='function',
    carved_files='[]', docker_image='taskgen-r:local',
)


# ------------------------------------------------------------ the dataclass --


def test_a_plain_checkout_has_no_provenance():
    assert make_provenance(None, None, None) is None
    assert make_provenance(None, 'a' * 40, 'pinned') is None


def test_a_pinned_provenance_keeps_its_commit():
    prov = make_provenance('https://github.com/o/r', 'a' * 40, 'pinned')
    assert (prov.repo_url, prov.commit, prov.clone_kind) == (
        'https://github.com/o/r', 'a' * 40, 'pinned'
    )
    assert prov.pinned


def test_a_floating_provenance_records_an_empty_commit():
    prov = make_provenance('https://github.com/o/r', '', 'floating')
    assert prov.commit == ''
    assert not prov.pinned


def test_a_pinned_provenance_without_a_commit_is_refused():
    """The sha IS the pin; 'pinned' with nothing to pin to is a false claim."""
    with pytest.raises(SystemExit, match='reproducible'):
        make_provenance('https://github.com/o/r', '', 'pinned')


def test_an_unknown_clone_kind_is_refused():
    with pytest.raises(SystemExit, match='unknown clone_kind'):
        make_provenance('https://github.com/o/r', 'a' * 40, 'rebased')


# ------------------------------------------------------------- the template --


def test_task_toml_without_provenance_ends_exactly_as_it_always_did():
    text = templates.render_task_toml(**TOML_KW)
    assert '[provenance]' not in text
    assert text.endswith('gpus = 0\n')


def test_task_toml_with_provenance_parses_back_to_the_recorded_pin():
    text = templates.render_task_toml(
        provenance=templates.render_provenance(
            make_provenance('https://github.com/o/r', 'a' * 40, 'pinned')
        ),
        **TOML_KW,
    )
    doc = tomllib.loads(text)
    assert doc['provenance'] == {
        'repo_url': 'https://github.com/o/r',
        'commit': 'a' * 40,
        'clone_kind': 'pinned',
    }
    assert doc['environment']['docker_image'] == 'taskgen-r:local'


def test_a_url_with_a_quote_is_escaped_rather_than_breaking_the_toml():
    text = templates.render_task_toml(
        provenance=templates.render_provenance(
            make_provenance('file:///tmp/we"ird\\path', 'a' * 40, 'pinned')
        ),
        **TOML_KW,
    )
    assert tomllib.loads(text)['provenance']['repo_url'] == 'file:///tmp/we"ird\\path'


def test_a_control_character_is_refused_rather_than_emitted():
    with pytest.raises(SystemExit, match='control character'):
        templates.render_provenance(
            make_provenance('https://h/o/r\nname = "x"', 'a' * 40, 'pinned')
        )


# ------------------------------------------------------------- end to end ----


def _emit_one(repo, out, **kw):
    entries = emit_all(
        repo=repo, out=out, package_base='src/', file=FROZEN_FILE, func=FROZEN_FUNC,
        context_types=('no_context',), **kw,
    )
    return tomllib.loads((entries[0].path / 'task.toml').read_text())


def test_generate_from_a_plain_repo_emits_no_provenance_block(repo, tmp_path):
    assert 'provenance' not in _emit_one(repo, tmp_path / 'plain')


def test_generate_from_a_clone_records_the_source_and_commit(repo, tmp_path):
    doc = _emit_one(
        repo, tmp_path / 'cloned',
        repo_url='https://github.com/a2aproject/a2a-python',
        commit='c' * 40, clone_kind='pinned',
    )
    assert doc['provenance'] == {
        'repo_url': 'https://github.com/a2aproject/a2a-python',
        'commit': 'c' * 40,
        'clone_kind': 'pinned',
    }
