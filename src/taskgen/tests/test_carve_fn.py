"""Function-level carve: stub the body, keep the file importable, keep the original."""

from __future__ import annotations

import hashlib

import pytest

from taskgen.carve_fn import carve_function

from .conftest import FROZEN_FUNC, FROZEN_SHA256


@pytest.fixture(scope='module')
def carve(repo, target):
    return carve_function(repo, target)


def test_original_is_the_frozen_bytes(carve):
    digest = hashlib.sha256(carve.original_text.encode('utf-8')).hexdigest()
    assert digest == FROZEN_SHA256


def test_stub_replaces_the_body(carve):
    assert 'raise NotImplementedError' in carve.stubbed_text


def test_stub_keeps_the_def_line(carve):
    assert f'def {FROZEN_FUNC}' in carve.stubbed_text


def test_stub_differs_from_original(carve):
    assert carve.stubbed_text != carve.original_text


def test_stub_is_shorter_than_original(carve):
    assert len(carve.stubbed_text) < len(carve.original_text)


def test_stub_still_parses_as_python(carve):
    import ast

    ast.parse(carve.stubbed_text)


def test_stub_drops_the_carved_body(carve):
    body = carve.receipt['carved_body']
    distinctive = [
        line.strip()
        for line in body.splitlines()
        if len(line.strip()) >= 24 and 'NotImplementedError' not in line
    ]
    assert distinctive, 'frozen target has no distinctive body lines to check'
    survivors = {line.strip() for line in carve.stubbed_text.splitlines()}
    assert not [d for d in distinctive if d in survivors]


def test_receipt_records_the_frozen_byte_ranges(carve):
    r = carve.receipt
    assert r['relpath'] == 'src/a2a/utils/task.py'
    assert r['func_name'] == FROZEN_FUNC
    assert (r['func_start_byte'], r['func_end_byte']) == (969, 2070)
    assert (r['body_start_byte'], r['body_end_byte']) == (1061, 2070)
    assert r['original_sha256'] == FROZEN_SHA256


def test_docstring_is_preserved_for_the_prompt(carve):
    assert 'Applies history_length parameter' in carve.docstring


def test_carve_never_writes_to_the_repo(repo, carve):
    digest = hashlib.sha256((repo / 'src/a2a/utils/task.py').read_bytes()).hexdigest()
    assert digest == FROZEN_SHA256
