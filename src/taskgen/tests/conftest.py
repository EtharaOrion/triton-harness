"""Put HARNESS/src on sys.path so `import taskgen` works from a bare checkout.

Also exposes the frozen dry-run target as fixtures. The frozen target lives in
the TESTS, never in the library -- select.py must stay repo-agnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HARNESS_SRC = Path(__file__).resolve().parents[2]
if str(HARNESS_SRC) not in sys.path:
    sys.path.insert(0, str(HARNESS_SRC))

ROOT = HARNESS_SRC.parents[2]
REPO = ROOT / 'harbor-tasks' / 'repos-src' / 'python-a2a-python'

FROZEN_FILE = 'src/a2a/utils/task.py'
FROZEN_FUNC = 'apply_history_length'
FROZEN_SHA256 = '96e2a55554b30626e34b4b4635db7a0b954c8905fe488f32cab8bf7506ad9820'
#: sha256 of the CARVED (stubbed) file. Unlike FROZEN_SHA256 this one is safe to
#: pin inside the agent-facing image: it identifies the question, not the answer.
FROZEN_STUB_SHA256 = 'cab7cab089659222acc24dea4d6e56a283f22e629cf4e920db32c82fe5fd7b87'
FROZEN_NODEIDS = [
    'tests/utils/test_task.py::TestApplyHistoryLength::test_large_history_length_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_none_config_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_positive_history_length_truncates',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_unset_history_length_returns_full_history',
    'tests/utils/test_task.py::TestApplyHistoryLength::test_zero_history_length_returns_empty_history',
]


@pytest.fixture(scope='session')
def repo() -> Path:
    if not REPO.is_dir():
        pytest.skip(f'frozen dry-run repo not present: {REPO}')
    return REPO


@pytest.fixture(scope='session')
def target(repo):
    """The frozen target, selected through the public select.py API."""
    from taskgen.select import select_target

    return select_target(repo, package_base='src/', file=FROZEN_FILE, func=FROZEN_FUNC)
