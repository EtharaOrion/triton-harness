"""Language plugins: one module per language, one shared contract in `base`.

`base.LangPlugin` owns the eight axes and, deliberately, the two renders that
are security invariants rather than language details (`render_solve_sh`,
`render_dockerfile`). Each plugin module calls `register()` at import time;
`get()` imports the module on first use, so nothing here has to know which of
them exist yet.

python and go are imported EAGERLY below because they are Increment 1 and are
the two the registry is expected to be complete for -- `available()` should not
depend on whether someone happened to call `get()` first. java, rust, c, cpp and
csharp stay lazy: they are later increments, and `get()` explains that rather
than raising ImportError.
"""

from __future__ import annotations

from .base import (
    LANGS,
    DepWarmSpec,
    EnvSpec,
    FLOOR_MODES,
    GradedSet,
    LangError,
    LangPlugin,
    MEASURE_KEYS,
    REWARD_KEYS,
    RewardCounts,
    SOLUTION_MOUNT,
    ToolchainSpec,
    available,
    get,
    register,
)
from .c import CPlugin
from .cpp import CppPlugin
from .go import GoPlugin
from .java import JavaPlugin
from .python import PythonPlugin
from .rust import RustPlugin

__all__ = [
    'CPlugin',
    'CppPlugin',
    'DepWarmSpec',
    'GoPlugin',
    'JavaPlugin',
    'PythonPlugin',
    'RustPlugin',
    'EnvSpec',
    'FLOOR_MODES',
    'GradedSet',
    'LANGS',
    'LangError',
    'LangPlugin',
    'MEASURE_KEYS',
    'REWARD_KEYS',
    'RewardCounts',
    'SOLUTION_MOUNT',
    'ToolchainSpec',
    'available',
    'get',
    'register',
]
