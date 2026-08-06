"""Ported upstream generation half: the modules that AUTHOR a verifier bundle.

Nothing here evaluates an agent trajectory — the results/eval half (judge, reexec,
mutation, evaluator, differential, trajectory, orchestrate, ...) is not ported, so
this subpackage stays free of the git/worktree and trajectory-reader dependencies.
"""
from __future__ import annotations
