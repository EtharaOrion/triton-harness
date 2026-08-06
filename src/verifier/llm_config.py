"""`.llm_config` loading — the only place the verifier learns where its LLM lives.

The dotfile is a JSON object (seeded from the proxy's `claude-code-oauth.json`)::

    {"model": "anthropic/claude-opus-4-8", "base_url": "http://127.0.0.1:8765",
     "api_key": "...", "timeout": 600, "num_retries": 2}

The JSON key ``base_url`` is litellm's ``api_base``; ``client_from_config`` does
that mapping so no caller has to remember it. Nothing here reads ambient
credentials or OAuth state: if the file is absent the caller is told to provide
one rather than silently falling back to some other identity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .generators.model_client import LiteLLMClient

DEFAULT_CONFIG_NAME = ".llm_config"
REQUIRED_KEYS = ("model", "base_url", "api_key")


class LLMConfigError(RuntimeError):
    """`.llm_config` is missing, unreadable, or does not describe a usable client."""


def _repo_root() -> Path:
    """The repo root — three parents up from src/verifier/llm_config.py."""
    return Path(__file__).resolve().parents[2]


def load_llm_config(path: str | Path = DEFAULT_CONFIG_NAME) -> dict[str, Any]:
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as e:
        raise LLMConfigError(
            f"cannot read the LLM config at {p}: {e}. The verifier requires an LLM "
            f"config/proxy; write a {DEFAULT_CONFIG_NAME} JSON file with "
            f"{{{', '.join(REQUIRED_KEYS)}}}.") from e
    try:
        cfg = json.loads(raw)
    except ValueError as e:
        raise LLMConfigError(f"malformed LLM config at {p}: not valid JSON ({e})") from e
    if not isinstance(cfg, dict):
        raise LLMConfigError(f"malformed LLM config at {p}: expected a JSON object, "
                             f"got {type(cfg).__name__}")
    missing = [k for k in REQUIRED_KEYS if not str(cfg.get(k) or "").strip()]
    if missing:
        raise LLMConfigError(f"incomplete LLM config at {p}: missing/empty {missing}")
    for key in ("timeout", "num_retries"):
        if cfg.get(key) is not None and not isinstance(cfg[key], (int, float)):
            raise LLMConfigError(f"malformed LLM config at {p}: {key} must be a number, "
                                 f"got {cfg[key]!r}")
    return cfg


def client_from_config(cfg: dict[str, Any]) -> LiteLLMClient:
    timeout = cfg.get("timeout")
    retries = cfg.get("num_retries")
    return LiteLLMClient(
        model=str(cfg["model"]),
        api_base=str(cfg["base_url"]),
        api_key=str(cfg["api_key"]),
        timeout=float(timeout) if timeout is not None else None,
        num_retries=int(retries) if retries is not None else None,
    )


def resolve_config(explicit_path: str | Path | None = None) -> dict[str, Any]:
    """Explicit path > `.llm_config` in the repo root > a loud failure."""
    if explicit_path is not None:
        return load_llm_config(explicit_path)
    root_cfg = _repo_root() / DEFAULT_CONFIG_NAME
    if root_cfg.is_file():
        return load_llm_config(root_cfg)
    raise LLMConfigError(
        f"the verifier requires an LLM config/proxy: no config given and no "
        f"{DEFAULT_CONFIG_NAME} at {root_cfg}. Seed one from the proxy's "
        f"claude-code-oauth.json (it is git-ignored).")
