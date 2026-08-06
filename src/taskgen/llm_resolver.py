"""The `EnvResolver` that actually asks a model: gather, prompt, parse.

`env_resolver` owns the prompt and the parse but is deliberately client-free;
`refine` owns the loop but is deliberately IO-free; `resolver_inputs` owns the
walk. This module is the one place those three meet a real transport, and it is
kept small on purpose -- everything it does is arrange existing pieces, so there
is no second copy of the prompt assembly to drift out of sync with the first.

THE ONE JUDGEMENT IT MAKES: what a failure MEANS.

  * a `Refuse` is a property of the REPOSITORY. The model looked at the
    manifests and said this environment cannot be built. That is an ANSWER, it
    is returned as a value, and `refine` turns it into `ResolveRefused`.
  * a transport failure is a property of the MACHINE. The bridge is not
    running, the port is wrong, the socket died mid-stream. Nothing was learned
    about the repo, so answering `Refuse` would be a lie that gets recorded as
    "this repo is unresolvable" and poisons every later run. It raises
    `ResolverTransportError` instead, which no retry path catches.

An EMPTY completion is filed under the second heading, not the first. A bridge
that returns 200 with no body has not refused; it has failed while looking like
it succeeded, which is the failure mode most worth naming loudly.

NO MODEL SDK IS IMPORTED HERE. The client arrives constructed (the cli builds it
from `.llm_config`), so the offline suite drives this module end to end with a
mock and `litellm` stays out of the generate import graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .depplan import DepPlan
from .env_resolver import (
    MAX_RESPONSE_TOKENS,
    SYSTEM_PROMPT,
    Refuse,
    ResolverClient,
    build_user_prompt,
    parse_resolution,
)
from .resolver_inputs import ResolverInputs, gather_resolver_inputs

__all__ = [
    'BRIDGE_HINT',
    'LlmEnvResolver',
    'ResolverTransportError',
]

#: Appended to every transport failure. The single most common cause by far is
#: "the bridge is not up", and a message that names the health check turns a
#: stack trace into an instruction.
BRIDGE_HINT = (
    'This is an OPERATIONAL failure, not a REFUSE: nothing was learned about '
    'the repository, so the run stops rather than recording it as unresolvable. '
    'Check the LLM bridge is running and healthy '
    '(proxy/claude_code_bridge.sh status; curl -sS {endpoint}/healthz) and that '
    "--llm-config's base_url points at it."
)


class ResolverTransportError(RuntimeError):
    """The resolver could not be REACHED. Never a statement about the repo."""


@dataclass(frozen=True)
class LlmEnvResolver:
    """`EnvResolver` bound to a real client, one round trip per call.

    Stateless between calls by construction: the checkout is re-walked every
    time, so a repaired ask sees the same repo the first ask did and the only
    thing that differs between attempt 1 and attempt 2 is the repair block --
    which is the invariant `refine`'s prompt test pins.
    """

    client: ResolverClient
    capabilities: tuple[str, ...] = ()
    required_slots: tuple[str, ...] = ()
    endpoint: str = ''
    echo: Callable[[str], object] = field(default=print)

    def gather(self, *, lang: str, repo: Path, base_image: str) -> ResolverInputs:
        """The leak-safe inputs for one ask. Exposed so a caller can inspect
        exactly what would be sent without sending it."""
        inputs = gather_resolver_inputs(
            repo=Path(repo), lang=lang, capabilities=self.capabilities,
        )
        if inputs.omitted_manifests:
            self.echo(
                f'resolve-env: {len(inputs.omitted_manifests)} manifest(s) over the '
                f'cap were not sent: {", ".join(inputs.omitted_manifests)}'
            )
        return inputs

    def __call__(self, *, lang: str, repo: Path, base_image: str,
                 repair: str | None = None) -> DepPlan | Refuse:
        inputs = self.gather(lang=lang, repo=repo, base_image=base_image)
        user = build_user_prompt(
            lang=lang,
            # The base image is a capability, and the most load-bearing one:
            # principle 2 says pin to the repo's declared toolchain ONLY if the
            # base provides it, which the model cannot judge without knowing
            # which base it is answering about.
            toolchain_capabilities=[*inputs.capabilities, f'base image: {base_image}'],
            manifest_files=dict(inputs.manifest_files),
            test_paths=list(inputs.test_paths),
            framework=inputs.framework,
            # The plugin's own pre-render gate rejects a plan missing one of
            # these, so a model not shown them is being marked against a rubric
            # it was never given.
            required_slots=list(self.required_slots),
            repair=repair,
        )

        try:
            response = self.client.complete(
                SYSTEM_PROMPT, user, max_tokens=MAX_RESPONSE_TOKENS,
            )
        # Bare `Exception` on purpose: a bridge fails as a socket error, a
        # timeout, or an SDK wrapper, and the type we forgot to enumerate is
        # exactly the one that would escape this classification.
        except Exception as exc:
            raise ResolverTransportError(
                f'the environment resolver could not reach its model '
                f'({type(exc).__name__}: {exc}). ' + self._hint()
            ) from exc

        text = getattr(response, 'text', '') or ''
        if not text.strip():
            raise ResolverTransportError(
                'the environment resolver returned an EMPTY completion. A model '
                'that declines answers with a REFUSE disposition; an empty body '
                'is the transport failing while looking like it succeeded. '
                + self._hint()
            )
        return parse_resolution(text, lang)

    def _hint(self) -> str:
        return BRIDGE_HINT.format(endpoint=self.endpoint or 'http://127.0.0.1:8765')
