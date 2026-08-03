"""The nine deterministic context-provisioning bodies.

Five come from the harness `context_type` enum (`src/eval/eval_llm.py`) and four
from its `rag_type` enum (`src/eval/eval_rag.py`):

    no_context   nothing is inlined
    callee_func  full source of every first-party function the target calls
    callee_sig   signatures only of those callees
    in_file      the target's own file, with the target's body stubbed out
    project      whole surviving repo, path-sorted, truncated at the budget
    bm25         Okapi BM25 over ~512-token line chunks
    embedding    TF-IDF + seeded truncated-SVD (LSA) cosine over the same chunks
    mix          reciprocal rank fusion (k=60) of the bm25 and LSA rankings
    repo_coder   one-shot Jaccard overlap of query and chunk token sets

Rendering (block headers, packing, truncation) is deliberately borrowed from
`harbor-tasks/shared/tooling/gen_context.py` so a taskgen entry and a shipped
harbor entry are formatted identically.

DETERMINISM. Every ranking breaks ties on `chunk_id`; every file list is
path-sorted; the LSA index is seeded; the tokenizer is chars4. Nothing consults
a clock, a model or the network.

CORRECTNESS. The corpus is the SURVIVING repo: the target's file appears in its
STUBBED form, so the carved body is not in any corpus by construction. Every
body is re-checked against content tripwires before it is used.

`embedding` is a dense-LSA stand-in, not a pretrained neural retriever -- there
is no model available offline. `repo_coder` is RepoCoder's iteration-0 retrieval
only: the real method re-queries with model predictions, which is neither
deterministic nor LLM-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import _tooling_path  # noqa: F401  (side effect: sys.path + tree-sitter shim)

import sigextract  # noqa: E402
from count_tokens import get_counter  # noqa: E402
from gen_context import (  # noqa: E402
    FOOTER_RESERVE_TOKENS,
    LeakageError,
    assert_no_leakage,
    build_tripwires,
    pack_files,
    render_chunk_block,
    render_file_block,
)
from manifest import Manifest, walk_repo  # noqa: E402
from retrieval import tokenize  # noqa: E402
from retrieval.chunk import TARGET_TOKENS, build_corpus  # noqa: E402

from .ids import CONTEXT_TYPES

__all__ = [
    'CONTEXT_TYPES',
    'ContextInputs',
    'LeakageError',
    'assert_no_leakage',
    'build_body',
    'common_root',
    'fence',
    'stub_phrase',
]

#: Never inlined in any condition -- VCS internals, caches, build detritus and
#: the lockfile. Same list the shipped harbor manifests use.
CONTEXT_EXCLUDE = (
    '.git/**',
    '**/__pycache__/**',
    '**/*.pyc',
    '**/.pytest_cache/**',
    '**/.ruff_cache/**',
    '**/*.egg-info/**',
    'uv.lock',
)

RRF_K = 60

#: How each language's stub reads in prose. Go must never be described as
#: raising NotImplementedError: the prompt has to match the stub the solver
#: actually finds on disk, which for go is a panic (spike gotcha G2).
_STUB_PHRASE = {
    'python': 'raise NotImplementedError',
    'go': 'panic("not implemented")',
    'java': 'throw new UnsupportedOperationException()',
}

_FENCE = {'python': 'python', 'go': 'go', 'java': 'java'}


def stub_phrase(language: str) -> str:
    return _STUB_PHRASE.get(language, 'a stub')


def fence(language: str) -> str:
    return _FENCE.get(language, '')


def _synth_manifest(repo_name: str, relpaths, language: str = 'python') -> Manifest:
    """A Manifest just complete enough for `walk_repo`.

    taskgen has no carve manifest -- the carve is a byte range or a CLI glob,
    not a `*.carve.toml` -- but `walk_repo` is the read-only repository walker we
    want to reuse verbatim (binary sniffing, size cap, symlink refusal, `**`
    glob semantics).
    """
    rels = [Path(r).as_posix() for r in relpaths]
    return Manifest(
        path=Path(f'<synthetic:{repo_name}>'),
        repo=repo_name,
        language=language,
        carve_root=common_root(rels),
        test_command='uv run pytest',
        description='',
        include=rels,
        context_exclude=list(CONTEXT_EXCLUDE),
    )


def common_root(relpaths) -> str:
    parents = [Path(rel).parent for rel in relpaths]
    common = parents[0].parts
    for parent in parents[1:]:
        parts = parent.parts
        keep = 0
        while keep < min(len(common), len(parts)) and common[keep] == parts[keep]:
            keep += 1
        common = common[:keep]
    return '/'.join(common) or '.'


@dataclass
class ContextInputs:
    """Everything the nine builders share, computed once per target."""

    repo: Path
    repo_name: str
    target: object
    carve: object
    nodeids: list[str]
    budget: int
    seed: int
    counter: object
    surviving: list[tuple[str, str]]
    tripwires: dict
    spec: object
    language: str = 'python'
    header_tokens: int = 0
    _corpus: list = field(default=None, repr=False)

    @property
    def target_relpath(self) -> str:
        return self.carve.relpath

    @property
    def carved_relpaths(self) -> tuple[str, ...]:
        return tuple(self.carve.carved_relpaths)

    @property
    def context_budget(self) -> int:
        """Budget left for inlined content once header and footer are paid for."""
        return max(0, self.budget - self.header_tokens - FOOTER_RESERVE_TOKENS)

    @property
    def corpus(self) -> list:
        """~512-token line-aligned chunks of the surviving repo. Built once."""
        if self._corpus is None:
            self._corpus = build_corpus(self.surviving, self.counter, TARGET_TOKENS)
        return self._corpus

    @property
    def query(self) -> str:
        """Retrieval query: signature + docstring + path + callee names."""
        parts = [
            self.carve.signature,
            self.carve.docstring,
            self.target_relpath,
            self.target.qualname,
        ]
        if not self.carve.is_function_scope:
            parts.extend(self.carve.carved_relpaths)
            parts.extend(f.qualname for f in self.carve.carved_functions)
        parts.extend(self.target.callee_qualnames())
        return '\n'.join(p for p in parts if p)

    @classmethod
    def build(cls, repo, repo_name, target, carve, nodeids, budget=100000, seed=0,
              tokenizer='chars4', header_tokens=0, language='python') -> 'ContextInputs':
        from .carve import CarveSet

        repo = Path(repo)
        counter = get_counter(tokenizer)
        if not isinstance(carve, CarveSet):
            carve = CarveSet.from_function_carve(carve, language=language, target=target)
        if target is None:
            target = carve.target_view
        mf = _synth_manifest(repo_name, carve.carved_relpaths, language)

        # The surviving repo = everything on disk, with EVERY carved file swapped
        # for its stub and every deleted file dropped. This is the single place
        # the carve enters the corpus, so no builder can index the answer -- and
        # it has to cover the whole carve set, not just the primary file, or a
        # multi-file carve leaks N-1 answers straight into the retrievers.
        surviving = []
        for rel, text in walk_repo(repo, mf):
            replacement = carve.surviving_text(rel, text)
            if replacement is not None:
                surviving.append((rel, replacement))

        seen = {rel for rel, _ in surviving}
        missing = [
            rel for rel in carve.carved_relpaths
            if rel not in seen and not carve.is_deleted(rel)
        ]
        if missing:
            raise ValueError(
                f'{missing} were excluded by the context walker, so their stubs '
                'never entered the corpus. Refusing to generate.'
            )

        spec = sigextract.extract(carve.original_text, carve.relpath, language)
        # The docstring is the task PROMPT, not the answer: MRG-Bench hands the
        # model the annotation and asks for the body. Whitelisting its lines
        # keeps `build_tripwires` from flagging our own instruction text.
        allowed = set(spec.signature_lines) | set(carve.docstring_lines)
        quotable = sigextract.FileSpec(
            relpath=spec.relpath,
            language=spec.language,
            symbols=spec.symbols,
            approximate=spec.approximate,
            error=spec.error,
            signature_lines=allowed,
        )
        tripwires = build_tripwires(
            [(rel, carve.originals[rel]) for rel in carve.carved_relpaths],
            {carve.relpath: quotable},
            surviving,
        )

        return cls(
            repo=repo,
            repo_name=repo_name,
            target=target,
            carve=carve,
            nodeids=list(nodeids),
            budget=budget,
            seed=seed,
            counter=counter,
            surviving=surviving,
            tripwires=tripwires,
            spec=spec,
            language=language,
            header_tokens=header_tokens,
        )


# --------------------------------------------------------------------------
# Static builders
# --------------------------------------------------------------------------


def _body_no_context(inp: ContextInputs):
    intro = (
        '## Pre-loaded context: none\n\n'
        'No repository content is inlined in this condition. The complete '
        'surviving repository -- including the file the function was carved out '
        'of -- is nonetheless present on disk; read it yourself with the tools '
        'you have. The signature, docstring, test command and reward mechanics '
        'above are all you are given up front.\n'
    )
    return intro, [], {'selection': 'none', 'context_tokens': 0, 'files_eligible': 0}


def _callee_sources(inp: ContextInputs) -> list[tuple[str, str, int, int]]:
    """(relpath, source, start_line, end_line) per callee, path-sorted.

    A callee whose own body the carve removed is dropped. For function scope
    that is the recursive self-reference only -- every other function in the
    target's file still has its body on disk and is legitimate context. For
    file/folder scope it is anything inside the carve set, because those bodies
    are gone and inlining one would hand over part of the answer.
    """
    out = []
    for callee in inp.target.callees:
        rel = Path(callee.file_path).resolve()
        try:
            rel = rel.relative_to(inp.repo.resolve()).as_posix()
        except ValueError:
            rel = Path(callee.file_path).as_posix()
        if inp.carve.carved_body_owner(rel, callee.name):
            continue
        source = callee.get_func()
        start = callee.func_node.start_point[0] + 1
        end = callee.func_node.end_point[0] + 1
        out.append((rel, source, start, end))
    out.sort(key=lambda t: (t[0], t[2], t[3]))
    return out


def _body_callee_func(inp: ContextInputs):
    sources = _callee_sources(inp)
    blocks: list[str] = []
    used = 0
    for rel, source, start, end in sources:
        block = render_file_block(rel, source, note=f'lines {start}-{end}')
        cost = inp.counter.count(block)
        if used + cost > inp.context_budget:
            continue
        blocks.append(block)
        used += cost
    intro = (
        '## Pre-loaded context: callee function bodies\n\n'
        'The full source of every first-party function the carved function calls, '
        'as resolved from this repository\'s import graph and call sites. '
        f'{len(sources)} callee(s) were resolved; {len(blocks)} were inlined within '
        'the token budget. Nothing else from the repository is pre-loaded.\n'
    )
    return intro, blocks, {
        'selection': 'callee function bodies',
        'context_tokens': used,
        'files_eligible': len(sources),
        'callees_resolved': len(sources),
        'callees_inlined': len(blocks),
    }


def _signature_of(source: str, name: str) -> str:
    """`def name(...) -> T: ...` for one function, via the stdlib ast. Exact.

    Methods arrive indented, so the source is dedented before parsing. On a
    syntax error we fall back to the literal declaration text up to the colon,
    which is still signature-only.
    """
    import ast
    import textwrap

    dedented = textwrap.dedent(source)
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        head = dedented.split(':\n', 1)[0]
        return f'{head.strip()}: ...'
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return sigextract._py_func_sig(node)
    head = dedented.split(':\n', 1)[0]
    return f'{head.strip()}: ...'


def _brace_signature_of(source: str) -> str:
    """Everything up to the opening brace, for a brace-delimited language.

    Deliberately textual rather than a second parse: the body is the answer, so
    the only requirement is that nothing after `{` survives, and cutting at the
    first brace guarantees that whatever the declaration looks like.
    """
    head = source.split('{', 1)[0].rstrip()
    return f'{head} {{ ... }}' if head else source.splitlines()[0]


def _body_callee_sig(inp: ContextInputs):
    sources = _callee_sources(inp)
    blocks: list[str] = []
    used = 0
    for rel, source, start, end in sources:
        name = source.split('(', 1)[0].split()[-1] if '(' in source else ''
        sig = (
            _signature_of(source, name) if inp.language == 'python'
            else _brace_signature_of(source)
        )
        block = render_file_block(rel, sig, note=f'signature only, lines {start}-{end}')
        cost = inp.counter.count(block)
        if used + cost > inp.context_budget:
            continue
        blocks.append(block)
        used += cost
    intro = (
        '## Pre-loaded context: callee signatures\n\n'
        'Signatures ONLY of every first-party function the carved function calls. '
        'Bodies, docstrings and default values are withheld -- treat these as the '
        f'contract the carved code is written against. {len(sources)} callee(s) '
        f'were resolved; {len(blocks)} were inlined within the token budget.\n'
    )
    return intro, blocks, {
        'selection': 'callee signatures',
        'context_tokens': used,
        'files_eligible': len(sources),
        'callees_resolved': len(sources),
        'callees_inlined': len(blocks),
    }


def _body_in_file(inp: ContextInputs):
    rel = inp.target_relpath
    if inp.carve.is_deleted(rel):
        intro = (
            f'## Pre-loaded context: containing file `{rel}`\n\n'
            f'`{rel}` was DELETED outright by the carve, so there is no surviving '
            'version of it to inline. Recreate it from its callers, its tests and '
            'the rest of the repository, which is all still on disk.\n'
        )
        return intro, [], {
            'selection': 'containing file (deleted)',
            'context_tokens': 0,
            'files_eligible': 0,
        }

    text = inp.carve.stubbed_text
    stub = stub_phrase(inp.language)
    blocks, stats = pack_files([(rel, text)], inp.counter, inp.context_budget)
    if inp.carve.is_function_scope:
        intro = (
            f'## Pre-loaded context: containing file `{rel}`\n\n'
            'The file the carved function lives in, exactly as it is on disk right '
            f'now: imports, module constants and sibling definitions intact, with '
            f'`{inp.target.qualname}`\'s body replaced by `{stub}`. '
            'Nothing from the rest of the repository is pre-loaded.\n'
        )
    else:
        intro = (
            f'## Pre-loaded context: containing file `{rel}`\n\n'
            'The skeleton of the primary carved file, exactly as it is on disk '
            'right now: imports, module-level code and every signature intact, '
            f'with every function body replaced by `{stub}`. The other '
            f'{len(inp.carved_relpaths) - 1} carved file(s) are on disk in the same '
            'skeletonised form. Nothing else is pre-loaded.\n'
        )
    stats['files_eligible'] = 1
    stats['selection'] = 'containing file (stubbed)'
    return intro, blocks, stats


def _body_project(inp: ContextInputs):
    ordered = sorted(inp.surviving, key=lambda p: p[0])
    blocks, stats = pack_files(ordered, inp.counter, inp.context_budget)
    intro = (
        '## Pre-loaded context: whole project (path-sorted prefix)\n\n'
        'The entire surviving repository concatenated in deterministic path-sorted '
        f'order and truncated at the token budget. {len(ordered)} files were '
        f'eligible; {stats["files_inlined_whole"]} were inlined whole. Coverage is '
        'broad but shallow, and stops wherever the budget runs out in the sort '
        'order -- later paths are absent from this prompt (they remain on disk).\n'
    )
    stats['files_eligible'] = len(ordered)
    stats['selection'] = 'path-sorted project prefix'
    return intro, blocks, stats


# --------------------------------------------------------------------------
# Retrieval builders
# --------------------------------------------------------------------------


def _pack_ranked(inp: ContextInputs, ranked, label: str):
    blocks: list[str] = []
    used = 0
    files_seen: set[str] = set()
    for rank_i, (score, idx) in enumerate(ranked, start=1):
        chunk = inp.corpus[idx]
        block = render_chunk_block(chunk, rank_i, score, label)
        cost = inp.counter.count(block)
        if used + cost > inp.context_budget:
            continue
        blocks.append(block)
        used += cost
        files_seen.add(chunk.relpath)
    return blocks, {
        'selection': label,
        'chunks_retrieved': len(blocks),
        'chunks_in_corpus': len(inp.corpus),
        'chunks_ranked': len(ranked),
        'distinct_files_retrieved': len(files_seen),
        'context_tokens': used,
    }


def _retrieval_outro(stats: dict) -> str:
    return (
        f'\n{stats["chunks_retrieved"]} chunks from '
        f'{stats["distinct_files_retrieved"]} files were retrieved out of '
        f'{stats["chunks_in_corpus"]} in the corpus. Each block below carries its '
        'source path, line range, rank and score. Chunks are fragments -- read the '
        'full file from disk when you need more.\n'
    )


def _rank_bm25(inp: ContextInputs):
    from retrieval.bm25 import BM25Index

    return BM25Index(inp.corpus).score(inp.query)


def _rank_lsa(inp: ContextInputs):
    from retrieval.dense_lsa import LSAIndex

    return LSAIndex(inp.corpus, seed=inp.seed).score(inp.query)


def _body_bm25(inp: ContextInputs):
    blocks, stats = _pack_ranked(inp, _rank_bm25(inp), 'bm25')
    intro = (
        '## Pre-loaded context: BM25 retrieval\n\n'
        f'The surviving repository was split into ~{TARGET_TOKENS}-token '
        'line-aligned chunks and ranked with Okapi BM25 (k1=1.5, b=0.75), queried '
        'with the carved function\'s signature, docstring, path and callee names. '
        'This is lexical retrieval: it matches identifiers and literals, not '
        'meaning.\n'
    )
    return intro + _retrieval_outro(stats), blocks, stats


def _body_embedding(inp: ContextInputs):
    from retrieval.dense_lsa import N_COMPONENTS

    blocks, stats = _pack_ranked(inp, _rank_lsa(inp), 'dense-lsa')
    intro = (
        '## Pre-loaded context: dense (LSA) retrieval\n\n'
        f'The surviving repository was split into ~{TARGET_TOKENS}-token '
        'line-aligned chunks and ranked by cosine similarity in a '
        f'{N_COMPONENTS}-dimensional latent space built with TF-IDF plus truncated '
        'SVD (LSA), queried with the carved function\'s signature, docstring, path '
        'and callee names.\n\n'
        'This **approximates** dense retrieval; it is not a pretrained neural '
        'embedding model. No such model was available in the generation '
        'environment (no HF token, no GPU, no network), so its semantics come '
        'purely from term co-occurrence within this one repository. Expect it to '
        'be **weaker** than a real dense retriever, and read its ranking with that '
        'in mind.\n'
    )
    return intro + _retrieval_outro(stats), blocks, stats


def _rrf(rankings: list[list[tuple[float, int]]], k: int = RRF_K) -> list[tuple[float, int]]:
    """Reciprocal rank fusion. score(d) = sum_lists 1/(k + rank_list(d)).

    Rank-based, so the two retrievers' incomparable score scales (BM25 is
    unbounded, cosine is [-1,1]) never need normalising. Ties break on chunk
    index, which IS the chunk_id, so the fused order is a pure function of the
    inputs.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank_i, (_score, idx) in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank_i)
    out = [(score, idx) for idx, score in fused.items()]
    out.sort(key=lambda p: (-p[0], p[1]))
    return out


def _body_mix(inp: ContextInputs):
    bm25 = _rank_bm25(inp)
    lsa = _rank_lsa(inp)
    blocks, stats = _pack_ranked(inp, _rrf([bm25, lsa]), 'rrf-mix')
    stats['bm25_ranked'] = len(bm25)
    stats['lsa_ranked'] = len(lsa)
    stats['rrf_k'] = RRF_K
    intro = (
        '## Pre-loaded context: hybrid retrieval (BM25 + dense-LSA, RRF)\n\n'
        f'The same ~{TARGET_TOKENS}-token chunks were ranked twice -- once with '
        'Okapi BM25 (lexical) and once with TF-IDF + truncated-SVD cosine (dense '
        f'stand-in) -- and the two rankings fused with reciprocal rank fusion '
        f'(k={RRF_K}): a chunk\'s fused score is the sum over both lists of '
        f'1/({RRF_K} + rank). RRF is rank-based, so the retrievers\' incomparable '
        'score scales never have to be normalised, and a chunk that both methods '
        'rate highly outranks one that only a single method loves.\n'
    )
    return intro + _retrieval_outro(stats), blocks, stats


def _body_repo_coder(inp: ContextInputs):
    q = set(tokenize(inp.query))
    ranked: list[tuple[float, int]] = []
    if q:
        for idx, chunk in enumerate(inp.corpus):
            c = set(tokenize(chunk.text))
            if not c:
                continue
            union = len(q | c)
            score = len(q & c) / union if union else 0.0
            if score > 0.0:
                ranked.append((score, idx))
        ranked.sort(key=lambda p: (-p[0], inp.corpus[p[1]].chunk_id))
    blocks, stats = _pack_ranked(inp, ranked, 'repo-coder-jaccard')
    intro = (
        '## Pre-loaded context: RepoCoder-style similarity retrieval (one-shot)\n\n'
        f'The surviving repository was split into ~{TARGET_TOKENS}-token '
        'line-aligned chunks and ranked by Jaccard overlap between the query\'s '
        'identifier token set and each chunk\'s -- the similarity function '
        'RepoCoder uses for its retrieval step.\n\n'
        'This is RepoCoder\'s **first iteration only**. The published method then '
        're-queries with the model\'s own draft completion and repeats. That loop '
        'needs a model in the generation pipeline, which would make this task '
        'neither deterministic nor reproducible offline, so it is deliberately not '
        'run. Read this as the un-augmented retrieval baseline.\n'
    )
    return intro + _retrieval_outro(stats), blocks, stats


_BUILDERS = {
    'no_context': _body_no_context,
    'callee_func': _body_callee_func,
    'callee_sig': _body_callee_sig,
    'in_file': _body_in_file,
    'project': _body_project,
    'bm25': _body_bm25,
    'embedding': _body_embedding,
    'mix': _body_mix,
    'repo_coder': _body_repo_coder,
}


def build_body(context_type: str, inputs: ContextInputs):
    """(intro, blocks, stats) for one context type. Leakage-checked before return."""
    builder = _BUILDERS.get(context_type)
    if builder is None:
        raise ValueError(
            f'unknown context_type {context_type!r}; expected one of {CONTEXT_TYPES}'
        )
    intro, blocks, stats = builder(inputs)
    assert_no_leakage(intro + ''.join(blocks), inputs.tripwires)
    return intro, blocks, stats
