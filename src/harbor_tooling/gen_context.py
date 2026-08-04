"""Generate instruction.md for one repo under one of five context conditions.

    python gen_context.py --repo repos-src/<repo> \
        --manifest shared/manifests/<repo>.carve.toml \
        --condition longctx-folder --budget 100000 \
        --tokenizer tiktoken --seed 0 --out out/instruction.md

The five conditions differ ONLY in what repository content is pre-inlined:

  longctx-folder      non-carved files in the carve root's parent package,
                      packed breadth-first nearest-first to budget
  longctx-repoprefix  whole surviving repo, path-sorted, truncated to budget
  bm25-rag            ~512-token chunks ranked by Okapi BM25
  dense-lsa-rag       same chunks ranked by TF-IDF + truncated-SVD cosine
  nocontext           no repository content at all

Read-only over the repository. Deterministic: same inputs and seed produce a
byte-identical file.

CORRECTNESS PROPERTY: no carved file's BODY may appear in the output. Carved
files are excluded from every corpus, only signatures are emitted, and
`assert_no_leakage` re-checks the finished document against every distinctive
body line before it is written.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sigextract  # noqa: E402
from count_tokens import get_counter  # noqa: E402
from manifest import Manifest, resolve_carved, walk_repo  # noqa: E402
from retrieval.chunk import TARGET_TOKENS, build_corpus  # noqa: E402

CONDITIONS = (
    'longctx-folder',
    'longctx-repoprefix',
    'bm25-rag',
    'dense-lsa-rag',
    'nocontext',
)

FRAMING_BLOCK = (
    'These conditions vary context PROVISIONING (pre-loaded into the prompt vs. '
    'agent-retrieved from disk) under a fixed turn/token budget. They measure '
    'provisioning EFFICIENCY, not context AVAILABILITY as in MRG-Bench. The full '
    'repository (minus the files you must regenerate) is present on disk in every '
    'condition; conditions differ only in what is pre-inlined here. This does not '
    "replicate MRG-Bench's ablation."
)

FENCE_LANG = {
    '.py': 'python', '.go': 'go', '.rs': 'rust', '.java': 'java', '.c': 'c',
    '.h': 'c', '.cc': 'cpp', '.cpp': 'cpp', '.hpp': 'cpp', '.toml': 'toml',
    '.md': 'markdown', '.json': 'json', '.yaml': 'yaml', '.yml': 'yaml',
    '.cfg': 'ini', '.ini': 'ini', '.sh': 'bash', '.sql': 'sql', '.txt': 'text',
}

MIN_TRIPWIRE_CHARS = 24
FOOTER_RESERVE_TOKENS = 1200


class LeakageError(AssertionError):
    pass


def fence_for(relpath: str) -> str:
    return FENCE_LANG.get(Path(relpath).suffix, 'text')


def norm_line(line: str) -> str:
    return ' '.join(line.split())


# --------------------------------------------------------------------------
# Leakage tripwires
# --------------------------------------------------------------------------


def build_tripwires(carved: list[tuple[str, str]], specs: dict, surviving: list[tuple[str, str]]):
    """Distinctive body lines of carved files that must never appear in output.

    A carved line that ALSO occurs somewhere in the surviving repo is not a
    tripwire: the model can legitimately see it via the surviving file, so
    flagging it would be a false positive. Only lines unique to carved bodies,
    excluding the lines signature extraction is allowed to quote, count.
    """
    survivor_lines = set()
    for _rel, text in surviving:
        for line in text.splitlines():
            survivor_lines.add(norm_line(line))

    tripwires: dict[str, tuple[str, int]] = {}
    for rel, text in carved:
        spec = specs.get(rel)
        siglines = spec.signature_lines if spec is not None else set()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if lineno in siglines:
                continue
            norm = norm_line(line)
            if len(norm) < MIN_TRIPWIRE_CHARS or norm in survivor_lines:
                continue
            tripwires.setdefault(norm, (rel, lineno))
    return tripwires


def assert_no_leakage(document: str, tripwires: dict) -> None:
    doc_lines = {norm_line(line) for line in document.splitlines()}
    hits = sorted(t for t in tripwires if t in doc_lines)
    if hits:
        detail = '\n'.join(
            f'  {tripwires[h][0]}:{tripwires[h][1]}  {h[:100]}' for h in hits[:10]
        )
        raise LeakageError(
            f'CARVED BODY LEAKED INTO OUTPUT: {len(hits)} distinctive line(s) from '
            f'carved files appear in the generated document. Refusing to write.\n{detail}'
        )


# --------------------------------------------------------------------------
# Static sections
# --------------------------------------------------------------------------


def render_header(mf: Manifest, condition: str, carved: list[str], specs: dict, counter) -> str:
    parts: list[str] = []
    a = parts.append

    a(f'# Task: regenerate the `{mf.carve_root}` subsystem of `{mf.repo}`\n')
    a('## What happened\n')
    if mf.description:
        a(mf.description + '\n')
    a(
        f'The {len(carved)} files listed below were deleted from this repository. '
        f'Everything else is still on disk exactly as it was. Your job is to write '
        f'those files again, from scratch, so that the repository\'s surviving test '
        f'suite passes.\n'
    )
    a(
        'The rest of the repository is your ground truth: surviving code imports '
        'these modules, and the tests exercise them. Read whatever you need from '
        'disk -- the working tree is complete apart from the deleted files.\n'
    )

    a('## Context condition\n')
    a(f'This instance runs under condition **`{condition}`**.\n')
    a('> ' + FRAMING_BLOCK + '\n')

    a('## Files you must regenerate\n')
    a(f'Carve root: `{mf.carve_root}` -- {len(carved)} files.\n')
    a('```')
    for rel in carved:
        a(rel)
    a('```\n')

    a('## Signature spec (public symbols of the deleted files)\n')
    a(
        'Signatures ONLY. Bodies, docstrings, and default values are deliberately '
        'withheld -- reproducing the behaviour is the task. Treat these as the '
        'public API contract the surviving code and tests are written against.\n'
    )
    approximate = any(s.approximate for s in specs.values())
    if approximate:
        a(
            'NOTE: signature extraction for this language is regex-based and '
            'APPROXIMATE; declarations may be partially captured. Verify against '
            'call sites in the surviving code.\n'
        )
    else:
        a('Extracted exactly via the Python `ast` module.\n')
    a('```')
    for rel in carved:
        spec = specs.get(rel)
        if spec is not None:
            a(spec.render().rstrip('\n'))
    a('```\n')

    a('## How you are graded\n')
    a(f'Test command, run from the repository root:\n\n```bash\n{mf.test_command}\n```\n')
    a(
        '- Reward is computed **only** from that command\'s result on the surviving '
        'test suite.\n'
        '- If collection fails (import errors, syntax errors, missing modules), the '
        'reward is **0.0** -- a repo that will not import scores nothing, however '
        'much code you wrote.\n'
        '- Otherwise reward is the fraction of tests that pass, and **1.0** only if '
        'the entire suite is green.\n'
        '- Editing, deleting, skipping, or `xfail`-ing tests earns no credit; the '
        'grader restores the test tree before scoring.\n'
        '- Only files under the carve root should be created. Do not modify '
        'surviving source to make tests pass.\n'
    )
    return '\n'.join(parts) + '\n'


def render_footer(mf: Manifest, condition: str, counter, seed: int, budget: int, stats: dict) -> str:
    parts = ['## Generation metadata\n', '```']
    rows = [
        ('repo', mf.repo),
        ('condition', condition),
        ('budget_tokens', budget),
        ('tokenizer', counter.method),
        ('seed', seed),
        ('carve_root', mf.carve_root),
        ('carved_files', stats.get('carved_files', 0)),
        ('surviving_files_considered', stats.get('surviving_files', 0)),
    ]
    for key, value in rows:
        parts.append(f'{key} = {value}')
    for key in sorted(stats):
        if key in ('carved_files', 'surviving_files'):
            continue
        parts.append(f'{key} = {stats[key]}')
    parts.append('```\n')
    parts.append(f'Token counts above use {counter.describe()}.\n')
    parts.append(
        'Leakage check: the generator verified that no distinctive body line from '
        'any carved file appears anywhere in this document.\n'
    )
    return '\n'.join(parts) + '\n'


# --------------------------------------------------------------------------
# Condition bodies
# --------------------------------------------------------------------------


def render_file_block(rel: str, text: str, note: str = '') -> str:
    n_lines = len(text.splitlines())
    head = f'### `{rel}` (lines 1-{n_lines})' if not note else f'### `{rel}` ({note})'
    return f'{head}\n```{fence_for(rel)}\n{text}\n```\n'


def pack_files(files: list[tuple[str, str]], counter, budget: int) -> tuple[list[str], dict]:
    """Fill `budget` with whole files in the given order; truncate the one that
    crosses the boundary, then stop. This is the ordered-prefix semantics both
    longctx conditions specify."""
    blocks: list[str] = []
    used = 0
    whole = 0
    truncated = 0
    for rel, text in files:
        block = render_file_block(rel, text)
        cost = counter.count(block)
        if used + cost <= budget:
            blocks.append(block)
            used += cost
            whole += 1
            continue
        remaining = budget - used
        if remaining > 200:
            keep = counter.truncate(text, remaining - 120)
            keep = keep[: keep.rfind('\n') + 1] if '\n' in keep else keep
            if keep.strip():
                kept_lines = len(keep.splitlines())
                block = render_file_block(
                    rel, keep.rstrip('\n'), note=f'lines 1-{kept_lines}, TRUNCATED at budget'
                )
                blocks.append(block)
                used += counter.count(block)
                truncated = 1
        break
    return blocks, {'files_inlined_whole': whole, 'files_truncated': truncated, 'context_tokens': used}


def tree_distance(relpath: str, carve_root: str) -> int:
    a = Path(relpath).parent.parts
    b = Path(carve_root).parts
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return (len(a) - common) + (len(b) - common)


def body_longctx_folder(mf: Manifest, surviving, counter, budget: int):
    root = mf.folder_root
    prefix = f'{root}/' if root else ''
    scoped = [(r, t) for r, t in surviving if r.startswith(prefix)]
    scoped.sort(key=lambda p: (tree_distance(p[0], mf.carve_root), p[0]))
    blocks, stats = pack_files(scoped, counter, budget)
    intro = (
        f'## Pre-loaded context: `{root or "."}` package (breadth-first, nearest-first)\n\n'
        f'Every non-deleted file in `{root or "."}` -- the package that directly '
        f'contains the carve root -- ordered by tree distance from `{mf.carve_root}` '
        f'(nearest first, then path-sorted), packed to the token budget. '
        f'{len(scoped)} files were eligible; {stats["files_inlined_whole"]} were '
        f'inlined whole.\n'
    )
    stats['files_eligible'] = len(scoped)
    stats['selection'] = 'parent-package breadth-first nearest-first'
    return intro, blocks, stats


def body_longctx_repoprefix(mf: Manifest, surviving, counter, budget: int):
    ordered = sorted(surviving, key=lambda p: p[0])
    blocks, stats = pack_files(ordered, counter, budget)
    intro = (
        '## Pre-loaded context: repository prefix (path-sorted)\n\n'
        'The entire surviving repository concatenated in deterministic path-sorted '
        f'order and truncated at the token budget. {len(ordered)} files were '
        f'eligible; {stats["files_inlined_whole"]} were inlined whole. Coverage is '
        'broad but shallow, and stops wherever the budget runs out in the sort '
        'order -- later paths are absent from this prompt (they remain on disk).\n'
    )
    stats['files_eligible'] = len(ordered)
    stats['selection'] = 'path-sorted repository prefix'
    return intro, blocks, stats


def build_query(mf: Manifest, carved: list[str], specs: dict) -> str:
    parts = [mf.description, mf.carve_root, mf.repo]
    parts.extend(carved)
    for rel in carved:
        spec = specs.get(rel)
        if spec is not None:
            parts.append(spec.render())
    return '\n'.join(parts)


def render_chunk_block(chunk, rank: int, score: float, label: str) -> str:
    head = (
        f'### [{label} rank {rank}] `{chunk.relpath}` lines {chunk.start_line}-'
        f'{chunk.end_line} (score {score:.4f})'
    )
    return f'{head}\n```{fence_for(chunk.relpath)}\n{chunk.text}\n```\n'


def _pack_ranked(chunks, ranked, counter, budget: int, label: str):
    blocks: list[str] = []
    used = 0
    taken = 0
    files_seen: set[str] = set()
    for rank_i, (score, idx) in enumerate(ranked, start=1):
        chunk = chunks[idx]
        block = render_chunk_block(chunk, rank_i, score, label)
        cost = counter.count(block)
        if used + cost > budget:
            continue
        blocks.append(block)
        used += cost
        taken += 1
        files_seen.add(chunk.relpath)
    return blocks, {
        'chunks_retrieved': taken,
        'chunks_in_corpus': len(chunks),
        'distinct_files_retrieved': len(files_seen),
        'context_tokens': used,
    }


def body_rag(mf, surviving, counter, budget, carved, specs, seed, dense: bool):
    corpus = build_corpus(surviving, counter, TARGET_TOKENS)
    query = build_query(mf, carved, specs)
    if dense:
        from retrieval.dense_lsa import N_COMPONENTS, LSAIndex

        ranked = LSAIndex(corpus, seed=seed).score(query)
        label = 'dense-lsa'
        intro = (
            '## Pre-loaded context: dense-LSA retrieval\n\n'
            f'The surviving repository was split into ~{TARGET_TOKENS}-token '
            'line-aligned chunks and ranked by cosine similarity in a '
            f'{N_COMPONENTS}-dimensional latent space built with TF-IDF plus '
            'truncated SVD (LSA), queried with the signature spec and the carved '
            'file list.\n\n'
            'This **approximates** dense retrieval; it is not a pretrained neural '
            'embedding model. No such model was available in the generation '
            'environment (no HF token, no GPU), so its semantics come purely from '
            'term co-occurrence within this one repository. Expect it to be '
            '**weaker** than a real dense retriever, and read its ranking with that '
            'in mind.\n'
        )
    else:
        from retrieval.bm25 import BM25Index

        ranked = BM25Index(corpus).score(query)
        label = 'bm25'
        intro = (
            '## Pre-loaded context: BM25 retrieval\n\n'
            f'The surviving repository was split into ~{TARGET_TOKENS}-token '
            'line-aligned chunks and ranked with Okapi BM25 (k1=1.5, b=0.75), '
            'queried with the signature spec and the carved file list. This is '
            'lexical retrieval: it matches identifiers and literals, not meaning.\n'
        )
    blocks, stats = _pack_ranked(corpus, ranked, counter, budget, label)
    stats['selection'] = label
    intro += (
        f'\n{stats["chunks_retrieved"]} chunks from '
        f'{stats["distinct_files_retrieved"]} files were retrieved out of '
        f'{stats["chunks_in_corpus"]} in the corpus. Each block below carries its '
        'source path, line range, rank, and score. Chunks are fragments -- read the '
        'full file from disk when you need more.\n'
    )
    return intro, blocks, stats


def body_nocontext(mf: Manifest):
    intro = (
        '## Pre-loaded context: none\n\n'
        'No repository content is inlined in this condition. The complete surviving '
        'repository is nonetheless present on disk; read it yourself with the tools '
        'you have. The task description, carved file list, signature spec, test '
        'command, and reward mechanics above are all you are given up front.\n'
    )
    return intro, [], {'selection': 'none', 'context_tokens': 0}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def generate(repo: Path, mf: Manifest, condition: str, budget: int, tokenizer: str,
             seed: int) -> tuple[str, dict]:
    counter = get_counter(tokenizer)
    all_files = walk_repo(repo, mf)
    carved = resolve_carved(repo, mf)
    if not carved:
        raise SystemExit(
            f'carve globs matched zero files under {repo}. Check [carve].include '
            f'in {mf.path}.'
        )
    carved_set = set(carved)

    surviving = [(r, t) for r, t in all_files if r not in carved_set]
    carved_texts: list[tuple[str, str]] = []
    for rel in carved:
        try:
            carved_texts.append((rel, (repo / rel).read_text(encoding='utf-8')))
        except (OSError, UnicodeDecodeError):
            carved_texts.append((rel, ''))

    specs = {rel: sigextract.extract(text, rel, mf.language) for rel, text in carved_texts}
    tripwires = build_tripwires(carved_texts, specs, surviving)

    header = render_header(mf, condition, carved, specs, counter)
    header_tokens = counter.count(header)
    context_budget = budget - header_tokens - FOOTER_RESERVE_TOKENS
    if context_budget < 0:
        context_budget = 0

    if condition == 'longctx-folder':
        intro, blocks, stats = body_longctx_folder(mf, surviving, counter, context_budget)
    elif condition == 'longctx-repoprefix':
        intro, blocks, stats = body_longctx_repoprefix(mf, surviving, counter, context_budget)
    elif condition == 'bm25-rag':
        intro, blocks, stats = body_rag(
            mf, surviving, counter, context_budget, carved, specs, seed, dense=False
        )
    elif condition == 'dense-lsa-rag':
        intro, blocks, stats = body_rag(
            mf, surviving, counter, context_budget, carved, specs, seed, dense=True
        )
    else:
        intro, blocks, stats = body_nocontext(mf)

    stats['carved_files'] = len(carved)
    stats['surviving_files'] = len(surviving)
    stats['leak_tripwires_checked'] = len(tripwires)
    stats['header_tokens'] = header_tokens

    def assemble(chosen: list[str]) -> str:
        footer = render_footer(mf, condition, counter, seed, budget, stats)
        mid = intro + '\n' + ''.join(chosen) if chosen else intro + '\n'
        return header + '\n' + mid + '\n' + footer

    document = assemble(blocks)
    # The footer quotes selection stats, so its true size is only known after
    # packing; drop trailing blocks if the reserve turned out to be too small.
    while blocks and counter.count(document) > budget:
        blocks.pop()
        stats['context_tokens'] = sum(counter.count(b) for b in blocks)
        document = assemble(blocks)

    total = counter.count(document)
    if total > budget:
        raise SystemExit(f'internal error: document {total} tokens exceeds budget {budget}')

    assert_no_leakage(document, tripwires)
    if FRAMING_BLOCK not in document:
        raise SystemExit('internal error: required framing block missing from document')

    stats['total_tokens'] = total
    return document, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description='Generate instruction.md for one context condition.')
    ap.add_argument('--repo', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--condition', required=True, choices=CONDITIONS)
    ap.add_argument('--budget', type=int, default=100000)
    ap.add_argument('--tokenizer', choices=('tiktoken', 'chars4'), default='tiktoken')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)

    repo = args.repo.resolve()
    if not repo.is_dir():
        raise SystemExit(f'--repo is not a directory: {repo}')
    mf = Manifest.load(args.manifest)

    document, stats = generate(
        repo, mf, args.condition, args.budget, args.tokenizer, args.seed
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(document, encoding='utf-8')

    print(
        f'{args.condition:<19} {stats["total_tokens"]:>7} tokens  '
        f'(budget {args.budget}, {stats["leak_tripwires_checked"]} tripwires clean)  '
        f'-> {args.out}'
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
