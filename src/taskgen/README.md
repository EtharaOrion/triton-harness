# taskgen — deterministic harbor-format task generation

Given a repository checkout, `taskgen` carves **one function body** out of one
file and emits a **complete harbor task entry per context-provisioning
condition** — nine of them, matching the `triton/harness` enums
(`src/eval/eval_llm.py`'s `context_type` plus `src/eval/eval_rag.py`'s
`rag_type`).

No LLM and no network are used at generation time.

## Setup

```bash
cd triton/harness
uv venv --python 3.12 .venv-taskgen
VIRTUAL_ENV=.venv-taskgen uv pip install -r src/taskgen/requirements-dev.txt
```

Python 3.12 is a floor, not a preference: `harbor-tasks/shared/tooling` uses
`tomllib` (3.11+) and the pinned `tree_sitter_python` wheel has no 3.13/3.14
arm64 build.

## Generate

```bash
cd triton/harness
PYTHONPATH=src ./.venv-taskgen/bin/python -m taskgen.cli generate \
  --repo ../../harbor-tasks/repos-src/python-a2a-python \
  --package-base src/ \
  --file src/a2a/utils/task.py \
  --func apply_history_length \
  --out .taskgen_out
```

Omit `--file`/`--func` to take the first eligible function under a total order
on `(relpath, class, name)`.

## Test

```bash
cd triton/harness
./.venv-taskgen/bin/python -m pytest src/taskgen/tests -q
```

## The nine conditions

| context type  | what is pre-inlined                                                |
|---------------|--------------------------------------------------------------------|
| `no_context`  | nothing                                                            |
| `callee_func` | full source of every first-party function the target calls          |
| `callee_sig`  | signatures only of those callees                                    |
| `in_file`     | the target's own file, with the target's body stubbed out           |
| `project`     | whole surviving repo, path-sorted, truncated at the budget          |
| `bm25`        | Okapi BM25 over ~512-token line chunks                              |
| `embedding`   | TF-IDF + seeded truncated-SVD (LSA) cosine over the same chunks     |
| `mix`         | reciprocal rank fusion (k=60) of the bm25 and LSA rankings          |
| `repo_coder`  | one-shot Jaccard overlap of query and chunk token sets              |

Two are deliberate, documented approximations:

* **`embedding`** is a dense-LSA stand-in. There is no pretrained embedding
  model available offline, and shipping one would make generation depend on a
  download. The condition is never labelled as a neural retriever.
* **`repo_coder`** is RepoCoder's **iteration 0** only. The published method
  re-queries with the model's own draft completion; that needs an LLM in the
  generation pipeline, which would break both determinism and the no-LLM rule.

## Determinism

Two runs produce byte-identical trees, directory names included:

```bash
cd triton/harness
for d in /tmp/tg-a /tmp/tg-b; do
  PYTHONPATH=src ./.venv-taskgen/bin/python -m taskgen.cli generate \
    --repo ../../harbor-tasks/repos-src/python-a2a-python \
    --file src/a2a/utils/task.py --func apply_history_length --out "$d"
done
diff -r /tmp/tg-a /tmp/tg-b   # empty
```

What makes that hold: entry ids are `uuid5` over
`repo|relpath|class|func|context_type`; every ranking breaks ties on `chunk_id`;
every file list is path-sorted; the LSA index is seeded (`seed=0`); the token
counter is `chars4`; and no emitted file carries a timestamp or a host path. All
nine conditions are also given the *same* context budget — computed from the
widest of the nine headers — so a longer condition name cannot buy a condition
fewer context tokens than its neighbours.

## Emitted entry layout

```
<entry_id>/                              # uuid5, so the name is content-derived
  task.toml                              # Family-A schema, schema_version 1.4
  instruction.md                         # 8 sections
  environment/Dockerfile                 # two-stage: intact -> carved
  environment/Dockerfile.dockerignore
  environment/carve/<relpath>            # the STUBBED file
  tests/test.sh                          # binary reward over the linked ids
  tests/allowlist.txt                    # the linked pytest node ids
  tests/harbor_filter.py                 # vendored verbatim
  solution/solve.sh                      # restores the original, asserts sha256
  solution/carved/<relpath>              # the INTACT original file
```

Only `instruction.md`'s *Pre-loaded context* block and `task.toml`'s
`[metadata].condition` differ across the nine — the carve, the image and the
oracle are shared.

### Building the image

The Dockerfile bakes in no host path. The repo checkout and the entry assets
arrive as BuildKit **named contexts**:

```bash
ENTRY=.taskgen_out/<entry_id>
docker build \
  --build-context repo-src=../../harbor-tasks/repos-src/python-a2a-python \
  --build-context entry="$ENTRY" \
  -f "$ENTRY/environment/Dockerfile" --target carved \
  -t taskgen-python-a2a-python:local "$ENTRY/environment"
```

## Why the oracle is correct by construction

`solution/carved/<relpath>` is the **untouched original file**, not a patch, so
`solve.sh` returns the working tree to a state byte-identical to the pristine
checkout (it asserts the sha256). `reward=1` therefore reduces to *"the linked
tests pass on the pristine repo and `harbor_filter` selects exactly them"* — a
property of target selection, not of emission.

## Leakage

The corpus every builder indexes is the **surviving** repo: the target's file
appears in its stubbed form, so the carved body is absent by construction. On
top of that, every generated document is re-checked against content tripwires —
distinctive (≥24 char) carved body lines that appear in no surviving file — and
generation is refused if one shows up. The signature and the docstring are
quoted deliberately: they are the prompt, not the answer, so their source lines
are whitelisted.

## Borrowed, not forked

`taskgen/_tooling_path.py` puts two vendored trees on `sys.path` rather than
copying them:

* `triton/harness/src` → `parser.py_parser` (target selection, call graph)
* `harbor-tasks/shared/tooling` → `sigextract`, `count_tokens`, `manifest`,
  `retrieval/{bm25,dense_lsa,chunk}`, `gen_context` renderers

Forking those would silently decouple generated tasks from the reference harbor
corpus. `taskgen/_ts_compat.py` **is** vendored, because it is a shim over a
third-party API break rather than shared logic.
