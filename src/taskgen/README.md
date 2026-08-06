# taskgen — deterministic harbor-format task generation

Given a repository checkout, `taskgen` carves **one function body** out of one
file and emits a **complete harbor task entry per context-provisioning
condition** — eleven of them: the nine `triton/harness` enums
(`src/eval/eval_llm.py`'s `context_type` plus `src/eval/eval_rag.py`'s
`rag_type`) and the `caller_*` pair taskgen adds by inverting the call graph.

No LLM and no network are used at generation time. The one exception is opt-in
and additive: `--verifier` (off by default) also authors a
[verifier bundle](#verifier-bundle-opt-in) through an LLM proxy.

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

Three further flags are off by default and documented in
[Verifier bundle](#verifier-bundle-opt-in):

* `--verifier` — also author one verifier bundle and copy it into every entry's
  `solution/verifier/`. Needs an LLM proxy; not `diff -r`-reproducible.
* `--llm-config PATH` — the JSON naming that proxy (default: `.llm_config` at the
  repo root).
* `--verifier-min-criteria INT` — how many rubric criteria must survive the
  soundness filter before a bundle ships (default `6`).

### Cloning the source instead of pointing at one

`--repo-url` replaces `--repo` when the checkout does not exist yet (exactly one
of the two is required):

```bash
... generate --repo-url https://github.com/a2aproject/a2a-python \
  --commit <40-char sha> --out .taskgen_out
```

The clone lands in `--repos-cache` (default `../../harbor-tasks/repos-src`)
under the source basename with `.git` stripped, so it is indistinguishable from
a hand-placed checkout: that basename seeds the uuid5 entry ids, the slug and
the image tag. `--repo-name` overrides it.

* **`--commit` is mandatory.** Two sha256 gates (verify's oracle-integrity,
  measure's tree provenance) pin an emitted task to exact upstream bytes, so an
  unpinned clone generates today and fails `verify` the moment upstream moves.
  `--allow-floating` accepts that trade explicitly and marks the task
  `clone_kind = "floating"`.
* **Public sources only.** No token is ever used or accepted; credential helpers
  and terminal prompts are disabled per invocation, so a private URL fails fast
  instead of succeeding on one machine.
* **Reruns are offline.** An existing checkout already at `--commit` is reused
  without touching the network, so `diff -r` between two runs still works.
* A local directory with no `.git` — every sample under `repos-src/` — is used
  **directly**, since `git clone` cannot read a plain snapshot.

A cloned task records its pin in `task.toml`, which is the only place it can
live (`.git` is stripped from the staged tree and from the measure digest):

```toml
[provenance]
repo_url = "https://github.com/a2aproject/a2a-python"
commit = "0123abcd..."          # "" when floating
clone_kind = "pinned"           # or "floating"
```

`verify` reads that block and re-clones the pinned commit itself when no local
checkout exists. A task generated from `--repo` carries no `[provenance]` block
at all and is byte-identical to what taskgen emitted before cloning existed.

## Test

```bash
cd triton/harness
./.venv-taskgen/bin/python -m pytest src/taskgen/tests -q
```

## The eleven conditions

| context type  | what is pre-inlined                                                |
|---------------|--------------------------------------------------------------------|
| `no_context`  | nothing                                                            |
| `callee_func` | full source of every first-party function the target calls          |
| `callee_sig`  | signatures only of those callees                                    |
| `caller_func` | full source of every first-party function that calls the target     |
| `caller_sig`  | signatures only of those callers                                    |
| `in_file`     | the target's own file, with the target's body stubbed out           |
| `project`     | whole surviving repo, path-sorted, truncated at the budget          |
| `bm25`        | Okapi BM25 over ~512-token line chunks                              |
| `embedding`   | TF-IDF + seeded truncated-SVD (LSA) cosine over the same chunks     |
| `mix`         | reciprocal rank fusion (k=60) of the bm25 and LSA rankings          |
| `repo_coder`  | one-shot Jaccard overlap of query and chunk token sets              |

`caller_func`/`caller_sig` are the only pair with no `triton/harness`
counterpart. The parser writes forward `callee` edges for every function
(`py_parser.py:161-197`) and nothing ever reads them backwards, so "who calls
the target" is one deterministic transpose away — no LLM, no network, no new
dependency (gap **B1** in `TASKGEN_GAP_ANALYSIS.md` §3). The universe inverted
is *every* parsed function, not the eligible subset: a caller that lacks a
docstring or a linked test is still a caller. A test function that calls the
target directly is one too, and is inlined — it is on disk regardless, and
`project` already inlines it.

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
eleven conditions are also given the *same* context budget — computed from the
widest of the eleven headers — so a longer condition name cannot buy a condition
fewer context tokens than its neighbours.

## Verifier bundle (opt-in)

`--verifier` authors **one** bundle per `generate` — through an LLM behind a
local proxy — *after* every entry has been emitted, then copies that single
frozen bundle into each entry's `solution/verifier/`:

```
solution/verifier/
  TRUTH.md                        # path-agnostic answer key; may not quote the golden
  coverage.json                   # taxonomy coverage, derived, deterministic
  target_manifest.json            # the files a correct solve must touch
  pytest/test_truth_generated.py  # executable tests authored from TRUTH.md
  pytest/predicates.json          # decidable checks, vacuous ones pruned
  rubric/rubric.json              # judge criteria
```

The bundle is **sound, not merely generated**: every candidate test is executed
and every candidate criterion judged against *both* the oracle (golden) tree and
the carved stub tree, and only checks that **pass on golden and fail on stub** are
frozen — the rest are discarded. The floors are `MIN_SOUND_TESTS = 4` and
`MIN_SOUND_CRITERIA = 6` (`--verifier-min-criteria`). Under either floor **nothing
is written**: the task ships without a bundle, the reason is logged, and the run
still succeeds. The bundle is additive and non-blocking.

That is the only quiet path. A missing, malformed or unreachable `--llm-config`
proxy — or a language with no soundness executor — fails **loud**; an empty or
mocked bundle is never emitted in its place. `.llm_config` is a git-ignored JSON
object, seeded from `proxy/claude-code-oauth.json`, whose `base_url` points at the
local proxy bridge:

```json
{"model": "...", "base_url": "http://127.0.0.1:8765", "api_key": "...",
 "timeout": 600, "num_retries": 2}
```

Two caveats, both structural:

* **Determinism.** The bundle is LLM-authored, so a `--verifier` run is *not*
  `diff -r`-reproducible. The flag is off by default exactly so the default
  `generate` stays byte-deterministic and offline — run the proof above **without**
  `--verifier`.
* **Leakage.** The bundle lives only under `solution/`, which never enters an image
  layer, so `verify --all` still clears all six gates with it present. Oracle lines
  the bundle quotes are additionally merged into `trip/tripwires.txt`, so the
  in-build leakscan and verify's layer archaeology both harden around it.

The generators live in the verifier generation package (`src/verifier`) and are
imported lazily: without `--verifier`, none of it — and no `litellm` — is imported
at all.

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
  solution/verifier/                     # --verifier only; see above
```

`tests/test.sh` (python) also persists its JUnit report as `results.xml`, under
the run's results dir (`/logs/verifier/results/results.xml`). It is created inside
the container at **run time**, so it is neither an emitted asset nor an image
layer, and it is a copy taken *after* the reward has been read off `junit.xml` —
nothing consumes it, so the contract is byte-for-byte the one that was proven:
RED still scores `0.0`, GREEN still `1.0`. The other five languages are a
documented follow-up, not implemented (`gotestsum --junitfile` for go, Surefire's
`TEST-*.xml` for java, `cargo nextest` for rust, `ctest --output-junit` for c,
doctest's junit reporter for cpp).

Only `instruction.md`'s *Pre-loaded context* block and `task.toml`'s
`[metadata].condition` differ across the eleven — the carve, the image and the
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
