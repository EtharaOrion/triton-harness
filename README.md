# MRGBench
This is the Official Repository for the repository level code generation benchmark: MRGBench

## Structure
```
.
├── data                                 # MRGBench dataset (518 entries: Python, Java, Go)
│   ├── all_context_project_dict.json    # project-level context
│   ├── context_info.json                # in-file context
│   ├── go_data_final.xlsx               # data for Go
│   ├── java_data_final.xlsx             # data for Java
│   └── py_data_final.xlsx               # data for Python
├── repo                                 # source repositories
│   ├── go_data.7z                       # MRGBench Go repos (compressed)
│   ├── java_data.7z                     # MRGBench Java repos (compressed)
│   ├── py_data.7z                       # MRGBench Python repos (compressed)
│   ├── python-a2a-python/               # trimmed sample repo (taskgen test fixture)
│   ├── go-multigres/                    # trimmed sample repo (taskgen test fixture)
│   └── vendor/                          # pinned toolchain assets (wabt tarball)
├── result                               # MRGBench experiment results
│   ├── cache_result                     # per-model results across contexts
│   ├── llms
│   └── rag
├── proxy                                # Anthropic-compatible LLM bridge (for --verifier)
│   ├── claude_code_bridge/
│   ├── claude_code_bridge.sh
│   └── README.md
├── .llm_config                          # verifier LLM client config
│   └── example.json                     # committed template (real config git-ignored)
├── fig                                  # README images
├── src
│   ├── eval                             # MRGBench evaluation / RAG harness
│   ├── parser                           # repository -> dataset parsers (call graph)
│   ├── taskgen                          # deterministic harbor task generator
│   │   ├── cli.py                       # generate / verify entry point
│   │   ├── clone.py                     # repo clone (URL + pinned commit)
│   │   ├── select.py                    # target selection (eligibility)
│   │   ├── carve.py                     # carve the target body -> stub + oracle
│   │   ├── contexts.py                  # the eleven context conditions
│   │   ├── emit.py                      # write the harbor task entries
│   │   ├── measure.py                   # whole-suite graded-set measurement
│   │   ├── verify.py                    # six-gate verify (docker)
│   │   ├── langs/                       # per-language plugins (python/go/rust/c/cpp/java)
│   │   └── tests/                       # self-contained pytest suite
│   ├── verifier                         # per-task verifier bundle generator (opt-in --verifier)
│   │   ├── bundle.py                    # drive generators + soundness loop
│   │   ├── inputs.py                    # build TruthInputs from carve artifacts
│   │   ├── exec_env.py                  # run generated tests vs golden / stub trees
│   │   ├── llm_config.py                # .llm_config loader + client factory
│   │   └── generators/                  # TRUTH / rubric / pytest / predicates / coverage
│   └── harbor_tooling                   # vendored harbor retrieval + staging tooling
├── LICENSE                              # MIT
├── pyproject.toml                       # taskgen uv project (harbor pinned from PyPI)
├── uv.lock
├── requirement.txt                      # separate MRGBench GPU stack (untouched)
└── README.md
```

## Usage

MRGBench is a dataset designed for evaluating repository-level code generation tasks for large language models, as well as related Retrieval-Augmented Generation (RAG) and Agent methods. MRGBench encompasses three programming languages: Python, Java, and Go, comprising a total of 518 entries. Each entry includes a natural language description, repository information, and corresponding function test cases. Additionally, we provide a runnable Docker container, enabling researchers to quickly validate related algorithms using MRGBench.

![MRGBench Structure](fig/image.png)

The use of MRGBench can be divided into two steps:

1. Generate the corresponding function body for each use case using methods such as LLM, RAG, or Agent.
2. Verify the correctness of the generated function body in the Docker runtime environment and compute Pass@k.

### Generate corresponding code

In the experimental phase, we evaluated the performance of various large language models in different contexts and basic RAG methods. These evaluation scripts can be quickly executed to reproduce our results. 

#### Prepare Python Env
`pip install -r requirements.txt`

#### API key and url

We implemented the OpenAI API interface within `AIClient.py`. Users can configure their own URL and API key (e.g., for a locally deployed VLLM) in `config.json` to access the large language model.

#### Reproducing Results
For instance, to generate results for Go language samples using the DeepSeek model with infile context, you can run the following command:
```bash
python eval_llm.py -context_type <context_type> -lang_list <language_list> -model_name <model_name>
```
or 
```bash
python eval_llm.py --context_type in_file -lang_list go -model_name deepseek
```
Alternatively, to build a RAG system using the DeepSeek model and BM25 search, and generate results, you can run the following command:

```bash
python eval_rag.py -rag_type <rag_type> -lang_list <language_list> -model_name <model_name>
```
Here is an example command to run the evaluation script:
```bash
python eval_rag.py -rag_type bm25 -lang_list py,java -model_name deepseek
```

You can also design a new algorithm yourself, as long as the final output is in JSON format and contains the following fields:
```json
{
    "task-id": "?" // must the same as the task-id in data/*_final__data.xlsx
    "response": ["code-1", "code-2"] // list of code generated by LLM
}
```

### Verifying Correctness
To verify the correctness of the generated code, we provide a Docker container that can be used to run the code in a controlled environment.

1. Pull the Docker image (note that the data volume is large, so this may take some time).
```bash
docker pull mrgbench/mrgbench:v1
```
2. Use the following command to run the Docker image:
```bash
docker run -it mrgbench/mrgbench:v1 /bin/bash -v ../MRG-Bench:/root/MRG-Bench
```
3. After the container is created, run the following script to initiate the test:
```bash
cd /root/MRG-Bench
python run_test.py -df_path <data_path> -result_path <result_path> -lan <language>
```
For example:
```bash
python run_test.py -df_path data/py_data_final.xlsx -result_path result/llms/your_result.json -lan py
```
4. The test results will be saved in `result/llms/your_result.json_testresult.json`.

---

# taskgen: deterministic harbor task generator

`taskgen` (under `src/taskgen/`) turns a real source repository into harbor evaluation tasks. It carves a target out of the repo, ships the carved tree in a Docker image, and proves two things with real container runs: the carved stub scores reward 0.0, and the original code (the oracle) scores reward 1.0. The generator is deterministic, and the verifier is leak-audited so the answer never reaches any Docker layer.

## What a task is

For each run, `generate` writes one output directory holding eleven entries, one per retrieval context type (no_context, callee_func, callee_sig, caller_func, caller_sig, in_file, project, bm25, embedding, mix, repo_coder). caller_func and caller_sig inline the first-party functions that call the target, the deterministic inverse of the callee pair. All eleven share a single carved image and oracle; only the instruction context differs. Each entry contains:

- `environment/Dockerfile` and `environment/contexts.json` (named build contexts: entry, repoctx, tooling, trip)
- `solution/solve.sh` and `solution/carved/...` (the oracle payload, restored at run time from a read only mount)
- `tests/test.sh`, `tests/graded.json`, and for the measured languages `tests/graded.lock.json`
- `task.toml`, `instruction.md`

The python `tests/test.sh` additionally persists its JUnit report as `results.xml` at run time. That file is runtime only, not an emitted asset, and the reward contract is unchanged. The other five languages are a follow-up.

## Languages and carve scopes

Two families of language plugin live in `src/taskgen/langs/`.

Parser-backed (tree-sitter), function / file / folder scope:

- python
- go

Whole-suite (no parser), folder scope with `--delete-whole-file`:

- rust
- c
- cpp
- java

csharp is planned but not yet implemented.

A parser-backed language can carve a single function body, or every function body in a glob-selected set of files. A whole-suite language deletes a whole subtree and grades the entire test suite; its pass floor is measured once against the intact tree and pinned into `graded.lock.json`.

## Environment

The generator is a uv project rooted at this directory (the taskgen harness). It pins the taskgen dependencies and the `harbor` runtime CLI from PyPI (`harbor==0.20.0`); the shared harbor tooling is vendored in-tree under `src/harbor_tooling`. The MRGBench GPU stack in `requirement.txt` is separate and untouched.

```bash
uv sync
```

That creates `.venv` and installs the pinned dependencies (tree-sitter, numpy, tqdm, and the `harbor` CLI from PyPI) plus the dev tools (pytest); the harbor tooling package is vendored in-tree (`src/harbor_tooling`). Run everything through `uv run`, which keeps the environment in sync:

```bash
uv run taskgen --help    # console entry point
PY="uv run python"       # the commands below use $PY -m taskgen.cli
```

Source repositories live under `../../harbor-tasks/repos-src/`. The harbor base image plus the per-repo images must be loaded into Docker for `verify`, and also for `generate` on the whole-suite languages, which build a throwaway measure image.

## generate

```bash
$PY -m taskgen.cli generate --repo <repo> --out <dir> [options]
$PY -m taskgen.cli generate --repo-url <src> --commit <sha> --out <dir> [options]
```

The source is either a local checkout (`--repo`) or a clone (`--repo-url`); exactly one is required.

Key options:

- `--repo PATH` source checkout to carve
- `--repo-url SRC` clone source, a public HTTPS URL or a local git repo; mutually exclusive with `--repo`
- `--commit SHA` commit to check out (required for a URL unless `--allow-floating`)
- `--allow-floating` clone the default branch HEAD instead of a pin (non-reproducible)
- `--repo-name NAME` override the derived repo basename (identity)
- `--repos-cache DIR` where clones land (default: `../../harbor-tasks/repos-src`)
- `--out DIR` output directory (required)
- `--lang {python,go,rust,c,cpp,java,csharp}` language plugin (default: python)
- `--carve-scope {function,file,folder}` how much to carve (default: function)
- `--file PATH` target file, repo relative
- `--func NAME` target function name
- `--class NAME` target class name for methods
- `--package-base DIR` import root inside the repo (default: `src/`)
- `--include GLOB` file/folder scope carve glob, repeatable
- `--exclude GLOB` subtract a glob from `--include`, repeatable
- `--delete-whole-file` delete carved files outright instead of skeleton stubbing (required for whole-suite languages)
- `--receiver TYPE` go method receiver disambiguation
- `--project SEG` go.mod project segment (derived when omitted)
- `--contexts all|bm25,mix,...` which context entries to write (default: all eleven)
- `--budget INT`, `--seed INT`
- `--verifier` also author one sound verifier bundle per generate and copy it into each entry's `solution/verifier/` (opt-in; needs an LLM proxy; not `diff -r` reproducible)
- `--llm-config PATH` path to the LLM config used by `--verifier` (default: repo-root `.llm_config`)
- `--verifier-min-criteria INT` sound-rubric floor for the bundle (default: 6)

### Cloning and provenance

A pinned `--repo-url ... --commit ...` clone is deterministic, and the resolved `repo_url`, `commit`, and `clone_kind` are recorded in `task.toml` under `[provenance]`. `verify` self-clones the pinned commit when no local checkout exists.

### Verifier bundle

`--verifier` is off by default so a default `generate` stays offline and deterministic. The bundle lives only under `solution/verifier`, never in an image layer, so `verify` still passes every gate. Below the sound floor the task ships without a bundle, which is non-blocking; a missing or unreachable proxy fails loud. The LLM config is a git-ignored `.llm_config/claude-code-oauth.json` (copy the committed `.llm_config/example.json`).

For python and go, `generate` is fully offline and deterministic: run it twice into two directories and `diff -r` them. For rust, c, cpp, and java, `generate` builds a never-ship measure image once to count the intact test suite, pins that count into `graded.lock.json`, deletes the image, and reuses the lock on later runs when the repo and base image are unchanged. Those runs need Docker and network access.

## verify

```bash
$PY -m taskgen.cli verify --all <out-dir> --lang <lang> --carve-scope <scope> --repo <repo>
$PY -m taskgen.cli verify --entry <entry-dir> --lang <lang> --carve-scope <scope> --repo <repo>
```

`--all` builds the one image the eleven entries share, proves they are byte identical, then runs the full gate on that image:

1. oracle-integrity: every `solution/carved/...` file matches the upstream repo by sha256, checked on the host
2. build the graded image from the named build contexts
3. image hygiene: no oracle solution baked into the image
4. RED run: the carved stub must score reward 0.0
5. GREEN run: the oracle (bind mounted read only, restored by solve.sh) must score reward 1.0 and binary 1.0
6. layer archaeology: `docker save` and scan every layer for carved bytes and tripwire digests

The command exits non-zero if any bar is missed. `verify` does not auto-detect language or scope, so pass `--lang` and `--carve-scope` for every non-python entry. `--repo` is optional and is located under `../../harbor-tasks/repos-src/` from the entry when omitted. `--keep-image` keeps the built image for inspection.

## Worked matrix

python, function / file / folder scope:

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/python-a2a-python \
    --package-base src/ --file src/a2a/utils/task.py --func apply_history_length \
    --out .taskgen_out/py-fn
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/python-a2a-python \
    --package-base src/ --carve-scope file --include 'src/a2a/utils/task.py' \
    --out .taskgen_out/py-file
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/python-a2a-python \
    --package-base src/ --carve-scope folder --include 'src/a2a/utils/**' \
    --out .taskgen_out/py-folder
$PY -m taskgen.cli verify --all .taskgen_out/py-fn --lang python --carve-scope function \
    --repo ../../harbor-tasks/repos-src/python-a2a-python
```

go, function scope:

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/go-multigres --lang go \
    --file go/common/pgprotocol/server/listener.go --func assignConnectionID \
    --out .taskgen_out/go-fn
$PY -m taskgen.cli verify --all .taskgen_out/go-fn --lang go --carve-scope function \
    --repo ../../harbor-tasks/repos-src/go-multigres
```

rust, whole src tree:

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/rust-spacewasm --lang rust \
    --carve-scope folder --include 'src/**' --delete-whole-file --out .taskgen_out/rust-folder
$PY -m taskgen.cli verify --all .taskgen_out/rust-folder --lang rust --carve-scope folder \
    --repo ../../harbor-tasks/repos-src/rust-spacewasm
```

c, whole runtime:

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/c-xs --lang c \
    --carve-scope folder --include 'src/runtime/**' --delete-whole-file --out .taskgen_out/c-folder
$PY -m taskgen.cli verify --all .taskgen_out/c-folder --lang c --carve-scope folder \
    --repo ../../harbor-tasks/repos-src/c-xs
```

cpp, compiler middle-end (three narrow includes):

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/cpp-Rux --lang cpp \
    --carve-scope folder \
    --include 'Compiler/Semantic/**' --include 'Compiler/Ir/**' --include 'Compiler/CodeGen/**' \
    --delete-whole-file --out .taskgen_out/cpp-folder
$PY -m taskgen.cli verify --all .taskgen_out/cpp-folder --lang cpp --carve-scope folder \
    --repo ../../harbor-tasks/repos-src/cpp-Rux
```

java, widget library:

```bash
$PY -m taskgen.cli generate --repo ../../harbor-tasks/repos-src/java-tamboui --lang java \
    --carve-scope folder --include 'tamboui-widgets/src/main/java/**' --delete-whole-file \
    --out .taskgen_out/java-folder
$PY -m taskgen.cli verify --all .taskgen_out/java-folder --lang java --carve-scope folder \
    --repo ../../harbor-tasks/repos-src/java-tamboui
```

Verified floors: python and go carry function-level graded sets; rust grades 92 tests, c grades 91, cpp grades 170 doctest cases, and java grades 823 tests across 69 suites. Each combo produces RED reward 0.0 and GREEN reward 1.0 with the leak and integrity gates passing.

## Tests

```bash
uv run pytest -q
```

The suite covers the carve, staging, emit, measure, and verify paths plus one plugin test module per language. Current count: 823 passing.

## License

Released under the MIT License; see the `LICENSE` file. Copyright (c) 2026 Ethara.AI.

The vendored sample repositories under `../../harbor-tasks/repos-src/` and other third-party or upstream components retain their own original licenses.

