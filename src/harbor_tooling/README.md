# harbor_tooling (vendored)

Verbatim copy of `harbor-tasks/shared/tooling`, vendored so the harness
resolves every import from inside its own worktree with no outside mapping.

    source: harbor-tasks/shared/tooling
    commit: 18e4155570f222aa145b899b853ed2ea3a7db4a9

taskgen imports these by bare name (`import sigextract`,
`from retrieval.chunk import build_corpus`, ...); `taskgen/_tooling_path.py`
puts this directory on sys.path.

DRIFT MATTERS: these are the same implementations the shipped harbor dataset
was generated with. If they diverge from upstream, generated tasks silently
decouple from the reference corpus. Refresh with:

    rsync -a --exclude __pycache__ ../../harbor-tasks/shared/tooling/ src/harbor_tooling/
    # then re-run: uv run pytest -q
