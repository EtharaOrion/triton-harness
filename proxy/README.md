# Claude Code OAuth Proxy — aurora harness

Run Multi-SWE-bench **trajectory generation** on your **Claude Code subscription**
(Pro/Max, via OAuth) instead of a metered `ANTHROPIC_API_KEY`.

The proxy is a small Anthropic-compatible HTTP server that runs on the host. It
reads the same OAuth token the `claude` CLI uses, and injects it into every
request. You point the harness's LLM config `base_url` at the proxy; from the
harness's perspective nothing else changes.

---

## Table of contents

- [How it works](#how-it-works)
- [What's in this folder](#whats-in-this-folder)
- [Requirements & constraints](#requirements--constraints)
- [Quick start (local / macOS)](#quick-start-local--macos)
- [Running on EC2 (Linux) — full integration](#running-on-ec2-linux--full-integration)
- [Configuration reference](#configuration-reference)
- [Verifying the proxy](#verifying-the-proxy)
- [Troubleshooting](#troubleshooting)
- [Security & Terms](#security--terms)

---

## How it works

The harness talks to Claude through **litellm** using the LLM config's
`base_url`. Point that at the proxy and the request flow becomes:

```
run_eval.sh  /  uv run multi-swebench-infer
  └─ OpenHands SDK  LLM(model="anthropic/claude-opus-4-8", base_url=<proxy>)
      └─ litellm "anthropic/" provider
          └─ POST {base_url}/v1/messages          headers: x-api-key: <stub>
              └─ [ claude_code_bridge PROXY on the host ]
                   • strips  x-api-key
                   • adds    Authorization: Bearer <your OAuth token>
                   • adds    anthropic-beta: oauth-2025-04-20
                   • injects "You are Claude Code…" system prefix (required for OAuth)
                   • auto-refreshes the token near expiry
                   └─ POST https://api.anthropic.com/v1/messages   ← billed to your subscription
```

Because the eval **agent runs inside a Docker container**, its LLM calls must
reach the proxy *on the host*. How the container addresses the host differs by
OS — see [EC2 networking](#3-point-the-config-at-the-docker-gateway).

The proxy also classifies Anthropic errors (429 transient vs. subscription cap,
401 token-invalid, 529 overloaded, 5xx) and retries/fails over accordingly. With
a multi-account pool it rotates accounts on cap exhaustion.

---

## What's in this folder

```
proxy/
├── claude_code_bridge/            # the proxy (standalone Python package, relative imports)
│   ├── __main__.py                #   CLI entry:  python -m claude_code_bridge
│   ├── bridge.py                  #   FastAPI app: forwarding, retries, streaming, failover
│   ├── credentials.py             #   OAuth load / refresh / cache + multi-account pool
│   └── errors.py                  #   Anthropic error classification
├── claude_code_bridge.sh          # start | stop | status | check | monitor (background + watchdog)
├── claude-code-oauth.json         # the LLM config you pass to the harness
├── .gitignore                     # ignores runtime artifacts (*.pid, logs/)
└── README.md                      # this file
```

No new dependencies: the proxy only needs `fastapi`, `uvicorn`, `httpx`, which
are already in the harness `.venv`.

---

## Requirements & constraints

- **Harness venv built** (`cd harness && make build`). Verify:
  `.venv/bin/python -c "import fastapi, uvicorn, httpx"`.
- **A Claude Code subscription** (Pro/Max) authenticated with the `claude` CLI
  at least once (that produces the OAuth credential this proxy reads).
- **`--workspace docker` only.** The proxy is host-local; it is **incompatible
  with `--workspace remote`** (the remote runtime cannot reach the host). This
  is the same constraint the harness's `--compression headroom` proxy has.

---

## Quick start (local / macOS)

macOS Docker Desktop resolves `host.docker.internal` automatically, so no
gateway lookup is needed.

```bash
cd harness

# 1. start the proxy (reads creds from your macOS Keychain)
proxy/claude_code_bridge.sh start
proxy/claude_code_bridge.sh status          # /healthz -> {"ok": true, ...}

# 2. set base_url for macOS in proxy/claude-code-oauth.json:
#      "base_url": "http://host.docker.internal:8765"

# 3. generate trajectories
LANGUAGE=python uv run multi-swebench-infer proxy/claude-code-oauth.json \
    --dataset bytedance-research/Multi-SWE-Bench --split python_verified \
    --workspace docker --max-iterations 100

# 4. stop when done
proxy/claude_code_bridge.sh stop
```

---

## Running on EC2 (Linux) — full integration

Two things differ from macOS: **there is no Keychain** (so you must place the
credential on the box as a file), and **`host.docker.internal` does not resolve
inside containers** (so you address the host by its docker-bridge gateway IP).

### 1. Provision & build

- An EC2 instance (Amazon Linux / Ubuntu) with **Docker installed and running**
  and the eval Docker images available (ECR login or local build, as usual).
- Clone the aurora repo and build the harness:
  ```bash
  cd aurora/harness
  make build
  .venv/bin/python -c "import fastapi, uvicorn, httpx; print('proxy deps OK')"
  ```

### 2. Put your Claude credential on the box (no Keychain on Linux)

`claude` login is a browser OAuth flow, so do it on a machine with a browser
(your laptop) and copy the resulting credential to EC2 as a file.

**On your Mac** (where `claude` is already logged in):
```bash
security find-generic-password -s "Claude Code-credentials" -w > claude_creds.json
```
**Copy to EC2 securely** (never commit this file):
```bash
scp claude_creds.json ec2-user@<host>:~/.config/aurora/claude_creds.json
ssh ec2-user@<host> 'chmod 600 ~/.config/aurora/claude_creds.json'
```
**Point the proxy at it using the pool form.** The pool's file provider writes
the *rotated* token **back to this same file** on every refresh, so the
credential survives the ~8-hour token refresh **and** proxy restarts:
```bash
export AURORA_CC_ACCOUNT_POOL="$HOME/.config/aurora/claude_creds.json"
```

> **Account sharing caveat.** Anthropic rotates the refresh token on every
> refresh. An account used by this proxy therefore can't also be used by an
> interactive `claude` CLI elsewhere (each refresh logs the other side out).
> Use a **dedicated account** for the eval box, or stop using that account's
> `claude` CLI once its credential lives on EC2.

### 3. Point the config at the docker gateway

On Linux the container reaches the host via the **docker0 bridge gateway**,
normally `172.17.0.1`. Confirm it:
```bash
ip -4 addr show docker0 | awk '/inet/ {print $2}'          # e.g. 172.17.0.1/16
# or authoritatively, the gateway of the default bridge network:
docker network inspect bridge -f '{{range .IPAM.Config}}{{.Gateway}}{{end}}'
```
Set that IP as `base_url` in [`proxy/claude-code-oauth.json`](claude-code-oauth.json)
(the shipped default is already `172.17.0.1:8765`):
```json
{
  "model": "anthropic/claude-opus-4-8",
  "base_url": "http://172.17.0.1:8765",
  "api_key": "sk-ant-oauth-bridge-stub",
  "timeout": 600,
  "num_retries": 2
}
```

### 4. Start the proxy

```bash
export AURORA_CC_ACCOUNT_POOL="$HOME/.config/aurora/claude_creds.json"
harness/proxy/claude_code_bridge.sh start
harness/proxy/claude_code_bridge.sh status     # expect: {"ok": true, "token_prefix": "sk-ant-oat01-..."}
```
The proxy binds `0.0.0.0:8765` (so the container can reach it via the gateway);
a watchdog restarts it if `/healthz` fails 3× in a row.

### 5. Generate trajectories

```bash
cd harness
LANGUAGE=python uv run multi-swebench-infer proxy/claude-code-oauth.json \
    --dataset bytedance-research/Multi-SWE-Bench --split python_verified \
    --workspace docker --num-workers 4 --max-iterations 100

# ...or the full pipeline:
./run_eval.sh --llm-config proxy/claude-code-oauth.json \
    --dataset <bundle>.jsonl --ecr-prefix <acct>.dkr.ecr.<region>.amazonaws.com/<repo>
```
Every worker's agent container routes its LLM calls through the one host proxy.

### 6. (Recommended) run the proxy as a systemd service

For unattended, reboot-surviving operation, run the proxy under systemd instead
of the background script. `Type=simple` + the package's foreground entrypoint:

```ini
# /etc/systemd/system/claude-code-bridge.service    (adjust User/paths)
[Unit]
Description=Claude Code OAuth proxy (aurora harness)
After=network-online.target docker.service
Wants=network-online.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user/aurora/harness/proxy
Environment=AURORA_CC_ACCOUNT_POOL=/home/ec2-user/.config/aurora/claude_creds.json
ExecStart=/home/ec2-user/aurora/harness/.venv/bin/python -m claude_code_bridge --host 0.0.0.0 --port 8765
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now claude-code-bridge
curl -fsS http://127.0.0.1:8765/healthz          # {"ok": true, ...}
journalctl -u claude-code-bridge -f              # logs
```
(When using systemd, don't also run `claude_code_bridge.sh start` — pick one.)

### 7. Confirm the container can reach the proxy

The harness's in-container **egress filter is a denylist** (it only 403s the
task's own repo/package to prevent cheating) — it does **not** block the LLM
endpoint. If a run still can't reach the proxy, test host reachability from a
throwaway container on the same bridge network:
```bash
docker run --rm curlimages/curl -sS http://172.17.0.1:8765/healthz
```
If that fails: the proxy isn't bound to `0.0.0.0`, the gateway IP is different,
or a host firewall is dropping container→host traffic on 8765. Escape hatch for
the egress filter (host env, forwarded into the container): `EGRESS_FILTER_DISABLE=1`.

---

## Configuration reference

All optional. Set in the environment before `start` (or in the systemd unit).

| Variable | Default | Purpose |
|---|---|---|
| `AURORA_CC_ACCOUNT_POOL` | *(unset)* | Colon-separated credential sources; enables rotation/failover **and** write-back to file sources. Entries: an absolute file path, `keychain:<service>`, or `default`. **Recommended on EC2** (single file path). |
| `AURORA_CC_CREDS_PATH` | *(unset)* | Single credentials JSON path for the default provider (read-only source; rotations cache to `~/.cache/aurora-harness/`, not back to this file — prefer the pool form for durability). |
| `AURORA_CC_WRITE_BACK_KEYCHAIN` | `0` | macOS only: on refresh, write the rotated token back to Keychain so the `claude` CLI stays logged in. |
| `AURORA_CC_BRIDGE_HOST` | `0.0.0.0` | Interface the proxy binds. Keep `0.0.0.0` for container reachability. |
| `AURORA_CC_BRIDGE_PORT` | `8765` | Proxy port. |
| `AURORA_CC_PYTHON` | `<harness>/.venv/bin/python` | Python used to run the proxy. |
| `AURORA_CC_DISABLE_MONITOR` | `0` | `1` disables the health watchdog. |
| `AURORA_CC_UPSTREAM` | `https://api.anthropic.com` | Upstream Anthropic base. |
| `AURORA_BRIDGE_READ_TIMEOUT` | `180` | Per-chunk read timeout (s). Raise for very long extended-thinking turns. |
| `AURORA_BRIDGE_REQUEST_TIMEOUT` | `600` | Overall request timeout (s). |

The **LLM config** ([`claude-code-oauth.json`](claude-code-oauth.json)) must use
an `anthropic/` model (so litellm speaks the native Messages API the proxy
proxies) and a non-empty stub `api_key` (litellm requires one; the proxy
discards it).

---

## Verifying the proxy

```bash
# credentials load (no traffic)
proxy/claude_code_bridge.sh check

# real round-trip through the proxy with only a STUB key — a 200 proves the
# proxy injected your OAuth token (a stub key can't authenticate on its own):
curl -sS -X POST http://127.0.0.1:8765/v1/messages \
  -H 'x-api-key: stub' -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \
  -d '{"model":"claude-opus-4-8","max_tokens":16,"messages":[{"role":"user","content":"reply pong"}]}'
```

This path was validated end-to-end (raw `/v1/messages`, litellm `anthropic/`,
and the OpenHands SDK `LLM` object the harness builds) — each returned a real
completion with only a stub key.

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `401 credentials_unavailable` | No creds found or refresh token dead. Re-seed the creds file (step 2); check `AURORA_CC_ACCOUNT_POOL` points at it. |
| Container can't reach the proxy (Linux) | Wrong gateway IP — run `ip -4 addr show docker0`; set that in `base_url`. Ensure the proxy binds `0.0.0.0`. |
| Works on macOS with `host.docker.internal`, fails on EC2 | Linux has no `host.docker.internal`; use the `172.17.0.1` gateway IP. |
| `claude` CLI logged out after using the proxy | Refresh-token rotation on a shared account. Use a dedicated account, or `AURORA_CC_WRITE_BACK_KEYCHAIN=1` (macOS). |
| `model not found` / 404 | The config `model` must be one your subscription serves (e.g. `anthropic/claude-opus-4-8`). |
| `--workspace remote` run gets no traffic | Not supported — the remote runtime can't reach a host-local proxy. Use `--workspace docker`. |
| Proxy stalls on very long turns | Raise `AURORA_BRIDGE_READ_TIMEOUT` (e.g. `300`). |

---

## Security & Terms

- The `api_key` in the config is a **stub**. The real secret is your OAuth token,
  read at runtime from the creds file — keep that file `chmod 600` and out of git.
  (This folder's `.gitignore` already excludes runtime artifacts; the creds file
  lives outside the repo, under `~/.config/aurora/`.)
- Do not commit `claude_creds.json`, `.pid`, or `logs/`.
- **Terms of service:** this routes eval traffic through your Claude Code
  subscription programmatically. Confirm that use is within your plan's terms
  before running at scale.
