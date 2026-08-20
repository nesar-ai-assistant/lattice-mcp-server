# Using this server from any MCP client

Same drill as the other tutorial servers — see
[`spectra-mcp-server/docs/mcp-clients.md`](https://github.com/HEP-KE/spectra-mcp-server/blob/main/docs/mcp-clients.md)
for the full walkthrough (per-app configs, tunnels, hosting options). Only
the name and port differ here.

## The two facts every config needs

1. **stdio launch command**: `python -m mcp_server --transport stdio`, run so
   that this repo is importable — either the working directory is the repo
   root, or `PYTHONPATH` points at it.
2. **HTTP endpoint**: run `python -m mcp_server --transport streamable-http
   --port 8002` in a terminal; clients connect to
   `http://127.0.0.1:8002/mcp`.

`python` must be an environment with this repo's dependencies. GUI apps do
not inherit your shell, so use the absolute interpreter path
(`conda activate <env> && which python`).

## Claude Code

A project config (`.mcp.json`) is checked into this repo: activate the env,
`cd` here, run `claude`. Or add it yourself:

```bash
claude mcp add lattice -- python -m mcp_server --transport stdio   # from this repo root
claude mcp add --transport http lattice http://127.0.0.1:8002/mcp  # server already running
```

## Claude desktop app / Codex / Cursor

Same JSON/TOML shapes as the spectra doc, with the entry named `lattice`:

```json
{
  "mcpServers": {
    "lattice": {
      "command": "/ABS/PATH/TO/envs/<env>/bin/python",
      "args": ["-m", "mcp_server", "--transport", "stdio"],
      "env": { "PYTHONPATH": "/ABS/PATH/TO/lattice-mcp-server" }
    }
  }
}
```

## Port convention across the tutorial servers

| server | port |
|---|---|
| spectra | 8000 |
| gaia | 8001 |
| **lattice** | **8002** |
| bbn | 8003 |

Distinct ports are mandatory (each server clears its own port on startup),
and don't `pip install -e` several of these repos into one environment —
they export the same package names; per-server `PYTHONPATH`/working
directory keeps them apart.

For serving beyond localhost (`MCP_PUBLIC=1`, tunnels, persistent hosting,
`MCP_OUTPUT_ROOT`/`MCP_ARTIFACT_URL`), see the spectra doc — this wrapper is
identical.
