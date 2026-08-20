# lattice-mcp-server

An MCP server that exposes science tools — the thermodynamics of hot,
strongly interacting matter from lattice QCD — to any LLM agent. Built on
the same pattern as
[`spectra-mcp-server`](https://github.com/HEP-KE/spectra-mcp-server) and
[`gaia-mcp-server`](https://github.com/HEP-KE/gaia-mcp-server); the
multi-agent client in
[`multiagent-client-demo`](https://github.com/HEP-KE/multiagent-client-demo)
drives any of these servers unchanged.

## The science

Heat nuclear matter past about 156 MeV (~2 × 10¹² K) and hadrons dissolve
into a quark–gluon plasma. This is not speculation: it is a first-principles
result of lattice QCD, it is recreated on Earth in heavy-ion collisions at
RHIC and the LHC, and the universe itself cooled through this transition
about 10 μs after the Big Bang. The tools here compute the two descriptions
that meet at the crossover:

- the **equation of state** of QCD matter — the published parameterization of
  continuum-extrapolated (2+1)-flavor lattice results (Bazavov et al.,
  HotQCD collaboration, [PRD 90, 094503 (2014)](https://arxiv.org/abs/1407.6387)),
- the **hadron resonance gas** (HRG) — a non-interacting gas of ~1200 hadrons
  and resonances (QMHRG2020 list), the confined-phase description that must
  fail once quarks are liberated.

Where the dashed HRG curves peel away from the lattice curves *is*
deconfinement:

![QCD equation of state vs hadron resonance gas](output/qcd_eos.png)

Everything is computed with the
[**LatticeQCD AnalysisToolbox**](https://github.com/LatticeQCD/AnalysisToolbox)
(`pip install latqcdtools`), the HotQCD collaboration's public analysis
package. The hadron list bundled in `data/` comes from that repository (MIT
license).

## Tools

| tool | what it does |
|---|---|
| `describe_qcd_thermodynamics()` | physics context, validity ranges, references — call it first |
| `compute_qcd_eos(output_dir, ...)` | lattice EoS: p/T⁴, ε/T⁴, s/T³, trace anomaly, c_s² on a T grid → CSV |
| `compute_hrg_thermodynamics(output_dir, ...)` | the same columns from the hadron resonance gas → CSV |
| `plot_eos(eos_file, output_dir, hrg_file=None)` | the classic ε/T⁴, 3p/T⁴, (3/4)s/T³ figure, crossover band, ideal-gas limit, optional HRG overlay |
| `plot_speed_of_sound(eos_file, output_dir, ...)` | c_s²(T) with the softest point + trace anomaly panels |
| `compute_charge_cumulants(output_dir, ...)` | HRG baryon-number susceptibilities χ₂ᴮ, χ₄ᴮ (heavy-ion fluctuation observables) → CSV |
| `plot_charge_cumulants(cumulants_file, output_dir)` | χₙᴮ(T) and the kurtosis ratio χ₄ᴮ/χ₂ᴮ against the hadron-gas expectation |
| `compute_running_coupling(output_dir, ...)` | 2-loop α_s(μ): asymptotic freedom → CSV + PNG |

Two conventions worth copying into any science MCP server:

1. Every tool returns `{status, files, message, metadata}` (`ArtifactResult`).
2. Arrays move between tools **as file paths**, never through the agent's
   context window.

## Install

```bash
conda create -n lattice-tutorial python=3.12 -y
conda activate lattice-tutorial
pip install -e ".[dev]"
pytest
```

Already have the `spectra-tutorial` env from the other servers? You can
reuse it: `pip install latqcdtools`, then always launch this server (and
pytest) **from this repo's root directory** — all tutorial servers export
packages named `tools` and `mcp_server`, so don't `pip install -e` several
of them into one env; running from the repo root makes the local packages
win.

## Run the server

```bash
python -m mcp_server --transport streamable-http --port 8002
```

Clients connect to `http://127.0.0.1:8002/mcp`. Port **8002** by convention,
so this server can run alongside spectra (8000), gaia (8001) and bbn (8003).
Stop with **Ctrl+C**.

To use this server from Claude Code, the Claude desktop app, Codex, Cursor,
or any other MCP client, see [`docs/mcp-clients.md`](docs/mcp-clients.md) —
a checked-in `.mcp.json` already wires it into Claude Code.

## A question to ask an agent

> "At what temperature does hadronic matter melt? Compute the lattice QCD
> equation of state and the hadron resonance gas, overlay them, and tell me
> where and why they disagree."
