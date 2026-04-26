# APIs vs MCP vs A2A — Hands-on Tutorials

Three Jupyter notebooks that teach **architecture decisions** between traditional APIs, the **Model Context Protocol (MCP)**, and **Agent-to-Agent (A2A)** collaboration. Built to accompany the slide deck `apis_mcp_a2a_course_deck_boston_analytics_v3.pptx`.

Every notebook uses **live LLM calls** (Google Gemini) — no canned answers. Two of the three also use **Gemini embeddings** with FAISS for retrieval.

## What's inside

| Notebook | What it builds | Key concepts |
|---|---|---|
| `api_mcp_a2a_tutorial.ipynb` | Open-source dependency health triage, three ways | API contracts, MCP host loop, A2A peer agents, tool discovery, task lifecycle, latency comparison |
| `catalog_rag_via_mcp.ipynb` | Retail catalog assistant with semantic + structured retrieval | Gemini embeddings, FAISS, MCP tool discovery, hybrid retrieval, naive-RAG vs MCP comparison |
| `contract_risk_highlighter_mcp.ipynb` | Vendor-contract reviewer that flags risky clauses | Layered tools (deterministic + embeddings + LLM), playbook-grounded judgment, auditable highlights, redline drafting |

## Requirements

- Python 3.10+
- A **Gemini API key** — get one free at https://aistudio.google.com/apikey
- Pip-installs (run inside each notebook):
  - `google-genai` — Gemini chat + embeddings
  - `faiss-cpu` — open-source vector index (notebooks 2 + 3)
  - `requests` — live GitHub REST calls (notebook 1)
  - `numpy`, `pandas`

Optional: `GITHUB_TOKEN` env var for higher GitHub API rate limit (notebook 1 only — public endpoints work without auth at 60 req/hr).

## How to run

```bash
git clone https://github.com/Praneeth16/apis-mcp-a2a-tutorial.git
cd apis-mcp-a2a-tutorial
pip install jupyter
jupyter lab
```

Open any `.ipynb`, run cells top-to-bottom. The first code cell prompts for your `GEMINI_API_KEY` via `getpass` so the key never lands in the file.

## Notebook 1 — `api_mcp_a2a_tutorial.ipynb`

**Scenario:** rate a public open-source repo's health (HEALTHY / AT_RISK / CRITICAL) and draft a Slack note for the team depending on it.

**Real data source:** `https://api.github.com` — repo metadata, recent commits, open issues, releases. Change `REPO = "owner/name"` to test any public project.

Three implementations of the same workflow:
- **API path** — four typed HTTP calls, deterministic rule scoring, Gemini drafts the message.
- **MCP path** — same four functions advertised as tools; Gemini host loop discovers + picks tools at runtime.
- **A2A path** — three opaque specialist agents (commits / issues / releases), each with own LLM judgment; orchestrator synthesizes from artifacts only.

Ends with a side-by-side comparison table showing **verdict, score, latency, tool calls** per path.

## Notebook 2 — `catalog_rag_via_mcp.ipynb`

**Scenario:** a retail-catalog assistant that handles fuzzy questions ("waterproof shell for alpine climbing") and hard constraints ("only headlamps over 500 lumens, in stock").

**Stack:** synthetic catalog of 50 SKUs across 5 categories with realistic price/stock/policy variance → Gemini embeddings (`gemini-embedding-001` with asymmetric `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types) → FAISS `IndexFlatIP` (cosine).

Four MCP-style tools: `search_catalog`, `get_product`, `filter_products`, `check_return_policy`. The Gemini host picks per query.

Includes a **naive-RAG baseline** so students see what MCP buys: naive RAG hardcodes one retrieval path; the MCP host loop routes hard constraints to `filter_products` and fuzzy intent to `search_catalog`.

## Notebook 3 — `contract_risk_highlighter_mcp.ipynb`

**Scenario:** review a vendor MSA, classify each clause, score risk against a company playbook, propose redlines, output an executive summary.

**Stack:** synthetic MSA with deliberately mixed clauses (uncapped indemnity, 3-month liability cap, perpetual feedback license, AS-IS, etc.) + curated playbook (11 categories) + 20-clause precedent library labeled low/medium/high.

Five layered MCP tools — each uses the simplest mechanism that suffices:
- `split_clauses` — deterministic regex (no LLM)
- `classify_clause` — Gemini classifier
- `find_precedents` — FAISS over Gemini embeddings of the precedent library
- `get_playbook_position` — table lookup
- `score_clause` — Gemini scorer **grounded** on playbook + nearest precedents

Output: ANSI-colored risk-level highlights, verbatim trigger phrases, redline suggestions, executive summary with overall posture (vendor-favorable / balanced / customer-favorable).

Includes a **real-contract loader** (`load_contract_from_url`) so the pipeline can run on SEC EDGAR exhibits, GitHub-hosted templates, or the CUAD dataset.

## Deck mapping

| Deck section | Notebook coverage |
|---|---|
| §3 Decision rule of thumb | All three NB1 paths exemplify it |
| §4 Mental model (call / equip / collaborate) | NB1 §Path1 / §Path2 / §Path3 |
| §13–§18 MCP primitives, host–client–server, security | NB1 §Path2 + NB2 + NB3 |
| §20–§25 A2A (Agent Card, Task lifecycle, artifacts) | NB1 §Path3 |
| §27 Comparison at a glance | NB1 §4 (computed live) |
| §34 Hybrid pattern (A2A outside, MCP inside, APIs underneath) | Stretch exercises in NB2 + NB3 |
| §43 Local directory structure | This README |

## Decision rule

Pick the **least autonomous protocol** that delivers the outcome with acceptable governance:

1. Can deterministic service calls solve it? → **API**
2. Does an AI app need dynamic tool/context access? → **MCP**
3. Must independent agents coordinate stateful work? → **A2A**
4. Multiple layers? → **Hybrid**: A2A outside, MCP inside agents, APIs underneath.

## License

MIT. Built for educational use.
