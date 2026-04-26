# APIs vs MCP vs A2A — Hands-on Tutorials (real MCP servers + clients)

Three Jupyter notebooks that teach **architecture decisions** between traditional APIs, the **Model Context Protocol (MCP)**, and **Agent-to-Agent (A2A)** collaboration. Companion to the slide deck `apis_mcp_a2a_course_deck_boston_analytics_v3.pptx`.

**Real MCP**, not simulated:
- Each tutorial spawns a standalone **MCP server** (`mcp_servers/*.py`) built with the official `mcp` SDK + `FastMCP`.
- Each notebook is an **MCP client** that speaks the JSON-RPC protocol over stdio.
- Tools are loaded into LangChain via `langchain-mcp-adapters`; a LangGraph ReAct agent powered by `langchain-google-genai` (Gemini) drives the loop.
- Same servers can plug into Claude Desktop, VS Code Copilot, Cursor, or any MCP-compliant client without code changes.

## Layout

```
.
├── README.md
├── apis_mcp_a2a_course_deck_boston_analytics_v3.pptx
├── api_mcp_a2a_tutorial.ipynb              # API vs MCP vs A2A on live GitHub data
├── catalog_rag_via_mcp.ipynb               # Retail catalog RAG via real MCP
├── contract_risk_highlighter_mcp.ipynb     # Vendor-contract reviewer via real MCP
└── mcp_servers/
    ├── dependency_health_server.py         # GitHub REST tools (used by NB1)
    ├── catalog_server.py                   # Gemini-embedded FAISS catalog (NB2)
    └── contract_server.py                  # Playbook + precedent library (NB3)
```

## Notebooks

| Notebook | What it builds | Key concepts |
|---|---|---|
| `api_mcp_a2a_tutorial.ipynb` | Open-source dependency health triage in three architectures | API contracts, real MCP server + LangChain client, A2A peer agents, latency comparison |
| `catalog_rag_via_mcp.ipynb` | Retail catalog assistant with semantic + structured retrieval | MCP server owns FAISS index over Gemini embeddings; LangGraph agent drives tool selection; naive-RAG vs MCP comparison |
| `contract_risk_highlighter_mcp.ipynb` | Vendor MSA reviewer that flags risky clauses | Layered MCP tools (deterministic + embeddings + LLM), playbook grounding, ANSI-colored highlights, redline drafting |

## Requirements

- Python 3.10+
- A **Gemini API key** — get one free at https://aistudio.google.com/apikey
- Pip-installs (each notebook has its own `%pip install` cell):
  - `mcp` — official Python MCP SDK (server + client)
  - `langchain-mcp-adapters` — bridges MCP tools into LangChain
  - `langchain-google-genai` — Gemini chat model
  - `langgraph` — prebuilt ReAct agent
  - `google-genai` — Gemini embeddings (used inside the servers)
  - `faiss-cpu`, `numpy`, `pandas`
  - `requests` (NB1 only)

Optional: `GITHUB_TOKEN` env var for higher GitHub rate limit (NB1 only).

## How to run

```bash
git clone https://github.com/Praneeth16/apis-mcp-a2a-tutorial.git
cd apis-mcp-a2a-tutorial
pip install jupyter
jupyter lab
```

Open any `.ipynb`, run cells top-to-bottom. The first code cell prompts for `GEMINI_API_KEY` via `getpass` so the key never lands in the notebook. The notebook itself spawns the MCP server as a subprocess; nothing extra to start.

## How the MCP path works

Each notebook follows the same pattern:

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

server_params = StdioServerParameters(
    command="python",
    args=["mcp_servers/<server>.py"],
    env={**os.environ},               # passes GEMINI_API_KEY to the server
)

async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()        # MCP handshake
        tools = await load_mcp_tools(session)   # tools/list + adapt to LangChain
        agent = create_react_agent(llm, tools)
        result = await agent.ainvoke({"messages": [...]})
```

Wire-level steps you can see in the output:
1. **Spawn** the server subprocess.
2. **Initialize** — protocol version + capability handshake.
3. `tools/list` — server returns advertised tools and their JSON schemas.
4. **Agent loop** — LangGraph ReAct picks tools; each call goes via stdio JSON-RPC to the server; server executes (HTTP / FAISS / Gemini); result returns.

## Notebook 1 — `api_mcp_a2a_tutorial.ipynb`

**Scenario:** rate a public open-source repo's health (HEALTHY / AT_RISK / CRITICAL) and draft a Slack note for the team depending on it.

**Real data source:** `https://api.github.com` — repo metadata, commits, issues, releases. Change `REPO = "owner/name"` to test any public project.

Three implementations of the same workflow:
- **API path** — four typed HTTP calls, deterministic rule scoring, Gemini drafts the message.
- **MCP path** — `mcp_servers/dependency_health_server.py` exposes the same four functions as MCP tools; the notebook is a LangGraph agent over MCP.
- **A2A path** — three opaque specialist agents (commits / issues / releases), each with its own LLM judgment; orchestrator synthesizes from artifacts only.

Final cells: side-by-side comparison table (verdict, score, latency, tool-call count) + a contrast run on a dormant repo.

## Notebook 2 — `catalog_rag_via_mcp.ipynb`

**Scenario:** retail-catalog assistant that handles fuzzy questions ("waterproof shell for alpine climbing") and hard constraints ("only headlamps over 500 lumens, in stock").

**Server (`mcp_servers/catalog_server.py`):** builds a synthetic 50-SKU catalog (seed=42), embeds with `gemini-embedding-001` using asymmetric `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types, indexes in FAISS `IndexFlatIP` (cosine). Exposes `search_catalog`, `get_product`, `filter_products`, `check_return_policy`.

**Client:** LangGraph ReAct agent with `langchain-google-genai`. Includes a **naive-RAG baseline** (no MCP, single retrieval path) so students see what MCP buys: hard-constraint queries break naive RAG; the MCP agent routes them to `filter_products`.

## Notebook 3 — `contract_risk_highlighter_mcp.ipynb`

**Scenario:** review a vendor MSA, classify each clause, score risk against a company playbook, propose redlines, output an executive summary.

**Server (`mcp_servers/contract_server.py`):** 11-category playbook + 20-clause precedent library labeled low/medium/high, embedded with `gemini-embedding-001` and indexed in FAISS. Exposes five layered tools — each uses the simplest mechanism that suffices:

- `split_clauses` — deterministic regex (no LLM)
- `classify_clause` — Gemini classifier
- `find_precedents` — FAISS over Gemini embeddings
- `get_playbook_position` — table lookup
- `score_clause` — Gemini scorer **grounded** on playbook + nearest precedents

**Client:** the notebook shows two consumers of the same server — a deterministic batch pipeline (raw `session.call_tool(...)` in fixed order) and an ad-hoc LangGraph agent for lawyer questions. ANSI-colored output: red=high, yellow=medium, green=low; trigger phrase bolded inside clause text. Executive summary block: count per risk level, high-risk clause numbers, overall posture (vendor-favorable / balanced / customer-favorable).

Includes `load_contract_from_url` so the pipeline can run on SEC EDGAR exhibits, GitHub-hosted templates, or the CUAD dataset.

## Deck mapping

| Deck section | Notebook coverage |
|---|---|
| §3 Decision rule of thumb | All three NB1 paths exemplify it |
| §4 Mental model (call / equip / collaborate) | NB1 §Path1 / §Path2 / §Path3 |
| §13–§18 MCP primitives, host–client–server, security | All three notebooks (real server + client) |
| §20–§25 A2A (Agent Card, Task lifecycle, artifacts) | NB1 §Path3 |
| §27 Comparison at a glance | NB1 §4 (computed live) |
| §34 Hybrid pattern (A2A outside, MCP inside, APIs underneath) | Stretch exercises in all three |

## Decision rule

Pick the **least autonomous protocol** that delivers the outcome with acceptable governance:

1. Can deterministic service calls solve it? → **API**
2. Does an AI app need dynamic tool/context access? → **MCP**
3. Must independent agents coordinate stateful work? → **A2A**
4. Multiple layers? → **Hybrid**: A2A outside, MCP inside agents, APIs underneath.

## License

MIT. Built for educational use.
