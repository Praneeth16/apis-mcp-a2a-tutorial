# APIs vs MCP vs A2A — Hands-on Tutorials

Three Jupyter notebooks that teach **architecture decisions** between traditional APIs, the **Model Context Protocol (MCP)**, and **Agent-to-Agent (A2A)** collaboration. Companion to the slide deck `apis_mcp_a2a_course_deck_boston_analytics_v3.pptx`.

**Real MCP, fully self-contained per notebook.** Each notebook defines a `FastMCP` server inline, connects a real `ClientSession` **in-process** via `mcp.shared.memory.create_connected_server_and_client_session`, and drives the loop with a LangGraph ReAct agent (`langchain-google-genai` Gemini) over `langchain-mcp-adapters`. Every tool call goes through the actual MCP wire protocol (`initialize`, `tools/list`, `tools/call`) — no subprocess, no transport plumbing, no separate Python files.

## Layout

```
.
├── README.md
├── apis_mcp_a2a_course_deck_boston_analytics_v3.pptx
├── api_mcp_a2a_tutorial.ipynb              # API vs MCP vs A2A on live GitHub data
├── catalog_rag_via_mcp.ipynb               # Retail catalog RAG via real MCP
└── contract_risk_highlighter_mcp.ipynb     # Vendor-contract reviewer via real MCP
```

## Notebooks

| Notebook | What it builds | Key concepts |
|---|---|---|
| `api_mcp_a2a_tutorial.ipynb` | Open-source dependency health triage in three architectures | Live GitHub REST, in-process MCP server + LangChain client, A2A peer agents, latency comparison |
| `catalog_rag_via_mcp.ipynb` | Retail catalog assistant with semantic + structured retrieval | Inline FAISS index over Gemini embeddings, MCP tool discovery, naive-RAG baseline for comparison |
| `contract_risk_highlighter_mcp.ipynb` | Vendor MSA reviewer that flags risky clauses | Layered MCP tools (regex + lookup + embeddings + LLM), playbook grounding, ANSI-colored highlights, redline drafting |

## Requirements

- Python 3.10+
- A **Gemini API key** — get one free at https://aistudio.google.com/apikey
- Pip-installs (each notebook has its own `%pip install` cell):
  - `mcp` — official Python MCP SDK
  - `langchain-mcp-adapters` — bridges MCP tools into LangChain
  - `langchain-google-genai` — Gemini chat model
  - `langgraph` — prebuilt ReAct agent
  - `google-genai` — Gemini chat + embeddings
  - `faiss-cpu`, `numpy`, `pandas`
  - `requests`

Optional: `GITHUB_TOKEN` env var for higher GitHub rate limit (NB1 only).

## How to run

```bash
git clone https://github.com/Praneeth16/apis-mcp-a2a-tutorial.git
cd apis-mcp-a2a-tutorial
pip install jupyter
jupyter lab
```

Open any `.ipynb`, run cells top-to-bottom. The first code cell prompts for `GEMINI_API_KEY` via `getpass` so the key never lands in the notebook.

## How the in-notebook MCP works

Every notebook follows the same shape:

```python
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

# 1. Define the server inline. Tools read notebook globals (DataFrames, indexes, ...).
mcp_server = FastMCP("my-server")

@mcp_server.tool()
def my_tool(arg: str) -> dict:
    """Schema is auto-derived from this docstring + type hints."""
    ...

# 2. Open a real ClientSession in-process. Real MCP handshake runs.
async with create_connected_server_and_client_session(mcp_server) as session:
    listing = await session.list_tools()       # wire-level tools/list
    tools = await load_mcp_tools(session)      # adapt to LangChain Tools

    # 3. Drive the agent.
    agent = create_react_agent(llm, tools)
    result = await agent.ainvoke({"messages": [...]})
```

Want to run the same server externally? Add `if __name__ == "__main__": mcp_server.run(transport="stdio")` in a separate file. Tools and schemas don't change — Claude Desktop / VS Code / Cursor see the same MCP server.

## Notebook 1 — `api_mcp_a2a_tutorial.ipynb`

Rate a public open-source repo's health (HEALTHY / AT_RISK / CRITICAL) and draft a Slack note. Live data from `api.github.com`. Three implementations of the same workflow:
- **API path** — four typed HTTP calls, deterministic rule scoring, Gemini drafts the message.
- **MCP path** — same four functions exposed as MCP tools by an inline `FastMCP` server; LangGraph agent drives.
- **A2A path** — three opaque specialist agents (commits / issues / releases), each with own LLM judgment; orchestrator synthesizes.

Ends with a side-by-side comparison (verdict, score, latency, tool-call count) and a contrast run on a dormant repo.

## Notebook 2 — `catalog_rag_via_mcp.ipynb`

Retail-catalog assistant. Synthetic 50-SKU catalog (seed=42), embedded with `gemini-embedding-001` using asymmetric `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task types, indexed in FAISS `IndexFlatIP`.

Inline MCP server exposes `search_catalog`, `get_product`, `filter_products`, `check_return_policy`. LangGraph agent picks per query.

Includes a **naive-RAG baseline** (no MCP, single retrieval path) so students see what MCP buys: hard-constraint queries break naive RAG; the MCP agent routes them to `filter_products`.

## Notebook 3 — `contract_risk_highlighter_mcp.ipynb`

Vendor MSA reviewer. 11-category playbook + 20-clause precedent library labeled low/medium/high, embedded with Gemini and indexed in FAISS.

Inline MCP server exposes five layered tools — each uses the simplest mechanism that suffices:
- `split_clauses` — deterministic regex (no LLM)
- `classify_clause` — Gemini classifier
- `find_precedents` — FAISS over Gemini embeddings
- `get_playbook_position` — table lookup
- `score_clause` — Gemini scorer **grounded** on playbook + nearest precedents

Two consumers of the same server: a deterministic batch pipeline (raw `session.call_tool(...)` in fixed order) and an ad-hoc LangGraph agent for lawyer questions. ANSI-colored output: red=high, yellow=medium, green=low; trigger phrase bolded inside clause text. Executive summary block with overall posture.

`load_contract_from_url` lets you swap the synthetic contract for any plain-text contract on the web (SEC EDGAR / CUAD / GitHub-hosted templates).

## Decision rule

Pick the **least autonomous protocol** that delivers the outcome with acceptable governance:

1. Can deterministic service calls solve it? → **API**
2. Does an AI app need dynamic tool/context access? → **MCP**
3. Must independent agents coordinate stateful work? → **A2A**
4. Multiple layers? → **Hybrid**: A2A outside, MCP inside agents, APIs underneath.

## Deck mapping

| Deck section | Notebook coverage |
|---|---|
| §3 Decision rule of thumb | All three NB1 paths exemplify it |
| §4 Mental model (call / equip / collaborate) | NB1 §Path1 / §Path2 / §Path3 |
| §13–§18 MCP primitives, host–client–server, security | All three notebooks (real server + client) |
| §20–§25 A2A (Agent Card, Task lifecycle, artifacts) | NB1 §Path3 |
| §27 Comparison at a glance | NB1 §4 (computed live) |
| §34 Hybrid pattern | Stretch exercises in all three |

## License

MIT. Built for educational use.
