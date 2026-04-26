"""
MCP server: contract risk reviewer.

Bundles a curated playbook + a precedent library (20 labeled clauses),
embeds the precedents with gemini-embedding-001, and indexes in FAISS.

Exposes five tools over stdio:
  - split_clauses          (deterministic regex)
  - classify_clause        (Gemini classifier)
  - find_precedents        (FAISS over precedent library)
  - get_playbook_position  (table lookup)
  - score_clause           (Gemini scorer grounded on playbook + precedents)

Spawn:  python mcp_servers/contract_server.py
Requires GEMINI_API_KEY in environment.
"""
from __future__ import annotations

import json
import os
import re
import sys

import faiss
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("contract-risk")

if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set in server env", file=sys.stderr)
    sys.exit(1)

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
CHAT_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-001"


def _embed(texts: list[str], task_type: str) -> np.ndarray:
    out = []
    BATCH = 32
    for i in range(0, len(texts), BATCH):
        resp = _client.models.embed_content(
            model=EMBED_MODEL,
            contents=texts[i : i + BATCH],
            config=types.EmbedContentConfig(task_type=task_type),
        )
        out.extend([e.values for e in resp.embeddings])
    return np.asarray(out, dtype="float32")


# ---------------------------------------------------------------------------
# Playbook (company standards)
# ---------------------------------------------------------------------------
PLAYBOOK = {
    "fees": {"acceptable": "Fees fixed during initial term. Increases capped at CPI or 5% per renewal, with 60-day notice.",
             "red_flags": ["unilateral mid-term increase", "no cap on price increases", "<60 days notice"]},
    "termination": {"acceptable": "Customer may terminate for convenience with 30-day notice. Renewal opt-out window 30-60 days.",
                    "red_flags": ["no termination for convenience", "opt-out window > 60 days", "auto-renewal > 12 months"]},
    "data": {"acceptable": "Vendor uses Customer Data only to provide the Services. Aggregated/anonymized use only with consent.",
             "red_flags": ["training ML on customer data without consent", "broad derived-data rights"]},
    "ip": {"acceptable": "Customer retains ownership of feedback. Vendor receives a limited license to incorporate non-confidential feedback.",
           "red_flags": ["perpetual irrevocable license", "rights extend to user-submitted content"]},
    "warranties": {"acceptable": "Vendor warrants the Services will perform materially as described in the Documentation.",
                   "red_flags": ["AS IS with full disclaimer", "no documentation warranty"]},
    "indemnity": {"acceptable": "Mutual IP indemnity by Vendor. Customer indemnifies only for misuse, capped at fees paid.",
                  "red_flags": ["customer-only indemnity", "unlimited indemnity", "no IP indemnity from vendor"]},
    "liability": {"acceptable": "Aggregate liability cap >= 12 months of fees. Mutual carve-outs for confidentiality and data breach.",
                  "red_flags": ["cap < 12 months fees", "asymmetric carve-outs favoring vendor"]},
    "governing_law": {"acceptable": "Neutral venue; arbitration with class-action waiver only if customer-favorable seat.",
                      "red_flags": ["vendor home-state arbitration", "class waiver in consumer-facing contracts"]},
    "assignment": {"acceptable": "Mutual consent required. Permitted assignment to affiliates with notice.",
                   "red_flags": ["asymmetric - vendor may assign freely while customer cannot"]},
    "sla": {"acceptable": "99.9% uptime with service credits. Defined incident response times.",
            "red_flags": ["no SLA", "commercially reasonable efforts only", "no service credits"]},
    "confidentiality": {"acceptable": "Mutual NDA. Survives 5 years post-termination; perpetual for trade secrets.",
                        "red_flags": ["survival < 3 years", "asymmetric obligations"]},
}


# ---------------------------------------------------------------------------
# Precedent library
# ---------------------------------------------------------------------------
PRECEDENTS = pd.DataFrame([
    {"id": "P-001", "clause_type": "fees",          "verdict": "low",    "text": "Fees are fixed for the initial term. Renewal increases shall not exceed 5% or CPI, whichever is lower, with 60 days notice."},
    {"id": "P-002", "clause_type": "fees",          "verdict": "high",   "text": "Vendor may adjust fees at any time upon written notice; no cap on increases."},
    {"id": "P-003", "clause_type": "termination",   "verdict": "low",    "text": "Either party may terminate for convenience with 30 days written notice."},
    {"id": "P-004", "clause_type": "termination",   "verdict": "high",   "text": "Auto-renews for successive 12-month terms; 90-day non-renewal notice required."},
    {"id": "P-005", "clause_type": "data",          "verdict": "low",    "text": "Vendor processes Customer Data solely to provide the Services; no training use without explicit consent."},
    {"id": "P-006", "clause_type": "data",          "verdict": "high",   "text": "Vendor may use anonymized customer data to train its machine learning models for any purpose."},
    {"id": "P-007", "clause_type": "ip",            "verdict": "medium", "text": "Customer grants Vendor a non-exclusive license to use feedback, with no obligation of attribution."},
    {"id": "P-008", "clause_type": "ip",            "verdict": "high",   "text": "Customer grants a perpetual, irrevocable, royalty-free license to all submitted content."},
    {"id": "P-009", "clause_type": "warranties",    "verdict": "high",   "text": "Services are provided AS IS; all warranties expressly disclaimed."},
    {"id": "P-010", "clause_type": "warranties",    "verdict": "low",    "text": "Vendor warrants the Services will perform materially in accordance with the Documentation."},
    {"id": "P-011", "clause_type": "indemnity",     "verdict": "high",   "text": "Customer shall indemnify Vendor from all claims arising from use, without limit."},
    {"id": "P-012", "clause_type": "indemnity",     "verdict": "low",    "text": "Vendor indemnifies Customer for third-party IP claims; Customer indemnifies for misuse, capped at fees paid in the prior 12 months."},
    {"id": "P-013", "clause_type": "liability",     "verdict": "high",   "text": "Vendor's liability is capped at fees paid in the prior 3 months."},
    {"id": "P-014", "clause_type": "liability",     "verdict": "low",    "text": "Liability is mutually capped at 12 months of fees, with carve-outs for confidentiality breach and gross negligence."},
    {"id": "P-015", "clause_type": "governing_law", "verdict": "medium", "text": "Disputes resolved by arbitration in vendor's home state; class actions waived."},
    {"id": "P-016", "clause_type": "assignment",    "verdict": "high",   "text": "Vendor may assign freely. Customer may not assign without prior consent."},
    {"id": "P-017", "clause_type": "assignment",    "verdict": "low",    "text": "Either party may assign to a successor with notice; otherwise mutual written consent required."},
    {"id": "P-018", "clause_type": "sla",           "verdict": "high",   "text": "Vendor will use commercially reasonable efforts; no uptime guarantee or service credits."},
    {"id": "P-019", "clause_type": "sla",           "verdict": "low",    "text": "99.9% monthly uptime; service credits up to 25% of monthly fees for breaches."},
    {"id": "P-020", "clause_type": "confidentiality", "verdict": "low",  "text": "Mutual confidentiality; obligations survive 5 years post-termination, perpetual for trade secrets."},
])

print(f"[contract-server] embedding {len(PRECEDENTS)} precedents...", file=sys.stderr)
_prec_vecs = _embed(PRECEDENTS["text"].tolist(), task_type="RETRIEVAL_DOCUMENT")
faiss.normalize_L2(_prec_vecs)
_prec_index = faiss.IndexFlatIP(_prec_vecs.shape[1])
_prec_index.add(_prec_vecs)
print(f"[contract-server] index ready: {_prec_index.ntotal} vectors", file=sys.stderr)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@mcp.tool()
def split_clauses(contract_text: str) -> list[dict]:
    """Split a numbered contract into clauses. Returns list of {n, heading, text}. Deterministic, no LLM."""
    pattern = re.compile(
        r"^\s*(\d+)\.\s+([^.\n]+?)\.\s+(.+?)(?=^\s*\d+\.|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return [
        {"n": int(m.group(1)), "heading": m.group(2).strip(), "text": m.group(3).strip()}
        for m in pattern.finditer(contract_text)
    ]


@mcp.tool()
def classify_clause(clause_text: str) -> dict:
    """Classify a clause into one playbook category: fees, termination, data, ip, warranties, indemnity, liability, governing_law, assignment, sla, confidentiality."""
    cats = list(PLAYBOOK.keys())
    prompt = (
        f"Classify this contract clause into exactly one of: {cats}. "
        f'Reply JSON {{"category": "..."}}.\n\nClause: {clause_text}'
    )
    r = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(r.text)


@mcp.tool()
def find_precedents(clause_text: str, k: int = 3) -> list[dict]:
    """Semantic search over the precedent library. Returns top-k similar past clauses with risk verdicts."""
    qv = _embed([clause_text], task_type="RETRIEVAL_QUERY")
    faiss.normalize_L2(qv)
    scores, ids = _prec_index.search(qv, k)
    out = []
    for s, i in zip(scores[0], ids[0]):
        row = PRECEDENTS.iloc[int(i)]
        out.append({
            "id": row["id"],
            "clause_type": row["clause_type"],
            "verdict": row["verdict"],
            "text": row["text"],
            "score": float(s),
        })
    return out


@mcp.tool()
def get_playbook_position(category: str) -> dict:
    """Return the company's standard 'acceptable' position and 'red_flags' for a clause category."""
    return PLAYBOOK.get(category, {"error": f"unknown category {category}"})


@mcp.tool()
def score_clause(clause_text: str, category: str) -> dict:
    """Score a clause against playbook + precedents. Returns {risk, rationale, highlight, redline_suggestion}."""
    pb = PLAYBOOK.get(category, {})
    qv = _embed([clause_text], task_type="RETRIEVAL_QUERY")
    faiss.normalize_L2(qv)
    scores, ids = _prec_index.search(qv, 3)
    prec = []
    for s, i in zip(scores[0], ids[0]):
        row = PRECEDENTS.iloc[int(i)]
        prec.append({"id": row["id"], "verdict": row["verdict"], "text": row["text"]})
    prompt = (
        "You are a contract reviewer. Score the clause against the company playbook and "
        "the most similar precedents.\n\n"
        f"CLAUSE:\n{clause_text}\n\n"
        f"PLAYBOOK ({category}):\n{json.dumps(pb)}\n\n"
        f"PRECEDENTS:\n{json.dumps(prec, indent=2)}\n\n"
        'Reply JSON: {"risk": "low|medium|high", '
        '"rationale": "...", '
        '"highlight": "verbatim phrase from clause that triggers risk, or empty", '
        '"redline_suggestion": "proposed replacement text"}'
    )
    r = _client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    return json.loads(r.text)


if __name__ == "__main__":
    mcp.run(transport="stdio")
