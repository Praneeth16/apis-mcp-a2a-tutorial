"""
MCP server: retail catalog with Gemini-embedded vector search.

Builds a synthetic catalog (50 SKUs, 5 categories, reproducible seed=42),
embeds product text via gemini-embedding-001, and indexes in FAISS.

Exposes four tools over stdio:
  - search_catalog       (semantic)
  - get_product          (exact lookup)
  - filter_products      (structured)
  - check_return_policy  (policy lookup)

Spawn:  python mcp_servers/catalog_server.py
Requires GEMINI_API_KEY in environment.
"""
from __future__ import annotations

import os
import random
import sys

import faiss
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("retail-catalog")

# ---------------------------------------------------------------------------
# Gemini client (embeddings only)
# ---------------------------------------------------------------------------
if not os.environ.get("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY not set in server env", file=sys.stderr)
    sys.exit(1)

_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
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
# Synthetic catalog (deterministic seed)
# ---------------------------------------------------------------------------
random.seed(42)

TEMPLATES = {
    "running_shoes": {
        "names": [
            "Velocity 5", "AirGlide Pro", "Trail Runner X", "Marathon Lite", "Cushion Max",
            "Sprint Carbon", "Daily Trainer", "Race Flat 2", "Recovery Plush", "All-Terrain Run",
        ],
        "price": (75, 220),
        "return_days": 30,
        "materials": ["engineered mesh", "flyknit", "recycled polyester"],
        "use_cases": ["road running", "speed work", "long distance", "recovery runs", "trail running"],
    },
    "hiking_boots": {
        "names": [
            "Summit GTX", "Ridge Walker", "Alpine Pro", "Backcountry Mid", "Canyon Boot",
            "Glacier 8", "Trail Master", "Peakbagger", "Forest Light", "Tundra Insulated",
        ],
        "price": (140, 380),
        "return_days": 30,
        "materials": ["full-grain leather", "nubuck", "synthetic mesh + leather"],
        "use_cases": ["day hikes", "backpacking", "mountaineering", "winter hiking", "wet conditions"],
    },
    "jackets": {
        "names": [
            "Stormshield Hardshell", "DownPuff 800", "Wind Breaker Lite", "Rain Cell", "Insulated Parka",
            "Fleece Mid", "Alpha Hybrid", "Packable Rain", "3-in-1 Travel", "Soft-Shell Trek",
        ],
        "price": (90, 600),
        "return_days": 30,
        "materials": ["GORE-TEX 3L", "800-fill goose down", "Pertex Quantum", "recycled nylon"],
        "use_cases": ["alpine climbing", "winter commuting", "shoulder season hiking", "backpacking", "city + travel"],
    },
    "backpacks": {
        "names": [
            "Daybreak 22", "Trekker 45", "Summit Haul 65", "Commuter 18", "Ultralight 35",
            "Hydration Vest 8", "Camera Pack 24", "Travel Carry 40", "Kid Pack 12", "Climber 30",
        ],
        "price": (60, 320),
        "return_days": 30,
        "materials": ["420D ripstop nylon", "Dyneema composite", "recycled polyester"],
        "use_cases": ["day hikes", "multi-day trips", "daily commute", "travel", "climbing"],
    },
    "headlamps": {
        "names": [
            "Beam 200", "Trail Lite USB", "Alpine Pro 600", "Runner Strap", "Camp Lantern Combo",
            "MicroBeam", "Storm 800", "Bivy Light", "Kid Glow", "Tactical 1000",
        ],
        "price": (25, 180),
        "return_days": 14,
        "materials": ["polycarbonate", "aluminum housing"],
        "use_cases": ["trail running", "camping", "alpine starts", "emergency kit", "around camp"],
    },
}


def _make_product(cat: str, idx: int) -> dict:
    t = TEMPLATES[cat]
    name = t["names"][idx]
    price = round(random.uniform(*t["price"]), 2)
    material = random.choice(t["materials"])
    use_case = random.choice(t["use_cases"])
    waterproof = random.choice([True, False]) if cat in ("hiking_boots", "jackets", "backpacks") else False
    weight_g = random.randint(200, 1800)
    in_stock = random.random() > 0.15
    final_sale = (price < t["price"][0] * 1.1) and random.random() < 0.2
    desc = (
        f"The {name} is built for {use_case}. Made from {material}"
        f"{', fully waterproof' if waterproof else ''}, weighing {weight_g}g. "
        f"{'Clearance - final sale.' if final_sale else 'Tested for everyday performance.'}"
    )
    return {
        "sku": f"{cat[:3].upper()}-{idx + 1:03d}",
        "name": name,
        "category": cat,
        "price_usd": price,
        "in_stock": in_stock,
        "material": material,
        "use_case": use_case,
        "waterproof": waterproof,
        "weight_g": weight_g,
        "description": desc,
        "return_policy": (
            "Final sale - non-returnable."
            if final_sale
            else f"Returns accepted within {t['return_days']} days unworn."
        ),
    }


CATALOG = pd.DataFrame(
    [_make_product(cat, i) for cat in TEMPLATES for i in range(10)]
)


# ---------------------------------------------------------------------------
# Build FAISS index at startup
# ---------------------------------------------------------------------------
def _doc_text(row) -> str:
    return f"{row['name']} | {row['category']} | {row['use_case']} | {row['material']} | {row['description']}"


print(f"[catalog-server] embedding {len(CATALOG)} products...", file=sys.stderr)
_doc_vecs = _embed([_doc_text(r) for _, r in CATALOG.iterrows()], task_type="RETRIEVAL_DOCUMENT")
faiss.normalize_L2(_doc_vecs)
_index = faiss.IndexFlatIP(_doc_vecs.shape[1])
_index.add(_doc_vecs)
print(f"[catalog-server] index ready: {_index.ntotal} vectors, dim={_doc_vecs.shape[1]}", file=sys.stderr)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------
@mcp.tool()
def search_catalog(query: str, k: int = 5) -> list[dict]:
    """Semantic search over the product catalog. Use for fuzzy intent (e.g. 'rain jacket for hiking'). Returns top-k candidates with snippet and similarity score."""
    qv = _embed([query], task_type="RETRIEVAL_QUERY")
    faiss.normalize_L2(qv)
    scores, ids = _index.search(qv, k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        r = CATALOG.iloc[int(idx)]
        results.append(
            {
                "sku": r["sku"],
                "name": r["name"],
                "category": r["category"],
                "price_usd": float(r["price_usd"]),
                "in_stock": bool(r["in_stock"]),
                "snippet": r["description"],
                "score": float(score),
            }
        )
    return results


@mcp.tool()
def get_product(sku: str) -> dict:
    """Fetch the full product record for an exact SKU. Use after search_catalog when you need price/stock/weight/material/return_policy."""
    hit = CATALOG[CATALOG["sku"] == sku]
    if hit.empty:
        return {"error": f"sku {sku} not found"}
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in hit.iloc[0].to_dict().items()}


@mcp.tool()
def filter_products(
    category: str | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
    waterproof: bool | None = None,
) -> list[dict]:
    """Structured filter. Use when the user specifies hard constraints (category, max_price, in_stock_only, waterproof). Categories: running_shoes, hiking_boots, jackets, backpacks, headlamps."""
    df = CATALOG.copy()
    if category:
        df = df[df["category"] == category]
    if max_price:
        df = df[df["price_usd"] <= max_price]
    if in_stock_only:
        df = df[df["in_stock"]]
    if waterproof is not None:
        df = df[df["waterproof"] == waterproof]
    return df[["sku", "name", "price_usd", "in_stock"]].head(20).to_dict(orient="records")


@mcp.tool()
def check_return_policy(sku: str) -> dict:
    """Return the return-policy text for a SKU. Read-only."""
    hit = CATALOG[CATALOG["sku"] == sku]
    if hit.empty:
        return {"error": f"sku {sku} not found"}
    return {"sku": sku, "return_policy": hit.iloc[0]["return_policy"]}


if __name__ == "__main__":
    mcp.run(transport="stdio")
