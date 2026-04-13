#!/usr/bin/env python3
"""
search_advanced.py - Diversity-First RAG Pipeline
Proposed: Dense → MMR → CrossEncoder Reranker

Usage:
    python search_advanced.py "your query"
    python search_advanced.py "your query" --verbose
    python search_advanced.py "your query" --json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Configuration ────────────────────────────────────────────────────────────
# chroma_path and collection_name can be overridden via environment variables.
# Set CHROMA_PATH and CHROMA_COLLECTION in your environment (or .env file)
# instead of hardcoding local paths in source code.
DEFAULT_CONFIG = {
    "embedding_model": "intfloat/multilingual-e5-base",
    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    # For Japanese corpora, consider:
    # "reranker_model": "hotchpotch/japanese-reranker-cross-encoder-small-v1"
    "chroma_path": os.environ.get("CHROMA_PATH", "./chroma_db"),
    "collection_name": os.environ.get("CHROMA_COLLECTION", "my_collection"),
    "vector_k": 20,
    "mmr_k": 7,
    "final_k": 3,
    "lambda_mult": 0.7,
}


# ── Security: ChromaDB path validation ───────────────────────────────────────
def _validate_chroma_path(path: str) -> str:
    """
    Validate the ChromaDB path before use.

    Rejects:
    - URL schemes (http/https/ftp) — ChromaDB requires a local filesystem path
    - Null bytes — prevent path injection on some OS implementations

    Resolves the path to its absolute form so that the caller always sees
    the concrete directory that will be opened, making accidental sensitive-
    path exposure visible in logs and error messages rather than hidden.
    """
    if not isinstance(path, str) or not path.strip():
        raise ValueError("chroma_path must be a non-empty string.")

    if "\x00" in path:
        raise ValueError("chroma_path must not contain null bytes.")

    lowered = path.strip().lower()
    for scheme in ("http://", "https://", "ftp://", "s3://"):
        if lowered.startswith(scheme):
            raise ValueError(
                f"chroma_path must be a local filesystem path, not a URL. Got: {path!r}\n"
                "Set CHROMA_PATH in your environment to point to a local directory."
            )

    resolved = Path(path).resolve()
    return str(resolved)

# ── Model caches (reused within the same process) ────────────────────────────
_embed_model = None
_rerank_model = None
_collection = None


def get_embed_model(config: dict):
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config["embedding_model"], device="cpu")
    return _embed_model


def get_rerank_model(config: dict):
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(config["reranker_model"])
    return _rerank_model


def get_collection(config: dict):
    global _collection
    if _collection is None:
        safe_path = _validate_chroma_path(config["chroma_path"])
        client = chromadb.PersistentClient(path=safe_path)
        _collection = client.get_collection(config["collection_name"])
    return _collection


# ── Stage 1: Dense Vector Search ─────────────────────────────────────────────
def vector_search(query: str, top_k: int, config: dict):
    model = get_embed_model(config)
    collection = get_collection(config)

    # E5 models require "query: " prefix for query encoding
    query_emb = model.encode(f"query: {query}", normalize_embeddings=True)

    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "id": results["ids"][0][i],
            "vector_score": round(1 - results["distances"][0][i], 4),
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "embedding": np.array(results["embeddings"][0][i]),
        })

    return hits, query_emb


# ── Stage 2: MMR Diversity Filter ────────────────────────────────────────────
def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def mmr_filter(hits: list, query_emb: np.ndarray, top_n: int, lambda_mult: float) -> list:
    """
    Maximal Marginal Relevance selection.

    score = lambda_mult * cosine(doc, query)
          - (1 - lambda_mult) * max cosine(doc, selected_doc)

    lambda_mult = 1.0 → pure relevance ordering (no diversity)
    lambda_mult = 0.7 → relevance-biased (recommended default)
    lambda_mult = 0.0 → maximum diversity
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError(f"lambda_mult must be in [0, 1], got {lambda_mult}")
    if not hits:
        return []

    selected_idx, selected_hits, remaining = [], [], list(range(len(hits)))

    for _ in range(min(top_n, len(hits))):
        if not remaining:
            break

        scores = []
        for idx in remaining:
            relevance = cosine_sim(query_emb, hits[idx]["embedding"])
            if not selected_idx:
                score = relevance
            else:
                max_sim = max(
                    cosine_sim(hits[idx]["embedding"], hits[s]["embedding"])
                    for s in selected_idx
                )
                score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            scores.append(score)

        best_local = int(np.argmax(scores))
        best_global = remaining.pop(best_local)

        hit = dict(hits[best_global])
        hit["mmr_score"] = round(scores[best_local], 4)
        selected_idx.append(best_global)
        selected_hits.append(hit)

    return selected_hits


# ── Stage 3: CrossEncoder Reranker ───────────────────────────────────────────
def _sigmoid(x: float) -> float:
    return float(1 / (1 + np.exp(-x))) if x >= 0 else float(np.exp(x) / (1 + np.exp(x)))


def rerank(query: str, hits: list, top_n: int, config: dict) -> list:
    """
    CrossEncoder reranking.
    rerank_score_raw:  model logit (used for ranking)
    rerank_score_norm: sigmoid-normalized 0–1 (for display)
    """
    if not hits:
        return []

    model = get_rerank_model(config)
    pairs = [(query, h["document"]) for h in hits]
    raw_scores = model.predict(pairs)

    ranked = sorted(zip(hits, raw_scores), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (hit, raw) in enumerate(ranked[:top_n], 1):
        # Exclude embeddings (internal vectors) and raw document text (may contain sensitive content)
        h = {k: v for k, v in hit.items() if k not in ("embedding", "document")}
        h["rerank_score_raw"] = round(float(raw), 4)
        h["rerank_score_norm"] = round(_sigmoid(float(raw)), 4)
        h["final_rank"] = rank
        results.append(h)

    return results


# ── Main Pipeline: MMR → Reranker ────────────────────────────────────────────
def search_advanced(query: str, config: dict = None, verbose: bool = False) -> dict:
    """
    Diversity-First RAG pipeline.

    Pipeline: Dense(vector_k=20) → MMR(mmr_k=7) → Reranker(final_k=3)

    Returns stage-level latency alongside results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    t0 = time.time()
    hits, query_emb = vector_search(query, config["vector_k"], config)
    t1 = time.time()

    mmr_hits = mmr_filter(hits, query_emb, config["mmr_k"], config["lambda_mult"])
    t2 = time.time()

    final = rerank(query, mmr_hits, config["final_k"], config)
    t3 = time.time()

    if verbose:
        print(f"[Stage 1] Vector search : {len(hits)} hits  ({(t1-t0)*1000:.0f}ms)", file=sys.stderr)
        print(f"[Stage 2] MMR filter    : {len(mmr_hits)} selected  ({(t2-t1)*1000:.1f}ms)", file=sys.stderr)
        print(f"[Stage 3] Reranker      : {len(final)} final  ({(t3-t2)*1000:.0f}ms)", file=sys.stderr)

    return {
        "query": query,
        "pipeline": "mmr_then_rerank",
        "config": {k: v for k, v in config.items() if k not in ("chroma_path", "collection_name")},
        "results": final,
        "mmr_candidates": [
            {k: v for k, v in h.items() if k not in ("embedding", "document")} for h in mmr_hits
        ] if verbose else [],
        "latency_ms": {
            "vector": round((t1 - t0) * 1000, 1),
            "mmr":    round((t2 - t1) * 1000, 1),
            "rerank": round((t3 - t2) * 1000, 1),
            "total":  round((t3 - t0) * 1000, 1),
        },
    }


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Diversity-First RAG: MMR → Reranker")
    parser.add_argument("query")
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--collection", default="my_collection")
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--mmr-k", type=int, default=7)
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_mult", type=float, default=0.7)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG,
              "chroma_path": args.chroma_path,
              "collection_name": args.collection,
              "vector_k": args.vector_k,
              "mmr_k": args.mmr_k,
              "final_k": args.final_k,
              "lambda_mult": args.lambda_mult}

    result = search_advanced(args.query, config, verbose=args.verbose)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    lat = result["latency_ms"]
    print(f"\n{'='*60}")
    print(f"Diversity-First RAG  (MMR → Reranker)")
    print(f"Query   : {result['query']}")
    print(f"Pipeline: {config['vector_k']} → {config['mmr_k']} → {config['final_k']}")
    print(f"Latency : total={lat['total']}ms  (vector={lat['vector']} / mmr={lat['mmr']} / rerank={lat['rerank']})")
    print(f"{'='*60}")
    for r in result["results"]:
        print(f"\n#{r['final_rank']}  {r['metadata'].get('file_name', r['id'])}")
        print(f"   vector score  : {r['vector_score']:.4f}")
        print(f"   rerank (norm) : {r['rerank_score_norm']:.4f}")
        print(f"   title         : {r['metadata'].get('title', '')}")


if __name__ == "__main__":
    main()
