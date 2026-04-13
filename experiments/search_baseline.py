#!/usr/bin/env python3
"""
search_baseline.py - Standard RAG Baseline Pipeline
Baseline: Dense → Reranker → MMR  (quality-first, diversity-second)

Used for comparison against Diversity-First (MMR → Reranker) pipeline.
"""

import sys
import json
import time
import argparse

from search_advanced import (
    DEFAULT_CONFIG, get_embed_model, get_rerank_model,
    get_collection, vector_search, mmr_filter, rerank, cosine_sim, _sigmoid
)


def search_baseline(query: str, config: dict = None, verbose: bool = False) -> dict:
    """
    Standard pipeline: Dense → Reranker → MMR.

    The reranker selects top candidates by quality first.
    MMR then tries to diversify — but the reranker has already
    collapsed the pool to semantically similar documents.
    """
    if config is None:
        config = DEFAULT_CONFIG

    t0 = time.time()
    hits, query_emb = vector_search(query, config["vector_k"], config)
    t1 = time.time()

    # Stage 2 (reversed): Reranker first
    reranked = rerank(query, hits, config["mmr_k"], config)
    t2 = time.time()

    # Stage 3 (reversed): MMR after reranking
    # Restore embeddings for MMR computation
    id_to_emb = {h["id"]: h["embedding"] for h in hits}
    reranked_with_emb = []
    for r in reranked:
        h = dict(r)
        h["embedding"] = id_to_emb[r["id"]]
        reranked_with_emb.append(h)

    diverse = mmr_filter(reranked_with_emb, query_emb, config["final_k"], config["lambda_mult"])
    t3 = time.time()

    # Exclude embeddings (internal vectors) and raw document text (may contain sensitive content)
    final = [{k: v for k, v in h.items() if k not in ("embedding", "document")} for h in diverse]

    if verbose:
        print(f"[Stage 1] Vector search : {len(hits)} hits  ({(t1-t0)*1000:.0f}ms)", file=sys.stderr)
        print(f"[Stage 2] Reranker      : {len(reranked)} selected  ({(t2-t1)*1000:.0f}ms)", file=sys.stderr)
        print(f"[Stage 3] MMR filter    : {len(final)} final  ({(t3-t2)*1000:.1f}ms)", file=sys.stderr)

    return {
        "query": query,
        "pipeline": "rerank_then_mmr",
        "results": final,
        "latency_ms": {
            "vector": round((t1 - t0) * 1000, 1),
            "rerank": round((t2 - t1) * 1000, 1),
            "mmr":    round((t3 - t2) * 1000, 1),
            "total":  round((t3 - t0) * 1000, 1),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Standard RAG Baseline: Reranker → MMR")
    parser.add_argument("query")
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--collection", default="my_collection")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG,
              "chroma_path": args.chroma_path,
              "collection_name": args.collection}

    result = search_baseline(args.query, config, verbose=args.verbose)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    lat = result["latency_ms"]
    print(f"\n{'='*60}")
    print(f"Standard RAG Baseline  (Reranker → MMR)")
    print(f"Query  : {result['query']}")
    print(f"Latency: total={lat['total']}ms")
    print(f"{'='*60}")
    for i, r in enumerate(result["results"], 1):
        print(f"\n#{i}  {r.get('metadata', {}).get('file_name', r.get('id', ''))}")
        print(f"   mmr_score : {r.get('mmr_score', 'N/A')}")


if __name__ == "__main__":
    main()
