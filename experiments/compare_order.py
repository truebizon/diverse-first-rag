#!/usr/bin/env python3
"""
compare_order.py - Single-query pipeline order comparison

Shows side-by-side results for:
  [A] MMR → Reranker  (Diversity-First, proposed)
  [B] Reranker → MMR  (Quality-First, baseline)

Usage:
    python compare_order.py "your query"
    python compare_order.py "your query" --chroma-path ./chroma_db --collection my_collection
    python compare_order.py "your query" --json
"""

import sys
import json
import argparse
import numpy as np

from search_advanced import search_advanced, DEFAULT_CONFIG, cosine_sim
from search_baseline import search_baseline


def avg_pairwise_sim(results: list, hits_with_emb: list) -> float:
    """Average pairwise cosine similarity among retrieved documents (lower = more diverse)."""
    id_to_emb = {h["id"]: h["embedding"] for h in hits_with_emb}
    embeddings = [id_to_emb[r["id"]] for r in results if r["id"] in id_to_emb]
    if len(embeddings) < 2:
        return 0.0
    sims = [cosine_sim(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))]
    return round(float(np.mean(sims)), 4)


def compare(query: str, config: dict, as_json: bool = False):
    # Run both pipelines
    result_a = search_advanced(query, config)
    result_b = search_baseline(query, config)

    ids_a = [r["id"] for r in result_a["results"]]
    ids_b = [r["id"] for r in result_b["results"]]
    overlap = len(set(ids_a) & set(ids_b))

    if as_json:
        print(json.dumps({
            "query": query,
            "mmr_then_rerank": result_a,
            "rerank_then_mmr": result_b,
            "overlap": overlap,
            "changed": config["final_k"] - overlap,
        }, ensure_ascii=False, indent=2))
        return

    k = config["final_k"]
    print(f"\n{'='*72}")
    print(f"ORDER COMPARISON")
    print(f"Query : \"{query}\"")
    print(f"{'='*72}")

    # Pipeline A
    lat_a = result_a["latency_ms"]
    print(f"\n[A] MMR → Reranker  (Diversity-First — proposed)")
    print(f"    Latency: {lat_a['total']}ms  (vector:{lat_a['vector']} / mmr:{lat_a['mmr']} / rerank:{lat_a['rerank']})")
    for r in result_a["results"]:
        fname = r.get("metadata", {}).get("file_name", r["id"])
        title = r.get("metadata", {}).get("title", "")
        print(f"    #{r['final_rank']}  [{r['vector_score']:.3f} | rr:{r['rerank_score_norm']:.3f}]  {fname}")
        if title:
            print(f"         {title[:70]}")

    # Pipeline B
    lat_b = result_b["latency_ms"]
    print(f"\n[B] Reranker → MMR  (Quality-First — baseline)")
    print(f"    Latency: {lat_b['total']}ms  (vector:{lat_b['vector']} / rerank:{lat_b['rerank']} / mmr:{lat_b['mmr']})")
    for i, r in enumerate(result_b["results"], 1):
        fname = r.get("metadata", {}).get("file_name", r.get("id", ""))
        title = r.get("metadata", {}).get("title", "")
        print(f"    #{i}  [mmr:{r.get('mmr_score','?')}]  {fname}")
        if title:
            print(f"         {title[:70]}")

    # Diff
    unique_a = [r for r in result_a["results"] if r["id"] not in set(ids_b)]
    unique_b = [r for r in result_b["results"] if r["id"] not in set(ids_a)]

    print(f"\n{'─'*72}")
    print(f"Overlap          : {overlap}/{k}  ({overlap/k*100:.0f}% same)")
    print(f"Order changed    : {k - overlap}/{k} documents")
    if unique_a:
        print(f"Only in A (MMR→Rerank): {[r.get('metadata',{}).get('file_name', r['id']) for r in unique_a]}")
    if unique_b:
        print(f"Only in B (Rerank→MMR): {[r.get('metadata',{}).get('file_name', r.get('id','')) for r in unique_b]}")


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline ordering: MMR→Rerank vs Rerank→MMR")
    parser.add_argument("query")
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--collection", default="my_collection")
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_mult", type=float, default=0.7)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = {**DEFAULT_CONFIG,
              "chroma_path": args.chroma_path,
              "collection_name": args.collection,
              "final_k": args.final_k,
              "lambda_mult": args.lambda_mult}

    compare(args.query, config, as_json=args.json)


if __name__ == "__main__":
    main()
