#!/usr/bin/env python3
"""
benchmark.py - Full 5-query benchmark: MMR→Rerank vs Rerank→MMR

Runs both pipelines on the eval query set and prints a summary table.
Results are saved to results/order_comparison.json.

Usage:
    python benchmark.py
    python benchmark.py --queries-file ../data/eval_queries.json
    python benchmark.py --chroma-path ./chroma_db --collection my_collection
"""

import sys
import json
import time
import argparse
from pathlib import Path

from search_advanced import search_advanced, DEFAULT_CONFIG
from search_baseline import search_baseline


def run_benchmark(queries: list, config: dict) -> dict:
    results = []
    total_overlap = 0

    print(f"\n{'='*72}")
    print(f"BENCHMARK: MMR→Rerank  vs  Rerank→MMR")
    print(f"Queries: {len(queries)}  |  final_k={config['final_k']}  |  λ={config['lambda_mult']}")
    print(f"{'='*72}")

    for i, q in enumerate(queries, 1):
        print(f"\n── Query {i}: \"{q['query']}\"  [{q.get('type', '')}]")

        res_a = search_advanced(q["query"], config)
        res_b = search_baseline(q["query"], config)

        ids_a = [r["id"] for r in res_a["results"]]
        ids_b = [r["id"] for r in res_b["results"]]
        overlap = len(set(ids_a) & set(ids_b))
        total_overlap += overlap
        changed = config["final_k"] - overlap

        print(f"  [MMR→Rerank]  {res_a['latency_ms']['total']}ms")
        for r in res_a["results"]:
            fname = r.get("metadata", {}).get("file_name", r["id"])
            print(f"    #{r['final_rank']} [vec:{r['vector_score']:.3f}|rr:{r['rerank_score_norm']:.3f}] {fname[:55]}")

        print(f"  [Rerank→MMR]  {res_b['latency_ms']['total']}ms")
        for j, r in enumerate(res_b["results"], 1):
            fname = r.get("metadata", {}).get("file_name", r.get("id", ""))
            print(f"    #{j} [mmr:{r.get('mmr_score','?')}] {fname[:55]}")

        marker = "⚠ ALL CHANGED" if changed == config["final_k"] else ""
        print(f"  → Overlap: {overlap}/{config['final_k']}  |  Changed: {changed}  {marker}")

        results.append({
            "query": q["query"],
            "type": q.get("type", ""),
            "mmr_then_rerank": res_a,
            "rerank_then_mmr": res_b,
            "overlap": overlap,
            "changed": changed,
        })

    avg_overlap = total_overlap / len(queries)
    avg_changed = config["final_k"] - avg_overlap

    print(f"\n{'='*72}")
    print(f"SUMMARY  ({len(queries)} queries, final_k={config['final_k']})")
    print(f"  Average overlap : {avg_overlap:.1f}/{config['final_k']}  ({avg_overlap/config['final_k']*100:.0f}% same)")
    print(f"  Average changed : {avg_changed:.1f} documents per query")
    verdict = "Pipeline order has SIGNIFICANT effect" if avg_changed >= 0.5 else "Pipeline order has minimal effect"
    print(f"  Verdict         : {verdict}")
    print(f"{'='*72}")

    return {
        "config": {k: v for k, v in config.items() if k not in ("chroma_path", "collection_name")},
        "summary": {
            "n_queries": len(queries),
            "avg_overlap": round(avg_overlap, 2),
            "avg_changed": round(avg_changed, 2),
            "overlap_rate": round(avg_overlap / config["final_k"], 3),
        },
        "per_query": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark: MMR→Rerank vs Rerank→MMR")
    parser.add_argument("--queries-file", default=str(Path(__file__).parent.parent / "data/eval_queries.json"))
    parser.add_argument("--chroma-path", default="./chroma_db")
    parser.add_argument("--collection", default="my_collection")
    parser.add_argument("--output", default=str(Path(__file__).parent.parent / "results/order_comparison.json"))
    parser.add_argument("--final-k", type=int, default=3)
    parser.add_argument("--lambda", dest="lambda_mult", type=float, default=0.7)
    args = parser.parse_args()

    # Load queries
    with open(args.queries_file, encoding="utf-8") as f:
        queries = json.load(f)

    config = {**DEFAULT_CONFIG,
              "chroma_path": args.chroma_path,
              "collection_name": args.collection,
              "final_k": args.final_k,
              "lambda_mult": args.lambda_mult}

    benchmark_result = run_benchmark(queries, config)

    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_result, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
