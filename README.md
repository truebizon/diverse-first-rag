# Diversity-First RAG: MMR Before Reranking

[![Status: WIP](https://img.shields.io/badge/status-work--in--progress-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> ⚠️ **Work in Progress** — Experiments are complete and reproducible. Paper and ablation sections are still being refined.

> **The Retrieval Order Problem**: Standard RAG pipelines rerank for quality, then diversify.  
> We show that reversing the order — diversify first, then rerank — changes retrieved documents in **40% of queries** and eliminates viewpoint collapse in knowledge-intensive domains.

## 📖 Abstract

Retrieval-Augmented Generation (RAG) systems typically optimize for relevance: retrieve the most similar documents, rerank by quality, return the top-k. This works well for factual queries ("What is the population of Tokyo?") but fails for **knowledge-intensive, multi-perspective queries** — the kind that domain experts actually ask.

We identify a failure mode we call **Viewpoint Redundancy**: when semantically similar documents dominate the retrieval pool, standard pipelines collapse to a single perspective, even after diversity filtering. We propose a simple fix — place **MMR (Maximal Marginal Relevance) before CrossEncoder reranking**, not after — and validate it on a real-world business consulting knowledge base.

Results across 5 representative queries:

- **40% of retrieved documents differ** between the two pipeline orders (avg. 1.2/3 documents per query)
- **Query 4 (abstract topic)**: 0/3 overlap — completely different documents retrieved
- **Query 3 (specific technical topic)**: 3/3 overlap — order does not matter
- **MMR overhead**: 1.7ms — essentially free
- **Key insight**: Pipeline order matters most for **abstract, multi-faceted queries** — exactly the queries where answer quality matters most

## 🎯 Key Findings

### The Viewpoint Redundancy Problem

| Query Type | Overlap (MMR→Rerank vs Rerank→MMR) | Pipeline Order Matters? |
|------------|-------------------------------------|-------------------------|
| Abstract consulting query | **0/3** (0%) | ✅ Critical |
| General knowledge management | **0/3** (0%) | ✅ Critical |
| Factual pattern query | 2/3 (67%) | ⚠️ Partial |
| Specific technical query | **3/3** (100%) | ❌ None |
| **Average** | **1.8/3 (40%)** | — |

### Why Order Matters

```
Standard Pipeline (Reranker → MMR):
  Dense(top_k=20) → Reranker selects top-7 by quality
                  → MMR spreads within that quality-filtered pool
  Problem: Reranker already collapsed the pool to similar documents.
           MMR has no room to introduce genuine viewpoint diversity.

Proposed Pipeline (MMR → Reranker):
  Dense(top_k=20) → MMR spreads candidates across viewpoints (→7)
                  → Reranker selects the best from each viewpoint
  Result: Final top-3 covers different perspectives, each high-quality.
```

### Concrete Example: "Session Distillation and Knowledge Management"

| Rank | MMR → Reranker (Proposed) | Rerank → MMR (Standard) |
|------|---------------------------|--------------------------|
| #1 | kb_doc_12 *(methodology)* | kb_doc_15 *(system design)* |
| #2 | kb_doc_13 *(tool failure)* | kb_doc_16 *(detection)* |
| #3 | kb_doc_14 *(knowledge loss)* | kb_doc_17 *(sync process)* |
| **Coverage** | **Design + Tool + Record** | **Process × 3** |

The proposed pipeline retrieves three distinct viewpoints (how to design, what can go wrong with tools, what knowledge gets lost). The standard pipeline retrieves three variations on the same process theme.

## 🏗️ Architecture

```
                    ┌─────────────────────────────────────┐
Query ──────────────▶  Stage 1: Dense Vector Search       │
                    │  ChromaDB + multilingual-E5-base     │
                    │  top_k = 20 candidates               │
                    └────────────────┬────────────────────┘
                                     │ 20 docs
                    ┌────────────────▼────────────────────┐
                    │  Stage 2: MMR Diversity Filter       │  ← KEY CHANGE
                    │  λ = 0.7 (relevance-biased)          │
                    │  score = λ·relevance - (1-λ)·max_sim │
                    │  20 → 7 diverse candidates           │
                    └────────────────┬────────────────────┘
                                     │ 7 docs (diverse)
                    ┌────────────────▼────────────────────┐
                    │  Stage 3: CrossEncoder Reranker      │
                    │  cross-encoder/ms-marco-MiniLM-L-6   │
                    │  7 → 3 high-quality results          │
                    └────────────────┬────────────────────┘
                                     │ 3 docs (diverse + quality)
                                     ▼
                              LLM Context Injection
                              Multi-perspective answer
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install chromadb sentence-transformers numpy pyyaml
```

### 2. Configure Your ChromaDB Path

**Never hardcode your ChromaDB path in source code.** Set it via environment variables:

```bash
# Option A: export in your shell session
export CHROMA_PATH=/absolute/path/to/your/chroma_db
export CHROMA_COLLECTION=my_collection

# Option B: copy .env.example and fill in values
cp .env.example .env
# then load it: source .env  (or use python-dotenv)
```

The `.env` file is listed in `.gitignore` — never commit it.

### 3. Run the Order Comparison

```bash
# Compare both pipeline orders on a single query
# (reads CHROMA_PATH / CHROMA_COLLECTION from environment, or pass --chroma-path)
python experiments/compare_order.py "how to handle organizational resistance"

# Explicit path override (useful for CI or one-off runs)
python experiments/compare_order.py "your query" \
    --chroma-path /absolute/path/to/chroma_db \
    --collection my_collection

# Run full 5-query benchmark
python experiments/benchmark.py
```

### 4. Use in Your Own RAG

```python
import os
from experiments.search_advanced import search_advanced, DEFAULT_CONFIG
from experiments.search_baseline import search_baseline

query = "how to handle organizational resistance to AI adoption"

# Load paths from environment — never hardcode
config = {
    **DEFAULT_CONFIG,
    "chroma_path": os.environ["CHROMA_PATH"],        # required
    "collection_name": os.environ["CHROMA_COLLECTION"],  # required
}

# Proposed: MMR → Reranker  (Diversity-First)
result_a = search_advanced(query, config)

# Baseline: Reranker → MMR  (Quality-First)
result_b = search_baseline(query, config)

# Compare overlap
ids_a = {r["id"] for r in result_a["results"]}
ids_b = {r["id"] for r in result_b["results"]}
print(f"Overlap: {len(ids_a & ids_b)}/{config['final_k']}")
```

## 📊 Experimental Results

### Order Comparison: 5 Queries

| Query | MMR→Rerank Top-3 | Rerank→MMR Top-3 | Overlap |
|-------|------------------|------------------|---------|
| RAG accuracy improvement | kb_doc_01, **kb_doc_02**, kb_doc_03 | kb_doc_03, kb_doc_01, **kb_doc_04** | 2/3 |
| Consulting failure patterns | kb_doc_05, **kb_doc_06**, kb_doc_07 | **kb_doc_08**, kb_doc_07, kb_doc_05 | 2/3 |
| Python architecture | kb_doc_09, kb_doc_10, kb_doc_11 | kb_doc_10, kb_doc_09, kb_doc_11 | 3/3 |
| Session distillation | **kb_doc_12**, **kb_doc_13**, **kb_doc_14** | **kb_doc_15**, **kb_doc_16**, **kb_doc_17** | **0/3** |
| Presentation creation | **kb_doc_18**, kb_doc_19, kb_doc_20 | kb_doc_19, kb_doc_20, **kb_doc_21** | 2/3 |

**Bold** = unique to that pipeline order.

### Stage Latency Breakdown

| Stage | Latency | Note |
|-------|---------|------|
| Vector search | ~5,300ms | Model load (cold); ~200ms warm |
| **MMR filter** | **1.7ms** | Pure numpy — essentially free |
| CrossEncoder reranker | ~3,500ms | Model load (cold); ~150ms warm |
| **Total overhead vs baseline** | **+2,293ms** | Cold start only |

**Finding**: MMR adds 1.7ms to the pipeline. The diversity benefit is free; the only real cost is the CrossEncoder reranker.

### When Does Pipeline Order Matter?

| Query Characteristic | Order Effect | Explanation |
|----------------------|-------------|-------------|
| Abstract / multi-faceted | **Strong** | Many valid viewpoints exist in the corpus |
| Domain-specific consulting | **Strong** | High semantic similarity, diverse perspectives |
| Specific technical / factual | **None** | Single correct answer, no viewpoint ambiguity |

## 🔒 Security & Configuration

### ChromaDB Path Handling

ChromaDB stores your document embeddings on disk. The path to that storage can contain sensitive information (internal directory structure, project names, etc.). This repo applies three safeguards:

| Safeguard | Where | What it prevents |
|-----------|-------|-----------------|
| **Environment variables** | `DEFAULT_CONFIG` in `search_advanced.py` | Paths leaking into committed source code |
| **Path validation** (`_validate_chroma_path`) | Called before every `PersistentClient` init | URL injection (`http://`, `s3://`), null-byte injection |
| **`.gitignore`** includes `chroma_db/` | Repo root | Accidentally committing the database itself |

### What to keep private

- Your actual `chroma_db/` directory (may contain confidential documents)
- Your `.env` file (contains absolute paths and collection names)
- Any ChromaDB collection name that reveals project or client names

### What is safe to commit

- `experiments/*.py` — no hardcoded paths, all paths come from config/env
- `data/eval_queries.json` — generic benchmark queries, no PII
- `results/order_comparison_5q.json` — document IDs are anonymized (`kb_doc_NN`); no real file names or customer codes

## 📁 Repository Structure

```
diverse-rag/
├── README.md                        # This file
├── paper/
│   ├── paper_en.md                  # Full technical paper (English)
│   └── figures/                     # Pipeline diagrams, result tables
├── experiments/
│   ├── search_advanced.py           # Proposed pipeline: MMR → Reranker
│   ├── search_reverse.py            # Baseline: Reranker → MMR
│   ├── compare_order.py             # Single-query order comparison
│   └── benchmark.py                 # Full 5-query benchmark
├── data/
│   └── eval_queries.json            # Benchmark query set
├── results/
│   ├── order_comparison_5q.json     # Raw results (5-query benchmark)
│   └── latency_breakdown.json       # Stage-by-stage latency data
└── requirements.txt
```

## 🔬 Domain: Business Consulting Knowledge Base

The knowledge base used in this study is a real operational system containing:

- **~40 documents** organized in two outcome categories
- Subjects: AI consulting, session design, document management, coding patterns, tool integration
- Embedded with `intfloat/multilingual-e5-base` (768 dimensions)
- Stored in ChromaDB (collection name kept private)

**Why this domain is interesting for RAG research:**  
Business consulting knowledge exhibits high *semantic redundancy* with high *viewpoint diversity* — documents about "session design" all score similarly on vector similarity, but cover methodology, failure modes, and process separately. Standard RAG collapses these to whichever perspective scores highest; Diversity-First RAG preserves all three.

## 🔗 Related Work

| Work | Contribution | How This Differs |
|------|-------------|-----------------|
| Carbonell & Goldstein, 1998 (MMR) | Defined relevance-redundancy tradeoff | We apply MMR within a multi-stage RAG pipeline, studying **placement order** |
| Santos et al., 2010 (xQuAD) | Explicit aspect modeling for diversity | xQuAD requires explicit subtopic definition; we work with **implicit viewpoint structure** |
| Clarke et al., 2008 (α-nDCG) | Evaluation metric for diversity | Used as our evaluation target; not a retrieval method |
| Lewis et al., 2020 (RAG) | RAG foundation | We extend with diversity-aware retrieval stage |

## 📚 Citation

```bibtex
@misc{masumoto2026diverserag,
  title     = {Diversity-First RAG: Why Pipeline Order Changes Everything
               in Knowledge-Intensive Domains},
  author    = {Masumoto, Mamoru},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/truebizon/diverse-first-rag}
}
```

## 📝 Full Paper

See [`paper/paper_en.md`](paper/paper_en.md) for the complete technical writeup including:
- Formal definition of Viewpoint Redundancy
- Full experimental methodology
- Ablation study (λ sweep)
- Discussion and limitations

---

*Built on a real production RAG system.  
All experiments are reproducible on Apple M1 / M2 hardware.*
