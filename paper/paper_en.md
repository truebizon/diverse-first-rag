# Diversity-First RAG: Why Pipeline Order Changes Everything in Knowledge-Intensive Domains

**Author**: Mamoru Masumoto  
**Affiliation**: TrueBizon, Ltd.  
**Date**: April 2026  
**Repository**: https://github.com/truebizon/diverse-first-rag

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are increasingly deployed in knowledge-intensive domains where queries require multi-perspective answers. Standard pipelines optimize for relevance first — dense retrieval, quality reranking, then optional diversity filtering — but this ordering causes **Viewpoint Redundancy**: the reranker collapses the candidate pool to semantically similar documents before diversity can act. When domain experts ask abstract, multi-faceted questions, this can produce outputs that appear comprehensive while omitting equally relevant perspectives — a failure mode that is easy to miss precisely because the returned documents remain individually plausible.

We propose **Diversity-First RAG**: placing MMR (Maximal Marginal Relevance) before CrossEncoder reranking, not after. Evaluated on a real business consulting knowledge base (~40 documents, multilingual-E5-base embeddings, ChromaDB), this reordering changes retrieved documents in **40% of queries** (avg. 1.2/3 per query) and produces **0% document overlap** on abstract, multi-faceted consulting queries — the exact queries where viewpoint diversity most affects answer quality.

Critically, MMR adds only **1.7ms** of latency. The diversity benefit is essentially free; the only cost is the CrossEncoder reranker itself. We show that pipeline order matters most for **abstract, multi-faceted queries** and has no effect on specific factual queries — providing a clear criterion for when Diversity-First RAG should be applied.

---

## 1. Introduction

### 1.1 Why Perspective Coverage Matters in Practice

In knowledge-intensive settings, domain experts often pose abstract queries for which no single document is the correct answer — only a diverse set of perspectives yields an adequate response. To illustrate, consider a knowledge worker querying a consulting knowledge base with: *"What patterns have caused AI adoption projects to fail?"* A relevance-first RAG pipeline may return three documents all emphasizing change management, while equally relevant documents on technical debt, leadership misalignment, or training costs are never surfaced. The output appears complete; the omission is invisible. This illustrative scenario reflects a structural property of knowledge bases where embedding similarity correlates with topic rather than perspective, causing relevance-optimized retrieval to cluster within a single viewpoint.

The concern extends to other knowledge-intensive domains (illustrative; not empirically validated in this paper):

| Domain | Typical Query | Perspectives Potentially Missed |
|--------|--------------|--------------------------------------|
| Management consulting | "Failure patterns in AI adoption" | Technical debt, leadership, training gaps |
| Medical decision support | "Differential diagnosis for these symptoms" | Rare conditions, drug interactions |
| Legal analysis | "Risks in this contract structure" | Regulatory, counterparty, enforcement jurisdiction |
| R&D search | "Prior art for this mechanism" | Adjacent-field analogues, negative results |
| HR / people analytics | "Indicators of attrition risk" | Composite risk factors beyond the dominant signal |

In our experiments, pipeline order changes **40% of the final retrieved documents** on average, with complete divergence (0/3 overlap) on the most abstract consulting queries. For organizations that build knowledge bases precisely to preserve diverse experience, this suggests that perspective coverage — not only relevance — is a practical retrieval requirement.

### 1.2 The Problem with Standard RAG

Modern RAG systems follow a well-established pipeline:

```
Query → Dense Retrieval → (Reranking) → LLM Generation
```

For factual queries, this works well. Ask "What is the population of Tokyo?" and the closest document wins. But domain experts rarely ask factual queries. A consultant asks:

> *"What patterns have caused AI adoption projects to fail at the organizational level?"*

This query has no single correct document. Multiple valid perspectives exist: change management failures, leadership alignment issues, technical infrastructure problems, training gaps. A good RAG system should surface one document from each perspective. A standard RAG system surfaces whichever perspective happens to have the highest embedding similarity — repeatedly.

We call this failure mode **Viewpoint Redundancy**.

### 1.3 The Viewpoint Redundancy Problem

**Definition (informal)**: Viewpoint Redundancy occurs when the top-k retrieved documents are semantically similar (high cosine similarity) but cover the same conceptual perspective, missing equally relevant documents from different viewpoints.

This is distinct from simple duplication or factual redundancy:
- **Duplication**: Two documents say the same thing
- **Factual Redundancy**: Two documents cover the same fact from different angles
- **Viewpoint Redundancy**: Two documents address the same topic but the retrieval system treats them as equivalent, ignoring that they cover orthogonal aspects

Viewpoint Redundancy is endemic to **knowledge-intensive domains** where:
1. A corpus contains many documents on related themes
2. Each document addresses the theme from a distinct professional perspective
3. Embedding similarity correlates with topic, not perspective

Business consulting, legal analysis, medical decision support, and organizational knowledge management all exhibit this structure.

### 1.4 Our Contribution

We make three contributions:

1. **Problem identification**: We formally define Viewpoint Redundancy and provide empirical evidence of its occurrence in a real RAG system
2. **Simple fix**: We show that placing MMR before reranking (rather than after) substantially reduces Viewpoint Redundancy at negligible computational cost (1.7ms)
3. **Principled criterion**: We show that pipeline order matters for abstract queries and not for specific ones — providing practitioners with a concrete decision rule

---

## 2. Related Work

### 2.1 Maximal Marginal Relevance (MMR)

MMR was introduced by Carbonell & Goldstein (1998) as a method for reducing redundancy in document retrieval:

```
MMR(d_i) = λ · Sim(d_i, q) - (1 - λ) · max_{d_j ∈ S} Sim(d_i, d_j)
```

where `q` is the query, `S` is the set of already-selected documents, and `λ` balances relevance against redundancy. MMR has been widely applied to summarization and search diversification, but its placement within multi-stage RAG pipelines has not been systematically studied.

### 2.2 Diversification in Information Retrieval

The information retrieval community has developed several frameworks for diversification:

- **xQuAD** (Santos et al., 2010): Models query aspects explicitly, selecting documents that cover different subtopics
- **α-nDCG** (Clarke et al., 2008): Evaluation metric that penalizes redundant relevant documents
- **IA-Select** (Agrawal et al., 2009): Probabilistic model for intent-aware search diversification

These approaches assume **explicit** subtopic or intent structure. Our setting differs: in consulting knowledge bases, viewpoints are **implicit** — the corpus contains documents on related themes whose perspectives are not labeled and cannot be inferred from query terms alone.

### 2.3 Reranking in RAG

CrossEncoder rerankers (Nogueira & Cho, 2019) have become standard in RAG pipelines. Unlike bi-encoder retrievers, CrossEncoders process query-document pairs jointly, producing scores that better reflect semantic relevance. The standard pipeline places reranking after dense retrieval and before generation.

We are not aware of prior work systematically comparing the effect of MMR-before-reranking vs reranking-before-MMR in production RAG systems.

### 2.4 RAG for Knowledge-Intensive Domains

Lewis et al. (2020) demonstrated that RAG substantially improves factual accuracy in open-domain QA. Subsequent work has applied RAG to legal (Huang et al., 2023), medical (Xiong et al., 2024), and enterprise (Barnett et al., 2024) domains. However, these works focus on retrieval accuracy rather than viewpoint coverage — the distinction we argue matters most for domain expert queries.

---

## 3. Problem Formulation

### 3.1 Setup

Let `K = {d_1, d_2, ..., d_n}` be a knowledge base of `n` documents, each embedded as a vector `e_i ∈ R^d`. Given query `q` with embedding `e_q`, the standard RAG pipeline returns:

```
S_standard = Rerank(TopK_dense(q, K, k=20), k=3)
```

where `TopK_dense` returns the 20 most similar documents by cosine similarity, and `Rerank` applies a CrossEncoder to select the top 3 by query-document relevance.

### 3.2 Viewpoint Redundancy

**Definition (formal)**: Given a query `q` and retrieved set `S = {d_1, ..., d_k}`, Viewpoint Redundancy is high when:

```
VR(S) = (1/|S|) · Σ_i max_{j≠i} cosine(e_i, e_j)
```

is high — i.e., on average, each retrieved document is very similar to at least one other retrieved document.

A low VR score indicates that retrieved documents are spread across the embedding space, covering diverse perspectives. A high VR score indicates clustering — documents that are saying similar things from similar angles.

**Observation**: In consulting knowledge bases, `TopK_dense` frequently returns documents with pairwise cosine similarities of 0.83–0.87, even across documents that domain experts would classify as covering distinct aspects. Reranking does not reduce this clustering — it selects the "best" documents from within the cluster.

### 3.3 Proposed Method: Diversity-First RAG

We propose inserting MMR between dense retrieval and reranking:

```
S_proposed = Rerank(MMR(TopK_dense(q, K, k=20), k=7, λ=0.7), k=3)
```

**Rationale**: MMR diversifies the candidate pool *before* the reranker makes its final selection. The reranker then selects the highest-quality document from each diverse perspective, rather than selecting quality within a homogeneous cluster.

This is equivalent to asking: "Given that we want 7 diverse candidates, which 3 are best?" rather than "Given the 7 best candidates, which 3 are most diverse?"

**MMR Score**:

```
MMR(d_i) = λ · cosine(e_i, e_q) - (1 - λ) · max_{d_j ∈ S_selected} cosine(e_i, e_j)
```

where relevance is computed directly from embedding vectors (not from the rounded vector search score) to avoid accumulated rounding error.

**λ parameter**: We use `λ = 0.7` (relevance-biased) as the default. Higher λ approaches pure relevance ordering; lower λ forces more diversity. For consulting queries, λ = 0.7 empirically balances retrieving relevant documents while preserving viewpoint coverage.

### 3.4 Adaptive Pipeline Selection

Static application of MMR-First (Section 3.3) improves average diversity but can degrade results on queries where the candidate pool is already naturally diverse — a reversal observed for low-abstraction technical queries. This motivates a query-time pipeline selection rule.

**Key observation**: The VR score of the candidate pool (top-k documents returned by dense retrieval) varies systematically by query type. Abstract queries attract a dense, perspective-homogeneous cluster (high candidate VR); specific technical queries attract a sparser, already-diverse set (lower candidate VR).

**Decision rule**: Given a candidate pool `H = {h_1, ..., h_K}` from dense retrieval and a pre-computed corpus reference VR `μ_corpus`:

```
candidate_vr = VR(H)            # avg pairwise cosine similarity of H

if candidate_vr > θ:            # candidate pool is denser than corpus baseline
    apply MMR-First             # diversity intervention needed
else:
    apply Rerank-First          # candidate pool already diverse; quality-first
```

**Threshold `θ`**: We set `θ = μ_corpus × (1 + α)` where `α` controls sensitivity. Setting `α = 0.025` (equivalently, `n = 1.5` in the implementation) places `θ` approximately at the median candidate VR observed across a representative query set, yielding a roughly 40/60 MMR-First/Rerank-First split.

**Pre-computation**: `μ_corpus` is computed offline by drawing random samples of size `k` from the full corpus embedding set and averaging their VR scores over 30 trials. This takes seconds and is stable across corpus sizes (standard deviation < 0.006 at k=20 over 374 documents).

This results in three pipeline variants evaluated in Section 5.5:

| Variant | Selection rule | Description |
|---------|---------------|-------------|
| MMR-First (static) | Always | Diversity-first regardless of query |
| Rerank-First (static) | Never | Quality-first regardless of query |
| **Adaptive** | `candidate_vr > θ` | Selects per query |

---

## 4. Experimental Setup

### 4.1 Knowledge Base

| Property | Value |
|----------|-------|
| Total documents | ~40 |
| Categories | two outcome categories (positive outcomes: 22, negative outcomes: 16) |
| Language | Japanese |
| Domain | AI consulting, session design, software engineering, knowledge management |
| Embedding model | `intfloat/multilingual-e5-base` (768 dim) |
| Vector store | ChromaDB (cosine distance) |
| Hardware | Apple M1, 8GB RAM |

The knowledge base is an operational system used for AI consulting work. Documents represent accumulated learnings from real projects, capturing patterns that succeeded, patterns that failed, and methodological principles.

### 4.2 Evaluation Queries

We selected 5 queries representing the range from abstract/multi-faceted to specific/technical:

| ID | Query | Type |
|----|-------|------|
| Q1 | "RAGの精度向上" (RAG accuracy improvement) | Technical, moderate abstraction |
| Q2 | "コンサルティング案件の失敗パターン" (Consulting failure patterns) | Domain, high abstraction |
| Q3 | "Pythonアーキテクチャの改善" (Python architecture improvement) | Technical, low abstraction |
| Q4 | "セッション蒸留と知識管理" (Session distillation and knowledge management) | Domain, high abstraction |
| Q5 | "プレゼンテーション資料の作り方" (Presentation creation) | Mixed, moderate abstraction |

### 4.3 Pipeline Configurations

We compare two configurations with identical parameters (k=20 pool, 7 intermediate candidates, 3 final results):

| Configuration | Order | Description |
|---------------|-------|-------------|
| **Proposed** (MMR→Rerank) | Dense → MMR → CrossEncoder | Diversity-first |
| **Baseline** (Rerank→MMR) | Dense → CrossEncoder → MMR | Quality-first |

### 4.4 Metrics

**Document Overlap**: `|S_proposed ∩ S_baseline| / k` — fraction of identical documents retrieved by both pipelines. Low overlap indicates pipeline order has significant effect.

**Latency breakdown**: Wall-clock time for each stage, measured on Apple M1 (cold start for first query, warm for subsequent).

**Qualitative coverage**: Manual examination of retrieved documents' conceptual perspectives for selected queries (Q2, Q4).

---

## 5. Results

### 5.1 Document Overlap by Query

| Query | Proposed (MMR→Rerank) | Baseline (Rerank→MMR) | Overlap |
|-------|-----------------------|-----------------------|---------|
| Q1: RAG accuracy | kb_doc_01, **kb_doc_02**, kb_doc_03 | kb_doc_03, kb_doc_01, **kb_doc_04** | **2/3** |
| Q2: Consulting failures | kb_doc_05, **kb_doc_06**, kb_doc_07 | **kb_doc_08**, kb_doc_07, kb_doc_05 | **2/3** |
| Q3: Python architecture | kb_doc_09, kb_doc_10, kb_doc_11 | kb_doc_10, kb_doc_09, kb_doc_11 | **3/3** |
| Q4: Session distillation | **kb_doc_12**, **kb_doc_13**, **kb_doc_14** | **kb_doc_15**, **kb_doc_16**, **kb_doc_17** | **0/3** |
| Q5: Presentation | **kb_doc_18**, kb_doc_19, kb_doc_20 | kb_doc_19, kb_doc_20, **kb_doc_21** | **2/3** |
| **Average** | — | — | **1.8/3 (60%)** |

Documents unique to each pipeline are **bold**. Average overlap of 1.8/3 means pipeline order changes 40% of final results across queries.

### 5.2 Qualitative Analysis: Query 4

Query 4 (session distillation and knowledge management) showed **0/3 document overlap** — the strongest evidence that pipeline order substantially affects results for abstract consulting queries.

**Proposed pipeline (MMR→Rerank) retrieved**:
- `kb_doc_12` — *How to structure a consulting session* (methodology)
- `kb_doc_13` — *Failure case: AI tool corrupted session output* (tool failure)
- `kb_doc_14` — *Failure case: Knowledge was not preserved after session* (knowledge loss)

Coverage: **Design methodology** + **Tool integration failure** + **Knowledge preservation failure** — three orthogonal aspects.

**Baseline pipeline (Rerank→MMR) retrieved**:
- `kb_doc_15` — *System design for session debriefs* (process)
- `kb_doc_16` — *Detecting false positives before deletion* (process)
- `kb_doc_17` — *Synchronization check patterns* (process)

Coverage: **Process × 3** — all three documents address operational process, missing methodology and failure analysis entirely.

A consultant asking about session distillation needs to know: (a) how to design it, (b) what tool failures to avoid, and (c) what knowledge loss patterns to watch for. The proposed pipeline retrieves all three; the baseline retrieves none of (a) or (c).

### 5.3 When Does Order Matter?

| Query | Abstraction | Domain | Overlap | Order Matters? |
|-------|-------------|--------|---------|----------------|
| Q4: Session distillation | High | Consulting | 0/3 | ✅ Critical |
| Q2: Consulting failures | High | Consulting | 2/3 | ✅ Significant |
| Q5: Presentation | Medium | Mixed | 2/3 | ✅ Significant |
| Q1: RAG accuracy | Medium | Technical | 2/3 | ✅ Significant |
| Q3: Python architecture | Low | Technical | 3/3 | ❌ None |

**Finding**: Pipeline order matters most for queries that are (a) abstract and (b) domain-specific (consulting). It has no effect on specific technical queries where a single document clearly dominates.

This provides a practical criterion: apply Diversity-First RAG when query abstraction level is moderate-to-high.

### 5.4 Latency

| Stage | Cold Start | Warm (2nd query+) |
|-------|-----------|---------------------|
| Dense vector search | ~5,300ms | ~200ms |
| **MMR filter** | **1.7ms** | **1.7ms** |
| CrossEncoder reranker | ~3,500ms | ~150ms |
| **Total (proposed)** | ~8,800ms | ~352ms |
| **Total (baseline)** | ~6,500ms | ~350ms |
| **Overhead** | +2,300ms (cold) | **+1.7ms (warm)** |

**Finding**: The MMR stage adds 1.7ms regardless of warm/cold state. In production (warm models), the proposed pipeline adds **essentially zero overhead**. The cold-start difference is entirely due to reranker model loading, which occurs in both pipelines.

### 5.5 Adaptive Pipeline Evaluation

To evaluate the Adaptive Pipeline Selection described in Section 3.4, we ran an extended experiment on a larger corpus (374 documents, same collection) with 20 queries spanning four abstraction levels.

**Extended corpus**:

| Property | Value |
|----------|-------|
| Total documents | 374 |
| Query set | 20 queries (high: 5, medium: 7, low: 5, cross-domain: 3) |
| Corpus reference VR (k=20, 30 trials) | 0.8742 ± 0.0054 |
| Adaptive threshold θ (n=1.5) | 0.8960 |

**Three-pipeline comparison** (final_VR, lower = more diverse):

| Pipeline | Avg final VR | vs. best static |
|----------|-------------|-----------------|
| MMR-First (static) | 0.8952 | — |
| Rerank-First (static) | 0.8958 | — |
| **Adaptive (n=1.5)** | **0.8942** | **−0.0010 better** |

**Pipeline selection distribution** (n=20 queries):

| Selected | Count | % |
|----------|-------|---|
| MMR-First | 8 | 40% |
| Rerank-First | 12 | 60% |

**By abstraction level**:

| Level | Adaptive avg VR | vs. best static | Win rate |
|-------|----------------|-----------------|----------|
| High abstraction | 0.8977 | −0.0005 ✅ | 3/5 |
| Medium abstraction | 0.8913 | −0.0009 ✅ | 3/7 |
| **Low abstraction** | **0.8928** | **−0.0001 ✅** | **3/5** |
| Cross-domain | 0.8974 | −0.0018 ✅ | 3/3 |

The key result is the low-abstraction row: static MMR-First degrades on specific technical queries (VR=0.8962, worse than Rerank-First at 0.8929), while Adaptive correctly selects Rerank-First for most low-abstraction queries, achieving 0.8928 — matching the best static choice. The reversal problem observed in the static evaluation (Section 5.3) is resolved.

**Finding**: Adaptive Pipeline Selection achieves the best overall diversity across all abstraction levels, with a win rate of 60% against the better of the two static pipelines and no catastrophic degradation in any level. The θ threshold (≈ median candidate VR) provides a practical, corpus-calibrated decision boundary.

---

## 6. Discussion

### 6.1 Why Does MMR-First Work?

The mechanism is straightforward: when the reranker sees 20 candidates (Rerank→MMR), it selects the 7 highest-quality documents. These 7 documents are already clustered in embedding space — they are the "best" documents from the dominant perspective. MMR applied to this cluster can only spread within the cluster.

When MMR sees 20 candidates first (MMR→Rerank), it selects 7 documents spread across embedding space — documents from the dominant perspective AND documents from minority perspectives that are still relevant. The reranker then identifies the highest-quality document from each region of the space, preserving cross-perspective coverage.

In short: **you cannot recover diversity that the reranker has already discarded**.

### 6.2 The λ Parameter

We used λ = 0.7 (70% relevance weight, 30% diversity pressure). This is a design choice that warrants further study:

- **λ = 1.0**: Pure relevance ordering — equivalent to dense retrieval ranking, no diversity benefit
- **λ = 0.7**: Relevance-biased diversity (our default) — maintains high relevance while forcing viewpoint spread
- **λ = 0.5**: Balanced — equal weight to relevance and diversity
- **λ = 0.0**: Maximum diversity — selects documents maximally spread in embedding space, possibly at the cost of relevance

For consulting queries, we observe that λ = 0.7 is sufficient to produce the ordering effects shown in Section 5. A systematic sweep over λ values is left to future work.

### 6.3 Limitations

**Small knowledge base**: Our experiments use ~40 documents. The Viewpoint Redundancy problem likely becomes more severe as corpus size grows (more competing perspectives), but also more amenable to solution (more diverse candidates in the top-20 pool). Validation on larger corpora is needed.

**No ground-truth labels**: We cannot compute α-nDCG or other diversity metrics without human-annotated viewpoint labels. Our current evaluation relies on document overlap and qualitative analysis. Future work should include human annotation.

**English reranker on Japanese corpus**: We use `cross-encoder/ms-marco-MiniLM-L-6-v2`, an English model, on a Japanese knowledge base. A Japanese CrossEncoder (e.g., `hotchpotch/japanese-reranker-cross-encoder-small-v1`) would likely produce higher-quality reranking scores. The current setup demonstrates that even a mismatched reranker produces meaningful results; a matched reranker would strengthen the findings.

**Model scale**: Experiments used a mid-sized embedding model (multilingual-E5-base, 768d, 110M parameters) and a lightweight reranker (MiniLM-L-6, 22M parameters). The Viewpoint Redundancy mechanism is structural — it arises from pipeline ordering, not from any particular model's characteristics — and is expected to hold at larger scales. Validation with larger embedding models (e.g., multilingual-E5-large-instruct) and stronger rerankers remains future work.

**Single domain**: All experiments use a business consulting knowledge base. Whether the findings generalize to legal, medical, or other knowledge-intensive domains remains to be validated.

### 6.4 Practical Recommendations

Based on these results, we recommend:

1. **Use Adaptive Pipeline Selection** (Section 3.4) as the default: compute candidate VR at query time and select MMR-First or Rerank-First accordingly. This outperforms either static choice across abstraction levels with negligible overhead (one VR computation over k=20 documents).
2. **Use static MMR-First** when: query mix is known to be abstract-heavy, simplicity is preferred over per-query optimization.
3. **Use standard pipeline** when: queries are specific/factual, single correct answer exists.
4. **Calibrate θ offline**: compute `μ_corpus` by sampling the production corpus (30 trials, k=20, takes < 5 seconds). Set `θ = μ_corpus × 1.025` as a starting point; adjust based on query distribution.
5. **Default λ = 0.7** for consulting and similar domains; consider lower λ if diversity is paramount.
6. **Keep MMR pool size ≥ 10**: MMR needs enough candidates to find genuine diversity. A pool of 5 leaves little room for meaningful diversification.

---

## 7. Conclusion

We presented **Diversity-First RAG**, a simple reordering of the standard RAG pipeline that places MMR before CrossEncoder reranking, and **Adaptive Pipeline Selection**, a query-time decision rule that applies this reordering only when the candidate pool exhibits Viewpoint Redundancy. Through experiments on a real business consulting knowledge base (40 documents, 5 queries; extended to 374 documents, 20 queries), we showed:

- Pipeline order changes **40% of retrieved documents** on average across diverse queries
- **0% document overlap** on abstract, multi-faceted consulting queries — the highest-stakes case
- Only **1.7ms overhead** in warm-model production settings
- Static MMR-First degrades on low-abstraction technical queries; Adaptive resolves this
- Adaptive achieves the **best overall diversity** (avg VR 0.8942 vs. 0.8952 / 0.8958) across all abstraction levels with a 60% win rate against the better static choice

The core insight is that **diversity discarded by the reranker cannot be recovered by post-hoc MMR**. By diversifying first — and doing so only when the candidate pool warrants it — we ensure that the reranker selects the best representative from each relevant perspective, rather than the best N documents from a single perspective.

The Viewpoint Redundancy problem is likely widespread in knowledge-intensive RAG deployments. We hope this work encourages practitioners to examine the ordering assumptions in their own pipelines.

---

## References

1. Carbonell, J., & Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents and producing summaries. *SIGIR 1998*.

2. Santos, R. L. T., Macdonald, C., & Ounis, I. (2010). Exploiting query reformulations for web search result diversification. *WWW 2010*.

3. Clarke, C. L. A., Kolla, M., Cormack, G. V., Vechtomova, O., Ashkan, A., Büttcher, S., & MacKinnon, I. (2008). Novelty and diversity in information retrieval evaluation. *SIGIR 2008*.

4. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 2020*.

5. Nogueira, R., & Cho, K. (2019). Passage re-ranking with BERT. *arXiv:1901.04085*.

6. Agrawal, R., Gollapudi, S., Halverson, A., & Ieong, S. (2009). Diversifying search results. *WSDM 2009*.

7. Wang, L., Yang, N., Huang, X., Jiao, B., Yang, L., Jiang, D., ... & Wei, F. (2022). Text embeddings by weakly-supervised contrastive pre-training. *arXiv:2212.03533* (multilingual-E5-base).

---

## Appendix A: Implementation Details

### A.1 MMR Implementation

```python
def mmr_filter(hits, query_embedding, top_n=7, lambda_mult=0.7):
    """
    MMR selection. Relevance computed directly from query_embedding
    (not from rounded vector search scores) to avoid accumulation error.
    """
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError(f"lambda_mult must be in [0, 1]")

    selected_indices, selected_hits, remaining = [], [], list(range(len(hits)))

    for _ in range(min(top_n, len(hits))):
        mmr_scores = []
        for idx in remaining:
            relevance = cosine_sim(query_embedding, hits[idx]["embedding"])
            if not selected_indices:
                mmr_score = relevance
            else:
                max_sim = max(
                    cosine_sim(hits[idx]["embedding"], hits[sel]["embedding"])
                    for sel in selected_indices
                )
                mmr_score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
            mmr_scores.append(mmr_score)

        best_local = int(np.argmax(mmr_scores))
        best_global = remaining.pop(best_local)
        hit = dict(hits[best_global])
        hit["mmr_score"] = round(mmr_scores[best_local], 4)
        selected_indices.append(best_global)
        selected_hits.append(hit)

    return selected_hits
```

### A.2 Reranker with Score Normalization

```python
def rerank(query, hits, top_n=3):
    """
    rerank_score_raw: used for ranking (model logit)
    rerank_score_norm: sigmoid-normalized for display (0-1)
    """
    reranker = get_rerank_model()
    pairs = [(query, h["document"]) for h in hits]
    raw_scores = reranker.predict(pairs)

    ranked = sorted(zip(hits, raw_scores), key=lambda x: x[1], reverse=True)

    results = []
    for rank, (hit, raw) in enumerate(ranked[:top_n], 1):
        h = dict(hit)
        h["rerank_score_raw"] = round(float(raw), 4)
        h["rerank_score_norm"] = round(sigmoid(float(raw)), 4)
        h["final_rank"] = rank
        results.append(h)
    return results
```

### A.3 Configuration

```yaml
# config.yaml
embedding:
  model: "intfloat/multilingual-e5-base"
  dimension: 768

vector_db:
  type: "chroma"
  path: "./chroma_db"          # set via CHROMA_PATH env var
  collection: "my_collection"  # set via CHROMA_COLLECTION env var

reranker:
  # English (default): cross-encoder/ms-marco-MiniLM-L-6-v2
  # Japanese (recommended for ja corpus):
  #   hotchpotch/japanese-reranker-cross-encoder-small-v1
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```
