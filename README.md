# Article Clustering Algorithm — Debug Guide

> **Purpose:** Understand exactly how `ArticleClusterer` groups news articles by topic, why it works better than the old approach, and how to debug when it goes wrong.

---

## Table of Contents

- [Article Clustering Algorithm — Debug Guide](#article-clustering-algorithm--debug-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Big Picture](#1-big-picture)
  - [2. Pipeline Overview](#2-pipeline-overview)
  - [3. Step-by-Step Breakdown](#3-step-by-step-breakdown)
    - [Step 1 — Text Representation](#step-1--text-representation)
    - [Step 2 — Embedding](#step-2--embedding)
    - [Step 3 — Similarity Matrix](#step-3--similarity-matrix)
    - [Step 4 — Community Detection](#step-4--community-detection)
      - [Phase A — Build adjacency graph](#phase-a--build-adjacency-graph)
      - [Phase B — Connected components (BFS)](#phase-b--connected-components-bfs)
    - [Step 5 — Coherence Enforcement](#step-5--coherence-enforcement)
    - [Step 6 — Cluster Validation](#step-6--cluster-validation)
  - [4. Why the Old Approach Failed](#4-why-the-old-approach-failed)
  - [5. Key Parameters \& Tuning](#5-key-parameters--tuning)
    - [Tuning guide](#tuning-guide)
  - [6. Debugging Playbook](#6-debugging-playbook)
    - [Enable debug logging](#enable-debug-logging)
    - [Inspect the similarity matrix](#inspect-the-similarity-matrix)
    - [Find all pairs above a threshold](#find-all-pairs-above-a-threshold)
    - [Check what gets discarded](#check-what-gets-discarded)
    - [Validate cluster quality](#validate-cluster-quality)
  - [7. Worked Example](#7-worked-example)
  - [8. Common Failure Modes](#8-common-failure-modes)
    - [❌ Two different stories end up in one cluster](#-two-different-stories-end-up-in-one-cluster)
    - [❌ Valid story pairs are not clustering](#-valid-story-pairs-are-not-clustering)
    - [❌ Same story from the same source appearing in multiple clusters](#-same-story-from-the-same-source-appearing-in-multiple-clusters)
    - [❌ All articles end up in one giant cluster](#-all-articles-end-up-in-one-giant-cluster)
    - [❌ `reliability` scores are all the same](#-reliability-scores-are-all-the-same)

---

## 1. Big Picture

The goal is simple: given a batch of news articles scraped from left-leaning and right-leaning sources, **group articles that cover the same real-world event** into clusters — so each cluster gets one left article and one right article to generate a dual-perspective piece.

```
[Raw Articles]
     │
     ▼
[Embed titles + snippets]
     │
     ▼
[Build similarity matrix]
     │
     ▼
[Community detection → connected components]
     │
     ▼
[Coherence enforcement → remove off-topic articles]
     │
     ▼
[Validate: need ≥1 left + ≥1 right]
     │
     ▼
[Valid Clusters]
```

---

## 2. Pipeline Overview

```
Input articles
    │
    ├─ _article_text()          # title + snippet → rich string
    │
    ├─ model.encode()           # SentenceTransformer → embeddings
    │
    ├─ cosine_similarity()      # n×n similarity matrix
    │
    ├─ _community_detection()
    │       │
    │       ├─ Build adjacency graph  (edge if sim ≥ threshold)
    │       ├─ BFS connected components
    │       └─ _enforce_coherence()   (trim outliers per component)
    │
    └─ _build_valid_cluster()   # pick best left + right per group
```

---

## 3. Step-by-Step Breakdown

### Step 1 — Text Representation

```python
def _article_text(self, article: RawArticle) -> str:
    parts = [article.title.strip()]
    snippet = getattr(article, "snippet", None) or getattr(article, "description", None)
    if snippet:
        parts.append(snippet.strip()[:200])
    return " | ".join(parts)
```

**What it does:**
Concatenates the article title and the first 200 characters of its snippet/description, separated by ` | `.

**Why not just the title?**
Titles alone are too ambiguous. Consider:

| Title only | Score |
|---|---|
| "Modi meets Biden" vs "PM holds foreign talks" | `0.71` — misses |
| "Modi meets Biden \| PM Modi met US President..." vs "PM holds foreign talks \| Prime Minister Modi held..." | `0.91` — hits ✓ |

The 200-char cap keeps it fast — you want topic signal, not full article content.

**Debug tip:**
```python
# Print what gets embedded for each article
for a in articles:
    print(clusterer._article_text(a))
```

---

### Step 2 — Embedding

```python
embeddings = self.model.encode(
    texts,
    show_progress_bar=False,
    normalize_embeddings=True   # ← important
)
```

**`normalize_embeddings=True`** converts raw vectors to unit length so cosine similarity equals dot product. This makes scores more reliable and comparable across different article lengths.

**Model used:** `paraphrase-multilingual-MiniLM-L12-v2`
- 12-layer transformer, ~500MB
- Supports Indian languages (Hindi, Tamil, Telugu, etc.)
- Output: 384-dimensional float vector per article

**Debug tip:**
```python
import numpy as np

embeddings = clusterer.model.encode(
    [clusterer._article_text(a) for a in articles],
    normalize_embeddings=True
)
print(f"Embedding shape: {embeddings.shape}")   # (n_articles, 384)
print(f"Sample norm: {np.linalg.norm(embeddings[0]):.4f}")  # Should be ~1.0
```

---

### Step 3 — Similarity Matrix

```python
similarity_matrix = cosine_similarity(embeddings)
```

Produces an **n × n matrix** where `matrix[i][j]` is the cosine similarity between article `i` and article `j`.

```
         Art0   Art1   Art2   Art3
Art0  [  1.00   0.91   0.43   0.38 ]
Art1  [  0.91   1.00   0.41   0.35 ]   ← Art0 & Art1 are same topic
Art2  [  0.43   0.41   1.00   0.89 ]
Art3  [  0.38   0.35   0.89   1.00 ]   ← Art2 & Art3 are same topic
```

**Score interpretation for `paraphrase-multilingual-MiniLM-L12-v2`:**

| Score range | Meaning |
|---|---|
| `0.95 – 1.00` | Near-duplicate (same article, maybe same source) |
| `0.88 – 0.95` | Same story, different framing ✅ ideal cluster |
| `0.82 – 0.88` | Same broad topic, possibly different angle |
| `< 0.82` | Different stories |

> ⚠️ These ranges are **model-specific**. Do not apply them to other models.

**Debug tip:**
```python
import pandas as pd

titles = [a.title[:40] for a in articles]
df = pd.DataFrame(similarity_matrix, index=titles, columns=titles)
print(df.round(2).to_string())
```

---

### Step 4 — Community Detection

This is the core improvement over the old greedy approach.

#### Phase A — Build adjacency graph

```python
for i in range(n):
    for j in range(i + 1, n):
        if similarity_matrix[i][j] >= self.similarity_threshold:
            adj[i].append(j)
            adj[j].append(i)
```

An edge is added between article `i` and `j` only if their similarity meets the threshold.

```
threshold = 0.88

Articles:  A   B   C   D   E
           │           │
           └─── 0.91 ──┘   ← edge A-C
               │
           B ──┴── 0.89 ── C   ← edge B-C
               │
           D ── 0.72 ── E   ← NO edge (below threshold)
```

#### Phase B — Connected components (BFS)

```python
for start in range(n):
    if visited[start] or not adj[start]:
        continue
    # BFS from start
    component = []
    queue = [start]
    while queue:
        node = queue.pop(0)
        component.append(node)
        for neighbour in adj[node]:
            if not visited[neighbour]:
                queue.append(neighbour)
```

Articles with **no neighbours** above the threshold are **discarded** as isolated — they don't belong to any story cluster.

**Worked graph example:**

```
Nodes: A(left), B(right), C(left), D(right), E(left)

Edges (sim ≥ 0.88):
  A ──── B   (sim=0.92)   ← same story
  A ──── C   (sim=0.89)   ← same story
  D ──── E   (sim=0.91)   ← different story

Connected components:
  Component 1: [A, B, C]
  Component 2: [D, E]
```

**Debug tip:**
```python
# Visualise the adjacency graph
for i, neighbours in enumerate(adj):
    if neighbours:
        for j in neighbours:
            if j > i:
                print(f"  {articles[i].title[:30]!r}  ←→  {articles[j].title[:30]!r}  ({similarity_matrix[i][j]:.3f})")
```

---

### Step 5 — Coherence Enforcement

Even after connected components, a cluster can have **transitively included outliers**:

```
A ──── B ──── C
       │
       └──── D   ← D is similar to B but NOT to A or C
```

If `A≈B`, `B≈C`, `B≈D` — all four end up in one component even though D doesn't belong.

`_enforce_coherence` fixes this:

```python
while len(indices) >= 2:
    sub = sim_matrix[np.ix_(indices, indices)]
    n = len(indices)
    off_diag_avg = (sub.sum() - np.trace(sub)) / (n * (n - 1))

    if off_diag_avg >= self.min_cluster_similarity:
        break  # ✅ cluster is coherent

    # Remove the article with lowest mean similarity to rest
    mean_sims = [(sub[i].sum() - 1.0) / (n - 1) for i in range(n)]
    worst = np.argmin(mean_sims)
    indices.pop(worst)
```

**Visual walkthrough:**

```
Iteration 1:
  Cluster: [A, B, C, D]
  Avg pairwise sim: 0.83  →  below min_cluster_similarity (0.85)

  Mean sims:
    A: (A-B + A-C + A-D) / 3 = (0.91 + 0.90 + 0.61) / 3 = 0.807
    B: (B-A + B-C + B-D) / 3 = (0.91 + 0.89 + 0.88) / 3 = 0.893
    C: (C-A + C-B + C-D) / 3 = (0.90 + 0.89 + 0.59) / 3 = 0.793
    D: (D-A + D-B + D-C) / 3 = (0.61 + 0.88 + 0.59) / 3 = 0.693  ← lowest

  → Remove D

Iteration 2:
  Cluster: [A, B, C]
  Avg pairwise sim: 0.90  →  above 0.85  ✅  stop
```

**Debug tip — see what gets removed:**
```python
import logging
logging.getLogger('clustering').setLevel(logging.DEBUG)
# You'll see lines like:
# DEBUG: Removing off-topic article: "Some unrelated headline" (mean_sim=0.693)
```

---

### Step 6 — Cluster Validation

```python
def _build_valid_cluster(self, group):
    left_in   = [a for a in group if a.lean == "left"]
    right_in  = [a for a in group if a.lean == "right"]
    centre_in = [a for a in group if a.lean == "centre"]

    if not left_in or not right_in:
        return None   # ← cluster discarded

    best_left   = max(left_in,   key=lambda a: a.reliability)
    best_right  = max(right_in,  key=lambda a: a.reliability)
    best_centre = max(centre_in, key=lambda a: a.reliability) if centre_in else None

    return Cluster(
        left_articles=[best_left],
        right_articles=[best_right],
        centre_articles=[best_centre] if best_centre else [],
    )
```

**Selection logic:**
From multiple left-leaning articles on the same story, pick the one with the highest `reliability` score. Same for right. This means your AI always gets the most credible source.

---

## 4. Why the Old Approach Failed

The original code used **greedy single-pass BFS**:

```python
# OLD CODE
for i in range(n):
    if visited[i]:
        continue
    cluster = [articles[i]]
    visited[i] = True
    for j in range(i + 1, n):
        if not visited[j] and similarity_matrix[i][j] >= threshold:
            cluster.append(articles[j])
            visited[j] = True   # ← marked visited even if only similar to i, not to others
```

**The failure scenario:**

```
Articles: A (Budget), B (Budget), C (Elections), D (Elections)

Similarities:
  A-B: 0.93  (same story ✅)
  A-C: 0.85  (different story, but above 0.82 threshold ⚠️)
  C-D: 0.91  (same story ✅)
  B-C: 0.71  (different story ✅)
  B-D: 0.69  (different story ✅)

Old greedy result starting from A:
  i=0 (A): cluster=[A], check j=1(B): 0.93≥0.82 → add B
                         check j=2(C): 0.85≥0.82 → add C  ← WRONG, C is different topic
                         check j=3(D): D already marked? No → 0.38, skip
  Result: ONE cluster [A, B, C] ← Budget + Elections mixed!

New community detection:
  Threshold=0.88:  A-B edge (0.93), C-D edge (0.91)
  No edge A-C (0.85 < 0.88)
  Component 1: [A, B]
  Component 2: [C, D]
  Result: TWO clean clusters ✅
```

---

## 5. Key Parameters & Tuning

| Parameter | Default | Effect |
|---|---|---|
| `similarity_threshold` | `0.88` | Min sim for graph edge. Higher = stricter, fewer clusters |
| `min_cluster_similarity` | `0.85` | Min avg pairwise sim to keep cluster intact |
| Snippet length | `200 chars` | More context per article. Increase if titles are very short |

### Tuning guide

**Getting too few clusters (stories not grouping):**
```python
# Lower the threshold
clusterer = ArticleClusterer(similarity_threshold=0.85, min_cluster_similarity=0.82)
```

**Still getting mixed topics in a cluster:**
```python
# Raise both thresholds
clusterer = ArticleClusterer(similarity_threshold=0.90, min_cluster_similarity=0.88)
```

**Articles with very short titles (< 5 words):**
- Increase snippet length from 200 → 400 chars in `_article_text()`
- Or consider a stronger model like `all-MiniLM-L6-v2` for English-only content

**Hindi/regional language articles not clustering:**
- `paraphrase-multilingual-MiniLM-L12-v2` handles this, but mixed-language titles can confuse it
- Consider lowering threshold to `0.85` for multilingual batches

---

## 6. Debugging Playbook

### Enable debug logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('clustering').setLevel(logging.DEBUG)
```

### Inspect the similarity matrix

```python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

clusterer = ArticleClusterer()
texts = [clusterer._article_text(a) for a in articles]
embeddings = clusterer.model.encode(texts, normalize_embeddings=True)
sim_matrix = cosine_similarity(embeddings)

titles = [a.title[:35] for a in articles]
df = pd.DataFrame(sim_matrix, index=titles, columns=titles)
print(df.round(3).to_string())
```

### Find all pairs above a threshold

```python
threshold = 0.88
for i in range(len(articles)):
    for j in range(i + 1, len(articles)):
        if sim_matrix[i][j] >= threshold:
            print(
                f"[{sim_matrix[i][j]:.3f}]  "
                f"{articles[i].lean:<8}  {articles[i].title[:40]}"
                f"\n          ↕"
                f"\n         {articles[j].lean:<8}  {articles[j].title[:40]}\n"
            )
```

### Check what gets discarded

```python
clusters = clusterer.cluster_articles(articles)
clustered_urls = {
    a.url
    for c in clusters
    for a in c.left_articles + c.right_articles + c.centre_articles
}
discarded = [a for a in articles if a.url not in clustered_urls]
print(f"\nDiscarded {len(discarded)} articles:")
for a in discarded:
    print(f"  [{a.lean}] {a.title}")
```

### Validate cluster quality

```python
for i, cluster in enumerate(clusters):
    left  = cluster.left_articles[0]
    right = cluster.right_articles[0]

    l_idx = articles.index(left)
    r_idx = articles.index(right)
    score = sim_matrix[l_idx][r_idx]

    print(f"\nCluster {i+1}  (L↔R sim: {score:.3f})")
    print(f"  LEFT  [{left.source_name}]  {left.title}")
    print(f"  RIGHT [{right.source_name}]  {right.title}")
    if score < 0.88:
        print(f"  ⚠️  Low similarity — may be different stories!")
```

---

## 7. Worked Example

**Input: 6 articles across 2 stories**

```
Story A — Union Budget
  A1 (left,  rel=0.85): "Budget 2025 fails the poor, say activists"
  A2 (right, rel=0.90): "Budget 2025 boosts infrastructure and growth"
  A3 (left,  rel=0.75): "Finance Minister presents Union Budget amid protests"

Story B — Delhi Elections
  B1 (right, rel=0.88): "BJP confident of Delhi sweep, cites development record"
  B2 (left,  rel=0.82): "AAP promises welfare push as Delhi votes"
  B3 (centre,rel=0.79): "Delhi election: key constituencies to watch"
```

**Similarity matrix (relevant pairs):**

```
       A1    A2    A3    B1    B2    B3
A1  [ 1.00  0.91  0.89  0.41  0.38  0.43 ]
A2  [ 0.91  1.00  0.87  0.39  0.35  0.40 ]
A3  [ 0.89  0.87  1.00  0.37  0.33  0.38 ]
B1  [ 0.41  0.39  0.37  1.00  0.92  0.90 ]
B2  [ 0.38  0.35  0.33  0.92  1.00  0.88 ]
B3  [ 0.43  0.40  0.38  0.90  0.88  1.00 ]
```

**Graph edges at threshold=0.88:**

```
A1 ── A2  (0.91)
A1 ── A3  (0.89)
B1 ── B2  (0.92)
B1 ── B3  (0.90)
B2 ── B3  (0.88)
```

**Connected components:**

```
Component 1: [A1, A2, A3]
Component 2: [B1, B2, B3]
```

**Coherence check (min=0.85):**

```
Component 1 avg sim:
  (0.91 + 0.89 + 0.87) × 2 / (3×2) = 0.89  ✅  above 0.85, no trimming

Component 2 avg sim:
  (0.92 + 0.90 + 0.88) × 2 / (3×2) = 0.90  ✅  above 0.85, no trimming
```

**Final clusters:**

```
Cluster 1 — Union Budget
  LEFT:   A1 (rel=0.85)  "Budget 2025 fails the poor, say activists"
  RIGHT:  A2 (rel=0.90)  "Budget 2025 boosts infrastructure and growth"
  (A3 discarded — lower reliability than A1)

Cluster 2 — Delhi Elections
  RIGHT:  B1 (rel=0.88)  "BJP confident of Delhi sweep..."
  LEFT:   B2 (rel=0.82)  "AAP promises welfare push..."
  CENTRE: B3 (rel=0.79)  "Delhi election: key constituencies..."
```

---

## 8. Common Failure Modes

### ❌ Two different stories end up in one cluster

**Symptom:** Generated perspective mixes unrelated facts.

**Diagnosis:**
```python
# Check the L↔R similarity score from debug snippet above
# If score is 0.85–0.88, you're in the "same broad topic" zone
```

**Fix:** Raise `similarity_threshold` to `0.90` and `min_cluster_similarity` to `0.87`.

---

### ❌ Valid story pairs are not clustering

**Symptom:** Fewer clusters than expected, some stories never appear.

**Diagnosis:**
```python
# Run the "find all pairs" snippet with a lower threshold (e.g. 0.80)
# to see what score the pair actually gets
```

**Fix options:**
- Lower `similarity_threshold` to `0.85`
- Increase snippet length (articles with very short titles lose signal)
- Check if one article is in Hindi/regional and the other in English — consider translating titles before embedding

---

### ❌ Same story from the same source appearing in multiple clusters

**Symptom:** Duplicate clusters with minor variation.

**Fix:** Add URL deduplication before clustering:
```python
seen_urls = set()
articles = [a for a in articles if not (a.url in seen_urls or seen_urls.add(a.url))]
```

---

### ❌ All articles end up in one giant cluster

**Symptom:** Single cluster with 20+ articles.

**Cause:** Threshold too low, or a very generic news day (all articles about one mega-story like budget day).

**Fix:** Raise threshold. If it's genuinely one mega-story, consider splitting by sub-topic using a secondary clustering pass on the filtered group.

---

### ❌ `reliability` scores are all the same

**Symptom:** `_best_article` is picking arbitrarily.

**Fix:** Ensure your scraper sets meaningful reliability scores per source. A simple mapping works:

```python
SOURCE_RELIABILITY = {
    "The Hindu": 0.90,
    "NDTV": 0.85,
    "Times of India": 0.82,
    "Republic World": 0.75,
    # etc.
}
article.reliability = SOURCE_RELIABILITY.get(article.source_name, 0.70)
```

---

*Generated for Spectrum News · Clustering v2 · `paraphrase-multilingual-MiniLM-L12-v2`*