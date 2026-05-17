from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

from scraper import RawArticle

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    left_articles: List[RawArticle]
    right_articles: List[RawArticle]
    centre_articles: List[RawArticle]


class ArticleClusterer:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.75,
        min_cluster_similarity: float = 0.65,
    ):
        """
        Args:
            similarity_threshold:     Min cosine similarity to consider two articles related.
            min_cluster_similarity:   Min *average* pairwise similarity required to keep a cluster
                                      intact. Articles that drag the average below this are split off.
        """
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.min_cluster_similarity = min_cluster_similarity

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def cluster_articles(self, articles: List[RawArticle]) -> List[Cluster]:
        """Cluster articles by topic and return clusters that have ≥1 left + ≥1 right article."""
        if not articles:
            return []


        # 1. Build rich text representations
        texts = [self._article_text(a) for a in articles]

        # 2. Embed
        logger.info(f"Encoding {len(texts)} articles...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        # 3. Compute full similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # DEBUG: inspect similarity distribution
        n = len(articles)
        upper = similarity_matrix[np.triu_indices(n, k=1)]
        logger.info(f"Similarity stats — min: {upper.min():.3f}, max: {upper.max():.3f}, "
                    f"mean: {upper.mean():.3f}, median: {np.median(upper):.3f}")
        logger.info(f"Pairs above 0.20: {(upper > 0.20).sum()}")
        logger.info(f"Pairs above 0.50: {(upper > 0.50).sum()}")
        logger.info(f"Pairs above 0.70: {(upper > 0.70).sum()}")
        flat_indices = np.argsort(upper)[::-1][:5]
        rows, cols = np.triu_indices(n, k=1)
        for idx in flat_indices:
            i, j = rows[idx], cols[idx]
            logger.info(f"  {similarity_matrix[i][j]:.3f} | {articles[i].title[:60]} | {articles[j].title[:60]}")

        # 4. Community detection — much better than greedy BFS
        raw_clusters = self._community_detection(articles, similarity_matrix)

        # 5. Validate and clean each cluster
        valid_clusters = []
        for group in raw_clusters:
            cluster = self._build_valid_cluster(group)
            if cluster:
                valid_clusters.append(cluster)

        logger.info(
            f"Formed {len(valid_clusters)} valid clusters from {len(articles)} articles "
            f"(threshold={self.similarity_threshold})"
        )
        return valid_clusters

    # ------------------------------------------------------------------ #
    #  Text representation                                                 #
    # ------------------------------------------------------------------ #

    def _article_text(self, article: RawArticle) -> str:
        """
        Combine title + snippet for a richer embedding.
        Falls back gracefully if snippet/description is missing.
        """
        parts = [article.title.strip()]

        # Support both .snippet and .description attribute names
        snippet = getattr(article, "snippet", None) or getattr(article, "description", None) or getattr(article, "summary", None)
        if snippet:
            from bs4 import BeautifulSoup
            clean = BeautifulSoup(snippet, "lxml").get_text(separator=" ")
            # First 200 chars is enough — we want topic signal, not full content
            parts.append(snippet.strip()[:300])

        return " | ".join(parts)

    # ------------------------------------------------------------------ #
    #  Community detection clustering                                      #
    # ------------------------------------------------------------------ #

    def _community_detection(
        self,
        articles: List[RawArticle],
        similarity_matrix: np.ndarray,
    ) -> List[List[RawArticle]]:
        """
        Build an adjacency graph where an edge exists between i and j iff
        similarity >= threshold, then extract connected components.

        Unlike greedy BFS this guarantees every member of a cluster has at
        least ONE neighbour inside that cluster above the threshold — so a
        stray article loosely related to the seed cannot drag in unrelated
        articles transitively.

        After connected components we do a second pass: any cluster whose
        *average* pairwise similarity is below min_cluster_similarity is
        split by removing the article with the lowest average similarity to
        the rest, repeating until the cluster is coherent or only 1 article
        remains.
        """
        n = len(articles)

        # Build adjacency (upper triangle only, symmetric)
        adj: List[List[int]] = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components via BFS
        visited = [False] * n
        components: List[List[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            if not adj[start]:
                # Isolated article — no neighbours above threshold, skip
                continue
            component = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbour in adj[node]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        queue.append(neighbour)
            components.append(component)
        

        # Convert index lists → article lists, then coherence-filter
        clusters: List[List[RawArticle]] = []
        for component in components:
            group = [articles[i] for i in component]
            sub_matrix = similarity_matrix[np.ix_(component, component)]
            coherent_groups = self._enforce_coherence(group, sub_matrix)
            clusters.extend(coherent_groups)

        return clusters

    def _enforce_coherence(
        self,
        group: List[RawArticle],
        sim_matrix: np.ndarray,
    ) -> List[List[RawArticle]]:
        """
        Repeatedly remove the article with the lowest mean similarity to the
        rest until the cluster's average pairwise similarity meets the
        min_cluster_similarity threshold.

        Removed articles are discarded (they are genuinely off-topic).
        Returns a list of groups (usually just one, occasionally zero if
        the cluster disintegrates below size 2).
        """

        logger.debug(f"Coherence check: group size {len(group)}, leans: {[a.lean for a in group]}")

        indices = list(range(len(group)))

        while len(indices) >= 2:
            sub = sim_matrix[np.ix_(indices, indices)]
            # Average pairwise sim (exclude self-similarity on diagonal)
            n = len(indices)
            off_diag = (sub.sum() - np.trace(sub)) / (n * (n - 1)) if n > 1 else 1.0

            if off_diag >= self.min_cluster_similarity:
                break  # Cluster is coherent

            # Find and remove the most-disruptive article
            mean_sims = [(sub[i].sum() - 1.0) / (n - 1) for i in range(n)]
            worst = int(np.argmin(mean_sims))
            removed = group[indices[worst]]
            logger.debug(
                f"Removing off-topic article from cluster: {removed.title!r} "
                f"(mean_sim={mean_sims[worst]:.3f})"
            )
            indices.pop(worst)

        if len(indices) < 2:
            return []

        return [[group[i] for i in indices]]

    # ------------------------------------------------------------------ #
    #  Cluster validation                                                  #
    # ------------------------------------------------------------------ #

    def _build_valid_cluster(self, group: List[RawArticle]) -> Optional[Cluster]:
        """
        From a raw group of articles, build a Cluster with the highest-
        reliability representative from each lean.  Returns None if the
        group lacks both a left and a right article.
        """

        if not group:
            return None

        left_in  = [a for a in group if a.lean == "left"]
        right_in = [a for a in group if a.lean == "right"]
        centre_in = [a for a in group if a.lean == "centre"]


        # Deduplicate within each lean by source — keep highest reliability
        best_left   = self._best_article(left_in)
        best_right  = self._best_article(right_in)
        best_centre = self._best_article(centre_in) if centre_in else None

        return Cluster(
            left_articles=[best_left] if best_left else [],
            right_articles=[best_right] if best_right else [],
            centre_articles=[best_centre] if best_centre else [],
        )

    @staticmethod
    def _best_article(articles: List[RawArticle]) -> RawArticle:
        """Return the article with the highest reliability score."""
        if len(articles) > 0 :
            return max(articles, key=lambda a: a.reliability)
        return None