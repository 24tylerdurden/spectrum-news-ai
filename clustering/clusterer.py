from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from typing import List
import logging

from scraper import RawArticle

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    left_articles: List[RawArticle]
    right_articles: List[RawArticle]
    centre_articles: List[RawArticle]


class ArticleClusterer:
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", similarity_threshold: float = 0.82):
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold

    def cluster_articles(self, articles: List[RawArticle]) -> List[Cluster]:
        """Cluster articles by similarity and return valid clusters with left+right perspectives."""
        if not articles:
            return []
        
        # Separate by lean
        left_articles = [a for a in articles if a.lean == "left"]
        right_articles = [a for a in articles if a.lean == "right"]
        centre_articles = [a for a in articles if a.lean == "centre"]
        
        if not left_articles or not right_articles:
            logger.info("Not enough articles for clustering (need at least one left and one right)")
            return []
        
        # Encode headlines
        all_headlines = [a.title for a in articles]
        embeddings = self.model.encode(all_headlines)
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group articles into clusters
        clusters = self._form_clusters(articles, similarity_matrix)
        
        # Filter clusters to only those with at least one left and one right
        valid_clusters = []
        for cluster in clusters:
            left_in_cluster = [a for a in cluster if a.lean == "left"]
            right_in_cluster = [a for a in cluster if a.lean == "right"]
            centre_in_cluster = [a for a in cluster if a.lean == "centre"]
            
            if left_in_cluster and right_in_cluster:
                # Pick highest reliability from each lean
                best_left = max(left_in_cluster, key=lambda x: x.reliability)
                best_right = max(right_in_cluster, key=lambda x: x.reliability)
                best_centre = max(centre_in_cluster, key=lambda x: x.reliability) if centre_in_cluster else None
                
                valid_cluster = Cluster(
                    left_articles=[best_left],
                    right_articles=[best_right],
                    centre_articles=[best_centre] if best_centre else []
                )
                valid_clusters.append(valid_cluster)
        
        logger.info(f"Found {len(valid_clusters)} valid clusters from {len(articles)} articles")
        return valid_clusters

    def _form_clusters(self, articles: List[RawArticle], similarity_matrix: np.ndarray) -> List[List[RawArticle]]:
        """Group articles into clusters based on similarity threshold."""
        n = len(articles)
        visited = [False] * n
        clusters = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # Find all similar articles
            cluster = [articles[i]]
            visited[i] = True
            
            for j in range(i + 1, n):
                if not visited[j] and similarity_matrix[i][j] >= self.similarity_threshold:
                    cluster.append(articles[j])
                    visited[j] = True
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
