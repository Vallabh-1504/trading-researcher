from sentence_transformers import CrossEncoder
from typing import List

# Reranking (cross-encoder)
class CrossEncoderReranker:
    """
    Reranks candidate paragraphs using a cross-encoder model.

    Cross-encoders process (query, document) pairs TOGETHER in the same
    forward pass - they see the full context of both simultaneously.
    This makes them significantly more accurate than bi-encoders for
    relevance scoring, at the cost of being slower.

    We only rerank after narrowing candidates via hybrid search, so the
    performance cost is bounded.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"[Reranker] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("[Reranker] Cross-encoder ready")


    def rerank(self, query: str, paragraphs: List[str], top_k: int) -> List[str]:
        """
        Scores each paragraph against the query and returns
        the top_k paragraphs sorted by relevance (highest first).
        """
        if not paragraphs:
            return []
            
        print(f"[Reranker] Scoring {len(paragraphs)} candidates...")
        
        # Build (query, doc) pairs for the cross-encoder
        pairs = [[query, para] for para in paragraphs]
        scores = self.model.predict(pairs)
        
        # Sort paragraphs by score descending
        scored = sorted(zip(scores, paragraphs), key=lambda x: x[0], reverse=True)

        top = [para for _, para in scored[:top_k]]

        print(
            f"[Reranker] Top scores: "
            f"{[round(float(s), 3) for s, _ in scored[:top_k]]}"
        )
        return top