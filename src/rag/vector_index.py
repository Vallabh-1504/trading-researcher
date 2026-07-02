import os
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

# Dual Indexing (Dense ChromaDB + Sparse BM25)
class HybridVectorIndex:
    """
    Manages two complementary indexes:
    - ChromaDB collection (dense vectors via SentenceTransformer)
    - BM25 index (sparse keyword scoring)

    Both are populated together during ingest so they stay in sync.
    """
    def __init__(self, encoder: SentenceTransformer):
        self.encoder = encoder

        # Optimization: Persistent disk-based vector storage
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "chroma_db")
        os.makedirs(db_path, exist_ok=True)

        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name="sec_filings", 
            metadata={"hnsw:space": "cosine"} # Use cosine distance (better for normalized sentence embeddings)
        )

        # BM25 state (must be rebuilt whenever documents are added)
        self._bm25_corpus: List[str] = []   # Raw text of each child chunk
        self._bm25_child_ids: List[str] = []    # Parallel list of child IDs
        self._bm25_index: Optional[BM25Okapi] = None
        self._child_id_to_parent_info: Dict[str, Dict[str, str]] = {}

        print("[VectorIndex] ChromaDB (in-memory) and BM25 initialized")


    def has_ticker(self, ticker: str) -> bool:
        """Checks if a ticker's embeddings are already computed and persisted."""
        results = self.collection.get(where={"ticker": ticker}, limit=1)
        return len(results["ids"]) > 0
    

    def add_documents(self, children: List[Dict]):
        """
        Adds child chunks to both ChromaDB and BM25.
        Encodes all children in a single batch (faster than one-by-one).
        """
        if not children:
            return

        texts = [c["text"] for c in children]
        ids = [c["id"] for c in children]
        metadata = [
            {"parent_id": c["parent_id"], "ticker": c["ticker"], "parent_text": c["parent_text"]}
            for c in children
        ]

        # 1. Dense Indexing (ChromaDB)
        print(f"  [VectorIndex] Encoding {len(texts)} chunks (dense)...")

        embeddings = self.encoder.encode(
            texts, 
            batch_size=256, 
            show_progress_bar=False,
        ).tolist()
        
        self.collection.add(
            ids=ids, 
            embeddings=embeddings, 
            documents=texts, 
            metadatas=metadata
        ) 
        
        print(f"[VectorIndex] Added {len(texts)} chunks to ChromaDB")


    def sync_bm25(self):
        """Rebuilds the BM25 sparse index rapidly from disk cache."""
        data = self.collection.get(include=["documents", "metadatas"])
        if not data or not data["ids"]:
            return
            
        self._bm25_corpus = data["documents"]
        self._bm25_child_ids = data["ids"]
        
        for c_id, meta in zip(data["ids"], data["metadatas"]):
            self._child_id_to_parent_info[c_id] = {
                "parent_id": meta["parent_id"],
                "parent_text": meta["parent_text"]
            }

        tokenized = [doc.lower().split() for doc in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized)

        print(f"[VectorIndex] BM25 rebuilt. Total corpus: {len(self._bm25_corpus)} chunks")


    def dense_search(self, query: str, top_k: int) -> Dict[str, str]:
        """Returns a deduplicated dictionary of {parent_id: parent_text}"""        
        total = self.collection.count()
        if total == 0:
            print("[VectorIndex] Dense search skipped — collection is empty.")
            return {}
            
        query_embedding = self.encoder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding, 
            n_results=min(top_k, total), # can never exceed what's stored
            include=["metadatas"]
        )
        
        parents = {}
        for meta in results["metadatas"][0]:
            parents[meta["parent_id"]] = meta["parent_text"]

        return parents
    

    def sparse_search(self, query: str, top_k: int) -> Dict[str, str]:
        """Returns top-K parent_ids via sparse (BM25 keyword) retrieval."""
        if self._bm25_index is None:
            return {}

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        parents = {}
        for idx in top_indices:
            if scores[idx] > 0:  # Skip zero-score results
                child_id = self._bm25_child_ids[idx]
                parent_info = self._child_id_to_parent_info.get(child_id)
                if parent_info:
                    parents[parent_info["parent_id"]] = parent_info["parent_text"]
                    
        return parents