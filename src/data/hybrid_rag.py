# src/data/hybrid_rag.py
# Hybrid RAG Pipeline
#
# This module handles turning raw SEC narrative text into retrieved context
# that the LangGraph nodes can feed to the LLM.
#
# THE PIPELINE (4 stages):
#
#   1. INGEST (Parent-Child Chunking)
#      - Split text into large "parent" chunks (~1000 chars) → full paragraphs
#      - Split each parent into small "child" chunks (~200 chars) → sentences
#      - Store parents in memory, index children in ChromaDB + BM25
#      WHY: We retrieve children (precise matches) but return parents (rich context)
#           This prevents context fragmentation while keeping retrieval sharp.
#
#   2. RETRIEVE (Hybrid Search)
#      - Dense search via ChromaDB (cosine similarity on sentence embeddings)
#        Good at: semantic meaning, paraphrasing, conceptual similarity
#      - Sparse search via BM25 (exact keyword scoring)
#        Good at: ticker symbols, specific terms, exact phrases
#      - Union the results → candidate parent paragraphs
#
#   3. RERANK (Cross-Encoder)
#      - Score each candidate paragraph against the query using a cross-encoder
#      - Cross-encoders are slower but much more accurate than bi-encoders
#        because they process query + document TOGETHER (not separately)
#      - Sort by score, return top K
#
#   4. GRADE (LLM Relevance Check) - done in graph.py node_grade_docs
#      - The LLM does a final pass on retrieved paragraphs
#      - Filters out noise that slipped through the embedding-based retrieval
#
# MODULAR DESIGN:
#   Each stage is a separate method so you can test and swap them independently.
# =============================================================================

import os
import uuid
import logging
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple, Optional

# Configure module-level logger
logger = logging.getLogger(__name__)

# =============================================================================
# Stage 1: Chunking (Parent-Child Strategy)
# =============================================================================
class DocumentChunker:
    """
    Implements Parent-Child chunking strategy.

    Parent chunks: ~1000 chars, preserve paragraph context.
    Child chunks:  ~200 chars, precise enough for good embedding retrieval.

    The parent_id links each child back to its parent, so when we retrieve
    a child, we can look up and return the full parent paragraph.
    """
    def __init__(
            self,
            parent_size: int = 1000, 
            parent_overlap: int = 100, 
            child_size: int = 200, 
            child_overlap: int = 30
        ):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=parent_overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=child_overlap
        )

    def chunk(
            self, 
            raw_text: str, 
            ticker: str
        ) -> Tuple[Dict[str, str], List[Dict]]:
        """
        Chunks raw text into parent and child chunks.

        Returns:
            parent_store: dict mapping parent_id → parent_text
            children:     list of dicts, each with keys:
                            - id:        unique child ID (for ChromaDB)
                            - text:      child chunk text
                            - parent_id: links back to parent_store
                            - ticker:    source company ticker
        """
        parents = self.parent_splitter.split_text(raw_text)
        print(
            f"  [Chunker] {ticker}: {len(raw_text):,} chars → "
            f"{len(parents)} parent chunks"
        )

        parent_store: Dict[str, str] = {}
        children: List[Dict] = []

        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            parent_store[parent_id] = parent_text
            
            child_texts = self.child_splitter.split_text(parent_text)
            for child_text in child_texts:
                children.append({
                    "id": str(uuid.uuid4()),
                    "text": child_text,
                    "parent_id": parent_id,
                    "parent_text": parent_text,
                    "ticker": ticker,
                })

        logger.debug(f"Chunker [{ticker}]: {len(parents)} parents -> {len(children)} children.")
        return parent_store, children
    
# =============================================================================
# Stage 2: Dual Indexing (Dense ChromaDB + Sparse BM25)
# =============================================================================
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

        print("  [VectorIndex] ChromaDB (in-memory) and BM25 initialized")


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

        # # Build child_id → parent_id lookup
        # for c in children:
        #     self._child_id_to_parent_id[c["id"]] = c["parent_id"]

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
        
        print(f"  [VectorIndex] Added {len(texts)} chunks to ChromaDB")

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

        print(f"  [VectorIndex] BM25 rebuilt. Total corpus: {len(self._bm25_corpus)} chunks")


    def dense_search(self, query: str, top_k: int) -> Dict[str, str]:
        """Returns a deduplicated dictionary of {parent_id: parent_text}"""        
        total = self.collection.count()
        if total == 0:
            print("  [VectorIndex] Dense search skipped — collection is empty.")
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

# =============================================================================
# Stage 3: Reranking (cross-encoder)
# =============================================================================
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
        print(f"  [Reranker] Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)
        print("  [Reranker] Cross-encoder ready")


    def rerank(self, query: str, paragraphs: List[str], top_k: int) -> List[str]:
        """
        Scores each paragraph against the query and returns
        the top_k paragraphs sorted by relevance (highest first).
        """
        if not paragraphs:
            return []
            
        logger.debug(f"[Reranker] Scoring {len(paragraphs)} candidates...")
        
        # Build (query, doc) pairs for the cross-encoder
        pairs = [[query, para] for para in paragraphs]
        scores = self.model.predict(pairs)
        
        # Sort paragraphs by score descending
        scored = sorted(zip(scores, paragraphs), key=lambda x: x[0], reverse=True)

        top = [para for _, para in scored[:top_k]]

        print(
            f"  [Reranker] Top scores: "
            f"{[round(float(s), 3) for s, _ in scored[:top_k]]}"
        )
        return top

# =============================================================================
# Main Pipeline Class (composes all stages)
# =============================================================================

class HybridRAGPipeline:
    """
    Orchestrates the full RAG pipeline: ingest → retrieve → rerank.

    Usage:
        rag = HybridRAGPipeline()
        rag.ingest(sec_text_ko, "KO")
        rag.ingest(sec_text_pep, "PEP")
        top_docs = rag.retrieve("What supply chain issues affect PEP?", top_k=4)
    """
    def __init__(
            self, 
            embedding_model: str = "all-MiniLM-L6-v2", 
            reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ):
        logger.info(f" [HybridRAG] Initializing HybridRAG Pipeline (Embeddings: {embedding_model})")
        self.encoder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoderReranker(reranker_model)
        
        self.chunker = DocumentChunker()
        self.index = HybridVectorIndex(encoder=self.encoder)
        # Global parent store: parent_id → full paragraph text
        # Accumulates across multiple ingest() calls
        self.parent_store: Dict[str, str] = {}

        print("[HybridRAG] Pipeline ready\n")


    def ingest(self, raw_text: str, ticker: str):
        """
        Processes raw SEC text for one company and adds it to the indexes.
        Call once per ticker before retrieving.
        """
        print(f"[HybridRAG] Ingesting SEC text for {ticker}...")

        if not raw_text or len(raw_text.strip()) < 100:
            print(f"  [HybridRAG] WARNING: No text to ingest for {ticker}. Skipping.")
            return
        
        # Optimization: Idempotency check. If we already calculated this, skip the 3-minute wait.
        if self.index.has_ticker(ticker):
            logger.info(f"Vector cache HIT for {ticker}. Skipping embeddings.")
        else:
            logger.info(f"Vector cache MISS for {ticker}. Computing embeddings...")
            new_parents, children = self.chunker.chunk(raw_text, ticker)
            self.parent_store.update(new_parents)
            self.index.add_documents(children)

        print(
            f"[HybridRAG] Ingestion complete for {ticker}. "
            f"Total parents in store: {len(self.parent_store)}\n"
        )

    def retrieve(self, query: str, top_k: int = 4) -> List[str]:
        """
        Runs the full 3-stage retrieval: Hybrid Search → Union → Rerank.

        Args:
            query:  Natural language question (e.g., "Supply chain issues for PEP")
            top_k:  Number of parent paragraphs to return after reranking.

        Returns:
            List of top_k most relevant parent paragraphs, sorted by relevance.
            Returns [] if the index is empty (nothing was ingested).
        """
        print(f"\n[HybridRAG] Retrieving for query: '{query}'")

        # if not self.parent_store:
        #     logger.error("[HybridRAG] Parent store is empty. Ingest data before retrieving.")
        #     return []

        # Sync the fast keyword search engine with the database
        self.index.sync_bm25()

        # How many candidates to gather before reranking
        # More candidates = better reranking quality, but slower
        candidate_count = top_k * 2

        # --- Stage 1: Dense retrieval ---
        dense_parents = self.index.dense_search(query, top_k=candidate_count)
        print(f"  [HybridRAG] Dense retrieval: {len(dense_parents)} parent IDs")

        # --- Stage 2: Sparse retrieval ---
        sparse_parents = self.index.sparse_search(query, top_k=candidate_count)
        print(f"  [HybridRAG] Sparse (BM25) retrieval: {len(sparse_parents)} parent IDs")

        # --- Stage 3: Union (deduplicate) ---
        # Union the dictionaries to automatically deduplicate overlapping chunks
        all_parents = {**dense_parents, **sparse_parents}
        print(f"  [HybridRAG] After union: {len(all_parents)} unique parent IDs")

        candidate_paragraphs = list(all_parents.values())
        if not candidate_paragraphs:
            print("  [HybridRAG] WARNING: No candidate paragraphs found!")
            return []

        # Reranking
        logger.info(f"Reranking {len(candidate_paragraphs)} candidate paragraphs...")
        
        # Stage 4: Use cross-encoder to rerank and directly return top_k paragraphs
        top_docs = self.reranker.rerank(query, candidate_paragraphs, top_k)
        
        print(f"[HybridRAG] Retrieved {len(top_docs)} final paragraphs\n")
        return top_docs
    
# =============================================================================
# Isolated Module Testing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    print("\n--- Starting Hybrid RAG Pipeline Test ---\n")
    
    rag = HybridRAGPipeline()

    fake_ko_text = """
    Item 2.02 Results of Operations.
    Coca-Cola reported strong Q3 results driven by pricing power in its sparkling beverages segment.
    However, the company faced meaningful headwinds from elevated aluminum and PET plastic input costs,
    which compressed gross margins by approximately 120 basis points year-over-year.
    Management highlighted ongoing supply chain normalization in North America but noted persistent
    bottlenecks in Latin American bottling operations due to port congestion.
    """

    fake_pep_text = """
    PepsiCo's quarterly earnings release noted significant supply chain disruptions in its
    Frito-Lay snacks division, driven by a labor strike at three regional distribution centers.
    The disruption impacted approximately 15% of North American snack volume for six weeks.
    Meanwhile, PepsiCo's international beverage segment delivered strong performance.
    Raw material costs, particularly cooking oil and corn, remained elevated, pressuring margins.
    """

    rag.ingest(fake_ko_text, "KO")
    rag.ingest(fake_pep_text, "PEP")

    query = "What specific supply chain and operational disruptions are affecting Latin America or Frito-Lay?"
    results = rag.retrieve(query, top_k=2)

    print("\n[SUCCESS] Retrieved Paragraphs:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.strip()}")
        
    assert len(results) == 2, "Failed to retrieve the requested number of documents."
    print("\n[SUCCESS] RAG tests passed.")