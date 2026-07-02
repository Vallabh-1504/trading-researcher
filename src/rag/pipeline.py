from sentence_transformers import SentenceTransformer
from typing import List, Dict

# Local imports from the split modules
from .chunker import DocumentChunker
from .vector_index import HybridVectorIndex
from .reranker import CrossEncoderReranker

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
        print(f" [HybridRAG] Initializing HybridRAG Pipeline (Embeddings: {embedding_model})")
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
            print(f"[HybridRAG] WARNING: No text to ingest for {ticker}. Skipping.")
            return
        
        # Check for Cache hit
        if self.index.has_ticker(ticker):
            print(f"Vector cache HIT for {ticker}. Skipping embeddings.")
        else:
            print(f"Vector cache MISS for {ticker}. Computing embeddings...")
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

        # Sync the fast keyword search engine with the database
        self.index.sync_bm25()

        # How many candidates to gather before reranking
        # More candidates = better reranking quality, but slower
        candidate_count = top_k * 2

        # Stage 1: Dense retrieval
        dense_parents = self.index.dense_search(query, top_k=candidate_count)
        print(f"[HybridRAG] Dense retrieval: {len(dense_parents)} parent IDs")

        # Stage 2: Sparse retrieval
        sparse_parents = self.index.sparse_search(query, top_k=candidate_count)
        print(f"[HybridRAG] Sparse (BM25) retrieval: {len(sparse_parents)} parent IDs")

        # Stage 3: Union (deduplicate)
        # Union the dictionaries to automatically deduplicate overlapping chunks
        all_parents = {**dense_parents, **sparse_parents}
        print(f"[HybridRAG] After union: {len(all_parents)} unique parent IDs")

        candidate_paragraphs = list(all_parents.values())
        if not candidate_paragraphs:
            print("[HybridRAG] WARNING: No candidate paragraphs found!")
            return []

        # Reranking
        print(f"Reranking {len(candidate_paragraphs)} candidate paragraphs...")
        
        # Stage 4: Use cross-encoder to rerank and directly return top_k paragraphs
        top_docs = self.reranker.rerank(query, candidate_paragraphs, top_k)
        
        print(f"[HybridRAG] Retrieved {len(top_docs)} final paragraphs\n")
        return top_docs
    
# Isolted Testing
if __name__ == "__main__":
    print("\nStarting Hybrid RAG Pipeline Test\n")
    
    rag = HybridRAGPipeline()

    fake_ko_text = """
    Item 2.02 Results of Operations.
    Coca-Cola reported strong Q3 results driven by pricing power in its sparkling beverages segment.
    However, the company faced meaningful headwinds from elevated aluminum and PET plastic input costs,
    which compressed gross margins by approximately 120 basis points year-over-year.
    Management highlighted ongoing supply chain normalization in North America but noted persistent
    bott bottlenecks in Latin American bottling operations due to port congestion.
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