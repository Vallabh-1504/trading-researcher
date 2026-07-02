import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple

# Chunking (Parent-Child Strategy)
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
            f"[Chunker] {ticker}: {len(raw_text):,} chars → "
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

        print(f"Chunker [{ticker}]: {len(parents)} parents -> {len(children)} children.")
        return parent_store, children