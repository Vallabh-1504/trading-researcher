import asyncio
import logging
import os
from typing import Optional
from edgar import Company, set_identity

from src.data.cache_manager import SQLiteCacheManager

# Configure module-level logger
logger = logging.getLogger(__name__)

class SECDataFetcher:
    """
    Handles fetching and cleaning of SEC EDGAR filings.
    Wraps the synchronous edgartools library in async thread pools and caches results.
    """
    
    def __init__(self, cache_manager: SQLiteCacheManager):
        self.cache = cache_manager
        
        # REQUIRED BY SEC LAW: Set user-agent identity.
        # Fallback provided for immediate testing.
        identity = os.getenv("EDGAR_IDENTITY", "QuantResearch Agent (student@university.edu)")
        set_identity(identity)
        logger.debug(f"SEC EDGAR identity set to: {identity}")

    def _extract_sync(self, ticker: str) -> str:
        """
        Synchronous edgartools call. Isolated to run in a separate thread.
        Natively compiles HTML, resolves iXBRL, and returns clean paragraphs.
        """
        logger.info(f"Querying SEC EDGAR for latest 10-K of {ticker}...")
        
        # 1. Initialize company lookup (handles Ticker -> CIK mapping)
        company = Company(ticker)
        
        # 2. Query 10-K filings
        filings = company.get_filings(form="10-K")
        if not filings:
            logger.warning(f"No 10-K filings found for {ticker}.")
            return ""
            
        # 3. Extract the latest
        latest_10k = filings.latest()
        
        # 4. Clean text extraction
        clean_text = latest_10k.text()
        
        if len(clean_text) < 500:
            logger.warning(f"Extracted text for {ticker} is unusually short: {len(clean_text)} chars.")
            
        return clean_text
    
    async def fetch_10k_text(self, ticker: str) -> str:
            """
            Retrieves SEC 10-K text, preferring cached data to prevent rate limits.
            """
            cache_key = f"sec_10k_text_{ticker}"
            
            # 10-Ks change annually; caching for 7 days during development is safe
            cached_data = await self.cache.get(cache_key, expiry_days=7)
            if cached_data:
                return cached_data
                
            try:
                # Execute blocking SEC API call in a thread pool
                clean_text = await asyncio.to_thread(self._extract_sync, ticker)
                
                if clean_text:
                    await self.cache.set(cache_key, clean_text)
                    logger.info(f"Successfully extracted and cached 10-K text for {ticker}.")
                    
                return clean_text
                
            except Exception as e:
                logger.error(f"Failed to fetch/parse SEC data for {ticker}: {e}")
                return ""
            
# =============================================================================
# Isolated Module Testing
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    
    async def test_sec_fetcher():
        print("\n--- Starting SECDataFetcher Test ---\n")
        
        cache = SQLiteCacheManager()
        await cache.initialize()
        
        fetcher = SECDataFetcher(cache)
        
        # Test: Fetch Coca-Cola (KO)
        try:
            text = await fetcher.fetch_10k_text("KO")
            print(f"\n[SUCCESS] Extracted {len(text):,} characters.")
            print(f"Preview (First 300 chars):\n{text[:300]}...\n")
            
            # Assertions to ensure we didn't just get an empty string or HTML tags
            assert len(text) > 5000, "Extraction failed or text is incomplete."
            assert "<html" not in text.lower(), "HTML tags were not stripped properly."
            print("[SUCCESS] SEC assertions passed.\n")
            
        except Exception as e:
            print(f"\n[FAILED] Exception caught: {e}\n")

    asyncio.run(test_sec_fetcher())