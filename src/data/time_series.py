import asyncio
import pandas as pd
import yfinance as yf
import logging
from datetime import datetime, timedelta
import io

from src.data.cache_manager import SQLiteCacheManager

# Configure module-level logger
logger = logging.getLogger(__name__)

class TimeSeriesFetcher:
    """
    Fetches historical pricing data for pairs trading analysis.
    Validates data integrity and handles fallback to cached data.
    """
    
    def __init__(self, cache_manager: SQLiteCacheManager):
        self.cache = cache_manager

    def _download_sync(self, ticker_a: str, ticker_b: str, start_str: str, end_str: str) -> pd.DataFrame:
            """
            Synchronous yfinance call. Isolated here to be run in a separate thread.
            """
            logger.info(f"Initiating yfinance network request for {ticker_a} & {ticker_b}...")

            return yf.download(
                tickers=[ticker_a, ticker_b], 
                start=start_str, 
                end=end_str,
                progress=False
            )
    
    async def fetch(self, ticker_a: str, ticker_b: str, lookback_days: int = 504) -> pd.DataFrame:
            """
            Retrieves adjusted close prices, preferring cache. Validates output shape.
            """
            cache_key = f"yfinance_{ticker_a}_{ticker_b}_{lookback_days}"
            
            # 1. Attempt to hit cache
            cached_json = await self.cache.get(cache_key, expiry_days=1)
            if cached_json:
                # We must specify orient and date formats depending on how pandas serialized it
                return pd.read_json(io.StringIO(cached_json))
    
            # 2. Cache Miss - Calculate dates.
            # factor with 1.5 for calender days -> trading days
            end_date = datetime.now()
            calendar_buffer = int(lookback_days * 1.5)
            start_date = end_date - timedelta(days=calendar_buffer)
    
            try:
                # 3. Execute blocking I/O in a separate thread
                df = await asyncio.to_thread(
                    self._download_sync,
                    ticker_a, ticker_b,
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # 4. Strict Data Validation
                if df.empty:
                    raise ValueError(f"yfinance returned an empty DataFrame for {ticker_a}, {ticker_b}")
                    
                if 'Close' not in df.columns:
                    raise KeyError("Could not find 'Close' prices in yfinance response. API structure may have changed.")
    
                # Extract just the Close prices. yfinance returns a MultiIndex column if multiple tickers are passed.
                prices_df = df['Close'].copy()
                
                if ticker_a not in prices_df.columns or ticker_b not in prices_df.columns:
                    raise ValueError("One or both tickers are missing from the parsed Close prices.")
    
                # 5. Data Imputation (Forward fill holidays/halts, then drop irrecoverable NaNs)
                initial_len = len(prices_df)
                prices_df.ffill(inplace=True)
                prices_df.dropna(inplace=True)
                
                if len(prices_df) < initial_len:
                    logger.warning(f"Dropped {initial_len - len(prices_df)} rows containing unfillable NaNs.")
    
                if len(prices_df) < 60:
                    raise ValueError(f"Insufficient data retrieved: Only {len(prices_df)} valid trading days found.")
    
                # 6. Trim to exactly lookback_days rows (tail = most recent dates).
                # converts the calendar-day over-fetch back to trading days
                if len(prices_df) > lookback_days:
                    prices_df = prices_df.iloc[-lookback_days:]
                elif len(prices_df) < lookback_days:
                    logger.warning(
                        f"Requested {lookback_days} trading days but yfinance only returned "
                        f"{len(prices_df)} days for {ticker_a}/{ticker_b}. "
                    )

                # 7. Cache the valid data
                await self.cache.set(cache_key, prices_df.to_json())
                logger.info(
                    f"Time-Series fetched, validated, and cached successfully. "
                    f"Returning {len(prices_df)} trading days."
                )

                return prices_df
    
            except Exception as e:
                logger.error(f"Failed to process Time-Series data: {e}")
                raise

# =============================================================================
# Isolated Module Testing
# =============================================================================
if __name__ == "__main__":
    # Set up stdout logging specifically for this test run
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    
    async def test_time_series():
        print("\n--- Starting TimeSeriesFetcher Test ---\n")
        
        cache = SQLiteCacheManager()
        await cache.initialize()
        
        fetcher = TimeSeriesFetcher(cache)
        
        # Test 1: Fetch KO and PEP
        try:
            df = await fetcher.fetch("KO", "PEP", lookback_days=100)
            print("\n[SUCCESS] DataFrame Output:")
            print(df.head())
            print(f"\nDataFrame Shape: {df.shape}")
            
            # Assertions to strictly verify our output
            assert "KO" in df.columns and "PEP" in df.columns, "Tickers missing from columns"
            assert not df.isnull().values.any(), "DataFrame contains NaNs after cleaning"
            print("[SUCCESS] All assertions passed.\n")
            
        except Exception as e:
            print(f"\n[FAILED] Exception caught: {e}\n")

    asyncio.run(test_time_series())