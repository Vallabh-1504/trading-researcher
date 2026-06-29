from typing import Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from src.orchestrator.schemas import QuantSignalResult

# Suppress warnings for perfect data or exact 0 variances in testing
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")


class StatisticalEngine:
    """
    A deterministic mathematical evaluator for pairs trading signals.
    Executes OLS Regression, Augmented Dickey-Fuller, Z-Scoring, and AR(1) Half-Life.
    """

    @staticmethod
    def _calculate_spread(series_a: pd.Series, series_b: pd.Series) -> Tuple[float, float, float, pd.Series]:
        """Calculates hedge ratios and the OLS residual spread.

        The no-intercept hedge ratio answers: "How many units of B should I short for every
        1 unit of A I buy, so that the portfolio is market-neutral?"

        Statistical tests use the intercept OLS residual:
            Spread = Price_A - (α + β × Price_B)

        The intercept OLS beta is kept as hedge_ratio for backward-compatible output.
        The no-intercept beta is reported separately as the intuitive trading ratio.

        Args:
            series_a: Price series for Asset A (the 'long' leg)
            series_b: Price series for Asset B (the 'short' leg)

        Returns:
            (hedge_ratio, hedge_intercept, hedge_ratio_no_intercept, spread_series)
        """
        X = sm.add_constant(series_b)    # Add constant (intercept) for OLS
        model = sm.OLS(series_a, X).fit()

        hedge_intercept = float(model.params.iloc[0])
        hedge_ratio = float(model.params.iloc[1])

        denominator = float((series_b * series_b).sum())

        hedge_ratio_no_intercept = 0
        if denominator != 0:
            hedge_ratio_no_intercept = float((series_a * series_b).sum()) / denominator

        spread = series_a - (hedge_intercept + hedge_ratio * series_b)

        return hedge_ratio, hedge_intercept, hedge_ratio_no_intercept, spread
    

    @staticmethod
    def _run_adf_test(spread: pd.Series) -> Tuple[float, bool]:
        """
        Runs the Augmented Dickey-Fuller test on the spread to check for stationarity.

        Null hypothesis (H0): The spread has a unit root (it is NON-stationary).
        We want to REJECT H0 (small p-value = evidence the spread IS stationary).

        Args:
            spread: The price spread series.

        Returns:
            (p_value, is_cointegrated)
            is_cointegrated = True if p_value < 0.05
        """
        adf_result = adfuller(spread, maxlag=1, autolag=None)

        p_value = round(float(adf_result[1]), 4)
        is_cointegrated = bool(p_value < 0.05)

        print(f"ADF p-value: {p_value} | Cointegrated: {is_cointegrated}")

        return p_value, is_cointegrated
    

    @staticmethod
    def _calculate_z_score(spread: pd.Series) -> float:
        """
        Calculates the current Z-Score of the spread.

        Z = (current_value - historical_mean) / historical_std

        The Z-score tells us how "extreme" today's spread deviation is relative
        to its own historical distribution.

        A Z-score of 2.0 means the spread is 2 standard deviations above its
        mean - historically, this has been a good entry point for mean-reversion.
        """
        mean_spread = spread.mean()
        std_spread = spread.std()
        current = spread.iloc[-1]

        if std_spread == 0:
            return 0.0

        z_score = round((current - mean_spread) / std_spread, 4)

        print(f"Z-Score components -> Current: {current:.4f}, Mean: {mean_spread:.4f}, Std: {std_spread:.4f}")

        return z_score
    
    @staticmethod
    def _calculate_half_life(spread: pd.Series) -> float:
        """
        Estimates the half-life of mean reversion using the Ornstein-Uhlenbeck
        process modeled as an AR(1) regression.

        The OU process says: dSpread = θ × (μ - Spread) × dt + σ × dW
        In discrete time (AR(1) form): ΔSpread_t = α + β × Spread_{t-1} + ε_t

        Where:
        β (beta) is the mean-reversion coefficient (should be negative)
        θ = -β is the mean-reversion rate
        Half-life = log(2) / θ = -log(2) / β

        The half-life is the expected number of days for the spread to close
        half the distance from its current value back to the mean.

        If β ≥ 0, the spread is diverging (not mean-reverting) → return infinity.
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align (both lose one obs from shift/diff)
        min_len = min(len(spread_lag), len(spread_diff))
        spread_lag = spread_lag.iloc[-min_len:]
        spread_diff = spread_diff.iloc[-min_len:]

        # AR(1) regression: ΔSpread = α + β × Spread_{t-1}
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_diff, X).fit()
        beta = float(model.params.iloc[1])

        if beta >= 0:
            print(f"AR(1) Beta = {beta:.4f} >= 0. Spread is diverging. Half-life = ∞")
            return float("inf")

        half_life = round(-np.log(2) / beta, 2)
        print(f"AR(1) Beta = {beta:.4f} | Half-life = {half_life} days")

        return half_life
    
    
    def analyze(self, ticker_a: str, ticker_b: str, prices_a: List[float], prices_b: List[float]) -> QuantSignalResult:
        """
        Runs the full three-test statistical analysis on a price pair.

        This is the function called by the LangGraph node. It is deterministic:
        the same inputs will always produce the same outputs.

        Args:
            ticker_a:  Ticker symbol for Asset A
            ticker_b:  Ticker symbol for Asset B
            prices_a:  List of daily adjusted close prices for A
            prices_b:  List of daily adjusted close prices for B

        Returns:
            QuantSignalResult: The complete mathematical verdict.
        """
        print(f"Executing statistical analysis for {ticker_a} vs {ticker_b}")
        
        if len(prices_a) != len(prices_b):
            raise ValueError(f"Length mismatch: {ticker_a}={len(prices_a)}, {ticker_b}={len(prices_b)}.")
        
        if len(prices_a) < 60:
            raise ValueError(f"Insufficient data: {len(prices_a)} observations. Need >= 60.")

        series_a = pd.Series(prices_a, dtype=float)
        series_b = pd.Series(prices_b, dtype=float)

        hedge_ratio, hedge_intercept, no_intercept, spread = self._calculate_spread(series_a, series_b)

        adf_p_value, is_cointegrated = self._run_adf_test(spread)

        z_score = self._calculate_z_score(spread)

        half_life = self._calculate_half_life(spread)

        # Tradability Logic Matrix
        is_tradable = True
        rejection_reason = ""

        if not is_cointegrated:
            is_tradable = False
            rejection_reason = "Not cointegrated (p-value >= 0.05)."

        elif abs(z_score) <= 2.0:
            is_tradable = False
            rejection_reason = f"Z-Score ({z_score}) within normal range (|Z| <= 2.0)."
            
        elif half_life == float("inf") or half_life < 5 or half_life > 45:
            is_tradable = False
            rejection_reason = f"Half-life ({half_life} days) outside tradable bounds (5-45 days)."

        if is_tradable:
            print(f"✓ TRADABLE SIGNAL DETECTED: {ticker_a}/{ticker_b}")

        else:
            print(f"✗ TRADE REJECTED: {rejection_reason}")

        return QuantSignalResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            hedge_ratio=round(hedge_ratio, 4),
            hedge_intercept=round(hedge_intercept, 4),
            hedge_ratio_no_intercept=round(no_intercept, 4),
            adf_p_value=adf_p_value,
            is_cointegrated=is_cointegrated,
            z_score=z_score,
            half_life_days=half_life,
            is_tradable=is_tradable,
            rejection_reason=rejection_reason
        )
    
# Isolated Module Testing
if __name__ == "__main__":
    print("\nStarting StatisticalEngine Test\n")
    engine = StatisticalEngine()
    
    # Generate Synthetic Cointegrated Data using a random walk and a mean-reverting spread
    np.random.seed(42)
    prices_b = np.cumsum(np.random.normal(0, 1, 200)) + 100
    
    # Create a perfectly mean-reverting spread
    spread = np.zeros(200)
    for i in range(1, 200):
        spread[i] = spread[i-1] * 0.85 + np.random.normal(0, 0.5) # Strong mean reversion
    
    # Force a current deviation (Z-score > 2.0)
    spread[-1] = spread.std() * 2.5
    
    prices_a = prices_b * 0.5 + spread
    
    try:
        result = engine.analyze(
            ticker_a="SYN_A", 
            ticker_b="SYN_B", 
            prices_a=prices_a.tolist(), 
            prices_b=prices_b.tolist()
        )
        
        print("\n[SUCCESS] Engine executed. Pydantic Output:\n")
        print(result.model_dump_json(indent=2))
        
        # Assertions
        assert result.is_cointegrated is True, "Synthetic data failed ADF test."
        assert abs(result.z_score) > 2.0, "Z-Score deviation failed."
        print("\n[SUCCESS] All mathematical proofs passed.")
        
    except Exception as e:
        print(f"\n[FAILED] Engine threw an exception: {e}")
