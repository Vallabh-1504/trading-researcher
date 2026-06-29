# src/orchestrator/schemas.py

# Three categories of Schema to be defined for the project
# 1. LLM Output Schemas  - what the AI is forced to produce (structured output)
# 2. Math Engine Schema  - what the math engine returns (deterministic)
# 3. Graph State         - the LangGraph state machine's memory

from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import TypedDict

# CATEGORY 2: Math Engine Schema
# The deterministic output of src/engine/math_engine.py.
class QuantSignalResult(BaseModel):
    """
    The deterministic, mathematical verdict on a pairs trade.
    Produced exclusively by the math engine - the LLM cannot modify this.

    Tradability matrix:
        is_cointegrated = True    (ADF p-value < 0.05)
        |z_score| > 2.0           (current spread is statistically extreme)
        5 <= half_life_days <= 45 (mean reversion is fast enough to be practical
                                   but slow enough for execution)
    """
    ticker_a: str
    ticker_b: str
    hedge_ratio: float = Field(
        description=(
            "OLS beta (β) with intercept. Kept for backward compatibility and "
            "used with hedge_intercept to form the statistical residual spread."
        )
    )

    hedge_intercept: float = Field(
        description="OLS intercept (α) used in the statistical residual spread."
    )

    hedge_ratio_no_intercept: float = Field(
        description=(
            "No-intercept beta (β), the intuitive unit hedge ratio for B per "
            "1 unit of A."
        )
    )

    adf_p_value: float = Field(
        description="p-value from Augmented Dickey-Fuller test. < 0.05 = cointegrated (good)."
    )

    is_cointegrated: bool = Field(
        description="True if adf_p_value < 0.05. The pair's spread is stationary."
    )

    z_score: float = Field(
        description=(
            "Current standard deviations from the historical mean of the spread. "
            "|Z| > 2.0 triggers a signal."
        )
    )

    half_life_days: float = Field(
        description=(
            "Expected days for the spread to revert halfway to its mean, "
            "derived from the Ornstein-Uhlenbeck AR(1) model."
        )
    )

    is_tradable: bool = Field(
        description=(
            "True only if ALL three conditions pass: "
            "cointegrated AND |z_score| > 2.0 AND 5 <= half_life_days <= 45."
        )
    )

    rejection_reason: str = Field(
        default="",
        description="If is_tradable=False, explains which condition failed."
    )


# CATEGORY 1: LLM Output Schemas
# These are passed to the LLM as response_schema, forcing rigid JSON output.
class DocumentRelevance(BaseModel):
    """
    Grader schema: the LLM evaluates one SEC document chunk and decides
    if it contains useful signal for understanding DIVERGENCE between the pair.
    """
    is_relevant: bool = Field(
        description=(
            "True if this document contains information about: supply chain issues, "
            "margin pressures, competitive dynamics, guidance changes, regulatory risk, "
            "or any structural factor that could explain WHY these two stocks are diverging. "
            "False for generic boilerplate, tax footnotes, or unrelated segments."
        )
    )

    reason: str = Field(
        description="One sentence explaining why this document is or is not relevant."
    )


class QualitativeThesis(BaseModel):
    """
    Thesis schema: after seeing all relevant documents, the LLM synthesizes
    one coherent investment thesis for the pairs trade.
    """
    primary_driver: str = Field(
        description=(
            "One concise sentence describing the main structural reason for the "
            "spread divergence between the two assets. Be specific and evidence-based."
        )
    )

    affected_ticker: str = Field(
        description=(
            "Which ticker is primarily experiencing the headwind/tailwind "
            "driving the divergence (e.g., 'PEP')."
        )
    )

    direction: str = Field(
        description=(
            "'widening' if the structural factor suggests the spread will continue to grow, "
            "'converging' if the factor is temporary and spread should revert to mean."
        )
    )

    seniiment_score: float = Field(
        description=(
            "Sentiment on the spread trade: -1.0 means highly negative/pairs trade looks bad, "
            "+1.0 means highly positive/strong convergence thesis. "
            "0.0 is neutral/inconclusive."
        )
    )

    supporting_evidence: List[str] = Field(
        description=(
            "2-4 specific pieces of evidence from the SEC filings that support this thesis. "
            "Each should be a paraphrased insight, NOT a direct quote."
        )
    )

    confidence: str = Field(
        description=(
            "'low' if evidence is weak or ambiguous, "
            "'medium' if evidence is present but limited, "
            "'high' if multiple documents clearly support the thesis."
        )
    )


# CATEGORY 3: LangGraph State
# This TypedDict is the shared memory of the entire LangGraph state machine.
# Every node reads from and writes to this. It is serialized to SQLite
# by the AsyncSqliteSaver checkpointer for crash recovery.
# IMP: All values must be JSON-serializable (no custom objects).

class GraphState(TypedDict):
    # Inputs (set by CLI before graph starts)
    ticker_a: str
    ticker_b: str
    prices_a: List[float]       # Daily adjusted close prices for ticker_a
    prices_b: List[float]       # Daily adjusted close prices for ticker_b
    raw_documents: List[str]    # Top-K paragraphs retrieved by the Hybrid RAG

    # Set by node_grade_docs
    filtered_documents: List[str]   # Subset of raw_documents that passed LLM grading

    # Set by node_generate_thesis
    qualitative_thesis: Optional[dict]   # QualitativeThesis.model_dump() or None

    # Set by node_quant_validate
    quant_metrics: Optional[dict]        # QuantSignalResult.model_dump() or None

    # Set by CLI *before* resuming from HITL interrupt
    human_approved: bool

    # Set by node_hitl_approval (final node)
    final_verdict: str   # "APPROVED_AND_LOGGED" | "REJECTED_BY_HUMAN" | "REJECTED_BY_MATH"
