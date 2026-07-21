from datetime import datetime
import json

from src.orchestrator.schemas import (
    GraphState,
    QualitativeThesis,
)
from src.orchestrator.prompts import get_thesis_prompt
from src.orchestrator.llm_client import get_llm_client
from src.engine.math_engine import StatisticalEngine


# Node 1: Document Grader
def node_grade_docs(state: GraphState) -> dict:
    """
    Filters the raw RAG output down to only relevant documents.

    Each document chunk is evaluated independently by the LLM.
    The LLM returns a DocumentRelevance schema (is_relevant: bool, reason: str).

    Why grade documents?
        The RAG retrieves the most "similar" chunks, but similarity ≠ relevance.
        A chunk about "supply chain of beverages" might be highly similar to the
        query but turn out to be a generic risk factor disclaimer.
        The LLM grader adds semantic judgment that pure embeddings miss.
    """
    print("NODE 1: GRADING DOCUMENTS")

    raw_docs = state.get("raw_documents", [])

    if not raw_docs:
        print("No raw documents provided to grade.")        
        return {"filtered_documents": []}
    
    # Disable the LLM Grading of the chunks retrieved
    return {"filtered_documents": raw_docs}


# Node 2: Thesis Generator
def node_generate_thesis(state: GraphState) -> dict:
    """
    Synthesizes a structured qualitative investment thesis from the filtered docs.

    The LLM sees all relevant document chunks together (combined into one prompt)
    and produces a single structured JSON thesis via the QualitativeThesis schema.

    The thesis captures:
      - primary_driver: the main structural reason for divergence
      - affected_ticker: which company is experiencing the headwind/tailwind
      - direction: is the spread widening or converging?
      - sentiment_score: -1 to +1
      - supporting_evidence: 2-4 specific pieces of evidence
      - confidence: low/medium/high
    """
    print("\n[Graph Node 2] Synthesizing qualitative investment thesis...")

    ticker_a = state["ticker_a"]
    ticker_b = state["ticker_b"]
    filtered_docs = state.get("filtered_documents", [])

    if not filtered_docs:
        print("  [Node 2] No relevant documents available. Thesis will be empty.")
        return {"qualitative_thesis": None}

    combined_context = "\n\n".join(filtered_docs)
    print(f"[Node 2] Feeding {len(filtered_docs)} documents ({len(combined_context):,} chars) to LLM...")

    system_prompt, user_prompt = get_thesis_prompt(ticker_a, ticker_b, combined_context)
    client = get_llm_client()

    try:
        result: QualitativeThesis = client.generate_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=QualitativeThesis,
        )

        print(f"\n[Node 2] Thesis generated:")
        print(f"  Primary driver:  {result.primary_driver}")
        print(f"  Affected ticker: {result.affected_ticker}")
        print(f"  Direction:       {result.direction}")
        print(f"  Sentiment:       {result.sentiment_score}")
        print(f"  Confidence:      {result.confidence}\n")

        return {"qualitative_thesis": result.model_dump()}

    except Exception as e:
        print(f"[Node 2] ERROR: Thesis generation failed: {e}")
        return {"qualitative_thesis": None}

# Node 3: Quantitative Validation
def node_quant_validate(state: GraphState) -> dict:
    """
    Runs the deterministic math engine on the price data.

    This node calls math_engine.run_statistical_analysis() directly
    as a Python function - no HTTP, no network, no chance of the LLM
    influencing the output.

    The result is a QuantSignalResult stored as a dict in the state.
    """
    print("\n[Graph Node 3] Running quantitative validation (Math Engine)...")

    ticker_a = state["ticker_a"]
    ticker_b = state["ticker_b"]
    prices_a = state.get("prices_a", [])
    prices_b = state.get("prices_b", [])

    if not prices_a or not prices_b:
        print("  [Node 3] ERROR: No price data in state. Cannot run math engine.")
        return {"quant_metrics": None}

    try:
        engine = StatisticalEngine()
        result = engine.analyze(ticker_a, ticker_b, prices_a, prices_b)

        # Store as dict (JSON-serializable for SQLite checkpointing)
        return {"quant_metrics": result.model_dump()}

    except Exception as e:
        print(f"  [Node 3] ERROR: Math engine failed: {e}")
        return {"quant_metrics": None}

# Node 4: HITL Approval (Runs AFTER human resumes the graph)
def node_hitl_approval(state: GraphState) -> dict:
    """
    Final node - executes AFTER the human resumes the graph.

    When LangGraph hits this node:
      - If the graph was compiled with interrupt_before=["hitl_approval"],
        execution PAUSED before this node ran.
      - The CLI displayed the verdict, got the human's decision, and resumed
        the graph by calling app.update_state() with human_approved=True/False.
      - Now this node runs and writes the final_verdict.

    This node does NOT ask for input - the CLI handles that.
    It reads human_approved from state and records the final decision.
    """
    print("\n[Graph Node 4] Recording final HITL decision...")

    quant = state.get("quant_metrics") or {}
    thesis = state.get("qualitative_thesis") or {}

    human_approved = state.get("human_approved", False)

    is_tradable = quant.get("is_tradable", False)

    if not is_tradable:
        verdict = "REJECTED_BY_MATH"
        print(f"  [Node 4] Verdict: {verdict}")
        print(f"  Reason: {quant.get('rejection_reason', 'Mathematical thresholds not met')}")

    elif not human_approved:
        verdict = "REJECTED_BY_HUMAN"
        print(f"  [Node 4] Verdict: {verdict}")
        print("  The human operator declined to authorize this trade.")

    else:
        verdict = "APPROVED_AND_LOGGED"
        print(f"  [Node 4] Verdict: {verdict}")
        print(
            f"  Trade authorized: {state['ticker_a']}/{state['ticker_b']} | "
            f"Z-Score: {quant.get('z_score')} | "
            f"Driver: {thesis.get('primary_driver', 'N/A')}"
        )

        # Log the approved trade
        _log_approved_trade(state)

    print(f"\n[Graph] Pipeline complete. Final verdict: {verdict}\n")

    return {"final_verdict": verdict}


def _log_approved_trade(state: GraphState):
    """
    Logs an approved trade signal. In production this would write to a database
    or send to an order management system. For now, appends to a local log file.
    """
    import json
    from datetime import datetime

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "ticker_a": state["ticker_a"],
        "ticker_b": state["ticker_b"],
        "quant_metrics": state.get("quant_metrics"),
        "qualitative_thesis": state.get("qualitative_thesis"),
        "verdict": "APPROVED_AND_LOGGED",
    }

    log_path = "trade_log.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry) + "\n")

    print(f"  [Node 4] Trade signal logged to {log_path}")

