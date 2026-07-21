from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from src.orchestrator.schemas import GraphState

from .nodes import (
    node_grade_docs,
    node_generate_thesis,
    node_quant_validate,
    node_hitl_approval
)

# Construct graph using nodes
def build_graph() -> StateGraph:
    """
    Builds and returns the compiled LangGraph StateGraph.
    """
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("grade_docs",     node_grade_docs)
    workflow.add_node("generate_thesis", node_generate_thesis)
    workflow.add_node("quant_validate", node_quant_validate)
    workflow.add_node("hitl_approval",  node_hitl_approval)

    # Set entry point
    workflow.set_entry_point("grade_docs")

    # Add edges (linear pipeline)
    workflow.add_edge("grade_docs",      "generate_thesis")
    workflow.add_edge("generate_thesis", "quant_validate")
    workflow.add_edge("quant_validate",  "hitl_approval")
    workflow.add_edge("hitl_approval",   END)

    print("[Graph] LangGraph workflow built: grade_docs → generate_thesis → quant_validate → [INTERRUPT] → hitl_approval → END")

    return workflow


# =============================================================================
# Isolated Module Testing
# =============================================================================
if __name__ == "__main__":
    import numpy as np
    
    print("\nStarting LangGraph State Machine Test\n")
    
    # 1. Setup mock data
    np.random.seed(42)
    prices_b = (np.cumsum(np.random.normal(0, 1, 100)) + 100).tolist()
    prices_a = (np.array(prices_b) * 0.5 + np.random.normal(0, 0.5, 100)).tolist()
    
    initial_state = {
        "ticker_a": "KO",
        "ticker_b": "PEP",
        "prices_a": prices_a,
        "prices_b": prices_b,
        "raw_documents": [
            "Coca-cola is facing severe supply chain issues in Latin America.",
            "Generic tax risk statement that is irrelevant."
        ],
        "human_approved": False, # Will inject True later
    }
    
    # 2. Compile graph with an in-memory checkpointer to test interruptions
    memory = MemorySaver()
    workflow = build_graph()
    app = workflow.compile(checkpointer=memory, interrupt_before=["hitl_approval"])
    
    thread_config: RunnableConfig = {"configurable": {"thread_id": "test_run_1"}}
    
    # 3. Run up to the interrupt
    print("\n>>> EXECUTING GRAPH (PHASE 1)...")
    for event in app.stream(initial_state, config=thread_config):
        pass # Stream prints log messages natively
        
    # Check if graph paused
    state = app.get_state(thread_config)
    print(f"\n>>> GRAPH PAUSED? {state.next == ('hitl_approval',)}")
    
    # 4. Simulate human typing 'APPROVE' in terminal
    print("\n>>> SIMULATING HUMAN APPROVAL AND RESUMING...")
    app.update_state(thread_config, {"human_approved": True})
    
    for event in app.stream(None, config=thread_config):
        pass
        
    final_state = app.get_state(thread_config).values
    print(f"\n>>> FINAL VERDICT: {final_state.get('final_verdict')}")
    assert final_state.get('final_verdict') is not None