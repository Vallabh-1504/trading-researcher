# the user-facing entry point.
#   1. Parses CLI arguments
#   2. Fetches all data (prices + SEC filings)
#   3. Runs the Hybrid RAG pipeline
#   4. Boots the LangGraph state machine
#   5. Streams node execution, displaying progress
#   6. PAUSES at the HITL interrupt for human approval
#   7. Resumes and logs the final decision
#
# HOW TO RUN:
#   uv run python cli.py --pair KO PEP
#   uv run python cli.py --pair KO PEP --lookback 252 --provider gemini
#
# THE HITL INTERRUPT MECHANISM:
#   The graph is compiled with interrupt_before=["hitl_approval"].
#   This means LangGraph saves state to SQLite and PAUSES before the last node.
#   The CLI reads the saved state, renders the verdict table, asks for input.
#   On APPROVE: CLI calls app.update_state() to inject human_approved=True,
#               then resumes graph execution (the last node runs and logs it).
#   On REJECT:  CLI calls app.update_state() with human_approved=False,
#               then resumes (the last node writes REJECTED_BY_HUMAN verdict).
# =============================================================================
import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.progress import Progress, SpinnerColumn, TextColumn

from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

# Internal Modules
from src.data.cache_manager import SQLiteCacheManager
from src.data.time_series import TimeSeriesFetcher
from src.data.sec_data import SECDataFetcher
from src.rag.pipeline import HybridRAGPipeline
from src.orchestrator.graph import build_graph

load_dotenv()

console = Console()

def print_banner():
    banner = """\
[bold cyan]
 ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗
██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝
██║   ██║██║   ██║███████║██╔██╗ ██║   ██║
██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║
╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║
 ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝
[/bold cyan][bold white]TRADE ADVISOR — AI-Assisted Quantitative Research Pipeline[/bold white]"""
    console.print(Panel(banner, border_style="cyan", expand=False))


async def fetch_and_prepare_data(ticker_a: str, ticker_b: str, lookback: int):
    """Orchestrates async data fetching and RAG ingestion."""
    console.print(f"[bold cyan]➤ Initializing Data Pipelines for {ticker_a} & {ticker_b}...[/bold cyan]")
    
    cache = SQLiteCacheManager()
    await cache.initialize()
    
    ts_fetcher = TimeSeriesFetcher(cache)
    sec_fetcher = SECDataFetcher(cache)
    rag = HybridRAGPipeline()
    
    # 1. Fetch data concurrently
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task(f"[bold yellow]Fetching Time-Series & SEC Filings for {ticker_a} & {ticker_b}...[/bold yellow]", total=None)
        prices_df_task = ts_fetcher.fetch(ticker_a, ticker_b, lookback_days=lookback)

        sec_a_task = sec_fetcher.fetch_10k_text(ticker_a)
        sec_b_task = sec_fetcher.fetch_10k_text(ticker_b)
        
        prices_df, sec_a, sec_b = await asyncio.gather(prices_df_task, sec_a_task, sec_b_task)
        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Prices: {len(prices_df)} trading days loaded")
    console.print(f"[green]✓[/green] SEC text {ticker_a}: {len(sec_a):,} chars")
    console.print(f"[green]✓[/green] SEC text {ticker_b}: {len(sec_b):,} chars")

    # 2. Ingest SEC texts into RAG
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[bold yellow]Ingesting documents into Hybrid Vector Index...[/bold yellow]", total=None)
        rag.ingest(sec_a, ticker_a)
        rag.ingest(sec_b, ticker_b)
        progress.update(task, completed=True)
    
    # 3. Retrieve relevant chunks
    query = (
        f"What are the most specific operational headwinds, regulatory risks, "
        f"margin pressures, or strategic shifts recently disclosed by {ticker_a} or {ticker_b}?"
    )

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("[bold yellow]Retrieving top-K relevant paragraphs...[/bold yellow]", total=None)
        raw_documents = rag.retrieve(query, top_k=8)
        progress.update(task, completed=True)
    
    console.print(f"[green]✓ Data ingestion and retrieval complete. Retrieved {len(raw_documents)} chunks.[/green]\n")
    
    # Extract price lists, applying robust checks
    prices_a = prices_df[ticker_a].dropna().tolist() if ticker_a in prices_df.columns else []
    prices_b = prices_df[ticker_b].dropna().tolist() if ticker_b in prices_df.columns else []

    if not prices_a or not prices_b:
        console.print(
            f"[bold red]ERROR:[/bold red] Could not extract price data for "
            f"{ticker_a} or {ticker_b}. Check that both tickers are valid."
        )
        sys.exit(1)

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "prices_a": prices_a,
        "prices_b": prices_b,
        "raw_documents": raw_documents
    }


def render_dashboard(state: dict):
    """Renders the state using Rich tables."""
    console.print("\n" + "=" * 70)
    console.print("[bold yellow]Human-in-the-Loop Review[/bold yellow]")
    console.print("=" * 70 + "\n")
    
    ticker_a = state.get("ticker_a", "A")
    ticker_b = state.get("ticker_b", "B")
    quant = state.get("quant_metrics") or {}
    thesis = state.get("qualitative_thesis") or {}

    table = Table(
        title=f"ASSISTANT VERDICT: {ticker_a} / {ticker_b}",
        style="cyan",
        show_lines=True,
        min_width=70,
    )
    table.add_column("Category", style="magenta bold", width=20)
    table.add_column("Metric", style="white", width=25)
    table.add_column("Value", style="yellow bold", width=15)
    table.add_column("Threshold", style="dim", width=15)

    # Quantitative Panel
    p_val = quant.get("adf_p_value", "N/A")
    coint = "✓ YES" if quant.get("is_cointegrated") else "✗ NO"
    
    # FIX: Force string casting for all rich table inputs
    table.add_row("MATH (ADF)", "Cointegration p-value", str(p_val), "< 0.05")
    table.add_row("", "Cointegrated?", str(coint), "")
    
    z = quant.get("z_score", "N/A")
    z_str = f"{z:+.3f}" if isinstance(z, float) else str(z)
    table.add_row("MATH (Z)", "Z-Score", z_str, "|Z| > 2.0")
    
    hl = quant.get("half_life_days", "N/A")
    hl_str = f"{hl:.1f} days" if isinstance(hl, float) and hl != float("inf") else str(hl)
    table.add_row("MATH (HL)", "Half-Life (OU AR1)", hl_str, "5-45 days")

    is_tradable = quant.get("is_tradable")

    # Qualitative Panel
    if thesis:
        driver = str(thesis.get("primary_driver", "N/A"))[:40]
        affected = str(thesis.get("affected_ticker", "N/A"))
        direction = str(thesis.get("direction", "N/A"))
        confidence = str(thesis.get("confidence", "N/A")).upper()

        table.add_row("AI (Driver)", "Primary Divergence Driver", driver, "")
        table.add_row("AI (Ticker)", "Affected Ticker", affected, "")
        table.add_row("AI (Direction)", "Spread Direction", direction, "")
        table.add_row("AI (Confidence)", "Evidence Confidence", confidence, "")
    else:
        table.add_row("AI", "Qualitative Thesis", "Not generated", "")

    console.print(table)
    
    if not is_tradable:
        reason = str(quant.get("rejection_reason", "Mathematical thresholds not met."))
        console.print(f"\n[bold red]SYSTEM LOCKED: {reason}[/bold red]")


async def main():
    print_banner()

    parser = argparse.ArgumentParser(description="QuantResearch Agent CLI")
    parser.add_argument("--pair", nargs=2, required=True, help="Two tickers to analyze (e.g., KO PEP)")
    parser.add_argument("--lookback", type=int, default=504, help="Trading days lookback (default: 504)")
    args = parser.parse_args()
    
    ticker_a, ticker_b = args.pair[0].upper(), args.pair[1].upper()
    
    # 1. Prepare Data
    initial_state = await fetch_and_prepare_data(ticker_a, ticker_b, args.lookback)
    
    # 2. Setup LangGraph
    memory = MemorySaver()
    workflow = build_graph()
    app = workflow.compile(checkpointer=memory, interrupt_before=["hitl_approval"])
    # thread_config = {"configurable": {"thread_id": f"{ticker_a}_{ticker_b}_session"}}
    thread_config: RunnableConfig = {"configurable": {"thread_id": "test_run_1"}}
    
    # 3. Run Pipeline up to the Circuit Breaker
    with console.status("[bold yellow]LangGraph Agents running analysis...[/bold yellow]"):
        for event in app.stream(initial_state, config=thread_config):
            pass # Silently stream through the nodes
            
    # 4. Check State & Render
    current_state = app.get_state(thread_config)
    current_values = current_state.values
    render_dashboard(current_values)
    
    # 5. Human-in-the-Loop Intercept
    quant = current_values.get("quant_metrics", {})
    if quant.get("is_tradable"):
        console.print("\n[bold yellow]BLINKING: [AWAITING HUMAN APPROVAL][/bold yellow]")
        decision = Prompt.ask("Authorize trade logging?", choices=["APPROVE", "REJECT"])
        
        is_approved = decision == "APPROVE"
        
        # Resume Graph
        app.update_state(thread_config, {"human_approved": is_approved})
        for event in app.stream(None, config=thread_config):
            pass
            
        final_state_values = app.get_state(thread_config).values
        console.print(f"\n[bold white]Final Pipeline Verdict:[/bold white] [bold cyan]{final_state_values.get('final_verdict')}[/bold cyan]")
    else:
        # Resume Graph automatically passing False to log the mathematical rejection
        app.update_state(thread_config, {"human_approved": False})
        for event in app.stream(None, config=thread_config):
            pass
        final_state_values = app.get_state(thread_config).values
        console.print(f"\n[bold white]Final Pipeline Verdict:[/bold white] [bold red]{final_state_values.get('final_verdict')}[/bold red]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Execution manually terminated.[/bold red]")