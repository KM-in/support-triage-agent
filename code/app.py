#!/usr/bin/env python3
"""
app.py — Interactive terminal UI for the Multi-Domain Support Triage Agent.

Run:  python app.py

Type a support query and receive the agent's classification, reasoning,
and grounded response. All interactions are logged to log.txt.
Type 'quit' or 'exit' to end the session.
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich.table import Table
from rich import box

from src.triage_agent import TriageAgent
from src.utils import log_interaction


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

console = Console()

BANNER = r"""
╔══════════════════════════════════════════════════════════════════╗
║         🎯  Multi-Domain Support Triage Agent  🎯               ║
║                                                                  ║
║  Domains: HackerRank  •  Claude (Anthropic)  •  Visa            ║
║  Type 'quit' or 'exit' to end the session.                      ║
╚══════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    console.print(BANNER, style="bold cyan")


def print_result(result):
    """Display the triage result in a rich formatted layout."""

    # Classification table
    table = Table(
        title="📋 Classification",
        box=box.ROUNDED,
        show_header=False,
        title_style="bold yellow",
        border_style="dim",
        padding=(0, 2),
    )
    table.add_column("Field", style="bold white", width=16)
    table.add_column("Value", style="bright_white")

    status_style = "bold green" if result.status == "replied" else "bold red"

    table.add_row("Company", result.company or "None")
    table.add_row("Product Area", result.product_area)
    table.add_row("Request Type", result.request_type)
    table.add_row("Language", result.language)
    table.add_row("Status", Text(result.status, style=status_style))
    table.add_row("Confidence", result.confidence)

    console.print()
    console.print(table)

    # Reasoning
    console.print()
    console.print(
        Panel(
            result.reasoning,
            title="🧠 Reasoning",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Response
    response_style = "green" if result.status == "replied" else "yellow"
    console.print()
    console.print(
        Panel(
            Markdown(result.response),
            title="💬 Response" if result.status == "replied" else "⚠️  Escalation",
            border_style=response_style,
            padding=(1, 2),
        )
    )

    # Sources
    if result.retrieved_sources:
        sources = ", ".join(set(result.retrieved_sources))
        console.print(f"\n  [dim]📄 Sources consulted: {sources}[/dim]")

    console.print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print_banner()

    # Initialise the agent (loads vector store)
    with console.status("[bold cyan]Loading triage agent and vector store...", spinner="dots"):
        try:
            agent = TriageAgent()
        except Exception as e:
            console.print(f"\n[bold red]Error initializing agent:[/bold red] {e}")
            console.print(
                "[yellow]Make sure GEMINI_API_KEY is set and dependencies are installed.[/yellow]"
            )
            sys.exit(1)

    console.print("[bold green]✓ Agent ready![/bold green]\n")

    while True:
        try:
            console.print("─" * 64, style="dim")
            query = console.input("[bold cyan]📩 Enter support query:[/bold cyan] ").strip()

            if not query:
                continue

            if query.lower() in ("quit", "exit", "q"):
                console.print("\n[bold cyan]👋 Goodbye! Session log saved to log.txt[/bold cyan]")
                break

            # Process the ticket
            with console.status("[bold yellow]Processing ticket...", spinner="dots"):
                result = agent.process_ticket(issue=query)

            # Display results
            print_result(result)

            # Log the interaction
            log_interaction(
                query=query,
                response=result.response,
                classification={
                    "company": result.company,
                    "request_type": result.request_type,
                    "product_area": result.product_area,
                },
                status=result.status,
            )

        except KeyboardInterrupt:
            console.print("\n\n[bold cyan]👋 Session interrupted. Log saved to log.txt[/bold cyan]")
            break
        except Exception as e:
            console.print(f"\n[bold red]Error processing query:[/bold red] {e}")
            console.print("[yellow]Please try again.[/yellow]\n")


if __name__ == "__main__":
    main()
