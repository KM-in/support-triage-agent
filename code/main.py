#!/usr/bin/env python3
"""
process_csv.py — Batch processor for support tickets.

Reads support_tickets/support_tickets.csv, runs each row through
the triage pipeline, and outputs support_tickets/output.csv.

Run:  python process_csv.py
"""

import sys
import os
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich import box

from src.triage_agent import TriageAgent
from src.utils import read_support_csv, write_output_csv, log_interaction


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT.parent / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = PROJECT_ROOT.parent / "support_tickets" / "output.csv"

console = Console()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    console.print("\n[bold cyan]╔══════════════════════════════════════════╗[/bold cyan]")
    console.print("[bold cyan]║   📋 Batch Support Ticket Processor      ║[/bold cyan]")
    console.print("[bold cyan]╚══════════════════════════════════════════╝[/bold cyan]\n")

    # Validate input
    if not INPUT_CSV.exists():
        console.print(f"[bold red]ERROR:[/bold red] Input CSV not found at {INPUT_CSV}")
        sys.exit(1)

    # Read tickets
    console.print(f"[dim]Reading tickets from {INPUT_CSV}[/dim]")
    tickets = read_support_csv(INPUT_CSV)
    console.print(f"[green]✓ Loaded {len(tickets)} ticket(s)[/green]\n")

    # Initialise agent
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

    # Process each ticket
    output_rows = []
    replied_count = 0
    escalated_count = 0
    error_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing tickets...", total=len(tickets))

        for i, ticket in enumerate(tickets):
            issue = ticket.get("issue", "")
            subject = ticket.get("subject", "")
            company = ticket.get("company", "")

            progress.update(
                task,
                description=f"Ticket {i + 1}/{len(tickets)}: {subject[:40] or issue[:40]}...",
            )

            try:
                result = agent.process_ticket(
                    issue=issue,
                    subject=subject,
                    company=company,
                )

                output_rows.append(result.to_csv_row())

                if result.status == "replied":
                    replied_count += 1
                else:
                    escalated_count += 1

                # Log each processed ticket
                log_interaction(
                    query=f"[CSV Row {i + 1}] {issue}",
                    response=result.response,
                    classification={
                        "company": result.company,
                        "request_type": result.request_type,
                        "product_area": result.product_area,
                    },
                    status=result.status,
                )

            except Exception as e:
                error_count += 1
                console.print(f"\n[red]Error on ticket {i + 1}: {e}[/red]")
                output_rows.append({
                    "issue": issue,
                    "subject": subject,
                    "company": company,
                    "response": f"Error processing ticket: {e}",
                    "product_area": "unknown",
                    "status": "escalated",
                    "request_type": "invalid",
                    "justification": f"Processing error: {e}",
                })

            progress.advance(task)

            # Small delay to respect rate limits
            time.sleep(0.5)

    # Write output
    write_output_csv(output_rows, OUTPUT_CSV)

    # Summary
    console.print()
    summary = Table(
        title="📊 Processing Summary",
        box=box.ROUNDED,
        show_header=False,
        border_style="cyan",
        title_style="bold cyan",
    )
    summary.add_column("Metric", style="bold white")
    summary.add_column("Value", style="bright_white")
    summary.add_row("Total Tickets", str(len(tickets)))
    summary.add_row("Replied", f"[green]{replied_count}[/green]")
    summary.add_row("Escalated", f"[yellow]{escalated_count}[/yellow]")
    summary.add_row("Errors", f"[red]{error_count}[/red]")
    summary.add_row("Output File", str(OUTPUT_CSV))
    console.print(summary)
    console.print()


if __name__ == "__main__":
    main()
