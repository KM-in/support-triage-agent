"""
utils.py — Logging and CSV handling utilities.
"""

import csv
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT.parent / "log.txt"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_interaction(
    query: str,
    response: str,
    classification: dict | None = None,
    status: str = "",
    filepath: Path | str | None = None,
):
    """
    Append a timestamped interaction to the log file.

    Args:
        query: The user's support query.
        response: The agent's response.
        classification: Optional classification dict (request_type, company, etc.).
        status: "Replied" or "Escalated".
        filepath: Log file path (defaults to project root log.txt).
    """
    filepath = Path(filepath) if filepath else LOG_FILE

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    separator = "=" * 80

    lines = [
        separator,
        f"Timestamp: {timestamp}",
        f"Query: {query}",
    ]

    if classification:
        lines.append(f"Company: {classification.get('company', 'N/A')}")
        lines.append(f"Request Type: {classification.get('request_type', 'N/A')}")
        lines.append(f"Product Area: {classification.get('product_area', 'N/A')}")

    if status:
        lines.append(f"Status: {status}")

    lines.append(f"Response:\n{response}")
    lines.append(separator)
    lines.append("")  # blank line between entries

    with open(filepath, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CSV Handling
# ---------------------------------------------------------------------------

def read_support_csv(filepath: str | Path) -> list[dict]:
    """
    Read a support issues CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of dicts, one per row.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize keys to lowercase and strip whitespace
            cleaned = {k.strip().lower(): v.strip() if v else "" for k, v in row.items()}
            rows.append(cleaned)

    return rows


def write_output_csv(rows: list[dict], filepath: str | Path):
    """
    Write triage results to an output CSV file.

    Args:
        rows: List of dicts with output columns.
        filepath: Output CSV path.
    """
    filepath = Path(filepath)

    fieldnames = [
        "issue",
        "subject",
        "company",
        "response",
        "product_area",
        "status",
        "request_type",
        "justification",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[utils] Output written to {filepath}")
