# Triage Agent Code

This directory contains the code for the Multi-Domain Support Triage Agent.

## Setup

1. Copy `.env.example` to `.env` and configure your API key:
   ```bash
   cp .env.example .env
   ```
   Add your `GEMINI_API_KEY` to the `.env` file.

2. Install dependencies (if not already installed in your virtual environment):
   ```bash
   pip install -r requirements.txt
   ```

3. Build the vector store from the local corpus:
   ```bash
   python src/ingest.py
   ```

## Running the Agent

To process the `support_tickets.csv` file and generate `output.csv`:

```bash
python main.py
```

To run the interactive terminal UI to test individual queries:

```bash
python app.py
```
