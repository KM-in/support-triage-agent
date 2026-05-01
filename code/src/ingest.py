"""
ingest.py — Load corpus documents and build a FAISS vector store.

Reads all .md, .txt, and .pdf files from the data/ directory,
splits them into chunks, embeds them, and persists a FAISS index.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT.parent / "data"
VECTORSTORE_DIR = PROJECT_ROOT / "vectorstore"

CHUNK_SIZE = 500       # tokens (approx characters / 4, but we use char count)
CHUNK_OVERLAP = 80
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_documents(data_dir: Path):
    """Load all supported documents from the data directory."""
    documents = []

    # Load markdown files
    md_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(md_loader.load())

    # Load text files
    txt_loader = DirectoryLoader(
        str(data_dir),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(txt_loader.load())

    return documents


def _split_documents(documents):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Add chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # Extract source filename for display
        source = chunk.metadata.get("source", "unknown")
        chunk.metadata["source_file"] = Path(source).name

    return chunks


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def build_vectorstore(rebuild: bool = False) -> FAISS:
    """
    Build (or load) the FAISS vector store from the corpus.

    Args:
        rebuild: If True, delete and rebuild even if store exists.

    Returns:
        A FAISS vector store instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Return existing store if available (and not rebuilding)
    if VECTORSTORE_DIR.exists() and not rebuild:
        print(f"[ingest] Loading existing vector store from {VECTORSTORE_DIR}")
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    # Validate data directory
    if not DATA_DIR.exists():
        print(f"[ingest] ERROR: Data directory not found at {DATA_DIR}")
        sys.exit(1)

    print(f"[ingest] Loading documents from {DATA_DIR} ...")
    documents = _load_documents(DATA_DIR)
    if not documents:
        print("[ingest] ERROR: No documents found in data directory.")
        sys.exit(1)

    print(f"[ingest] Loaded {len(documents)} document(s).")

    print("[ingest] Splitting into chunks ...")
    chunks = _split_documents(documents)
    print(f"[ingest] Created {len(chunks)} chunk(s).")

    print("[ingest] Embedding chunks and building FAISS index (using local HuggingFace embeddings) ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)
    vectorstore.save_local(str(VECTORSTORE_DIR))
    print(f"[ingest] Vector store saved to {VECTORSTORE_DIR}")

    return vectorstore


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the FAISS vector store from corpus data.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild even if vector store already exists.",
    )
    args = parser.parse_args()

    build_vectorstore(rebuild=args.rebuild)
    print("[ingest] Done.")
