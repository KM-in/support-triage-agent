"""
retriever.py — Fetch relevant chunks from the FAISS vector store.

Provides a simple interface to retrieve the top-k most relevant
document chunks for a given query.
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.ingest import VECTORSTORE_DIR, EMBEDDING_MODEL, build_vectorstore


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class CorpusRetriever:
    """Wrapper around the FAISS vector store for retrieval."""

    def __init__(self, top_k: int = 5):
        """
        Initialise the retriever.

        Args:
            top_k: Default number of chunks to retrieve per query.
        """
        self.top_k = top_k
        self._vectorstore = self._load_or_build()

    def _load_or_build(self) -> FAISS:
        """Load existing vector store or build one if missing."""
        if VECTORSTORE_DIR.exists():
            embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
            return FAISS.load_local(
                str(VECTORSTORE_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        else:
            print("[retriever] Vector store not found — building now ...")
            return build_vectorstore(rebuild=False)

    def retrieve(self, query: str, k: int | None = None) -> list:
        """
        Retrieve the top-k most relevant document chunks for *query*.

        Args:
            query: The user's support question or ticket text.
            k: Number of chunks to return (overrides default).

        Returns:
            List of LangChain Document objects, each with .page_content
            and .metadata (source_file, chunk_id).
        """
        k = k or self.top_k
        results = self._vectorstore.similarity_search(query, k=k)
        return results

    def retrieve_with_scores(self, query: str, k: int | None = None) -> list:
        """
        Like retrieve(), but also returns similarity scores.

        Returns:
            List of (Document, score) tuples — lower score = more similar.
        """
        k = k or self.top_k
        results = self._vectorstore.similarity_search_with_score(query, k=k)
        return results

    def format_context(self, documents: list) -> str:
        """
        Format retrieved documents into a single context string for the LLM.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            A formatted string with source references.
        """
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source_file", "unknown")
            context_parts.append(
                f"[Source {i}: {source}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(context_parts)
