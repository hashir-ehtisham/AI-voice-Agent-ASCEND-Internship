"""
rag.py — Standalone RAG (Retrieval-Augmented Generation) Pipeline
=================================================================
Loads a PDF, chunks the text, embeds chunks via Ollama's nomic-embed-text,
and answers questions by retrieving relevant context + querying Ollama's LLM.

No LangChain, no LlamaIndex, no external vector DB — just pypdf, numpy,
and plain HTTP requests to a locally-running Ollama server.

Usage (standalone):
    python rag.py
    → Loads data/knowledge.pdf, embeds all chunks, then starts an
      interactive Q&A loop in the terminal so you can test retrieval.

Usage (imported by agent.py):
    from rag import load_and_chunk, embed_texts, retrieve, build_rag_prompt
"""

import os
import sys
import time
import numpy as np
import requests
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# Configuration — tweak these if needed
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"        # Ollama embedding model
CHAT_MODEL = "llama3.2:3b"              # Ollama LLM for answering
CHUNK_SIZE = 500                        # Characters per chunk
CHUNK_OVERLAP = 50                      # Overlap between consecutive chunks
TOP_K = 3                               # Number of chunks to retrieve
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "knowledge.pdf")

# ---------------------------------------------------------------------------
# 1. Load PDF and chunk the text
# ---------------------------------------------------------------------------

def load_and_chunk(
    pdf_path: str = PDF_PATH,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Extract all text from a PDF and split it into overlapping character windows.

    Args:
        pdf_path:   Path to the PDF file.
        chunk_size: Number of characters per chunk.
        overlap:    Number of characters shared between consecutive chunks.

    Returns:
        A list of text chunks (strings).
    """
    # --- Extract raw text page by page ---
    reader = PdfReader(pdf_path)
    full_text = ""
    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    
    if not full_text.strip():
        print(f"[WARNING] No text extracted from {pdf_path}")
        return []

    print(f"[RAG] Extracted {len(full_text):,} characters from {len(reader.pages)} pages")

    # --- Chunk with sliding window ---
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end].strip()
        if chunk:  # skip empty chunks
            chunks.append(chunk)
        # Move forward by (chunk_size - overlap) so consecutive chunks share `overlap` chars
        start += chunk_size - overlap

    print(f"[RAG] Created {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ---------------------------------------------------------------------------
# 2. Embed texts using Ollama's /api/embed endpoint
# ---------------------------------------------------------------------------

def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed a list of text strings using Ollama's local embedding model.

    Sends texts in batches to avoid timeouts on large documents.

    Args:
        texts:      List of strings to embed.
        model:      Ollama embedding model name (e.g. "nomic-embed-text").
        ollama_url: Base URL of the running Ollama server.
        batch_size: Number of texts to embed per API call.

    Returns:
        numpy array of shape (len(texts), embedding_dim).
    """
    url = f"{ollama_url}/api/embed"
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": model, "input": batch}

        try:
            resp = requests.post(url, json=payload, timeout=120)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to Ollama at {ollama_url}")
            print("        Make sure Ollama is running: ollama serve")
            sys.exit(1)
        except requests.exceptions.HTTPError as e:
            print(f"[ERROR] Ollama embedding request failed: {e}")
            print(f"        Have you pulled the model?  ollama pull {model}")
            sys.exit(1)

        embeddings = resp.json().get("embeddings", [])
        all_embeddings.extend(embeddings)

        # Progress feedback for large documents
        done = min(i + batch_size, len(texts))
        print(f"[RAG] Embedded {done}/{len(texts)} chunks...", end="\r")

    print()  # newline after progress
    return np.array(all_embeddings, dtype=np.float32)


# ---------------------------------------------------------------------------
# 3. Retrieve top-k most relevant chunks via cosine similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between vector `a` (1D) and matrix `b` (2D).

    Returns a 1D array of similarity scores, one per row of `b`.
    """
    # Normalize vectors to unit length
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norms = np.linalg.norm(b, axis=1, keepdims=True) + 1e-10
    b_normed = b / b_norms
    # Dot product of unit vectors = cosine similarity
    return b_normed @ a_norm


def retrieve(
    query: str,
    chunks: list[str],
    chunk_vectors: np.ndarray,
    top_k: int = TOP_K,
    model: str = EMBED_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
) -> list[tuple[str, float]]:
    """
    Embed the query and find the top-k most similar chunks.

    Args:
        query:         The user's question (plain text).
        chunks:        The list of text chunks from the PDF.
        chunk_vectors: Pre-computed embeddings for all chunks (numpy array).
        top_k:         How many chunks to return.
        model:         Embedding model name.
        ollama_url:    Ollama server URL.

    Returns:
        List of (chunk_text, similarity_score) tuples, sorted best-first.
    """
    # Embed the query using the same model
    query_vector = embed_texts([query], model=model, ollama_url=ollama_url)[0]

    # Compute cosine similarity against all chunk vectors
    similarities = cosine_similarity(query_vector, chunk_vectors)

    # Get indices of top-k highest similarities
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((chunks[idx], float(similarities[idx])))

    return results


# ---------------------------------------------------------------------------
# 4. Build the RAG prompt
# ---------------------------------------------------------------------------

def build_rag_prompt(query: str, retrieved_chunks: list[tuple[str, float]]) -> str:
    """
    Construct a system-level prompt that includes the retrieved context
    and instructs the LLM to answer ONLY from that context.

    Args:
        query:            The user's question.
        retrieved_chunks: List of (chunk_text, score) from retrieve().

    Returns:
        A system prompt string to prepend to the LLM conversation.
    """
    context_parts = []
    for i, (chunk, score) in enumerate(retrieved_chunks, 1):
        context_parts.append(f"--- Context chunk {i} (relevance: {score:.3f}) ---\n{chunk}")

    context_block = "\n\n".join(context_parts)

    prompt = (
        "You are the Askari Bank HR Policy Assistant — a friendly, warm, and knowledgeable guide "
        "dedicated to helping Askari Bank employees with HR policies, leave rules, benefits, code of conduct, and related topics.\n\n"
        "Use the retrieved HR policy context below as your PRIMARY source of information. "
        "If the context clearly addresses the question, base your answer on it. "
        "If it only partially covers the question, enrich it with your broader knowledge of HR best practices to give the most helpful, complete answer. "
        "If the question is entirely unrelated to HR or Askari Bank, gently let the employee know your specialisation with a friendly tone.\n\n"
        "Always be warm, encouraging, and courteous — never give blunt refusals. "
        "Do NOT start your response with the word 'Assistant' or any role label. Speak naturally and directly.\n\n"
        f"=== RETRIEVED HR POLICY CONTEXT ===\n{context_block}\n"
        f"=== END CONTEXT ===\n\n"
        f"Employee's question: {query}"
    )
    return prompt


# ---------------------------------------------------------------------------
# 5. End-to-end ask function (retrieve + LLM call)
# ---------------------------------------------------------------------------

def ask(
    query: str,
    chunks: list[str],
    chunk_vectors: np.ndarray,
    top_k: int = TOP_K,
    model: str = CHAT_MODEL,
    ollama_url: str = OLLAMA_BASE_URL,
) -> str:
    """
    Full RAG pipeline: embed query → retrieve context → ask LLM → return answer.

    Args:
        query:         The user's question.
        chunks:        Pre-loaded text chunks from the PDF.
        chunk_vectors: Pre-computed chunk embeddings.
        top_k:         Number of context chunks to retrieve.
        model:         Ollama chat model name.
        ollama_url:    Ollama server URL.

    Returns:
        The LLM's answer string.
    """
    # Step 1: Retrieve relevant chunks
    retrieved = retrieve(query, chunks, chunk_vectors, top_k=top_k, ollama_url=ollama_url)

    # Step 2: Build the RAG prompt with context
    rag_prompt = build_rag_prompt(query, retrieved)

    # Step 3: Call Ollama's chat API
    url = f"{ollama_url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": rag_prompt},
            {"role": "user", "content": query},
        ],
        "stream": False,  # Get complete response at once
    }

    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return "[ERROR] Cannot connect to Ollama. Is it running?"
    except requests.exceptions.HTTPError as e:
        return f"[ERROR] Ollama chat request failed: {e}"

    answer = resp.json().get("message", {}).get("content", "")
    return answer.strip()


# ---------------------------------------------------------------------------
# 6. Interactive terminal Q&A (for standalone testing)
# ---------------------------------------------------------------------------

def main():
    """
    Interactive terminal loop: load PDF → embed chunks → ask questions.
    Run this to sanity-check that retrieval and answers work before
    wiring everything into the voice agent.
    """
    print("=" * 60)
    print("  RAG Pipeline — Standalone Test Mode")
    print("=" * 60)
    print(f"  PDF:       {PDF_PATH}")
    print(f"  Embed:     {EMBED_MODEL}")
    print(f"  LLM:       {CHAT_MODEL}")
    print(f"  Ollama:    {OLLAMA_BASE_URL}")
    print("=" * 60)
    print()

    # --- Load and chunk the PDF ---
    if not os.path.exists(PDF_PATH):
        print(f"[ERROR] PDF not found at: {PDF_PATH}")
        sys.exit(1)

    chunks = load_and_chunk(PDF_PATH)
    if not chunks:
        print("[ERROR] No chunks extracted. Check your PDF.")
        sys.exit(1)

    # --- Embed all chunks (one-time startup cost) ---
    print(f"\n[RAG] Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    t0 = time.time()
    chunk_vectors = embed_texts(chunks)
    elapsed = time.time() - t0
    print(f"[RAG] Embedding complete in {elapsed:.1f}s  |  Vector shape: {chunk_vectors.shape}")

    # --- Interactive Q&A loop ---
    print("\n" + "-" * 60)
    print("  Ready! Type your questions below (Ctrl+C to exit).")
    print("-" * 60 + "\n")

    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue

            # Show retrieved chunks for debugging
            retrieved = retrieve(query, chunks, chunk_vectors)
            print(f"\n  [Retrieved {len(retrieved)} chunks]")
            for i, (chunk, score) in enumerate(retrieved, 1):
                preview = chunk[:80].replace("\n", " ")
                print(f"    {i}. (score={score:.3f}) {preview}...")

            # Get the LLM answer
            print("\n  Thinking...", end="", flush=True)
            answer = ask(query, chunks, chunk_vectors)
            print(f"\r  {'':20}")  # clear "Thinking..."
            # Strip any leading "Assistant:" prefix the LLM may have added itself
            clean_answer = answer.lstrip()
            if clean_answer.lower().startswith("assistant:"):
                clean_answer = clean_answer[len("assistant:"):].lstrip()
            print(f"\nAssistant: {clean_answer}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
