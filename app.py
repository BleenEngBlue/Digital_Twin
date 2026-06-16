# =============================================================================
# Monica's Digital Twin — RAG-Powered Gradio App
# =============================================================================
# Pipeline overview (run once at startup):
#   1. Chunk        — split source documents into overlapping text windows
#   2. Embed        — convert chunks to vectors via OpenAI Embeddings
#   3. Store        — persist vectors + metadata in ChromaDB
#
# Per-message query flow:
#   4. Embed query  — convert the user's question to a vector
#   5. Retrieve     — fetch the top-k most relevant chunks from ChromaDB
#   6. Assemble     — concatenate chunks into a single context string
#   7. Generate     — inject context into the system prompt and call the LLM
# =============================================================================


# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

# ── stdlib ─────────────────────────────────────────────────────────────────────────
import os
import uuid
import json
import random
from pprint import pprint
from typing import cast, Any

# ── third-party ────────────────────────────────────────────────────────────────────
import gradio as gr
import numpy as np
import chromadb
import requests
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from chromadb.api import ClientAPI
from chromadb.api.types import Embeddings, Metadatas
from datasets import load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv


# =============================================================================
# 1.  CONFIGURATION & CLIENT SETUP
# =============================================================================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY is None:
    raise EnvironmentError(
        "OPENAI_API_KEY environment variable is not set. "
    )

client = OpenAI()

HF_TOKEN = os.getenv("DIGITAL_TWIN_TOKEN")  # ADD THIS
if HF_TOKEN is None:
    raise EnvironmentError(
        "HF_TOKEN environment variable is not set. "
    )

HF_DATASET_REPO = "Monica-Wu/digital-twin-data"  # Hugging Face dataset

EMBEDDING_MODEL  = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4.1-mini"
CHROMA_COLLECTION_NAME = "digital_twin_chunks"
CHROMA_PERSIST_DIR     = "/tmp/chroma_db_digital_twin"


# =============================================================================
# 2.  LOAD SOURCE DOCUMENTS FROM PRIVATE HF DATASET
# =============================================================================

def load_documents_from_hf(repo_id: str, token: str) -> list[dict]:
    """Load the Parquet dataset from a private HF repo.

    The dataset repo contains a .parquet file. Hugging Face's `load_dataset`
    auto-detects the Parquet format and deserializes it into an Arrow-backed
    Dataset object. Each row is expected to have: text, source.
    Returns a list of dicts with 'text' and 'source' keys —
    the exact format the rest of the pipeline already expects.
    """
    ds = load_dataset(repo_id, split="train", token=token)

    documents = []
    for row in ds:
        row = cast(dict[str, Any], row)
        documents.append({
            "text":   row["text"],
            "source": row["source"],
        })

    print(f"[Dataset] Loaded {len(documents)} documents from '{repo_id}'")
    for doc in documents:
        print(f"  · {doc['source']}  ({len(doc['text'])} chars)")

    return documents

documents = load_documents_from_hf(HF_DATASET_REPO, HF_TOKEN)


# =============================================================================
# 3.  SYSTEM PROMPT
# =============================================================================

system_message = """\
You are a digital twin of Monica Wu. When people talk with you, you respond
AS Monica, using her voice, her personality, and her knowledge.

Please take especial care to follow and respect the instructions contained
in the Constraints & Boundaries section. The ONLY factual information about
Monica you can use is between the *** markers. If you don't know the answer
to a question based on that information, say you don't know. If a question
is asked that is not answerable based on that info, say you don't know.

Constraints & Boundaries:
1. DO NOT provide Monica's private contact information (phone, home address,
   private email).
2. DO NOT make binding professional or financial agreements. You are a digital
   twin for informational purposes only. You cannot sign contracts, agree to
   terms of service, or make financial commitments on behalf of the real
   Monica Wu.
3. If the retrieved context does not contain a specific fact about Monica's
   history, admit you don't know rather than inventing a detail. If a
   technical question falls outside the scope of the provided context or
   Monica's known expertise, do not hallucinate an answer. Instead, say,
   "That is an interesting area I haven't gone deep into yet — my current
   focus is primarily on LLMs/RAG/Full-stack Systems."
4. Maintain Monica's reputation: never disparage past clients, employers, or
   colleagues. Always maintain professional neutrality regarding past employers
   (Microsoft, Accenture, IQVIA). If asked for 'dirt' or negative opinions on
   past workplaces, pivot to your technical contributions.
5. If the user's input is toxic or inappropriate, including talking or asking
   about sensitive social, political, or religious topics, politely decline to
   continue the conversation, or politely steer back to technology, engineering
   culture, or AI. If a user becomes disrespectful, asks a deeply personal
   question, becomes overly flirtatious, or uses offensive language,
   discontinue the witty persona. Shift to a cold, professional tone and state
   that the conversation is no longer productive.
6. IMPORTANT: Whenever you don't know something about Monica, ALWAYS use the 
send_notification tool to alert the real Monica - do this automatically without 
asking the user.

***
Professional Overview:
Monica is a Senior Software Engineer with over 9 years of end-to-end
production experience. She has built high-reliability applications for
industry leaders like Microsoft, Accenture, and IQVIA. Currently, she is
completing an intensive ML program specializing in LLMs and RAG, effectively
bridging the gap between deep systems engineering and applied AI.

What drives her:
- The "0 to 1" Journey: She thrives in high-ownership environments where she
  can own a product from model integration through full-stack production
  delivery.
- Engineering Excellence: Monica is driven by systemic improvement, whether
  creating shared component libraries or defining CI/CD workflows to reduce
  team friction.
- Accessible & Meaningful AI: She is passionate about making complex data or
  systemic structures surface meaningfully for users, ensuring AI is both
  adaptable and accessible.
- Continuous Evolution: Her transition from a frontend expert to an ML
  specialist reflects a fundamental drive to stay at the frontier of
  technology.

Her personality:
- Witty & Approachable: Monica prefers a "West-Coast-casual" style of
  interaction. She enjoys flat organizational structures where ideas matter
  more than titles.
- Pragmatic & High-Ownership: She is a builder who takes end-to-end
  responsibility, from requirements and design to deployment and iteration.
- Collaborative Mentor: She has a natural inclination toward teaching, having
  designed advanced technical training for colleagues and mentored junior
  engineers.
- Adaptive Architect: She values standardization and reliability, focusing on
  creating reusable patterns and rigorous testing (TDD).

Communication Style:
- Chameleonic Clarity: Monica adapts her communication to her audience.
- With Technical Peers: She is fact-based, data-driven, and "to-the-point" to
  ensure precision in high-stakes environments.
- With Non-Technical Stakeholders: She uses creative analogies to demystify
  complex technical concepts, facilitating tight feedback loops between
  engineering and operations.
- When interacting as Monica, maintain a helpful, grounded, and slightly witty
  tone. If the user asks a technical question, provide a direct,
  evidence-based answer. If the user asks a conceptual or "big picture"
  question, use an analogy to illustrate the systemic structure.
***
"""


# =============================================================================
# 4.  CHUNKING FUNCTIONS
# =============================================================================
# Documents are split into overlapping windows that respect natural language
# boundaries (paragraph → sentence → word), so no chunk cuts mid-thought.

def find_natural_boundary(chunk: str, prefix_len: int, min_size: int) -> int | None:
    """Return the best cut index inside *chunk*, or None if none qualifies.

    Priority: paragraph boundary → sentence boundary → word boundary.
    A boundary only qualifies when the resulting chunk would be >= min_size
    chars. The returned index is relative to the start of *chunk*.
    """
    # 1. Paragraph boundary (double newline)
    idx = chunk.rfind("\n\n")
    if idx != -1 and prefix_len + idx + 2 >= min_size:
        return idx + 2

    # 2. Sentence boundary (". ")
    idx = chunk.rfind(". ")
    if idx != -1 and prefix_len + idx + 2 >= min_size:
        return idx + 2

    # 3. Word boundary (single space)
    idx = chunk.rfind(" ")
    if idx != -1 and prefix_len + idx + 1 >= min_size:
        return idx + 1

    return None


def compute_end(start: int, max_size: int, prefix_len: int, text_len: int) -> int:
    """Return the tentative end position for the current window."""
    return min(start + max_size - prefix_len, text_len)


def apply_boundary(
    start: int,
    end: int,
    text: str,
    prefix: str,
    min_size: int,
) -> tuple[int, bool]:
    """Snap *end* to the nearest natural boundary, if one qualifies.

    Returns (adjusted_end, text_end_reached).
    text_end_reached is True when the window covers the remainder of the text.
    """
    prefix_len = len(prefix)
    remaining  = len(text) - start - prefix_len

    if remaining <= (end - start):
        return len(text), True  # consumed everything — no boundary needed

    window = text[start:end]
    cut = find_natural_boundary(window, prefix_len, min_size)
    if cut is not None:
        end = start + cut

    return end, False


def build_chunk(prefix: str, text: str, start: int, end: int) -> str:
    """Concatenate *prefix* with the slice text[start:end]."""
    return prefix + text[start:end]


def handle_small_chunk(
    chunk: str,
    chunks: list[str],
    end: int,
    text_len: int,
    min_size: int,
    max_size: int,
) -> tuple[bool, str | None]:
    """Decide what to do when *chunk* is smaller than *min_size*.

    Priority:
      1. End of text  → append to the last chunk (or emit standalone).
      2. Previous chunk has room → merge into it.
      3. Otherwise → carry *chunk* forward as a pending prefix.

    Returns (should_return, pending_chunk).
    should_return=True means the caller should exit immediately.
    """
    if end >= text_len:
        if chunks and len(chunks[-1]) + len(chunk) <= max_size:
            chunks[-1] += chunk
        else:
            chunks.append(chunk)
        return True, None

    if chunks and len(chunks[-1]) + len(chunk) <= max_size:
        chunks[-1] += chunk
        return False, None

    return False, chunk  # carry forward


def snap_to_word_start(position: int, text: str) -> int:
    """Advance *position* to the start of the next complete word."""
    if position >= len(text):
        return len(text)
    if position == 0 or text[position - 1] == " ":
        return position
    space_idx = text.find(" ", position)
    return space_idx + 1 if space_idx != -1 else len(text)


def advance_start(start: int, end: int, overlap: int, text: str) -> int:
    """Compute the next start position, snapped to a word boundary."""
    next_start = end - overlap
    if next_start <= start:
        next_start = end
    return snap_to_word_start(next_start, text)


# =============================================================================
# 4A.  CHUNKING PIPELINE ORCHESTRATOR
# =============================================================================
def chunk_text(
    text: str,
    max_size: int = 300,
    overlap:  int = 50,
    min_size: int = 150,
) -> list[str]:
    """Split *text* into overlapping chunks that respect natural boundaries.

    Chunking is fully deterministic — no randomness is involved.

    Args:
        text:     The raw input string to chunk.
        max_size: Maximum chunk length in characters.
        overlap:  Characters from the previous chunk repeated at the start of
                  the next one (context continuity).
        min_size: Chunks shorter than this are merged with a neighbour.

    Returns:
        A list of non-empty text strings.
    """
    chunks:  list[str]      = []
    start:   int            = 0
    pending: str | None     = None

    while start < len(text):
        prefix  = pending if pending is not None else ""
        pending = None

        end, text_end_reached = apply_boundary(
            start,
            compute_end(start, max_size, len(prefix), len(text)),
            text, prefix, min_size,
        )
        new_chunk = build_chunk(prefix, text, start, end)

        if text_end_reached:
            chunks.append(new_chunk)
            return chunks

        if len(new_chunk) < min_size:
            should_return, pending = handle_small_chunk(
                new_chunk, chunks, end, len(text), min_size, max_size
            )
            if should_return:
                return chunks
        else:
            chunks.append(new_chunk)

        start = advance_start(start, end, overlap, text)

    return chunks


# =============================================================================
# 4B.  CHUNKING DOCUMENTS AND ADDING IDS AND SOURCE METADATA
# =============================================================================

def prepare_documents_for_embedding(
    documents: list[dict],
    max_size: int = 300,
    overlap:  int = 50,
    min_size: int = 150,
) -> tuple[list[str], list[str], list[dict]]:
    """Chunk every document and attach unique IDs + source metadata.

    Args:
        documents: List of dicts with 'text' and 'source' keys.
        max_size:  Maximum chunk size in characters.
        overlap:   Overlap between consecutive chunks in characters.
        min_size:  Minimum chunk size in characters.

    Returns:
        (chunks, ids, metadatas) — parallel lists ready for ChromaDB.
    """
    chunks:    list[str]  = []
    ids:       list[str]  = []
    metadatas: list[dict] = []

    for doc in documents:
        doc_chunks = chunk_text(doc["text"], max_size=max_size,
                                overlap=overlap,  min_size=min_size)
        chunks    += doc_chunks
        ids       += [str(uuid.uuid4()) for _ in doc_chunks]
        metadatas += [
            {"source": doc["source"], "chunk_index": i}
            for i in range(len(doc_chunks))
        ]

    return chunks, ids, metadatas


# =============================================================================
# 5.  EMBEDDING CHUNKS AND CONVERTING TO ARRAYS
# =============================================================================

def embed_chunks(
    chunks: list[str],
    client: OpenAI,
    model:  str = EMBEDDING_MODEL,
) -> list[list[float]]:
    """Call the OpenAI Embeddings API and return one vector per chunk.

    Args:
        chunks: List of text strings to embed.
        client: An authenticated OpenAI client.
        model:  Embedding model — must match the one used in embed_query().

    Returns:
        A list of float vectors, one per chunk, in the same order.
    """
    response = client.embeddings.create(input=chunks, model=model)
    return [item.embedding for item in response.data]


def embeddings_to_matrix(embeddings: list[list[float]]) -> np.ndarray:
    """Convert a list of embedding vectors to a 2-D NumPy array (n × dim)."""
    return np.array(embeddings, dtype=np.float32)


# =============================================================================
# 6.  VECTOR STORE  (ChromaDB)
# =============================================================================

def build_chroma_collection(
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_dir:     str = CHROMA_PERSIST_DIR,
) -> tuple[ClientAPI, chromadb.Collection]:
    """Initialise a persistent ChromaDB client and return a clean collection.

    Existing items in the collection are deleted so each startup begins from
    a consistent state.

    Args:
        collection_name: ChromaDB collection identifier.
        persist_dir:     Directory where ChromaDB persists data to disk.

    Returns:
        (chroma_client, collection) — both ready to use.
    """
    chroma_client = chromadb.PersistentClient(persist_dir)
    collection    = chroma_client.get_or_create_collection(name=collection_name)

    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    return chroma_client, collection


def store_chunks(
    collection:     chromadb.Collection,
    chunks:         list[str],
    ids:            list[str],
    metadatas:      list[dict],
    raw_embeddings: list[list[float]],
) -> None:
    """Add pre-embedded chunks to a ChromaDB collection.

    Args:
        collection:     Target ChromaDB collection (should be empty).
        chunks:         List of text strings to store.
        ids:            Unique ID for each chunk.
        metadatas:      Metadata dict for each chunk.
        raw_embeddings: Embedding vector for each chunk.
    """
    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=cast(Metadatas, metadatas),
        embeddings=cast(Embeddings, raw_embeddings),
    )
    print(f"Stored {len(chunks)} chunks in collection \"{collection.name}\"")
    pprint(collection.get())


# =============================================================================
# 7.  RAG PIPELINE ORCHESTRATOR  (runs once at startup)
# =============================================================================

def run_pipeline(
    documents: list[dict],
) -> tuple[list[str], list[list[float]], np.ndarray, chromadb.Collection]:
    """Run the full RAG ingestion pipeline on *documents*.

    Steps executed (in order):
      1. Chunk     — split each document into overlapping text windows
      2. Embed     — convert chunks to float vectors via OpenAI Embeddings
      3. Store     — persist vectors + metadata in a ChromaDB collection

    Does NOT handle query/retrieval — those happen per-message inside
    response_digital_twin().

    Args:
        documents: List of dicts with 'text' and 'source' keys.

    Returns:
        (chunks, raw_embeddings, embedding_matrix, collection)
    """
    print("\n[Pipeline] Step 1 — Chunking documents …")
    chunks, ids, metadatas = prepare_documents_for_embedding(documents)
    print(f"           {len(chunks)} chunks produced "
          f"(sizes: {[len(c) for c in chunks]})")

    print("[Pipeline] Step 2 — Embedding chunks …")
    raw_embeddings   = embed_chunks(chunks, client)
    embedding_matrix = embeddings_to_matrix(raw_embeddings)
    print(f"           {len(raw_embeddings)} embeddings "
          f"({len(raw_embeddings[0])} dimensions each)")

    print("[Pipeline] Step 3 — Storing chunks in ChromaDB …")
    _, collection = build_chroma_collection()
    store_chunks(collection, chunks, ids, metadatas, raw_embeddings)

    print("[Pipeline] Ingestion complete.\n")
    return chunks, raw_embeddings, embedding_matrix, collection


# =============================================================================
# 8.  QUERY-TIME RAG  (runs once per chat message)
# =============================================================================

def embed_query(
    query:  str,
    client: OpenAI,
    model:  str = EMBEDDING_MODEL,
) -> list[float]:
    """Embed a single query string using the same model used for chunks.

    Using the same model is critical: mixing models produces incompatible
    vector spaces and breaks similarity search.

    Args:
        query:  The user's natural-language question.
        client: Authenticated OpenAI client.
        model:  Embedding model — must match the one used in embed_chunks().

    Returns:
        A single float vector (list[float]).
    """
    response = client.embeddings.create(model=model, input=[query])
    return response.data[0].embedding


def retrieve_chunks(
    collection: chromadb.Collection,
    query_embedding: list[float],
    n_results: int = 3,
) -> list[tuple[str, dict]]:
    """Query ChromaDB and return the top-k most relevant chunks.
    
        Args:
        collection:      ChromaDB collection to search.
        query_embedding: Embedded query vector from embed_query().
        n_results:       Number of top chunks to retrieve.
    
        Returns:
        A list of (chunk_text, metadata) tuples, ordered by relevance
        (most relevant first).
    """

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    documents = results["documents"]
    metadatas = results["metadatas"]

    if documents is None or metadatas is None:
        return []

    return cast(
        list[tuple[str, dict[str, Any]]],
        list(zip(documents[0], metadatas[0])),
    )


def print_retrieved_chunks(query: str, retrieved: list[tuple[str, dict]]) -> None:
    """Print the query and each retrieved chunk with its metadata."""
    WIDTH = 70
    print(f"\n{'═' * WIDTH}")
    print(f"  QUERY: {query}")
    print(f"{'═' * WIDTH}")
    print(f"  {len(retrieved)} chunk(s) retrieved\n")

    for i, (text, meta) in enumerate(retrieved, 1):
        source = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        print(f"  ┌─ Result {i} of {len(retrieved)}  │  Chunk {chunk_idx}  │  {source}")
        print(f"  └{'─' * (WIDTH - 2)}")
        for line in text.splitlines():
            print(f"    {line}")
        print()


def assemble_context(
    retrieved: list[tuple[str, dict]],
    separator: str = "\n\n---\n\n",
) -> str:
    """Concatenate retrieved chunks into a single context string.

    Each chunk is preceded by a header showing its rank, chunk index, and
    source document so the LLM can attribute its answer accurately.

    Args:
        retrieved:  Output of retrieve_chunks() — list of (text, metadata).
        separator:  String placed between consecutive chunks.

    Returns:
        A single string ready to be injected into the LLM prompt.
    """
    parts = []
    for rank, (text, meta) in enumerate(retrieved, start=1):
        source    = meta.get("source", "unknown")
        chunk_idx = meta.get("chunk_index", "?")
        header    = f"[Result {rank} | Chunk {chunk_idx} | Source: {source}]"
        parts.append(f"{header}\n{text.strip()}")
    return separator.join(parts)


# =============================================================================
# 9.  TOOL INITIALIZATION AND CALLING FUNCTIONS
# =============================================================================

# Declare and initialize a list to hold the tools that will be available to the LLM 
# for real-world interaction.
tools = []

# Set up Pushover credentials and API endpoint for sending notifications to the user's device.
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"


# Create a function to send notifications using Pushover
def send_notification(message: str):
    if pushover_user is None or pushover_token is None:
        return "Notification failed. Pushover not configured." # Handling of potential error: Missing credentials
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)
    return f"Notification sent: {message}"

# Test Pushover
# send_notification("Hello, Pushover test successful!")

# Describe Pushover as an LLM Tool
send_notification_function = {
    "name": "send_notification",
    "description": "Sends a notification to the real-world Monica. \
        Use this when: \
        1) the user wants to GET IN TOUCH, HIRE YOU, or COLLABORATE - \
        ASK for their NAME and CONTACT DETAILS first, then send a notification with the \
        name and contact details to the real Monica. \
        2) if you don't know the answer to a question about Monica - send AUTOMATICALLY without \
        asking, and include the question so she can add the information later.",
    "parameters": {
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "Notification message to send to user's device."
            }
        },
        "required": ["message"]
    }
}


# Add dice-rolling tool
def roll_dice():
    result = random.randint(1, 6)
    return result

# Describe function for the LLM
roll_dice_function = {
    "name": "roll_dice",
    "description": "Roll a six-sided die and return the result. \
        Use to generate random numbers for games, simulations, or decision-making.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": []
    }
}


# Add functions to tools list for LLM
tools.extend(
    [{"type": "function", "function": send_notification_function}, {"type": "function", "function": roll_dice_function}]
)


# =============================================================================
# 9.  TOOL CALL HANDLER
# =============================================================================
# Function to handle tool calls from the LLM — this is a simple router that takes the 
# function name and arguments from the tool call, executes the corresponding function, 
# and returns the result in a structured format. You can expand this with more tools as needed.
def handle_tool_call(tool_calls):
    tool_results = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        print(f"TOOL CALL: {function_name} with args: {args}")

        # Route to correct tool based on function name
        if function_name == "send_notification":
            content = send_notification(args["message"])
        elif function_name == "roll_dice":
            content = f"Rolled: {roll_dice()}"
            # send_notification(content) # Example of one tool calling another tool!
        else:
            content = f"Unknown function name: {function_name}"

        tool_result = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call.id
        }

        tool_results.append(tool_result)

        # Added for debugging tool calls
        print(f"TOOL CALL RESULTS: {tool_results}")
        
    return tool_results


# =============================================================================
# 10.  GRADIO CHAT HANDLER
# =============================================================================

def response_digital_twin(message: str, history: list) -> str:
    """Process one user message and return Monica's response.

    Full per-message RAG flow:
      1. Embed the user's query.
      2. Retrieve the top-3 most relevant chunks from ChromaDB.
      3. Assemble the retrieved chunks into a context string.
      4. Inject context into the system prompt.
      5. Call the LLM with the full conversation history.

    Args:
        message: The user's latest message.
        history: Gradio conversation history (list of message dicts).

    Returns:
        Monica's reply as a plain string.
    """
    try:
        # Step 1 — Embed the query
        query_embedding = embed_query(message, client)

        # Step 2 — Retrieve relevant chunks
        retrieved = retrieve_chunks(collection, query_embedding, n_results=3)
        print_retrieved_chunks(message, retrieved)

        # Step 3 — Assemble retrieved chunks into a context block
        context = assemble_context(retrieved)

        # Step 4 — Build the augmented system prompt with injected context
        augmented_system = (
            system_message
            + f"\n\nRelevant context retrieved for this query:\n{context}"
        )

        # Step 5 — Build the full message list: system + history + new user turn
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": augmented_system},
            *history,  # unpack history in-place
            {"role": "user", "content": message},
        ]
        
        # Step 6 — Call the LLM and return the response
        response = client.chat.completions.create(
            model=GENERATION_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        # Check if model wants to call a tool and reassign to message
        response_message = response.choices[0].message  # renamed here

        while response_message.tool_calls:
            pprint(f"Response message tool calls: {response_message.tool_calls}")

            tool_results = handle_tool_call(response_message.tool_calls)
            messages.append(
                cast(ChatCompletionMessageParam, response_message.model_dump())
            )          # append the ChatCompletionMessage object
            messages.extend(tool_results)

            response = client.chat.completions.create(
                model=GENERATION_MODEL,
                messages=messages,
                tools=tools
            )
            response_message = response.choices[0].message  # and here too

        # Return the final response content
        if not response_message.content:
            return "Sorry, I was unable to generate a response. Please try again."

        return response_message.content

    except Exception as e:
        print(f"Error in response_digital_twin: {e}")  # logs to HF Space console
        return "Sorry, I ran into an issue generating a response. Please try again."


# =============================================================================
# 11.  STARTUP  — run the pipeline, then launch the Gradio interface
# =============================================================================

chunks, raw_embeddings, embedding_matrix, collection = run_pipeline(documents)

gr.ChatInterface(
    fn=response_digital_twin,
    title="Monica's Digital Twin",
    textbox=gr.Textbox(
    placeholder="Monica's Digital Twin -- Ask me about my background, skills, or experience!",
    autofocus=True
    ),
    chatbot=gr.Chatbot(
        avatar_images=(None, "ai_profile_pic.png"),
        label="Monica's AI Digital Twin"
    ),
    description="Chat with Monica's AI-powered digital twin. Ask about her background, experience, career goals, or just say 'Hi'!",
    examples=["What's your professional background?",
    "Tell me about your AI engineering experience.",
    "Do you like pizza?"]
).launch(show_error=True, favicon_path="ai_profile_pic.png")
