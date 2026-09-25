"""
Local agentic RAG core + Gradio web UI.

Stack: Qwen 3 8B served by Ollama (localhost:11434) | LangGraph for the
agent's state machine | LlamaIndex for document ingestion/retrieval |
Gradio (web) or cli.py (terminal) for the streaming chat UI.

Nothing here imports `transformers` or loads a model into this Python
process - Ollama owns the model weights and inference, and this script
only ever talks to it over HTTP. That keeps the process's own RAM
footprint small enough to fit the ~12GB budget alongside the model.

Run:  python app.py   -> web UI at http://127.0.0.1:7860
      python cli.py   -> terminal UI
"""

import logging
import os
import shutil
from typing import Iterator, List, Literal, Optional, Tuple, TypedDict

import ollama
import psutil
from langgraph.graph import END, StateGraph
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.embeddings.ollama import OllamaEmbedding

# ---------------------------------------------------------------------------
# Config (every value can be overridden with an environment variable)
# ---------------------------------------------------------------------------
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
DATA_DIR = os.environ.get("DATA_DIR", "data")
PERSIST_DIR = os.environ.get("PERSIST_DIR", "storage")
TOP_K = 3
RAG_SCORE_THRESHOLD = float(os.environ.get("RAG_SCORE_THRESHOLD", "0.55"))
# Caps the KV cache Ollama allocates - the biggest RAM lever after model size.
NUM_CTX = int(os.environ.get("NUM_CTX", "8192"))
# Ollama picked 4 of the 4700U's 8 cores (it has no SMT); all 8 measured +25% prefill, +43% decode.
NUM_THREAD = int(os.environ.get("NUM_THREAD", psutil.cpu_count(logical=False) or os.cpu_count() or 4))
# Keep Qwen resident between requests; reloading 5GB from disk costs ~20-30s each time.
KEEP_ALIVE = os.environ.get("KEEP_ALIVE", "30m")
OLLAMA_OPTIONS = {"num_ctx": NUM_CTX, "num_thread": NUM_THREAD}
MAX_HISTORY_MESSAGES = 6
SUPPORTED_EXTS = [".pdf", ".md", ".txt"]

CLIENT = ollama.Client(host=OLLAMA_HOST)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Embeddings are also served by Ollama (not HuggingFace/transformers), so
# indexing never loads a second model copy into this process.
Settings.embed_model = OllamaEmbedding(model_name=EMBED_MODEL, base_url=OLLAMA_HOST)

RETRIEVER = None
CHUNK_COUNT = 0


# ---------------------------------------------------------------------------
# Ingestion / Indexing (LlamaIndex)
# ---------------------------------------------------------------------------
def load_index(rebuild: bool = False) -> int:
    """
    Build a VectorStoreIndex from DATA_DIR, or load the cached one from
    PERSIST_DIR. Returns the number of indexed chunks (0 = no documents,
    so every query is routed "direct").

    LlamaIndex's only job in this app is turning files into a searchable
    vector index and returning the top-k relevant chunks for a query. It
    never writes the final answer - that's our own streaming Ollama call,
    so the UI can show tokens as they're generated.
    """
    global RETRIEVER, CHUNK_COUNT

    if rebuild and os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)

    if os.path.isdir(PERSIST_DIR):
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        try:
            documents = SimpleDirectoryReader(
                DATA_DIR, recursive=True, required_exts=SUPPORTED_EXTS
            ).load_data()
        except ValueError:  # raised when the folder has no matching files
            RETRIEVER, CHUNK_COUNT = None, 0
            return 0
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)

    RETRIEVER = index.as_retriever(similarity_top_k=TOP_K)
    CHUNK_COUNT = len(index.docstore.docs)
    return CHUNK_COUNT


# ---------------------------------------------------------------------------
# LangGraph state machine
# ---------------------------------------------------------------------------
class AgentState(TypedDict, total=False):
    """
    The state LangGraph threads through every node in the graph.

    LlamaIndex is stateless per call - "given this query, what's
    relevant?" - it has no notion of where the app is in a multi-step
    decision. LangGraph supplies that: each node receives the current
    AgentState and returns only the keys it changed, which LangGraph
    merges back in. The edges (including the conditional one after
    "route") force State 1 -> State 2 -> State 3 in a fixed order.
    """

    query: str
    route: Literal["rag", "direct"]
    hits: List[NodeWithScore]
    top_score: Optional[float]
    context: str


def decide_route(query: str) -> Tuple[Literal["rag", "direct"], List[NodeWithScore], Optional[float]]:
    """
    Retrieval-score routing: embed the query, fetch the top-k chunks, and
    go "rag" only if the best cosine similarity clears RAG_SCORE_THRESHOLD.

    Costs one embedding call (milliseconds) instead of a full LLM
    classification pass, which matters when every generated token is
    CPU-bound. The hits are returned so the retrieve node reuses them
    instead of querying the index a second time.
    """
    if RETRIEVER is None:
        return "direct", [], None
    hits = RETRIEVER.retrieve(query)
    if not hits:
        return "direct", [], None
    top_score = hits[0].score or 0.0
    relevant = [h for h in hits if (h.score or 0.0) >= RAG_SCORE_THRESHOLD]
    return ("rag" if relevant else "direct"), relevant, top_score


def receive_query_node(state: AgentState) -> dict:
    """State 1: normalize the incoming query."""
    return {"query": state["query"].strip()}


def route_query_node(state: AgentState) -> dict:
    """State 2: pick the path and keep the retrieved hits for State 3a."""
    route, hits, top_score = decide_route(state["query"])
    return {"route": route, "hits": hits, "top_score": top_score}


def retrieve_node(state: AgentState) -> dict:
    """State 3a: turn the relevant hits into a source-labelled context block."""
    blocks = [
        f"[Source: {h.node.metadata.get('file_name', 'unknown')}]\n{h.node.get_content()}"
        for h in state["hits"]
    ]
    return {"context": "\n\n---\n\n".join(blocks)}


def direct_node(state: AgentState) -> dict:
    """State 3b: no document context."""
    return {"context": ""}


graph = StateGraph(AgentState)
graph.add_node("receive", receive_query_node)
graph.add_node("route", route_query_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("direct", direct_node)

graph.set_entry_point("receive")
graph.add_edge("receive", "route")
graph.add_conditional_edges("route", lambda s: s["route"], {"rag": "retrieve", "direct": "direct"})
graph.add_edge("retrieve", END)
graph.add_edge("direct", END)

APP_GRAPH = graph.compile()


def run_graph(query: str) -> AgentState:
    """Run States 1-3 and return the final state (route, hits, context)."""
    return APP_GRAPH.invoke({"query": query})


def sources(state: AgentState) -> List[Tuple[str, float]]:
    """Unique (file_name, best score) pairs for the hits used in a RAG answer."""
    best = {}
    for h in state.get("hits", []):
        name = h.node.metadata.get("file_name", "unknown")
        best[name] = max(best.get(name, 0.0), h.score or 0.0)
    return sorted(best.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# Streaming synthesis - deliberately outside the graph
# ---------------------------------------------------------------------------
def stream_answer(
    query: str, context: str, history: Optional[List[dict]] = None, think: bool = False
) -> Iterator[Tuple[str, object]]:
    """
    Stream Qwen 3's reply from Ollama as (kind, data) events:
      ("thinking", str)  reasoning tokens (only when think=True)
      ("answer", str)    answer tokens
      ("stats", dict)    token count and generation time, sent once at the end

    Kept outside the LangGraph graph on purpose: graph nodes return a
    finished state, but the UIs need a generator yielding token by token
    so nothing looks frozen during CPU inference.
    """
    if context:
        system_prompt = (
            "Answer the user's question using ONLY the context below. "
            "Mention which source file(s) you used. "
            "If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}"
        )
    else:
        system_prompt = "You are a helpful local assistant. Answer directly and concisely."

    messages = [{"role": "system", "content": system_prompt}]
    messages += (history or [])[-MAX_HISTORY_MESSAGES:]
    messages.append({"role": "user", "content": query})

    stream = CLIENT.chat(
        model=LLM_MODEL,
        messages=messages,
        stream=True,
        think=think,
        keep_alive=KEEP_ALIVE,
        options=OLLAMA_OPTIONS,
    )
    for chunk in stream:
        if chunk.message.thinking:
            yield "thinking", chunk.message.thinking
        if chunk.message.content:
            yield "answer", chunk.message.content
        if chunk.done:
            yield "stats", {
                "tokens": chunk.eval_count or 0,
                "seconds": (chunk.eval_duration or 0) / 1e9,
            }


# ---------------------------------------------------------------------------
# Gradio web UI
# ---------------------------------------------------------------------------
def _text(content) -> str:
    """Gradio history content can be a string or a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def launch_web() -> None:
    import gradio as gr  # imported here so cli.py doesn't pay gradio's RAM cost

    def respond(message: str, history: List[dict]):
        state = run_graph(message)
        past = [{"role": m["role"], "content": _text(m["content"])} for m in history]
        answer = ""
        for kind, data in stream_answer(state["query"], state["context"], past):
            if kind == "answer":
                answer += data
                yield answer
        if state["route"] == "rag":
            names = ", ".join(name for name, _ in sources(state))
            yield f"{answer}\n\n*Sources: {names}*"

    chunks = load_index()
    docs = f"{chunks} chunks from {DATA_DIR}/" if chunks else f"{DATA_DIR}/ is empty - answering without documents"
    gr.ChatInterface(
        fn=respond,
        title="Local Agentic RAG - Qwen 3 (Ollama)",
        description=f"Model: {LLM_MODEL} | {docs}",
    ).queue().launch(server_name="127.0.0.1")


if __name__ == "__main__":
    launch_web()
