"""
Local "System 1 + System 2" software factory, sized for a CPU-only laptop.

    python factory_cli.py

Layers
  1. Triage     Laya (421M ModernBERT, non-autoregressive) classifies each
                request: task kind (new code / bug fix / question) and
                whether it's multi-part (which decides whether Qwen 3's slow
                reasoning mode is worth turning on). ~1-2s and ~2GB RAM on CPU.
  2. Orchestr.  LangGraph state machine below. Cycles Generate -> Check ->
                Generate until the guardrails pass or MAX_ATTEMPTS is hit,
                then pauses with interrupt() for human review.
  3. Retrieval  LlamaIndex over WORKSPACE, Python split by CodeSplitter
                (function/class boundaries) instead of by sentence.
  4. Generation Qwen 3 8B via Ollama; tokens are pushed to the UI from
                inside the graph with LangGraph's custom stream writer.
  5. MCP server mcp_server.py, a separate process reached over stdio
                JSON-RPC (mcp_client.py). It owns every write and every
                execution: sandbox copies, ast -> placeholders -> ruff ->
                pytest, diffs, and committing approved files. This process
                only reads the workspace (for the LlamaIndex index).

The server process isolates hangs and crashes from the graph, not
security: tests still run as your user and can reach the network or any
file your account can. Review what you approve.
"""

import atexit
import os
import re
import shutil
import time
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import CodeSplitter, SentenceSplitter
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters

from app import CLIENT, KEEP_ALIVE, LLM_MODEL, NUM_CTX, OLLAMA_OPTIONS
from mcp_client import MCPConnectionLost, MCPToolError, WorkspaceMCP  # noqa: F401  (errors re-exported for the UIs)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
WORKSPACE = Path(os.environ.get("WORKSPACE", "workspace")).resolve()
LAYA_MODEL = os.environ.get("LAYA_MODEL", "convaiinnovations/laya")
LAYA_REVISION = os.environ.get("LAYA_REVISION", "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b")
LAYA_DIR = Path(__file__).resolve().with_name("models") / f"laya-{LAYA_REVISION[:7]}"
LOG_DIR = Path(__file__).resolve().with_name("logs")
MAX_ATTEMPTS = 3
CONTEXT_BUDGET_CHARS = 6000  # ~1.5k tokens: prompt prefill on CPU is slow too
CONTEXT_SCORE_MIN = 0.5  # min similarity for context on new-code tasks
MAX_NEW_TOKENS = 2048
THINK_BUDGET = 1200  # reasoning tokens before forcing an answer (~5 min at ~4 tok/s)
INDEX_EXTS = [".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml", ".cfg"]
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache"}
HEADER_RE = re.compile(r"^[#*\s]*(FILE|EDIT):\s*`?([^`*]+?)`?\s*\*{0,2}\s*$")
# Qwen often drops the words and writes bare git-conflict markers, so they're optional.
SEARCH_RE = re.compile(r"^<{5,9}( ?SEARCH)?\s*$")
DIVIDER_RE = re.compile(r"^={5,9}\s*$")
REPLACE_RE = re.compile(r"^>{5,9}( ?REPLACE)?\s*$")
FALLBACK_AFTER_FAILURES = 2  # failed EDITs on one file before asking for the whole file

WORKSPACE.mkdir(exist_ok=True)

_mcp: Optional[WorkspaceMCP] = None


def mcp() -> WorkspaceMCP:
    """Start the workspace MCP server on first use, and again if the old one died."""
    global _mcp
    if _mcp is not None and _mcp.dead:
        _mcp.close()
        _mcp = None
    if _mcp is None:
        _mcp = WorkspaceMCP(WORKSPACE)
        atexit.register(_mcp.close)
    return _mcp


def mcp_call(tool: str, **arguments):
    """
    Call an MCP tool, restarting a dead server and retrying once. commit is
    never retried: its sandbox lived in the old server process and is gone,
    so the task has to be re-validated rather than silently re-committed.
    """
    try:
        return mcp().call(tool, **arguments)
    except MCPConnectionLost:
        if tool == "commit":
            raise
        return mcp().call(tool, **arguments)


# ---------------------------------------------------------------------------
# Layer 1: Laya triage
# ---------------------------------------------------------------------------
TRIAGE_QUESTIONS = {
    "kind": {
        "type": "choice",
        "instructions": "What does the developer want done?",
        "criteria": {
            "new_code": "write new code: a new feature, script, function, class or file",
            "fix_bug": "fix a bug, an error or a failing test, or change existing code",
            "question": "explain something or answer a question; no code needs to be written",
        },
    },
    # A yes/no question separated simple from multi-part tasks far better than a
    # 3-level score did in testing (0.06-0.19 vs 0.89-0.91; the score put everything at 1.6-1.8).
    "multi_part": {
        "type": "noul",
        "instructions": "Does this task require building several separate components, classes or files that work together?",
    },
}

_laya = None


def laya_path() -> str:
    """
    Local folder of the pinned Laya revision. Pinned because the upstream repo
    changed 3 times in 2 days (one revision shipped without its config file and
    crashed load), and each change could shift the triage thresholds tuned below.
    Loading from a plain folder also skips a network check on every start, and
    avoids the HF cache, whose symlink step fails on Windows without admin/dev mode.
    """
    if (LAYA_DIR / "rl_agent_config.json").exists() and (LAYA_DIR / "model.safetensors").exists():
        return str(LAYA_DIR)
    cached = Path.home() / ".cache/huggingface/hub/models--convaiinnovations--laya/snapshots" / LAYA_REVISION
    if (cached / "rl_agent_config.json").exists() and (cached / "model.safetensors").exists():
        shutil.copytree(cached, LAYA_DIR, dirs_exist_ok=True)  # reuse the earlier download
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(LAYA_MODEL, revision=LAYA_REVISION, local_dir=LAYA_DIR,
                          allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"])
    return str(LAYA_DIR)


def load_laya():
    """Load only the English checkpoint (~2GB RAM) - Router(preload=True) would load all three."""
    global _laya
    if _laya is None:
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        import laya  # heavy (torch); imported on first use

        with warnings.catch_warnings():
            # Calibration warning only concerns choice questions with 11+ options; ours has 3.
            warnings.filterwarnings("ignore", message="laya: this checkpoint ships invalid temperatures")
            _laya = laya.load(laya_path(), device="cpu")
    return _laya


def workspace_files() -> List[str]:
    out = []
    for p in WORKSPACE.rglob("*"):
        if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(WORKSPACE).parts):
            out.append(p.relative_to(WORKSPACE).as_posix())
    return sorted(out)


def triage(task: str) -> dict:
    """
    Only the request text goes to Laya. Adding the workspace file list made
    multi_part score 0.95+ for every request (even one-line fixes), and
    Laya's own "needs project context?" answer stayed low even when a file
    was named - so context is decided deterministically in build_context().
    """
    answers = load_laya().predict({"request": task[:2000]}, TRIAGE_QUESTIONS)["answers"]
    kind = answers["kind"]["choice"]
    multi_p = float(answers["multi_part"]["noul"])
    return {
        "kind": kind,
        "kind_p": float(answers["kind"]["probabilities"][kind]),
        "multi_p": multi_p,
        "think": multi_p >= 0.5,
    }


# ---------------------------------------------------------------------------
# Layer 3: workspace retrieval
# ---------------------------------------------------------------------------
_index: Optional[VectorStoreIndex] = None
_index_dirty = True


def mark_index_dirty() -> None:
    global _index_dirty
    _index_dirty = True


def get_index() -> Optional[VectorStoreIndex]:
    """(Re)build the workspace index when files changed. Small projects re-embed in seconds."""
    global _index, _index_dirty
    if not _index_dirty:
        return _index
    _index_dirty = False
    try:
        docs = SimpleDirectoryReader(
            str(WORKSPACE),
            recursive=True,
            required_exts=INDEX_EXTS,
            exclude=[f"**/{d}/**" for d in SKIP_DIRS],
        ).load_data()
    except ValueError:  # no matching files
        _index = None
        return None
    for d in docs:
        d.metadata["rel_path"] = Path(d.metadata["file_path"]).resolve().relative_to(WORKSPACE).as_posix()
    py = [d for d in docs if d.metadata["rel_path"].endswith(".py")]
    other = [d for d in docs if not d.metadata["rel_path"].endswith(".py")]
    nodes = CodeSplitter(language="python", chunk_lines=60, max_chars=2000).get_nodes_from_documents(py)
    nodes += SentenceSplitter(chunk_size=512).get_nodes_from_documents(other)
    _index = VectorStoreIndex(nodes)
    return _index


def _file_chunks(index: VectorStoreIndex, rel: str, task: str, k: int = 3) -> List[str]:
    filters = MetadataFilters(filters=[ExactMatchFilter(key="rel_path", value=rel)])
    return [h.node.get_content() for h in index.as_retriever(similarity_top_k=k, filters=filters).retrieve(task)]


def _whole_lines(full: str, chunk: str) -> str:
    """
    Widen a chunk to complete lines. CodeSplitter can cut an oversized function
    mid-line ("self.state = next" from "...nextchar"), and Qwen then "fixes"
    the cut, producing SEARCH text that doesn't exist.
    """
    # CodeSplitter strips the first line's indentation, and the reader keeps CRLF that read_text() drops.
    chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
    core = chunk.strip()
    found = full.find(core)
    if not core or found < 0:
        return chunk
    start = full.rfind("\n", 0, found) + 1
    end = full.find("\n", found + len(core))
    return full[start:end if end >= 0 else len(full)]


def build_context(task: str, kind: str) -> Dict[str, str]:
    """
    Pick context deterministically. Small files go in whole; files over the
    budget go in as their most relevant chunks, which is enough because
    existing files are changed with EDIT blocks rather than rewritten.

    1. Files the request names ("fix pricing.py") always come first.
    2. Other retrieved chunks are kept only if clearly related, unless
       it's a bug fix/question that names no file - every context token
       is prefill time on a CPU.
    """
    index = get_index()
    if index is None:
        return {}
    paths, chunks = [], {}
    lowered = task.lower()
    for rel in workspace_files():
        if Path(rel).suffix in INDEX_EXTS and (rel.lower() in lowered or Path(rel).name.lower() in lowered):
            paths.append(rel)
    named = bool(paths)
    for h in index.as_retriever(similarity_top_k=4).retrieve(task):
        if (kind == "new_code" or named) and (h.score or 0.0) < CONTEXT_SCORE_MIN:
            continue
        rel = h.node.metadata.get("rel_path", "unknown")
        if rel not in paths:
            paths.append(rel)
        chunks.setdefault(rel, []).append(h.node.get_content())

    blocks, used = {}, 0
    for rel in paths:
        full = (WORKSPACE / rel).read_text(encoding="utf-8", errors="replace")
        if used + len(full) <= CONTEXT_BUDGET_CHARS:
            body, label = full, "full file"
        else:
            parts = chunks.get(rel) or _file_chunks(index, rel, task)
            if not parts:
                continue
            parts = list(dict.fromkeys(_whole_lines(full, p) for p in parts))
            body, label = "\n...\n".join(parts), "excerpt - change it only with EDIT blocks, never FILE"
        if used + len(body) > CONTEXT_BUDGET_CHARS:
            continue
        blocks[rel] = f"FILE: {rel} ({label})\n```\n{body}\n```"
        used += len(body)
    return blocks


# ---------------------------------------------------------------------------
# Output parsing (checks and writes live in the MCP server, mcp_server.py)
# ---------------------------------------------------------------------------
def parse_output(text: str) -> tuple:
    """
    Split Qwen's reply into whole files and SEARCH/REPLACE edits:

        FILE: path.py          EDIT: path.py
        ```python              <<<<<<< SEARCH
        ...whole file...       ...exact current lines...
        ```                    =======
                               ...new lines...
                               >>>>>>> REPLACE

    Code-fence lines around EDIT blocks are ignored; text inside blocks is kept verbatim.
    Returns ({path: text}, [{path, search, replace}]).
    """
    lines = text.split("\n")
    files, edits, edit_path, i = {}, [], None, 0
    while i < len(lines):
        header = HEADER_RE.match(lines[i])
        if header:
            # drop a copied context label like "pricing.py (full file)"
            kind, path = header.group(1), re.sub(r"\s*\(.*\)$", "", header.group(2).strip())
            if kind == "EDIT":
                edit_path, i = path, i + 1
                continue
            edit_path, j = None, i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].lstrip().startswith("```"):
                end = j + 1
                while end < len(lines) and lines[end].strip() != "```":
                    end += 1
                files[path] = "\n".join(lines[j + 1:end]) + "\n"
                i = end + 1
                continue
        elif edit_path and SEARCH_RE.match(lines[i].strip()):
            j = i + 1
            while j < len(lines) and not DIVIDER_RE.match(lines[j].strip()):
                j += 1
            k = j + 1
            while k < len(lines) and not REPLACE_RE.match(lines[k].strip()):
                k += 1
            if k < len(lines):  # ignore a block the model never closed
                edits.append({"path": edit_path, "search": "\n".join(lines[i + 1:j]),
                              "replace": "\n".join(lines[j + 1:k])})
            i = k + 1
            continue
        i += 1
    return files, edits


# ---------------------------------------------------------------------------
# Layer 4: generation (Qwen 3 via Ollama, streamed out of the graph)
# ---------------------------------------------------------------------------
CODER_PROMPT = """You are a senior Python engineer working inside a local project.

Rules:
- To change a file that ALREADY EXISTS, output only the changed parts as edit blocks:

EDIT: relative/path.py
<<<<<<< SEARCH
<lines copied exactly from the current file, same indentation>
=======
<the new lines>
>>>>>>> REPLACE

  Each SEARCH must match exactly one place in the file: include 2-3 unchanged lines
  around the change. Prefer several small blocks over one large one. To delete lines,
  leave the part after ======= empty. Never output a whole existing file.
- To create a NEW file (including new test files), output the complete file:

FILE: relative/path.py
```python
<entire file contents>
```

- Change only what the task asks for. Do not touch unrelated code, even if it looks odd.
- Always include pytest tests (in a test_*.py file) for the behaviour you wrote.
  Tests import modules by their path relative to the project root.
- Tests must never wait for real typing: when code calls input(), patch it, e.g.
  `with unittest.mock.patch("builtins.input", side_effect=["1", "3"]):`.
  If the code loops (a menu or a game), the side_effect list must END with the input that
  exits the loop, or the test fails with StopIteration when the list runs out.
- Put logic in plain functions that tests can call. Keep input()/print() interaction in a
  main() function run only under `if __name__ == "__main__":`, so tests never wait for typing.
- Never leave placeholders: no TODO, no `...` bodies, no NotImplementedError.
- Only the Python standard library plus pytest, unless the project context shows another dependency.
- Keep any explanation to one or two sentences after the blocks."""

ANSWER_PROMPT = "You are a senior engineer answering questions about a local code project. Be concise and concrete."


def _chat_stream(messages: List[dict], think: bool, writer, max_thinking: int) -> tuple:
    """One streamed Ollama call. Stops early (closing the stream) once thinking passes max_thinking chunks."""
    text, thinking, n_think, stats = "", "", 0, {}
    stream = CLIENT.chat(
        model=LLM_MODEL, messages=messages, stream=True, think=think, keep_alive=KEEP_ALIVE,
        options={**OLLAMA_OPTIONS, "num_predict": MAX_NEW_TOKENS + (max_thinking if think else 0), "temperature": 0.3},
    )
    for chunk in stream:
        if chunk.done:
            stats = {"tokens": chunk.eval_count or 0, "seconds": (chunk.eval_duration or 0) / 1e9,
                     "prompt_tokens": chunk.prompt_eval_count or 0,
                     "prompt_seconds": (chunk.prompt_eval_duration or 0) / 1e9}
        if chunk.message.thinking:
            thinking += chunk.message.thinking
            n_think += 1
            writer({"type": "thinking", "text": chunk.message.thinking})
            if n_think >= max_thinking and not text:
                stream.close()  # stops generation in Ollama too
                return text, thinking, True, stats
        if chunk.message.content:
            text += chunk.message.content
            writer({"type": "token", "text": chunk.message.content})
    return text, thinking, False, stats


def stream_llm(messages: List[dict], think: bool, phase: str, **meta) -> str:
    """
    Stream one completion, pushing tokens to the UI via LangGraph's custom stream.

    Reasoning is capped at THINK_BUDGET tokens (~5 min on this CPU). Qwen 3 has no
    built-in thinking limit, and in testing it spent the entire output budget
    thinking and returned an empty answer, twice in one task (~9 min each). When
    the cap hits, a second call with reasoning off gets the notes and writes the reply.
    """
    writer = get_stream_writer()
    writer({"type": "stream_start", "phase": phase, "think": think, **meta})
    text, thinking, cut, stats = _chat_stream(messages, think, writer, THINK_BUDGET)
    if cut:
        _debug_log(f"thinking budget ({THINK_BUDGET}) reached; answering from notes")
        writer({"type": "thinking_cut"})
        notes = thinking[-6000:]
        text, _, _, stats = _chat_stream(
            messages + [{"role": "user", "content": f"Stop planning now. Your notes so far:\n{notes}\n\n"
                                                    "Now write your reply in the required format."}],
            False, writer, THINK_BUDGET)
    writer({"type": "stream_end", "phase": phase, **stats})
    return text


# ---------------------------------------------------------------------------
# Layer 2: LangGraph orchestration
# ---------------------------------------------------------------------------
class FactoryState(TypedDict, total=False):
    """
    Everything the factory knows about one task. LangGraph checkpoints
    this after every node (InMemorySaver), which is what lets review()
    pause the whole graph mid-run and resume it later with your decision.
    """

    task: str
    notes: str  # optional background for Qwen (e.g. a design); not shown to Laya or used for retrieval
    triage: dict
    context: Dict[str, str]
    files: Dict[str, str]  # full text of every file changed so far (the draft)
    edits: List[dict]  # SEARCH/REPLACE blocks from the latest reply, applied by the MCP server
    rejected: List[str]  # whole-file blocks refused before validation (suspected truncation)
    applied: List[dict]  # post-edit text around each applied edit, for the next retry
    edit_failures: Dict[str, int]  # per file; triggers the whole-file fallback
    format_ok: bool
    checks: List[dict]
    sandbox_id: Optional[str]  # MCP server's staged copy of the current draft
    diffs: List[dict]
    messages: List[dict]  # the coder conversation, continued across retries and revisions
    attempt: int
    feedback: str
    decision: Literal["approve", "reject", "revise"]
    answer: str
    written: List[str]


def triage_node(state: FactoryState) -> dict:
    result = triage(state["task"])
    get_stream_writer()({"type": "triage", **result})
    return {"triage": result, "attempt": 0, "feedback": ""}


def retrieve_node(state: FactoryState) -> dict:
    context = build_context(state["task"], state["triage"]["kind"])
    get_stream_writer()({"type": "retrieve", "files": list(context)})
    return {"context": context}


def answer_node(state: FactoryState) -> dict:
    user = state["task"]
    if state.get("context"):
        context = "\n\n".join(state["context"].values())
        user = f"Project context:\n{context}\n\nQuestion: {state['task']}"
    text = stream_llm(
        [{"role": "system", "content": ANSWER_PROMPT}, {"role": "user", "content": user}],
        think=state["triage"]["think"], phase="answer",
    )
    return {"answer": text}


def _debug_log(entry: str) -> None:
    """Every prompt tail, raw Qwen reply and check result, for diagnosing bad edits (logs/factory.log)."""
    LOG_DIR.mkdir(exist_ok=True)
    with open(LOG_DIR / "factory.log", "a", encoding="utf-8") as f:
        f.write(f"\n{'=' * 30} {time.strftime('%Y-%m-%d %H:%M:%S')}\n{entry}\n")


def _current_text(state: FactoryState, rel: str) -> str:
    if rel in state.get("files", {}):
        return state["files"][rel]
    path = WORKSPACE / rel
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _followup_message(state: FactoryState) -> str:
    parts = []
    if state.get("feedback"):
        parts.append(f"The reviewer asked for these changes: {state['feedback']}")
    failed = [c for c in state.get("checks", []) if not c["ok"]]
    if failed:
        parts.append(
            "Your changes failed these automatic checks. A failing test can mean the code is wrong "
            "OR the test's expected value is wrong; check the task and fix whichever one is wrong:\n"
            + "\n".join(f"[{c['name']}] {c['detail']}" for c in failed)
        )
    # Edits that did apply changed the files: later SEARCH blocks must target the new text.
    regions = [a for a in state.get("applied", []) if a["region"]]
    if regions:
        parts.append("Your applied edits are already in the files. The text around them is now:\n" + "\n\n".join(
            f"{a['path']}:\n```\n{a['region']}\n```" for a in regions))
    # Fallback: after repeated SEARCH misses, ask for the whole file - with its full current
    # text, since the context may only have shown excerpts of it.
    stuck = [p for p, n in state.get("edit_failures", {}).items() if n >= FALLBACK_AFTER_FAILURES]
    for rel in stuck:
        parts.append(f"Your EDIT blocks for {rel} keep failing to match. Stop using EDIT for {rel}: "
                     f"reply with the complete updated file as a FILE block. Its current text is:\n"
                     f"```python\n{_current_text(state, rel)}\n```")
    parts.append("Reply with EDIT blocks for existing files and FILE blocks for new files, as before.")
    return "\n\n".join(parts)


def _fit(messages: List[dict]) -> List[dict]:
    """Drop the oldest draft/feedback pairs if the conversation would overflow num_ctx (~3 chars/token)."""
    budget = (NUM_CTX - MAX_NEW_TOKENS) * 3
    while len(messages) > 4 and sum(len(m["content"]) for m in messages) > budget:
        messages = messages[:2] + messages[4:]
    return messages


def generate_node(state: FactoryState) -> dict:
    """
    Retries and revisions continue ONE conversation - task, draft, errors, new
    draft - instead of rebuilding a fresh prompt each time. Ollama keeps the KV
    cache of the previous request, so a retry only has to prefill the short
    error message rather than the whole ~1.5k-token context again (prefill ran
    at ~11-14 tok/s on this CPU, i.e. ~2 minutes per full re-read).
    """
    attempt = state.get("attempt", 0) + 1
    messages = list(state.get("messages", []))
    if not messages:
        parts = [f"Task: {state['task']}"]
        if state.get("notes"):
            parts.append(f"Background (do only the task above, not the whole plan):\n{state['notes']}")
        if state.get("context"):
            parts.append("Project context (files as they are on disk now):\n" + "\n\n".join(state["context"].values()))
        # Qwen skipped the test file on its first reply in 3 of 5 logged runs; one reminder at the end helps.
        parts.append("Remember: your reply must include a test_*.py file for what you write or change.")
        messages = [{"role": "system", "content": CODER_PROMPT}, {"role": "user", "content": "\n\n".join(parts)}]
    else:
        messages.append({"role": "user", "content": _followup_message(state)})
    messages = _fit(messages)

    text = stream_llm(messages, think=state["triage"]["think"], phase="generate",
                      attempt=attempt, max_attempts=MAX_ATTEMPTS)
    _debug_log(f"attempt {attempt} - prompt\n{messages[-1]['content']}\n\nattempt {attempt} - Qwen reply\n{text}")
    new_files, edits = parse_output(text)

    # A whole-file block for a file ON DISK that shrank by half is almost always the model
    # re-typing only the excerpt it saw. Reject it rather than let it delete the rest of the file.
    # Files that exist only as this task's draft are the model's own work: it may shorten them.
    rejected = []
    for rel, new in list(new_files.items()):
        path = WORKSPACE / rel
        old = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
        if old and len(new.splitlines()) < 0.5 * len(old.splitlines()):
            rejected.append(f"FILE {rel} has {len(new.splitlines())} lines but the current file has "
                            f"{len(old.splitlines())}. Change existing files with EDIT blocks instead.")
            del new_files[rel]
    return {
        # Merge: a retry often re-sends only the file it fixed; replacing the draft would drop the rest.
        "files": {**state.get("files", {}), **new_files},
        "edits": edits,
        "rejected": rejected,
        "format_ok": bool(new_files or edits or rejected),
        "attempt": attempt,
        "feedback": "",
        "messages": messages + [{"role": "assistant", "content": text}],
    }


def check_node(state: FactoryState) -> dict:
    """MCP validate: the server applies the EDIT blocks to the draft in a fresh sandbox, then runs the guardrails."""
    if state.get("sandbox_id"):
        mcp_call("discard", sandbox_id=state["sandbox_id"])  # superseded draft
    if state.get("rejected"):
        result = {"checks": [{"name": "edits", "ok": False, "detail": "\n".join(state["rejected"])}],
                  "sandbox_id": None, "diffs": state.get("diffs", []), "files": state["files"],
                  "applied": [], "failed_paths": []}
    elif state["format_ok"]:
        result = mcp_call("validate", files=state["files"], edits=state.get("edits", []))
    else:
        result = mcp_call("validate")  # returns the "no FILE or EDIT blocks" format error
        result["files"] = state.get("files", {})

    _debug_log(f"attempt {state['attempt']} - checks\n" + "\n".join(
        f"[{'ok' if c['ok'] else 'FAIL'}] {c['name']}: {c['detail']}" for c in result["checks"]))
    failures = dict(state.get("edit_failures", {}))
    for rel in result["failed_paths"]:
        failures[rel] = failures.get(rel, 0) + 1
    get_stream_writer()({"type": "check", "checks": result["checks"], "attempt": state["attempt"],
                         "edits": len(state.get("edits", [])), "failed_paths": result["failed_paths"]})
    return {"checks": result["checks"], "sandbox_id": result["sandbox_id"], "diffs": result["diffs"],
            "files": result["files"], "applied": result["applied"], "edit_failures": failures,
            "edits": [], "rejected": []}


def review_node(state: FactoryState) -> dict:
    """Human-in-the-loop: interrupt() pauses the graph here until the UI resumes it."""
    reply = interrupt({
        "diffs": state["diffs"],
        "checks": state["checks"],
        "passed": all(c["ok"] for c in state["checks"]),
    })
    if reply == "approve":
        return {"decision": "approve"}
    if reply == "reject":
        if state.get("sandbox_id"):
            mcp_call("discard", sandbox_id=state["sandbox_id"])
        return {"decision": "reject", "sandbox_id": None}
    return {"decision": "revise", "feedback": reply, "attempt": 0}


def write_node(state: FactoryState) -> dict:
    """MCP commit: the server copies the exact files it tested from the sandbox into the workspace."""
    written = mcp_call("commit", sandbox_id=state["sandbox_id"])["written"] if state.get("sandbox_id") else []
    mark_index_dirty()
    get_stream_writer()({"type": "written", "files": written})
    return {"written": written, "sandbox_id": None}


def after_triage(state: FactoryState) -> str:
    if get_index() is not None:
        return "retrieve"
    return "answer" if state["triage"]["kind"] == "question" else "generate"


def after_retrieve(state: FactoryState) -> str:
    return "answer" if state["triage"]["kind"] == "question" else "generate"


def after_check(state: FactoryState) -> str:
    """The loop guard: retry only while checks fail AND attempts remain."""
    if all(c["ok"] for c in state["checks"]) or state["attempt"] >= MAX_ATTEMPTS:
        return "review"
    return "generate"


def after_review(state: FactoryState) -> str:
    return {"approve": "write", "reject": END, "revise": "generate"}[state["decision"]]


def build_graph():
    g = StateGraph(FactoryState)
    for name, fn in [("triage", triage_node), ("retrieve", retrieve_node), ("answer", answer_node),
                     ("generate", generate_node), ("check", check_node), ("review", review_node),
                     ("write", write_node)]:
        g.add_node(name, fn)
    g.set_entry_point("triage")
    g.add_conditional_edges("triage", after_triage, ["retrieve", "answer", "generate"])
    g.add_conditional_edges("retrieve", after_retrieve, ["answer", "generate"])
    g.add_edge("answer", END)
    g.add_edge("generate", "check")
    g.add_conditional_edges("check", after_check, ["generate", "review"])
    g.add_conditional_edges("review", after_review, ["write", "generate", END])
    g.add_edge("write", END)
    return g.compile(checkpointer=InMemorySaver())
