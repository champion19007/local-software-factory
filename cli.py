"""
Terminal chat UI for the local RAG agent, styled after Claude Code.

    python cli.py

Commands: /help  /think  /clear  /reindex  /exit
"""

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

ACCENT = "#D97757"
console = Console()

HELP = """\
  /think     toggle Qwen 3 reasoning mode (smarter, much slower on CPU)
  /clear     forget the conversation so far
  /reindex   rebuild the index after adding or changing files in data/
  /exit      quit  (Ctrl+C interrupts a running answer)"""


def banner(app, chunks: int, think: bool) -> None:
    docs = (
        f"{app.DATA_DIR}/ · {chunks} chunks indexed"
        if chunks
        else f"{app.DATA_DIR}/ is empty · answering without documents"
    )
    body = Text.assemble(
        ("✻ ", ACCENT), ("Local RAG Agent", "bold"), "\n\n",
        ("  model  ", "dim"), app.LLM_MODEL, "\n",
        ("  docs   ", "dim"), docs, "\n",
        ("  think  ", "dim"), "on" if think else "off", "\n\n",
        ("  /help for commands", "dim"),
    )
    console.print(Panel(body, border_style=ACCENT, expand=False, padding=(0, 2)))


def show_route(app, state) -> None:
    if state["route"] == "rag":
        console.print(Text.assemble(("● ", "green"), ("Retrieve", "bold"), f"({app.DATA_DIR}/)"))
        found = Text("  ⎿ ", style="dim")
        for i, (name, score) in enumerate(app.sources(state)):
            if i:
                found.append(", ", style="dim")
            found.append(name)
            found.append(f" {score:.2f}", style="dim")
        console.print(found)
    else:
        reason = "no documents indexed"
        if state.get("top_score") is not None:
            reason = f"best match {state['top_score']:.2f} < {app.RAG_SCORE_THRESHOLD}"
        console.print(Text.assemble(("● ", "dim"), ("Direct answer", "bold"), (f"  ({reason})", "dim")))


def run_turn(app, query: str, history: list, think: bool) -> str:
    with console.status("[dim]Routing…", spinner="dots", spinner_style=ACCENT):
        state = app.run_graph(query)
    show_route(app, state)

    answer, thinking, stats = "", "", None

    def render():
        if answer:
            return Markdown(answer)
        spinner = Spinner("dots", text=Text("Thinking…" if thinking else "Generating…", style=ACCENT), style=ACCENT)
        if not thinking:
            return spinner
        tail = "\n".join(thinking.strip().splitlines()[-3:])
        return Group(spinner, Text(tail, style="dim italic"))

    console.print()
    with Live(render(), console=console, refresh_per_second=10, vertical_overflow="visible") as live:
        for kind, data in app.stream_answer(state["query"], state["context"], history, think):
            if kind == "thinking":
                thinking += data
            elif kind == "answer":
                answer += data
            else:
                stats = data
            live.update(render())
        live.update(Markdown(answer) if answer else Text("(no answer)", style="dim"))

    if stats and stats["seconds"]:
        console.print(
            f"  ⎿ {stats['tokens']} tokens · {stats['tokens'] / stats['seconds']:.1f} tok/s · {stats['seconds']:.1f}s",
            style="dim",
        )
    return answer


def main() -> None:
    with console.status("[dim]Loading models and index…", spinner="dots", spinner_style=ACCENT):
        import app

        chunks = app.load_index()

    think = False
    history: list = []
    banner(app, chunks, think)

    while True:
        try:
            query = console.input(f"\n[bold {ACCENT}]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue

        if query.startswith("/"):
            cmd = query.lower()
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                console.print(HELP, style="dim")
            elif cmd == "/clear":
                history.clear()
                console.print("  ⎿ conversation cleared", style="dim")
            elif cmd == "/think":
                think = not think
                console.print(f"  ⎿ thinking {'on' if think else 'off'}", style="dim")
            elif cmd == "/reindex":
                with console.status("[dim]Re-indexing…", spinner="dots", spinner_style=ACCENT):
                    chunks = app.load_index(rebuild=True)
                console.print(f"  ⎿ {chunks} chunks indexed from {app.DATA_DIR}/", style="dim")
            else:
                console.print(f"  ⎿ unknown command {query} - try /help", style="dim")
            continue

        try:
            answer = run_turn(app, query, history, think)
        except KeyboardInterrupt:
            console.print("  ⎿ interrupted", style="dim")
            continue
        except Exception as e:  # surface Ollama/network errors without killing the session
            console.print(f"  ⎿ error: {e}", style="red")
            console.print("    is Ollama running?  try: ollama serve", style="dim")
            continue

        if answer:
            history += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]

    console.print("\n[dim]bye[/]")


if __name__ == "__main__":
    main()
