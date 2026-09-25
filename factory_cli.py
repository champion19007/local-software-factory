"""
Terminal UI for the software factory, styled after Claude Code.

    python factory_cli.py            (code goes in ./workspace)
    set WORKSPACE=C:\\path\\to\\project  then run it to work on another folder

Type a task. Commands: /help  /files  /reindex  /exit
"""

import os
import sys
import uuid

from rich.console import Console, Group

# On a non-UTF-8 output (old console code page, output piped to a file), print "?" for
# symbols like ● instead of crashing. kid_cli and dev_cli import this module too.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

ACCENT = "#D97757"
console = Console()

HELP = """\
  Type a coding task or a question about the workspace.
  /files     list files in the workspace
  /reindex   re-embed the workspace (after editing files outside the factory)
  /exit      quit  (Ctrl+C cancels a running task)"""


def pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def bullet(title: str, detail: str = "", color: str = "green") -> None:
    console.print(Text.assemble(("● ", color), (title, "bold"), (f"  {detail}" if detail else "", "dim")))


def sub(text, style: str = "dim") -> None:
    console.print(Text("  ⎿ ", style="dim") + (text if isinstance(text, Text) else Text(text, style=style)))


class View:
    """Renders the custom events that factory.py's graph nodes emit."""

    def __init__(self, factory):
        self.f = factory
        self.live = None
        self.buf = self.thinking = ""
        self.phase = None
        self.meta = {}

    def _render(self):
        if self.phase == "answer" and self.buf:
            return Markdown(self.buf)
        if self.buf:
            label = f"Writing code… attempt {self.meta.get('attempt')}/{self.meta.get('max_attempts')}"
        else:
            label = "Thinking…" if self.thinking else "Reading the prompt…"
        spinner = Spinner("dots", text=Text(label, style=ACCENT), style=ACCENT)
        tail_src = self.buf or self.thinking
        if not tail_src:
            return spinner
        tail = "\n".join(tail_src.rstrip().splitlines()[-12:])
        return Group(spinner, Text(tail, style="dim" if self.buf else "dim italic"))

    def stop(self):
        if self.live:
            self.live.stop()
            self.live = None

    def handle(self, ev: dict) -> None:
        t = ev["type"]
        if t == "triage":
            bullet("Triage", "(laya)")
            sub(f"{ev['kind']} {pct(ev['kind_p'])} · multi-part {pct(ev['multi_p'])} → reasoning {'on' if ev['think'] else 'off'}")
        elif t == "retrieve":
            bullet("Retrieve", f"({self.f.WORKSPACE.name}/)")
            sub(", ".join(ev["files"]) if ev["files"] else "nothing relevant found")
        elif t == "stream_start":
            self.phase, self.meta = ev["phase"], ev
            self.buf = self.thinking = ""
            if self.phase == "generate":
                bullet("Generate", f"(attempt {ev['attempt']}/{ev['max_attempts']}{', reasoning' if ev['think'] else ''})")
            else:
                console.print()
            self.live = Live(
                self._render(), console=console, refresh_per_second=8,
                transient=self.phase == "generate",
                vertical_overflow="visible" if self.phase == "answer" else "crop",
            )
            self.live.start()
        elif t in ("token", "thinking"):
            if t == "token":
                self.buf += ev["text"]
            else:
                self.thinking += ev["text"]
            self.live.update(self._render())
        elif t == "stream_end":
            if self.phase == "answer":
                self.live.update(Markdown(self.buf) if self.buf else Text("(no answer)", style="dim"))
            self.stop()
            if ev["seconds"]:
                prefix = ""
                if self.phase == "generate":
                    files, edits = self.f.parse_output(self.buf)
                    prefix = f"{len(edits)} edit(s), {len(files)} new file(s) · "
                # prompt_tokens counts cache hits too, so only the time says how much was really re-read
                read = f"prompt read in {ev['prompt_seconds']:.0f}s · " if ev.get("prompt_tokens") else ""
                sub(f"{prefix}{read}wrote {ev['tokens']} tokens at {ev['tokens'] / ev['seconds']:.1f} tok/s · {ev['seconds']:.0f}s")
        elif t == "check":
            checks = ev["checks"]
            passed = all(c["ok"] for c in checks)
            bullet("Check", "(mcp · validate)", "green" if passed else "red")
            line = Text()
            for c in checks:
                line.append("✓ " if c["ok"] else "✗ ", style="green" if c["ok"] else "red")
                line.append(f"{c['name']}  ")
            sub(line)
            for c in checks:
                if c["detail"] and (not c["ok"] or c["name"] == "pytest"):
                    for ln in c["detail"].splitlines()[-10:]:
                        console.print(f"    {ln}", style="dim" if c["ok"] else "red", highlight=False, markup=False)
            if any(c["name"] == "format" for c in checks):
                console.print("    the reply started with:", style="dim")
                for ln in self.buf.strip().splitlines()[:8]:
                    console.print(f"    | {ln}", style="dim", highlight=False, markup=False)
            if not passed:
                if ev["attempt"] < self.f.MAX_ATTEMPTS:
                    sub("sending the errors back to Qwen")
                else:
                    sub(f"still failing after {self.f.MAX_ATTEMPTS} attempts - handing over to you", "yellow")
        elif t == "written":
            bullet("Write", "(mcp · commit)")
            sub(", ".join(ev["files"]) or "nothing to write")

    def review(self, payload: dict) -> str:
        bullet("Review", "", ACCENT)
        for d in payload["diffs"]:
            console.print(Panel(
                Syntax(d["diff"] or "(no changes)", "diff", theme="ansi_dark", word_wrap=True),
                title=f"{d['path']} · {'new file' if d['is_new'] else 'modified'}", title_align="left",
                border_style="dim",
            ))
        if not payload["passed"]:
            console.print("  Checks are still failing - approving will write code that doesn't pass.", style="yellow")
        while True:
            try:
                ans = console.input(
                    f"  Apply to {self.f.WORKSPACE.name}/?  [bold]y[/]es · [bold]n[/]o · or type feedback to revise: "
                ).strip()
            except EOFError:
                ans = "n"
            if ans.lower() in ("y", "yes"):
                return "approve"
            if ans.lower() in ("n", "no"):
                sub("discarded")
                return "reject"
            if ans.startswith("/"):
                sub("answer y, n, or describe what to change first")
            elif ans:
                return ans


def run_task(factory, graph, view: View, task: str, notes: str = "") -> dict:
    """Drive one task through the graph, pausing for view.review(); returns the final state."""
    from langgraph.types import Command

    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 60}
    inp = {"task": task, "notes": notes}
    while True:
        pending = None
        try:
            for mode, chunk in graph.stream(inp, config, stream_mode=["custom", "updates"]):
                if mode == "custom":
                    view.handle(chunk)
                elif "__interrupt__" in chunk:
                    pending = chunk["__interrupt__"][0].value
        finally:
            view.stop()
        if pending is None:
            return graph.get_state(config).values
        inp = Command(resume=view.review(pending))


def main() -> None:
    with console.status("[dim]Loading Laya, the index, LangGraph and the MCP server… (first run downloads ~1.7GB)",
                        spinner="dots", spinner_style=ACCENT):
        import factory

        tools = factory.mcp().tool_names()
        factory.load_laya()
        factory.get_index()
        graph = factory.build_graph()

    ws = factory.WORKSPACE
    try:
        shown = os.path.relpath(ws)
    except ValueError:  # different drive on Windows
        shown = str(ws)
    body = Text.assemble(
        ("✻ ", ACCENT), ("Local Software Factory", "bold"), "\n\n",
        ("  coder      ", "dim"), factory.LLM_MODEL, " (ollama)\n",
        ("  router     ", "dim"), "laya (cpu)\n",
        ("  mcp        ", "dim"), f"factory-workspace · {', '.join(tools)}\n",
        ("  guardrails ", "dim"), f"ast · ruff · pytest (in mcp) · {factory.MAX_ATTEMPTS} attempts\n",
        ("  workspace  ", "dim"), f"{shown} · {len(factory.mcp_call('list_files'))} files\n\n",
        ("  /help for commands", "dim"),
    )
    console.print(Panel(body, border_style=ACCENT, expand=False, padding=(0, 2)))
    view = View(factory)

    while True:
        try:
            task = console.input(f"\n[bold {ACCENT}]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        if task.startswith("/"):
            cmd = task.lower()
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                console.print(HELP, style="dim")
            elif cmd == "/files":
                sub("\n    ".join(factory.mcp_call("list_files")) or "(empty)")
            elif cmd == "/reindex":
                factory.mark_index_dirty()
                with console.status("[dim]Re-indexing…", spinner="dots", spinner_style=ACCENT):
                    factory.get_index()
                sub("done")
            else:
                sub(f"unknown command {task} - try /help")
            continue
        try:
            run_task(factory, graph, view, task)
        except KeyboardInterrupt:
            sub("cancelled")
        except Exception as e:  # keep the session alive on Ollama/network errors
            console.print(f"  ⎿ error: {e}", style="red", markup=False)
            if isinstance(e, factory.MCPConnectionLost):
                console.print("    the MCP server stopped; it restarts automatically on the next task", style="dim")
            elif isinstance(e, ConnectionError):
                console.print("    is Ollama running?  try: ollama serve", style="dim")
            elif isinstance(e, factory.MCPToolError):
                console.print("    details in mcp_server.log", style="dim")

    console.print("\n[dim]bye[/]")


if __name__ == "__main__":
    main()
