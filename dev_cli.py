"""
Developer console: design a program first, then have the factory build it step by step.

    python dev_cli.py              (code goes in ./workspace, designs in ./designs)

    /design <idea>     Qwen writes a system design: goal, components, file layout,
                       data flow, a Mermaid diagram and numbered build steps.
                       Accept it, give feedback to revise it, or discard it.
    /plan              the current design's build steps and their status
    /build             build the next pending step      /build all   every remaining step
    /build <n>         build step n                     /skip <n>    mark step n done by hand
    /designs           saved designs                    /open <n>    switch to saved design n
    /think on|off|auto reasoning for design generation (auto = let Laya decide)
    /files  /help  /exit
    anything else      a normal factory task, like factory_cli.py

Each build step is an ordinary factory task (Laya -> Qwen edits -> MCP checks ->
your review). The design goes to Qwen as background notes, while Laya and
retrieval see only the step, so a design that lists many components doesn't
switch reasoning mode on for every small step. Designs live outside the
workspace, which keeps the MCP server as the only writer to the workspace.
"""

import json
import re
import time
from pathlib import Path

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from factory_cli import ACCENT, View, bullet, console, run_task, sub

DESIGN_DIR = Path(__file__).resolve().with_name("designs")
DESIGN_TOKENS = 2500

DESIGN_PROMPT = """You are a senior software architect planning a small Python program that a
local 8B coding model will then build one step at a time.

Reply with a design in exactly this Markdown structure:

# <short project name>

## Goal
Two or three sentences: what the program does and for whom.

## Components
- **name** - its single responsibility

## File layout
- `path/to/file.py` - what lives there (include the test file for each module)

## Data flow
A few sentences or bullets: how data moves between the components.

## Diagram
```mermaid
flowchart LR
    A[Component] --> B[Component]
```

## Build steps
1. One small task: name the exact file(s) it creates or changes and the tests to write.
2. ...

Rules:
- 3 to 7 build steps, in dependency order: core logic first, the entry point (main / CLI) last.
- The entry-point step creates the whole entry file, including its `if __name__ == "__main__":`
  block and tests that patch input(). Never split one file across two steps.
- Each step touches at most two source files plus their tests, and makes sense on its own.
- Python standard library only, unless the idea clearly needs something else; say so under Goal.
- Keep input()/print() interaction in a main() under `if __name__ == "__main__":`.
- Be concrete: real function names and file names, no placeholders."""

STEP_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)")


# ---------------------------------------------------------------------------
# Design documents
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48] or "design"


def section(md: str, heading: str) -> str:
    m = re.search(rf"^##\s+{heading}\s*$(.*?)(?=^##\s|\Z)", md, re.MULTILINE | re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def parse_steps(md: str) -> list:
    steps, current = [], None
    for line in section(md, "Build steps").splitlines():
        m = STEP_RE.match(line)
        if m:
            current = m.group(2).strip()
            steps.append(current)
        elif current is not None and line.strip():
            steps[-1] = f"{steps[-1]} {line.strip()}"
    return steps


def title_of(md: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", md, re.MULTILINE)
    return m.group(1).strip() if m else fallback


class Design:
    """A saved design (designs/<slug>.md) plus step status (designs/<slug>.status.json)."""

    def __init__(self, path: Path):
        self.path = path
        self.md = path.read_text(encoding="utf-8")
        self.title = title_of(self.md, path.stem)
        self.steps = parse_steps(self.md)
        status_file = path.with_suffix(".status.json")
        self.done = set(json.loads(status_file.read_text(encoding="utf-8"))) if status_file.exists() else set()

    def save_status(self) -> None:
        self.path.with_suffix(".status.json").write_text(json.dumps(sorted(self.done)), encoding="utf-8")

    def next_step(self):
        return next((i for i in range(1, len(self.steps) + 1) if i not in self.done), None)

    def notes_for(self, n: int) -> str:
        """Compact background for step n: the parts of the design a single step needs."""
        return "\n\n".join(part for part in (
            f"Project: {self.title}",
            "Goal:\n" + section(self.md, "Goal"),
            "File layout:\n" + section(self.md, "File layout"),
            "Components:\n" + section(self.md, "Components"),
            "Steps already done: " + (", ".join(str(i) for i in sorted(self.done)) or "none"),
        ) if part.strip())[:2500]


def save_design(md: str, idea: str) -> Design:
    DESIGN_DIR.mkdir(exist_ok=True)
    base = slugify(title_of(md, idea))
    path, n = DESIGN_DIR / f"{base}.md", 2
    while path.exists():
        path, n = DESIGN_DIR / f"{base}-{n}.md", n + 1
    path.write_text(md, encoding="utf-8")
    return Design(path)


def saved_designs() -> list:
    return sorted(DESIGN_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime) if DESIGN_DIR.exists() else []


# ---------------------------------------------------------------------------
# Design generation (streams straight from Ollama; no graph needed)
# ---------------------------------------------------------------------------
def stream_design(factory, messages: list, think: bool) -> str:
    text, thinking, stats, start = "", "", {}, time.monotonic()

    def render():
        label = "Designing…" if text else ("Thinking…" if thinking else "Reading the idea…")
        spinner = Spinner("dots", text=Text(f"{label}  {int(time.monotonic() - start)}s", style=ACCENT), style=ACCENT)
        tail = "\n".join((text or thinking).rstrip().splitlines()[-14:])
        return spinner if not tail else Panel(Text(tail, style="dim" if text else "dim italic"), title=spinner, border_style="dim")

    with Live(render(), console=console, refresh_per_second=6, transient=True) as live:
        msgs, n_think = messages, 0
        while True:
            stream = factory.CLIENT.chat(
                model=factory.LLM_MODEL, messages=msgs, stream=True, think=think, keep_alive=factory.KEEP_ALIVE,
                options={**factory.OLLAMA_OPTIONS, "temperature": 0.4,
                         "num_predict": DESIGN_TOKENS + (factory.THINK_BUDGET if think else 0)},
            )
            cut = False
            for chunk in stream:
                if chunk.message.thinking:
                    thinking += chunk.message.thinking
                    n_think += 1
                    if n_think >= factory.THINK_BUDGET and not text:  # same cap as the factory's coder
                        stream.close()
                        cut = True
                        break
                if chunk.message.content:
                    text += chunk.message.content
                if chunk.done:
                    stats = {"tokens": chunk.eval_count or 0, "seconds": (chunk.eval_duration or 0) / 1e9}
                live.update(render())
            if not cut:
                break
            think = False
            msgs = messages + [{"role": "user", "content": f"Stop planning now. Your notes so far:\n{thinking[-6000:]}\n\n"
                                                           "Now write the design in the required structure."}]
    if stats.get("seconds"):
        sub(f"wrote {stats['tokens']} tokens at {stats['tokens'] / stats['seconds']:.1f} tok/s · "
            f"{time.monotonic() - start:.0f}s total")
    return text.strip()


def design_flow(factory, idea: str, think_mode: str):
    """Generate -> show -> accept / revise / discard. Returns the saved Design or None."""
    if think_mode == "auto":
        tri = factory.triage(idea)
        think = tri["think"]
        bullet("Triage", "(laya)")
        sub(f"multi-part {tri['multi_p'] * 100:.0f}% → reasoning {'on' if think else 'off'}")
    else:
        think = think_mode == "on"

    existing = factory.mcp_call("list_files")
    user = f"Idea: {idea}"
    if existing:
        user += "\n\nThe project already has these files (extend them rather than starting over):\n" + \
                "\n".join(f"- {f}" for f in existing[:60])
    messages = [{"role": "system", "content": DESIGN_PROMPT}, {"role": "user", "content": user}]

    while True:
        bullet("Design", "(qwen" + (", reasoning" if think else "") + ")")
        md = stream_design(factory, messages, think)
        messages.append({"role": "assistant", "content": md})
        console.print(Panel(Markdown(md), border_style=ACCENT, padding=(1, 2)))
        steps = parse_steps(md)
        if steps:
            sub(f"{len(steps)} build step(s) found")
        else:
            console.print("  No numbered steps under '## Build steps' - ask for a revision.", style="yellow")
        try:
            ans = console.input("  [bold]a[/]ccept · [bold]d[/]iscard · or type feedback to revise: ").strip()
        except EOFError:
            ans = "d"
        if ans.lower() in ("a", "accept", "y", "yes") and steps:
            design = save_design(md, idea)
            bullet("Saved", f"({design.path.relative_to(DESIGN_DIR.parent)})")
            sub("run /build to start, or /build all")
            return design
        if ans.lower() in ("d", "discard", "n", "no", ""):
            sub("discarded")
            return None
        if ans.lower() in ("a", "accept", "y", "yes"):
            ans = "Add the numbered '## Build steps' section."
        messages.append({"role": "user", "content": f"Revise the design: {ans}\n"
                                                    "Reply with the complete revised design in the same structure."})


# ---------------------------------------------------------------------------
# Building steps through the factory
# ---------------------------------------------------------------------------
def show_plan(design) -> None:
    if not design:
        sub("no design yet - try /design <idea>")
        return
    bullet(design.title, f"({design.path.name})")
    nxt = design.next_step()
    for i, step in enumerate(design.steps, 1):
        mark, style = ("✓", "green") if i in design.done else (("▶", ACCENT) if i == nxt else ("·", "dim"))
        console.print(Text.assemble(f"   {mark} ", (f"{i}. ", "bold"), (step, "" if i != nxt else "bold")), style=style)


def build_step(factory, graph, view, design, n: int) -> bool:
    step = design.steps[n - 1]
    console.print()
    bullet(f"Step {n}/{len(design.steps)}", step[:110] + ("…" if len(step) > 110 else ""), ACCENT)
    final = run_task(factory, graph, view, step, notes=design.notes_for(n)) or {}
    passed = all(c["ok"] for c in final.get("checks", [])) and bool(final.get("checks"))
    if final.get("written") and passed:
        design.done.add(n)
        design.save_status()
        sub(f"step {n} done")
        return True
    if final.get("written"):
        # You approved it despite failing checks: saved, but later steps would build on broken code.
        sub(f"step {n} was saved with failing checks - fix it with a normal prompt, then /skip {n}", "yellow")
    else:
        sub(f"step {n} not saved - fix it with a normal prompt, re-run /build {n}, or /skip {n}", "yellow")
    return False


def main() -> None:
    with console.status("[dim]Loading Laya, the index, LangGraph and the MCP server…", spinner="dots",
                        spinner_style=ACCENT):
        import factory

        tools = factory.mcp().tool_names()
        factory.load_laya()
        factory.get_index()
        graph = factory.build_graph()

    designs = saved_designs()
    design = Design(designs[-1]) if designs else None
    think_mode = "auto"
    view = View(factory)
    console.print(Panel(Text.assemble(
        ("✻ ", ACCENT), ("Dev Console", "bold"), ("  design → build\n\n", "dim"),
        ("  coder      ", "dim"), f"{factory.LLM_MODEL} · router laya · mcp {', '.join(tools)}\n",
        ("  workspace  ", "dim"), f"{factory.WORKSPACE} · {len(factory.mcp_call('list_files'))} files\n",
        ("  design     ", "dim"), (f"{design.title} ({len(design.done)}/{len(design.steps)} steps done)" if design else "none yet"), "\n\n",
        ("  /design <idea>  ·  /plan  ·  /build [all|n]  ·  /help", "dim"),
    ), border_style=ACCENT, expand=False, padding=(0, 2)))

    while True:
        try:
            line = console.input(f"\n[bold {ACCENT}]>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        try:
            if cmd in ("/exit", "/quit"):
                break
            elif cmd == "/help":
                console.print(__doc__.split("\n\n", 2)[1], style="dim", highlight=False, markup=False)
            elif cmd == "/design":
                if not arg:
                    sub("usage: /design <what you want to build>")
                    continue
                design = design_flow(factory, arg, think_mode) or design
            elif cmd == "/plan":
                show_plan(design)
            elif cmd == "/build":
                if not design:
                    sub("no design yet - try /design <idea>")
                elif arg == "all":
                    while (n := design.next_step()) is not None:
                        if not build_step(factory, graph, view, design, n):
                            break
                    else:
                        bullet("All steps done", design.title, "green")
                elif arg:
                    if arg.isdigit() and 1 <= int(arg) <= len(design.steps):
                        build_step(factory, graph, view, design, int(arg))
                    else:
                        sub(f"pick a step between 1 and {len(design.steps)}")
                elif (n := design.next_step()) is not None:
                    build_step(factory, graph, view, design, n)
                else:
                    sub("every step is done")
            elif cmd == "/skip":
                if design and arg.isdigit() and 1 <= int(arg) <= len(design.steps):
                    design.done.add(int(arg))
                    design.save_status()
                    show_plan(design)
                else:
                    sub("usage: /skip <step number>")
            elif cmd == "/designs":
                for i, p in enumerate(saved_designs(), 1):
                    sub(f"{i}. {p.stem}" + ("   ← current" if design and p == design.path else ""))
            elif cmd == "/open":
                paths = saved_designs()
                if arg.isdigit() and 1 <= int(arg) <= len(paths):
                    design = Design(paths[int(arg) - 1])
                    show_plan(design)
                else:
                    sub("usage: /open <number from /designs>")
            elif cmd == "/think":
                if arg in ("on", "off", "auto"):
                    think_mode = arg
                sub(f"design reasoning: {think_mode}")
            elif cmd == "/files":
                sub("\n    ".join(factory.mcp_call("list_files")) or "(empty)")
            elif cmd.startswith("/"):
                sub(f"unknown command {cmd} - try /help")
            else:
                run_task(factory, graph, view, line)
        except KeyboardInterrupt:
            view.stop()
            sub("cancelled")
        except Exception as e:  # keep the console alive on Ollama/MCP errors
            view.stop()
            console.print(f"  ⎿ error: {e}", style="red", markup=False)
            if isinstance(e, factory.MCPConnectionLost):
                console.print("    the MCP server stopped; it restarts automatically on the next command", style="dim")
            elif isinstance(e, ConnectionError):
                console.print("    is Ollama running?  try: ollama serve", style="dim")

    console.print("\n[dim]bye[/]")


if __name__ == "__main__":
    main()
