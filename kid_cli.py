"""
Robo: a kid-friendly terminal front end for the software factory.

    python kid_cli.py

A child types what they want ("make a game where I guess a number"); Robo
builds it with the same Laya -> LangGraph -> Qwen -> MCP pipeline as
factory_cli.py, but talks in plain words, answers with single key presses,
hides technical detail unless a grown-up asks for it, and can run what it
made. Nothing is saved without a "keep it" key press, and code that failed
its checks can't be kept from this screen at all.
"""

import os
import subprocess
import sys
import time

from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.syntax import Syntax
from rich.text import Text

from factory_cli import run_task

console = Console()

FACES = {
    "hello": ("[^_^]", "bright_cyan"),
    "think": ("[o_o]", "yellow"),
    "work": ("[-_-]", "yellow"),
    "happy": ("[^o^]", "bright_green"),
    "oops": ("[;_;]", "bright_red"),
}
KID_ANSWER_PROMPT = (
    "You are Robo, a friendly robot talking to a 5-year-old. Answer in at most 4 short sentences, "
    "using simple everyday words and one fun, concrete example from a child's life. "
    "No jargon; if you must use a computer word, explain it in the same sentence."
)
EXIT_WORDS = {"bye", "goodbye", "stop", "exit", "quit", "/exit"}
EXAMPLES = [
    "make a game where I guess a number",
    "make a program that tells me a joke",
    "make a times table for the number 7",
    "what is a loop?",
]


def face(mood: str) -> Text:
    art, color = FACES[mood]
    return Text(art, style=f"bold {color}")


def say(mood: str, message: str, style: str = "") -> None:
    console.print(Text.assemble("  ", face(mood), "  ", (message, style or "bold")))


def get_key(prompt: Text, keys: str, default: str) -> str:
    """One key press, no Enter needed. Falls back to a typed line when input is piped (tests)."""
    console.print(prompt, end="")
    if sys.stdin.isatty() and os.name == "nt":
        import msvcrt

        while True:
            ch = msvcrt.getwch().lower()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in keys:
                console.print(ch.upper())
                return ch
    while True:
        try:
            line = input().strip().lower()
        except EOFError:
            return default
        if line[:1] in keys:
            return line[:1]


def button(key: str, label: str, color: str) -> Text:
    return Text.assemble((f" {key} ", f"bold black on {color}"), (f" {label}   ", "bold"))


class Status:
    """The one live line while Robo works: face, spinner, what it's doing, how long it's taken."""

    def __init__(self):
        self.mood, self.message, self.words, self.started = "think", "", 0, time.monotonic()
        self.spinner = Spinner("dots", style="yellow")

    def set(self, mood: str, message: str) -> None:
        self.mood, self.message, self.words = mood, message, 0

    def __rich__(self):
        secs = int(time.monotonic() - self.started)
        clock = f"{secs // 60} min {secs % 60} s" if secs >= 60 else f"{secs} s"
        extra = f"  {self.words} words written" if self.words else ""
        self.spinner.update(text=Text.assemble("  ", face(self.mood), "  ", (self.message, "bold"),
                                               (f"   {clock}{extra}", "dim")))
        lines = [self.spinner]
        if secs > 90:
            lines.append(Text("        This takes a few minutes. You can get a snack!", style="dim"))
        return Group(*lines)


class KidView:
    """Same interface as factory_cli.View (handle / review / stop), in kid language."""

    def __init__(self, factory):
        self.f = factory
        self.live = None
        self.status = Status()
        self.answer = ""
        self.phase = ""
        self.written = []

    def start_task(self) -> None:
        self.status = Status()
        self.written = []

    def _live(self):
        if self.live is None:
            self.live = Live(self.status, console=console, refresh_per_second=8, transient=True)
            self.live.start()
        return self.live

    def stop(self) -> None:
        if self.live:
            self.live.stop()
            self.live = None

    def handle(self, ev: dict) -> None:
        t = ev["type"]
        if t == "triage":
            if ev["kind"] == "question":
                self.status.set("think", "Thinking about your question...")
            else:
                self.status.set("think", "Thinking about what to build...")
                if ev["think"]:
                    self.stop()
                    say("think", "This is a big job, so I'll think extra hard. It may take a while.")
            self._live()
        elif t == "retrieve":
            if ev["files"]:
                self.status.set("think", "Looking at your files...")
            self._live()
        elif t == "stream_start":
            self.phase, self.answer = ev["phase"], ""
            if self.phase == "answer":
                self.status.set("think", "Thinking of an answer...")
            elif ev["attempt"] == 1:
                self.status.set("work", "Writing the code...")
            else:
                self.status.set("work", f"Fixing it (try {ev['attempt']} of {ev['max_attempts']})...")
            self._live()
        elif t == "thinking":
            self.status.message = "Thinking really hard..."
        elif t == "token":
            self.status.words += 1
            if self.phase == "answer":
                self.answer += ev["text"]
        elif t == "stream_end":
            if self.phase == "answer":
                self.stop()
                console.print(Panel(Markdown(self.answer or "Hmm, I don't know."), title=Text.assemble(face("happy"), " Robo says"),
                                    title_align="left", border_style="bright_cyan", padding=(1, 2)))
            else:
                self.status.set("work", "Checking my work...")
        elif t == "check":
            passed = all(c["ok"] for c in ev["checks"])
            self.stop()
            if passed:
                say("happy", "I tested it, and it works!", "bold bright_green")
            elif ev["attempt"] < self.f.MAX_ATTEMPTS:
                say("oops", "I found a little mistake. Let me fix it.", "bold yellow")
        elif t == "written":
            self.stop()
            self.written = ev["files"]
            if ev["files"]:
                say("happy", "Saved! " + ", ".join(ev["files"]), "bold bright_green")

    def _summary(self, diffs) -> Text:
        out = Text()
        for d in diffs:
            added = sum(1 for ln in d["diff"].splitlines() if ln.startswith("+") and not ln.startswith("+++"))
            removed = sum(1 for ln in d["diff"].splitlines() if ln.startswith("-") and not ln.startswith("---"))
            what = "a new file" if d["is_new"] else f"{added} line(s) added, {removed} removed"
            out.append(f"  - {d['path']}", style="bold")
            out.append(f"   ({what})\n", style="dim")
        return out

    def _show_code(self, diffs) -> None:
        for d in diffs:
            console.print(Panel(Syntax(d["diff"] or "(no changes)", "diff", theme="ansi_dark", word_wrap=True),
                                title=d["path"], title_align="left", border_style="dim"))

    def _ask_change(self) -> str:
        while True:
            try:
                text = console.input(Text.assemble("\n  ", face("think"), ("  What should I change? ", "bold"))).strip()
            except EOFError:
                return "reject"
            if text:
                return text

    def review(self, payload: dict) -> str:
        self.stop()
        diffs, passed = payload["diffs"], payload["passed"]
        if passed:
            body = Group(Text.assemble(face("happy"), ("  Here's what I made:\n", "bold")), self._summary(diffs))
            console.print(Panel(body, border_style="bright_green", padding=(1, 2)))
            keys, prompt = "ynts", Text.assemble("  ", button("Y", "Keep it", "bright_green"),
                                                 button("N", "Throw it away", "bright_red"),
                                                 button("T", "Tell me what to change", "bright_yellow"),
                                                 button("S", "Show the code", "bright_blue"), "\n  ")
        else:
            body = Text.assemble(face("oops"), ("  I tried 3 times, but I couldn't make it work.\n", "bold"),
                                 ("  You can tell me more about what you want, and I'll try again.", ""))
            console.print(Panel(body, border_style="bright_red", padding=(1, 2)))
            keys, prompt = "tns", Text.assemble("  ", button("T", "Tell me more", "bright_yellow"),
                                                button("N", "Throw it away", "bright_red"),
                                                button("S", "Show the code", "bright_blue"), "\n  ")
        while True:
            key = get_key(prompt, keys, default="n")
            if key == "s":
                self._show_code(diffs)
                continue
            if key == "y":
                return "approve"
            if key == "n":
                say("hello", "Okay, I threw it away.")
                return "reject"
            answer = self._ask_change()
            if answer != "reject":
                say("work", "Got it! Let me change that.")
            return answer


def runnable(factory, files) -> list:
    out = []
    for rel in files:
        path = factory.WORKSPACE / rel
        if rel.endswith(".py") and not os.path.basename(rel).startswith("test_") and path.exists():
            if "__main__" in path.read_text(encoding="utf-8", errors="replace"):
                out.append(rel)
    return out


def try_it(factory, rel: str) -> None:
    console.print(Panel(Text(f"Running {rel}!  (Press Ctrl+C to stop it.)", style="bold"),
                        border_style="bright_magenta", padding=(0, 2)))
    try:
        subprocess.run([sys.executable, rel], cwd=factory.WORKSPACE)
    except KeyboardInterrupt:
        pass
    say("happy", "That was fun!")


def welcome() -> None:
    lines = Text.assemble(
        face("hello"), ("  Hi! I'm Robo.\n\n", "bold bright_cyan"),
        ("  Tell me what to make, and I'll build it for you.\n", "bold"),
        ("  You can also ask me a question.\n\n", ""),
        ("  Try saying:\n", "dim"),
        *[(f'    "{e}"\n', "italic") for e in EXAMPLES],
        ("\n  Type ", "dim"), ("bye", "bold"), (" when you're done.", "dim"),
    )
    console.print(Panel(Align.left(lines), border_style="bright_cyan", padding=(1, 3)))


def main() -> None:
    with console.status(Text.assemble("  ", face("think"), ("  Waking up Robo... (about a minute)", "bold")),
                        spinner="dots"):
        import factory

        factory.ANSWER_PROMPT = KID_ANSWER_PROMPT  # answer_node reads this at call time
        factory.mcp()
        factory.load_laya()
        factory.get_index()
        graph = factory.build_graph()

    welcome()
    view = KidView(factory)
    while True:
        try:
            task = console.input(Text.assemble("\n  ", face("hello"), ("  What should I make? ", "bold bright_cyan"))).strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not task:
            continue
        low = task.lower().strip(" .!")
        if low in EXIT_WORDS:
            break
        if low in {"help", "/help", "?"}:
            welcome()
            continue
        if low in {"files", "my files", "what did you make", "what did you make?"}:
            names = factory.mcp_call("list_files")
            say("happy", "Here are your files:" if names else "We haven't made anything yet!")
            for n in names:
                console.print(f"      {n}", style="bold")
            continue

        view.start_task()
        try:
            run_task(factory, graph, view, task)
        except KeyboardInterrupt:
            view.stop()
            say("hello", "Okay, I stopped.")
            continue
        except Exception as e:  # keep Robo running whatever breaks
            view.stop()
            if isinstance(e, factory.MCPConnectionLost):
                say("oops", "I dropped my toolbox for a second. Please ask me again!", "bold bright_red")
            elif isinstance(e, ConnectionError):
                say("oops", "My brain isn't answering. Ask a grown-up to start Ollama.", "bold bright_red")
            else:
                say("oops", "Something went wrong. A grown-up can look in logs/factory.log.", "bold bright_red")
                console.print(f"      ({type(e).__name__}: {e})", style="dim", markup=False)
            continue

        games = runnable(factory, view.written)
        if games:
            key = get_key(Text.assemble("\n  ", button("P", f"Try {games[0]} now", "bright_magenta"),
                                        button("M", "Make something else", "bright_cyan"), "\n  "), "pm", default="m")
            if key == "p":
                try_it(factory, games[0])

    say("hello", "Bye! See you next time!", "bold bright_cyan")


if __name__ == "__main__":
    main()
