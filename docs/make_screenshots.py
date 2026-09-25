"""
Regenerate docs/robo.svg, the README's Robo screenshot.

    python docs/make_screenshots.py

It drives Robo's real rendering code (kid_cli.py) with a recording console, so
the image shows exactly what Robo draws. The content is from real test runs:
the joke program Robo built, the joke it told, and its answer to
"what is a robot?". No models are needed.
"""

import sys
import types
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import kid_cli  # noqa: E402

rec = Console(record=True, width=92, force_terminal=True, color_system="truecolor")
kid_cli.console = rec
face, button, say = kid_cli.face, kid_cli.button, kid_cli.say


def asks(text: str) -> None:
    rec.print(Text.assemble("\n  ", face("hello"), ("  What should I make? ", "bold bright_cyan"), text))


kid_cli.welcome()

asks("make a program that tells me a joke")
status = kid_cli.Status()
status.set("work", "Writing the code...")
status.words, status.started = 84, status.started - 72
rec.print(status)
say("oops", "I found a little mistake. Let me fix it.", "bold yellow")
say("happy", "I tested it, and it works!", "bold bright_green")

view = kid_cli.KidView(types.SimpleNamespace(MAX_ATTEMPTS=3, WORKSPACE=ROOT))
kid_cli.get_key = lambda prompt, keys, default: (rec.print(prompt, end=""), rec.print("Y"), "y")[-1]
view.review({"passed": True, "checks": [], "diffs": [
    {"path": "joke_teller.py", "diff": "", "is_new": True},
    {"path": "test_joke_teller.py", "diff": "", "is_new": True},
]})
say("happy", "Saved! joke_teller.py, test_joke_teller.py", "bold bright_green")

rec.print(Text.assemble("\n  ", button("P", "Try joke_teller.py now", "bright_magenta"),
                        button("M", "Make something else", "bright_cyan"), "\n  P"))
rec.print(Panel(Text("Running joke_teller.py!  (Press Ctrl+C to stop it.)", style="bold"),
                border_style="bright_magenta", padding=(0, 2)))
rec.print("Why did the scarecrow win an award? Because he was outstanding in his field!")
say("happy", "That was fun!")

asks("what is a robot?")
view.phase = "answer"
view.answer = ("A robot is a machine that can move and do things by itself. Like when you ride a toy car, "
               "it moves without you pushing it. Robots can help people do jobs, just like how your toy helps "
               "you play. Some robots even talk, like the ones in movies!")
view.handle({"type": "stream_end", "phase": "answer", "tokens": 0, "seconds": 0})

out = ROOT / "docs" / "robo.svg"
rec.save_svg(str(out), title="Robo")
print(f"wrote {out}")
