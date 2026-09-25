"""
Offline tests: the SEARCH/REPLACE matcher, the reply parser and the design parser.
No Ollama, Laya or network needed.

    pytest tests
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session", autouse=True)
def _workspace(tmp_path_factory):
    os.environ["WORKSPACE"] = str(tmp_path_factory.mktemp("workspace"))


SRC = (
    "def a():\n    return 1\n\ndef b():\n    return 10\n\n"
    "def c():\n    x = 1\n    return x\n\ndef d():\n    x = 1\n    return x\n"
)


def apply(search, replace):
    from mcp_server import apply_edit

    return apply_edit(SRC, search, replace)


@pytest.mark.parametrize("name,search,replace,applies", [
    ("exact", "def a():\n    return 1", "def a():\n    return 2", True),
    ("trailing whitespace", "def b():   \n    return 10  ", "def b():\n    return 11", True),
    ("blank edge lines", "\n\ndef b():\n    return 10\n\n", "def b():\n    return 11", True),
    ("blank lines dropped", "def b():\n    return 10\ndef c():", "def b():\n    return 11\n\ndef c():", True),
    ("uniform indent shift", "    def a():\n        return 1", "    def a():\n        return 5", True),
    ("deletion", "def c():\n    x = 1\n    return x\n", "", True),
    ("ambiguous", "    x = 1\n    return x", "    x = 2\n    return x", False),
    ("relative indent wrong", "def a():\nreturn 1", "def a():\nreturn 2", False),
    ("not found", "def b():\n    return 100", "x", False),
    ("empty", "\n", "x", False),
    ("all blank", "\n\n   \n", "x", False),
])
def test_matcher(name, search, replace, applies):
    new, err = apply(search, replace)
    assert (err is None) == applies, err


def test_matcher_never_edits_inside_a_line():
    new, err = apply("    return 1", "    return 99")
    assert err is None and "return 10" in new


def test_indent_shift_reindents_replacement():
    new, _ = apply("    def a():\n        return 1", "    def a():\n        return 5")
    assert new.startswith("def a():\n    return 5")


def test_not_found_shows_closest_text():
    _, err = apply("def b():\n    return 100", "x")
    assert "return 10" in err


def test_parse_output_handles_qwen_variations():
    from factory import parse_output

    reply = (
        "EDIT: lexer.py\n```python\n<<<<<<<\ndef quote(s):\n=======\ndef quote(s):  # x\n>>>>>>>\n```\n\n"
        "**EDIT: `lexer.py`**\n<<<<<<< SEARCH\nimport re\n=======\nimport re\nimport sys\n>>>>>>> REPLACE\n\n"
        "FILE: test_lexer.py (new file)\n```python\ndef test_x():\n    assert True\n```\n"
        "EDIT: never_closed.py\n<<<<<<< SEARCH\nx\n=======\ny\n"
    )
    files, edits = parse_output(reply)
    assert list(files) == ["test_lexer.py"]
    assert [e["path"] for e in edits] == ["lexer.py", "lexer.py"]
    assert not any("```" in e["search"] + e["replace"] for e in edits)


def test_design_steps_and_status(tmp_path):
    import dev_cli

    dev_cli.DESIGN_DIR = tmp_path
    md = (
        "Sure!\n\n# Todo Tracker\n\n## Goal\nA todo list.\n\n## File layout\n- `store.py` - storage\n\n"
        "## Build steps\n1. Create `store.py` with load/save;\n   test it in test_store.py.\n"
        "2) Add add_task to store.py.\n3. Create main.py.\n"
    )
    design = dev_cli.save_design(md, "todo")
    assert design.title == "Todo Tracker"
    assert design.steps[0].endswith("test it in test_store.py.")
    assert len(design.steps) == 3
    design.done.add(1)
    design.save_status()
    assert dev_cli.Design(design.path).next_step() == 2
    assert dev_cli.save_design(md, "todo").path.name == "todo-tracker-2.md"
