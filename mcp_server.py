"""
Local MCP server for the software factory: the only component that writes
to the workspace or executes generated code.

    python mcp_server.py --workspace PATH     (normally spawned by factory.py over stdio)

Tools
  list_files()          files in the workspace
  validate(files)       stage files in a fresh sandbox copy of the workspace and run
                        path check -> ast -> placeholder scan -> ruff -> pytest
  commit(sandbox_id)    copy the validated files from that sandbox into the workspace
  discard(sandbox_id)   delete a sandbox

What the process boundary buys: a hung test or a crash here can't freeze or
kill the LangGraph process, and every pytest run has a timeout. What it does
NOT buy: security isolation. Tests run as your Windows user and can reach the
network or any file your account can - review what you approve.
"""

import argparse
import ast
import atexit
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

TEST_TIMEOUT_S = 60
SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache"}
PLACEHOLDER_RE = re.compile(
    r"#\s*(TODO|FIXME)\b|implement (this|later|me)|raise NotImplementedError|^\s*\.\.\.\s*$",
    re.IGNORECASE | re.MULTILINE,
)

WORKSPACE = Path("workspace").resolve()
SANDBOX_ROOT = Path(tempfile.mkdtemp(prefix="factory_sandboxes_"))
atexit.register(shutil.rmtree, SANDBOX_ROOT, True)
_staged: Dict[str, Dict[str, str]] = {}

server = MCPServer(
    "factory-workspace",
    instructions="Validates generated code in a sandbox copy of the workspace and commits approved files.",
    log_level="WARNING",
)


def _safe_path(root: Path, rel: str) -> Optional[Path]:
    """Resolve a proposed relative path under root; None if it escapes it."""
    rel = rel.strip()
    if not rel or os.path.isabs(rel) or Path(rel).drive:
        return None
    target = (root / rel).resolve()
    if target == root or not target.is_relative_to(root):
        return None
    return target


def _is_test(rel: str) -> bool:
    name = Path(rel).name
    return name.startswith("test_") or name.endswith("_test.py")


def _check(name: str, errors: List[str], ok_detail: str = "") -> dict:
    return {"name": name, "ok": not errors, "detail": "\n".join(errors) if errors else ok_detail}


def _run_checks(sandbox: Path, files: Dict[str, str]) -> List[dict]:
    """Runs in order and stops at the first failing stage, so the model gets one clear problem at a time."""
    py = {rel: src for rel, src in files.items() if rel.endswith(".py")}

    errs = []
    for rel, src in py.items():
        try:
            ast.parse(src, filename=rel)
        except SyntaxError as e:
            # Show the offending line and a caret: with only "line 5: unterminated string"
            # Qwen resent the identical broken line on every retry.
            where = f"\n    {e.text.rstrip()}\n    {' ' * max((e.offset or 1) - 1, 0)}^" if e.text else ""
            errs.append(f"{rel}:{e.lineno}: SyntaxError: {e.msg}{where}")
    results = [_check("syntax", errs)]
    if errs:
        return results

    errs = [
        f"{rel}: placeholder '{m.group(0).strip()}' - write the complete implementation"
        for rel, src in py.items()
        for m in [PLACEHOLDER_RE.search(src)]
        if m
    ]
    results.append(_check("placeholders", errs))
    if errs:
        return results

    if py:
        lint = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--isolated", "--no-cache",
             "--select", "E9,F63,F7,F82,F811", "--output-format", "concise", *py],
            cwd=sandbox, capture_output=True, text=True, stdin=subprocess.DEVNULL,
        )
        results.append(_check("ruff", [lint.stdout.strip()] if lint.returncode else []))
        if lint.returncode:
            return results

    tests = [rel for rel in py if _is_test(rel)]
    if not tests:
        # No test in the draft: fall back to tests already on disk for the modules it changed.
        for rel in py:
            p = Path(rel)
            for cand in (p.with_name(f"test_{p.name}"), Path("tests") / f"test_{p.name}"):
                if (sandbox / cand).is_file():
                    tests.append(cand.as_posix())
    if not tests:
        results.append(_check("pytest", ["No test file was written. Include a test_*.py file with pytest tests."]))
        return results
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "--tb=short", *tests],
            cwd=sandbox, capture_output=True, text=True, timeout=TEST_TIMEOUT_S,
            stdin=subprocess.DEVNULL,  # never let tests inherit the JSON-RPC pipe on our stdin
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-40:])
        if proc.returncode == 0:
            results.append(_check("pytest", [], tail.splitlines()[-1] if tail else ""))
        else:
            results.append(_check("pytest", [tail]))
    except subprocess.TimeoutExpired:
        results.append(_check("pytest", [f"Tests timed out after {TEST_TIMEOUT_S}s (infinite loop or blocking call?)"]))
    return results


def _diffs(files: Dict[str, str]) -> List[dict]:
    out = []
    for rel, new in files.items():
        path = _safe_path(WORKSPACE, rel)
        old = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True), new.splitlines(keepends=True), f"a/{rel}", f"b/{rel}"
        ))
        out.append({"path": rel, "diff": diff, "is_new": not old})
    return out


@server.tool()
def list_files() -> List[str]:
    """List files in the workspace (relative paths)."""
    return sorted(
        p.relative_to(WORKSPACE).as_posix()
        for p in WORKSPACE.rglob("*")
        if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(WORKSPACE).parts)
    )


def _trim_blank(lines: List[str]) -> List[str]:
    while lines and not lines[0].strip():
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines = lines[:-1]
    return lines


def _find_block(lines: List[str], key: List[str], norm) -> List[int]:
    key = [norm(k) for k in key]
    n = len(key)
    return [i for i in range(len(lines) - n + 1) if [norm(x) for x in lines[i:i + n]] == key]


def _find_ignoring_blank(lines: List[str], search: List[str]) -> List[tuple]:
    """(start, end) spans whose non-blank lines equal search's non-blank lines."""
    key = [x.rstrip() for x in search if x.strip()]
    idx = [i for i, x in enumerate(lines) if x.strip()]
    vals = [lines[i].rstrip() for i in idx]
    n = len(key)
    if not n:
        return []
    return [(idx[j], idx[j + n - 1] + 1) for j in range(len(vals) - n + 1) if vals[j:j + n] == key]


def _indent(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _common_indent(lines: List[str]) -> str:
    indents = [_indent(x) for x in lines if x.strip()]
    return min(indents, key=len) if indents else ""


def _shift_match(lines: List[str], s_lines: List[str], r_lines: List[str]) -> tuple:
    """
    Match with a uniform indentation offset (the model indented the whole block
    one level too deep/shallow). Relative indentation must still match exactly;
    REPLACE is re-indented by the same offset. Returns (spans, new_replace_lines).
    """
    s_base = _common_indent(s_lines)
    body = [x[len(s_base):].rstrip() if x.strip() else "" for x in s_lines]
    key = [x for x in body if x]
    if not key:  # an all-blank SEARCH would otherwise "match" everywhere and index past the end
        return [], []
    idx = [i for i, x in enumerate(lines) if x.strip()]
    spans, f_base = [], ""
    for j in range(len(idx) - len(key) + 1):
        window = [lines[i] for i in idx[j:j + len(key)]]
        base = _common_indent(window)
        if [w[len(base):].rstrip() for w in window] == key:
            spans.append((idx[j], idx[j + len(key) - 1] + 1))
            f_base = base
    replaced = [f_base + (x[len(s_base):] if x.startswith(s_base) else x.lstrip()) if x.strip() else ""
                for x in _trim_blank(r_lines)]
    return spans, replaced


def _closest(lines: List[str], search: List[str]) -> str:
    """The window of the file that looks most like the failed SEARCH text, to show the model."""
    n, target, best, best_i = max(len(search), 1), "\n".join(search), 0.0, -1
    for i in range(max(len(lines) - n + 1, 1)):
        m = difflib.SequenceMatcher(None, "\n".join(lines[i:i + n]), target)
        if m.quick_ratio() > best and m.ratio() > best:
            best, best_i = m.ratio(), i
    if best_i < 0 or best < 0.5:
        return ""
    return "\n".join(lines[best_i:best_i + n])


def apply_edit(text: str, search: str, replace: str) -> tuple:
    """
    Replace the ONE place in text whose lines equal search's lines. Returns
    (new_text, None) or (None, error). Matching is line-based so "return 1"
    can never hit inside "return 10". Passes, strictest first: exact; ignoring
    trailing whitespace and blank edge lines; ignoring blank lines entirely
    (Qwen 8B routinely drops blank lines when copying code); a uniform
    indentation offset of the whole block. Relative indentation is never
    relaxed. Zero or 2+ matches is an error - picking one of several would
    silently edit the wrong code.
    """
    lines = text.split("\n")
    s_lines, r_lines = search.split("\n"), replace.split("\n") if replace else []
    for norm, s, r in ((lambda x: x, s_lines, r_lines),
                       (str.rstrip, _trim_blank(s_lines), _trim_blank(r_lines))):
        if not s:
            continue
        hits = _find_block(lines, s, norm)
        if len(hits) == 1:
            i = hits[0]
            return "\n".join(lines[:i] + r + lines[i + len(s):]), None
        if len(hits) > 1:
            return None, (f"SEARCH matches {len(hits)} places. Add more unchanged lines around it "
                          "so it matches exactly one place.")
    spans = _find_ignoring_blank(lines, s_lines)
    if len(spans) == 1:
        start, end = spans[0]
        return "\n".join(lines[:start] + _trim_blank(r_lines) + lines[end:]), None
    if len(spans) > 1:
        return None, (f"SEARCH matches {len(spans)} places. Add more unchanged lines around it "
                      "so it matches exactly one place.")
    spans, shifted = _shift_match(lines, s_lines, r_lines)
    if len(spans) == 1:
        start, end = spans[0]
        return "\n".join(lines[:start] + shifted + lines[end:]), None
    if len(spans) > 1:
        return None, (f"SEARCH matches {len(spans)} places. Add more unchanged lines around it "
                      "so it matches exactly one place.")
    if not _trim_blank(s_lines):
        return None, "SEARCH is empty. Copy the exact existing lines you want to change."
    near = _closest(lines, _trim_blank(s_lines))
    hint = f"\nThe closest text in the file is:\n{near}" if near else ""
    return None, f"SEARCH text not found in the file. Copy it exactly from the current file.{hint}"


def _region(text: str, replace: str, pad: int = 3) -> str:
    """Current text around an applied edit, so the next SEARCH targets post-edit code."""
    lines, r = text.split("\n"), _trim_blank(replace.split("\n"))
    hits = _find_block(lines, r, str.rstrip) if r else []
    if len(hits) != 1:
        return ""
    i = hits[0]
    return "\n".join(lines[max(i - pad, 0):i + len(r) + pad])


@server.tool()
def validate(files: Optional[Dict[str, str]] = None, edits: Optional[List[Dict[str, str]]] = None) -> dict:
    """
    Stage a draft in a fresh sandbox copy of the workspace and run the checks
    there. Nothing in the workspace changes.

    files: whole-file contents (relative path -> text), e.g. new files or the
           current draft of files changed by earlier attempts.
    edits: [{path, search, replace}] applied in order on top of `files` (or the
           workspace copy). Any edit that doesn't apply short-circuits the checks.

    Returns sandbox_id (for commit/discard), passed, checks, diffs, the
    resulting full `files`, `applied` regions and `failed_paths`.
    """
    files, edits = dict(files or {}), list(edits or [])
    if not files and not edits:
        return {"sandbox_id": None, "passed": False, "diffs": [], "files": {}, "applied": [], "failed_paths": [],
                "checks": [_check("format", ["No FILE or EDIT blocks found. Use the exact formats from the instructions."])]}
    bad = sorted({rel for rel in [*files, *(e["path"] for e in edits)] if _safe_path(WORKSPACE, rel) is None})
    if bad:
        return {"sandbox_id": None, "passed": False, "diffs": [], "files": files, "applied": [], "failed_paths": [],
                "checks": [_check("paths", [f"Paths outside the workspace are not allowed: {bad}"])]}

    sandbox_id = uuid.uuid4().hex[:12]
    sandbox = SANDBOX_ROOT / sandbox_id
    shutil.copytree(WORKSPACE, sandbox, ignore=shutil.ignore_patterns(*SKIP_DIRS))
    for rel, src in files.items():
        dest = _safe_path(sandbox, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src, encoding="utf-8")

    errors, failed_paths, applied = [], [], []
    for n, e in enumerate(edits, 1):
        rel, dest = e["path"], _safe_path(sandbox, e["path"])
        if not dest.exists():
            if e["search"].strip():
                new, err = None, "File does not exist. Create new files with a FILE: block."
            else:
                new, err = e["replace"], None
        else:
            try:
                current = dest.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                current = None
            if current is None:
                new, err = None, "File is not UTF-8 text, so it can't be edited safely. Leave it unchanged."
            else:
                new, err = apply_edit(current, e["search"], e["replace"])
        if err:
            errors.append(f"Edit {n} in {rel}: {err}")
            failed_paths.append(rel)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(new, encoding="utf-8")
        files[rel] = new
        applied.append({"path": rel, "replace": e["replace"]})

    applied = [{"path": a["path"], "region": _region(files[a["path"]], a["replace"])} for a in applied]
    if errors:
        shutil.rmtree(sandbox, ignore_errors=True)
        return {"sandbox_id": None, "passed": False, "diffs": _diffs(files), "files": files,
                "applied": applied, "failed_paths": sorted(set(failed_paths)),
                "checks": [_check("edits", errors)]}

    _staged[sandbox_id] = dict(files)
    checks = _run_checks(sandbox, files)
    return {"sandbox_id": sandbox_id, "passed": all(c["ok"] for c in checks), "checks": checks,
            "diffs": _diffs(files), "files": files, "applied": applied, "failed_paths": []}


@server.tool()
def commit(sandbox_id: str) -> dict:
    """Copy the files validated in sandbox_id into the workspace, then delete the sandbox."""
    if sandbox_id not in _staged:
        raise ToolError(f"Unknown or already used sandbox_id: {sandbox_id}")
    sandbox = SANDBOX_ROOT / sandbox_id
    written = []
    for rel in _staged.pop(sandbox_id):
        dest = _safe_path(WORKSPACE, rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(_safe_path(sandbox, rel), dest)
        written.append(rel)
    shutil.rmtree(sandbox, ignore_errors=True)
    return {"written": written}


@server.tool()
def discard(sandbox_id: str) -> dict:
    """Delete a sandbox without touching the workspace."""
    existed = _staged.pop(sandbox_id, None) is not None
    shutil.rmtree(SANDBOX_ROOT / sandbox_id, ignore_errors=True)
    return {"discarded": existed}


def main() -> None:
    global WORKSPACE
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=os.environ.get("WORKSPACE", "workspace"))
    WORKSPACE = Path(parser.parse_args().workspace).resolve()
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    server.run("stdio")


if __name__ == "__main__":
    main()
