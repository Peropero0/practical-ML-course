"""Runs the notebooks and checks that they behave as expected:

  * lessons and solutions: they must run from start to finish, without errors;
  * exercises with start "setup_only": the setup cell (imports, datasets) must work;
  * exercises with start "must_fail": the starter code runs, but the check cell MUST fail
    (otherwise the check cannot tell wrong code from right code).

With --numbers, solutions are also checked for stale figures: the decimals quoted in "What you learned" are compared
with the numbers printed by the executed notebook, and the ones that cannot be found are listed (information only: derived
figures such as differences or percentages legitimately do not appear in the output).

Usage:  python _tools/check_notebooks.py module_01_linear_regression [filter] [--numbers]
"""
import os
import re
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import List, Tuple

import nbformat
from nbclient import NotebookClient

# abspath and not resolve(): if the course sits in a very long path (over 260 characters on Windows) you can
# launch this script through a short-path junction (e.g. C:\mlp) and the kernels still start.
ROOT = Path(os.path.abspath(__file__)).parent.parent


def _execute(nb: nbformat.NotebookNode, path: Path, allow_errors: bool) -> nbformat.NotebookNode:
    """Run the notebook in a fresh kernel started in its own folder. Returns the executed notebook."""
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=allow_errors,
                            resources={"metadata": {"path": str(path.parent)}})
    client.execute()
    return nb


def _errors(nb: nbformat.NotebookNode) -> List[Tuple[int, List[str], str, str]]:
    """The error outputs of an executed notebook, as (cell index, cell tags, error name, error value)."""
    found = []
    for i, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                found.append((i, cell.get("metadata", {}).get("tags", []), out["ename"], out["evalue"]))
    return found


def _truncate_to_setup(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    """Copy of the notebook cut right after the cell tagged `setup`."""
    new = deepcopy(nb)
    for i, cell in enumerate(new.cells):
        if "setup" in cell.get("metadata", {}).get("tags", []):
            new.cells = new.cells[: i + 1]
            return new
    return new


NUMBER = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")


def _stale_numbers(nb: nbformat.NotebookNode) -> List[str]:
    """Decimals quoted in the 'What you learned' cell that no output of the notebook contains (after rounding)."""
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown" and c.source.startswith("## What you learned"))
    outputs = "\n".join(
        (o.get("text", "") if o.get("output_type") == "stream" else str(o.get("data", {}).get("text/plain", "")))
        for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []))
    available = [float(x) for x in NUMBER.findall(outputs)] + [float(x) for x in re.findall(r"(?<![\w.])\d+(?![\w.])", outputs)]
    missing = []
    for quoted in sorted(set(NUMBER.findall(text))):
        decimals = len(quoted.split(".")[1])
        value = float(quoted)
        if not any(round(a, decimals) == value or abs(a - value) < 10 ** -(decimals + 1) for a in available):
            missing.append(quoted)
    return missing


def check(path: Path) -> Tuple[bool, str]:
    """Run one notebook and judge it by its type: lessons and solutions must run clean, exercises must pass their setup
    or (start=must_fail) must fail the check on the starter code. Returns (ok, message)."""
    nb = nbformat.read(str(path), as_version=4)
    folder = path.parent.parent.name
    start = nb.metadata.get("practical_ml", {}).get("start")
    if folder == "exercises" or path.parent.parent.parent.name == "exercises":
        if start == "must_fail":
            nb = _execute(nb, path, allow_errors=True)
            errors = _errors(nb)
            before = [e for e in errors if "verify" not in e[1]]
            if before:
                return False, f"error before the check: {before[0][2]}: {before[0][3][:200]}"
            if not any("verify" in e[1] and e[2] == "AssertionError" for e in errors):
                return False, "the check does NOT fail on the starter code"
            return True, "the starter code runs and the check fails, as expected"
        _execute(_truncate_to_setup(nb), path, allow_errors=False)
        return True, "setup ok"
    nb = _execute(nb, path, allow_errors=False)
    if NUMBERS and folder == "solutions":
        missing = _stale_numbers(nb)
        if missing:
            return True, f"ran from start to finish | quoted numbers not found in the output: {', '.join(missing)}"
    return True, "ran from start to finish"


NUMBERS = "--numbers" in sys.argv


def main() -> None:
    """Run every notebook of a module (optionally only those whose path contains a filter) and exit with status 1 on failure."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    module = ROOT / "ML_course" / args[0]
    pattern = args[1] if len(args) > 1 else ""
    paths = sorted(p for p in module.rglob("*.ipynb") if pattern in str(p) and ".ipynb_checkpoints" not in str(p))
    failed = 0
    for p in paths:
        start = time.time()
        rel = p.relative_to(module)
        try:
            ok, msg = check(p)
        except Exception as exc:  # error in a solution or lesson cell
            ok, msg = False, f"{exc.__class__.__name__}: {str(exc)[:600]}"
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {rel}  ({time.time() - start:.1f}s)  {msg}", flush=True)
    print(f"\n{len(paths) - failed}/{len(paths)} notebooks ok")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
