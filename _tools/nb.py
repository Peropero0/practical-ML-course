"""Notebook generator for the practical course.

You don't need it to study: it exists to produce lessons, exercises and solutions in a consistent way
from compact specifications (see the `content/` folder).
"""
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
COURSE = ROOT / "ML_course"          # the module folders live here; common/, data/ and _tools/ stay beside it

KINDS = {
    "A": ("A_from_scratch", "Implement from scratch",
          "Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn."),
    "B": ("B_with_sklearn", "Implement with scikit-learn",
          "A complete problem to solve with the scikit-learn API, the way you would do it at work."),
    "C": ("C_break_the_model", "Break the model",
          "Build the case where the method fails yourself, and measure how badly it fails."),
    "D": ("D_interpret", "Interpret",
          "Read outputs, coefficients and plots, and answer questions that have a precise answer."),
    "E": ("E_find_the_bug", "Find the bug",
          "Code that runs without errors and produces plausible numbers, but is wrong."),
    "F": ("F_compare", "Compare",
          "A controlled experiment that compares two or more approaches."),
}

LEVELS = {1: ("●○○", "guided"), 2: ("●●○", "independent"), 3: ("●●●", "challenge")}

HEADER = '''\
import sys, pathlib
_root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir())
sys.path.insert(0, str(_root))

import warnings; warnings.filterwarnings("ignore")     # clean output: here the warnings are just noise
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from common import datasets as ds
from common.verify import check, close, done, verify
'''


# A cell specification is (kind, text, tag): kind is "md" or "code", tag is an optional notebook cell tag.
Cell = Tuple[str, str, Optional[str]]
# An exercise specification is the dictionary defined in `content/ex_*.py` (id, kind, level, goal, starter, checks...).
Spec = Dict[str, Any]


def _clean(text: str) -> str:
    """Remove the common indentation and the surrounding blank lines of a triple-quoted block."""
    return textwrap.dedent(text).strip("\n")


def M(text: str) -> Cell:
    """Markdown cell specification (the text is dedented)."""
    return ("md", _clean(text), None)


def C(text: str, tag: Optional[str] = None) -> Cell:
    """Code cell specification (the text is dedented). `tag` is stored in the cell metadata, e.g. "setup" or "verify"."""
    return ("code", _clean(text), tag)


def _cells(specs: Sequence[Cell]) -> list:
    """Turn cell specifications into nbformat cells."""
    cells = []
    for kind, text, tag in specs:
        cell = new_markdown_cell(text) if kind == "md" else new_code_cell(text)
        if tag:
            cell.metadata["tags"] = [tag]
        cells.append(cell)
    return cells


def write_notebook(path: Union[str, Path], specs: Sequence[Cell], meta: Optional[Dict[str, Any]] = None) -> None:
    """Write a validated notebook to `path` (parent folders are created); `meta` goes under metadata.practical_ml."""
    nb = new_notebook(cells=_cells(specs))
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    if meta:
        nb.metadata["practical_ml"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, str(path))


def details(question: str, answer: str) -> str:
    """Markdown <details> block: a collapsed question that opens on click to show the answer."""
    return f"<details><summary>{question}</summary>\n\n{_clean(answer)}\n\n</details>"


# ------------------------------------------------------------------------------ exercises

def _numbered(items: Sequence[str]) -> str:
    """Numbered markdown list (1., 2., ...)."""
    return "\n".join(f"{i}. {v}" for i, v in enumerate(items, 1))


def is_core(e: Spec) -> bool:
    """True if the exercise belongs to the core path: levels 1-2 (warm-ups and independent exercises) and every project of module 10."""
    return e["level"] <= 2 or e["module"].startswith("module_10_")


def _exercise_title(e: Spec) -> str:
    """Markdown title block of an exercise: id, type, level, time and path (core or extended)."""
    _, kind_name, _ = KINDS[e["kind"]]
    dots, level_name = LEVELS[e["level"]]
    path = "core path ★" if is_core(e) else "extended path"
    return (f"# {e['id']} · {e['title']}\n\n"
            f"**Type**: {kind_name} · **Level**: {dots} {level_name} · **Time**: about {e['minutes']} minutes · **{path}**")


def exercise_path(e: Spec, what: str) -> Path:
    """Notebook path of an exercise; `what` is "exercises" or "solutions"."""
    folder, _, _ = KINDS[e["kind"]]
    return COURSE / e["module"] / what / folder / f"{e['id']}_{e['slug']}.ipynb"


def _check_cell(checks: str, hint: Optional[str] = None) -> str:
    """Check cell: the checks live inside `with verify():` (technical errors become a readable message).
    The hint, if any, is revealed by `verify` only from the second failed run. Returns the cell source."""
    body = _clean(checks)
    arg = f"hint={' '.join(_clean(hint).split())!r}" if hint else ""
    return f"with verify({arg}):\n" + textwrap.indent(body, "    ") + "\ndone()"


def build_exercise(e: Spec) -> None:
    """Write the exercise notebook: description, setup, starter code, check cell and the pointer to the solution."""
    folder, _, _ = KINDS[e["kind"]]
    cells = [
        M(_exercise_title(e)),
        M(f"## Objective\n\n{_clean(e['goal'])}"),
        M(f"## Dataset\n\n{_clean(e['dataset'])}"),
        M(f"## Educational purpose\n\n{_clean(e['purpose'])}"),
        M(f"## Task\n\n{_numbered([_clean(v) for v in e['task']])}"),
        M("## Setup\n\nRun this cell without changing it."),
        C(HEADER + "\n" + _clean(e["setup"]), tag="setup"),
        M("## Your work"),
    ]
    cells += e["starter"]
    cells += [M("## Check\n\nRun the cell when you think you are done. If a check fails, the message tells you what is wrong. "
                "If it fails a second time, a hint appears under the message."),
              C(_check_cell(e["checks"], e.get("hint")), tag="verify")]
    cells.append(M(f"---\nThe commented solution is in `solutions/{folder}/{e['id']}_{e['slug']}.ipynb`. "
                   "Open it after you pass the check, or after you have really tried."))
    meta = {"id": e["id"], "start": e.get("start", "setup_only")}
    write_notebook(exercise_path(e, "exercises"), cells, meta)


def build_solution(e: Spec) -> None:
    """Write the solution notebook: commented solution, check, what you learned, pitfall and interview question."""
    _, kind_name, _ = KINDS[e["kind"]]
    cells = [
        M(f"# Solution {e['id']} · {e['title']}\n\n**Type**: {kind_name}"),
        M(f"## Objective\n\n{_clean(e['goal'])}"),
        C(HEADER + "\n" + _clean(e["setup"]), tag="setup"),
        M("## Commented solution"),
    ]
    cells += e["solution"]
    cells += [M("## Check"), C(_check_cell(e["checks"]), tag="verify"),
              M(f"## What you learned\n\n{_clean(e['learned'])}"),
              M(f"## Common pitfall\n\n{_clean(e['pitfall'])}")]
    if e.get("interview"):
        q, a = e["interview"]
        cells.append(M("## Interview question\n\n" + details(q, a)))
    write_notebook(exercise_path(e, "solutions"), cells, {"id": e["id"]})


def build_lesson(module: str, file_name: str, cells: Sequence[Cell]) -> None:
    """Write a lesson notebook into `<course>/<module>/lessons/`."""
    write_notebook(COURSE / module / "lessons" / file_name, cells, {"lesson": file_name})
