"""Builds the notebooks of one or more modules (lessons, exercises, solutions) and their README.

Usage:  python _tools/build.py 1 2 3      (module numbers)
        python _tools/build.py            (all the modules that have a lessons_mNN.py file)

Every module NN needs in `_tools/content/`:
  lessons_mNN.py     with MODULE, TITLE, LESSONS, PATH and the function build()
  ex_mNN_a.py ... ex_mNN_f.py     with EXERCISES (a list of specifications)
"""
import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "content"))

import nb  # noqa: E402

CONTENT = Path(__file__).resolve().parent / "content"


def collect(num: int) -> List[Dict[str, Any]]:
    """All the exercise specifications of module `num`: the ex_mNN_a..f files plus the warm-ups that belong to it."""
    exercises = []
    for kind in "abcdef":
        name = f"ex_m{num:02d}_{kind}"
        if (CONTENT / f"{name}.py").exists():
            exercises += importlib.import_module(name).EXERCISES
    # warm-up exercises (level 1): they live in a single file and are told apart by module
    module = importlib.import_module("ex_warmups")
    prefix = f"module_{num:02d}_"
    exercises += [e for e in module.EXERCISES if e["module"].startswith(prefix)]
    return exercises


def _link(e: Dict[str, Any], what: str) -> str:
    """Relative link, from the module folder, to the exercise or solution notebook (`what` is exercises or solutions)."""
    folder, _, _ = nb.KINDS[e["kind"]]
    return f"{what}/{folder}/{e['id']}_{e['slug']}.ipynb"


PREREQS = {
    1: "None. Basic Python, numpy and pandas.",
    2: "Module 1 (the regression idea and the train/test split).",
    3: "Modules 1 and 2: you need the train/test split and the preprocessing steps that a pipeline wraps.",
    4: "Modules 1-3: regression basics, scaling and encoding (module 2), pipelines and cross-validation (module 3).",
    5: "Modules 2-4: scaling (KNN is distance based), pipelines and cross-validation, and the classification metrics and "
       "threshold from module 4.",
    6: "Modules 2-3: scaling before PCA and pipelines. Module 4 is useful for the classification exercises.",
    7: "Modules 3-4: cross-validation and the classification metrics. Linear models (module 1) for the comparisons.",
    8: "Module 7 (a forest is an average of trees) and module 3 (cross-validation). Module 4 for the threshold exercises.",
    9: "Modules 7-8 (trees and forests) and module 4 (log-loss, probabilities and the threshold).",
    10: "All the previous modules. These are long projects that put the whole course together.",
}


is_core = nb.is_core


def write_readme(lez: ModuleType, exercises: List[Dict[str, Any]]) -> None:
    """Write the README of a module: time, core path, prerequisites, lessons, exercise tables and suggested path."""
    hours = sum(e["minutes"] for e in exercises) / 60
    core = [e for e in exercises if is_core(e)]
    core_hours = sum(e["minutes"] for e in core) / 60
    per_kind = {k: len([e for e in exercises if e["kind"] == k]) for k in nb.KINDS}
    lines = [
        f"# Module {lez.NUMBER} · {lez.TITLE}",
        "",
        f"{len(lez.LESSONS)} short lessons and {len(exercises)} exercises with a commented solution "
        f"({', '.join(f'{k}: {v}' for k, v in per_kind.items())}). "
        f"Estimated time: about {hours:.0f} hours of exercises plus a couple of hours of lessons.",
        "",
        f"**Short on time?** Do the **core path**: {len(core)} of the {len(exercises)} exercises (about {core_hours:.0f} hours), "
        "marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.",
        "",
        f"**Prerequisites:** {PREREQS[lez.NUMBER]}",
        "",
    ] + ([getattr(lez, "NOTE"), ""] if getattr(lez, "NOTE", "") else []) + [
        "## How to work",
        "",
        "1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.",
        "2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells "
        "under \"Your work\", in place of the `...`.",
        "3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself "
        "after the second failed run, so you always get one honest attempt first.",
        "4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is "
        "*What you learned*, not the code.",
        "",
        "## Lessons",
        "",
        "| # | Notebook | What it contains |",
        "|---|---|---|",
    ]
    for f, num, title, desc in lez.LESSONS:
        lines.append(f"| {num} | [{title}](lessons/{f}) | {desc} |")
    lines += ["", "## List of exercises", "",
              "Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.", ""]
    for key, (folder, name, descr) in nb.KINDS.items():
        items = [x for x in exercises if x["kind"] == key]
        if not items:
            continue
        lines += [f"### {key} · {name}", "", f"*{descr}*", "",
                  "| ID | Title | Level | Time | |", "|---|---|---|---|---|"]
        for e in items:
            dots, _ = nb.LEVELS[e["level"]]
            star = "★ " if is_core(e) else ""
            lines.append(f"| {star}{e['id']} | {e['title']} | {dots} | {e['minutes']} min | "
                         f"[exercise]({_link(e, 'exercises')}) · [solution]({_link(e, 'solutions')}) |")
        lines.append("")
    lines += ["## Suggested path", "",
              "The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at "
              "different points. This order follows the lessons.", ""]
    by_id = {e["id"]: e for e in exercises}
    path = [(lesson, list(ids)) for lesson, ids in lez.PATH]
    already = {i for _, ids in path for i in ids}
    warmups = [e["id"] for e in exercises if e["level"] == 1 and e["id"] not in already]      # the warm-up goes first
    if warmups and path:
        path[0] = (path[0][0], warmups + path[0][1])
    for lesson, ids in path:
        items = ", ".join(f"[{i}]({_link(by_id[i], 'exercises')})" for i in ids if i in by_id)
        lines.append(f"- **{lesson}** → {items}")
    lines.append("")
    (nb.COURSE / lez.MODULE / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build_module(num: int) -> Tuple[ModuleType, List[Dict[str, Any]]]:
    """Build lessons, exercises, solutions and README of module `num`. Returns the lessons module and the exercise list."""
    lez = importlib.import_module(f"lessons_m{num:02d}")
    lez.NUMBER = num
    lez.build()
    exercises = collect(num)
    for e in exercises:
        nb.build_exercise(e)
        nb.build_solution(e)
    write_readme(lez, exercises)
    print(f"module {num:02d}: {len(lez.LESSONS)} lessons, {len(exercises)} exercises, {len(exercises)} solutions")
    return lez, exercises


def write_indexes() -> None:
    """Two cross-cutting indexes, built from the content: the pitfalls (one per exercise) and the interview questions."""
    numbers = sorted(int(p.stem[-2:]) for p in CONTENT.glob("lessons_m*.py"))
    pitfalls = ["# Catalog of pitfalls", "",
                "Every exercise in the course ends with a **common pitfall**: the mistake that is really made and that usually "
                "gives no message. This is the complete list, by module, with a link to the solution where you see it live. "
                "Reread it before handing in any work.", ""]
    interview = ["# Interview questions", "",
                 "One question per exercise, with the answer in short form (click the question to open it). "
                 "Try to answer out loud before reading.", ""]
    n_pitfalls = n_questions = 0
    for num in numbers:
        lez = importlib.import_module(f"lessons_m{num:02d}")
        exercises = collect(num)
        heading = f"## Module {num} · {lez.TITLE}"
        pitfalls += [heading, "", "| ID | Exercise | Common pitfall |", "|---|---|---|"]
        interview += [heading, ""]
        for e in exercises:
            folder, _, _ = nb.KINDS[e["kind"]]
            link = f"ML_course/{lez.MODULE}/solutions/{folder}/{e['id']}_{e['slug']}.ipynb"
            text = " ".join(nb._clean(e["pitfall"]).split()).replace("|", "/")
            pitfalls.append(f"| {e['id']} | [{nb._clean(e['title'])}]({link}) | {text} |")
            n_pitfalls += 1
            if e.get("interview"):
                question, answer = e["interview"]
                interview += [f"<details><summary><b>{e['id']}</b> · {nb._clean(question)}</summary>", "",
                              " ".join(nb._clean(answer).split()), "", "</details>", ""]
                n_questions += 1
        pitfalls.append("")
    (nb.ROOT / "PITFALLS_INDEX.md").write_text("\n".join(pitfalls) + "\n", encoding="utf-8")
    (nb.ROOT / "INTERVIEW_QUESTIONS.md").write_text("\n".join(interview) + "\n", encoding="utf-8")
    print(f"indexes: {n_pitfalls} pitfalls, {n_questions} interview questions")


def build_start() -> None:
    """The 00_start_here.ipynb notebook: check of the environment, of the datasets, and a trial of the check cell."""
    C, M = nb.C, nb.M
    cells = [
        M("""
        # Start here

        This notebook checks that your environment is ready and shows you **how the exercises work**. Run it all, from top to bottom (Run all): it changes nothing.

        ## How to work, in four lines

        1. Every module has `lessons/` (read and **run the cells**), `exercises/` (yours) and `solutions/` (to open after you have tried).
        2. In an exercise you write the code in the cells under *Your work* and **give the results the names required by the Task**.
        3. The *Check* cell checks your results and says what is wrong. If a variable is missing or still `None`, it says so in plain words.
        4. The exercises with level ●○○ are **warm-ups**: start there if it is your first contact with the tool.
        """),
        C("""
        import sys, pathlib, platform
        _root = next(p for p in [pathlib.Path.cwd(), *pathlib.Path.cwd().parents] if (p / "common").is_dir())
        sys.path.insert(0, str(_root))
        import re, warnings; warnings.filterwarnings("ignore")

        MINIMUMS = {"numpy": (1, 20), "pandas": (1, 3), "sklearn": (1, 1), "scipy": (1, 7), "matplotlib": (3, 4), "xgboost": (1, 6)}
        print(f"Python {platform.python_version()} | course folder: {_root}")
        problems = []
        for name, minimum in MINIMUMS.items():
            try:
                module = __import__(name)
            except ImportError:
                problems.append(f"{name} is missing: pip install -r requirements.txt")
                print(f"  ✘ {name}: not installed")
                continue
            version = tuple(int(x) for x in re.findall(r"\\d+", module.__version__)[:2])
            ok = version >= minimum
            print(f"  {'✔' if ok else '✘'} {name} {module.__version__} (minimum {'.'.join(map(str, minimum))})")
            if not ok:
                problems.append(f"upgrade {name}")
        import sklearn
        sk_version = tuple(int(x) for x in re.findall(r"\\d+", sklearn.__version__)[:2])
        if sk_version >= (1, 4):
            problems.append("scikit-learn 1.4 or later: `OneHotEncoder(sparse=False)` no longer exists (see the README)")
        if sk_version >= (1, 6):
            problems.append("scikit-learn 1.6 or later: `squared=False` no longer exists (see the README)")
        print("\\nEnvironment ready." if not problems else "\\nTo fix: " + "; ".join(problems))
        """),
        M("""
        ## The datasets

        California housing is downloaded on first use and kept in `data/`. If there is no network the course falls back to a synthetic version: it works, but some numbers quoted in the comments change.
        """),
        C("""
        from common import datasets as ds

        X, y = ds.california()
        print("California housing:", X.shape, "| synthetic version" if ds.CALIFORNIA_SYNTHETIC else "| real data")
        print("loans:", ds.loans().shape, "| titanic:", ds.titanic().shape)
        """),
        M("""
        ## How a check works

        The check cells use `check(condition, ok_message, hint)`. Here is the trial: replace `answer = 2 + 2` with `answer = None`, run again and read the message; then with `answer = 5` and read that one.
        """),
        C("""
        from common.verify import check, done, verify

        answer = 2 + 2

        with verify():
            check(answer == 4, "correct answer", "2 + 2 is 4: check the calculation")
        done()
        """),
        M("""
        ## Where to go now

        | If you want to... | Start from |
        |---|---|
        | follow the course in order | `../module_01_linear_regression/README.md` |
        | review the typical mistakes | `PITFALLS_INDEX.md` (in the folder that contains `ML_course/`) |
        | prepare for an interview | `INTERVIEW_QUESTIONS.md` (same folder) |
        | do a complete project | `../module_10_projects/README.md` (after modules 1-9) |

        The suggested order is in the `README.md` of the folder that contains `ML_course/`.
        """),
    ]
    nb.write_notebook(nb.COURSE / "module_00_start_here" / "00_start_here.ipynb", cells, {"lesson": "00_start_here.ipynb"})
    readme0 = [
        "# Module 0 · Start here",
        "",
        "One notebook, `00_start_here.ipynb`, to open **before anything else**. It checks that your Python and your libraries are ready, "
        "checks the datasets, and lets you try an exercise check on a two-line example so you know how the exercises work.",
        "",
        "Run it from top to bottom (Run All). It changes nothing in the course. If it reports a problem, follow the "
        "installation steps in the README at the root of the project (virtual environment or Docker), then run it again.",
        "",
        "Next: [module 1](../module_01_linear_regression/README.md).",
    ]
    (nb.COURSE / "module_00_start_here" / "README.md").write_text("\n".join(readme0) + "\n", encoding="utf-8")
    print("written 00_start_here.ipynb")


def main() -> None:
    """Command line: no argument builds everything; numbers build those modules; `indexes` rebuilds only the two index files."""
    numbers = [int(a) for a in sys.argv[1:] if a.isdigit()]
    everything = not numbers and "indexes" not in sys.argv[1:]
    if not numbers:
        numbers = sorted(int(p.stem[-2:]) for p in CONTENT.glob("lessons_m*.py"))
    if "indexes" not in sys.argv[1:]:
        for n in numbers:
            build_module(n)
    if everything or "indexes" in sys.argv[1:]:
        write_indexes()
        build_start()


if __name__ == "__main__":
    main()
