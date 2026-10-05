"""Single entry point for building and testing the course (works the same on Windows, macOS and Linux).

    python _tools/run.py env                 check Python, libraries and the path length
    python _tools/run.py build [N ...]       rebuild notebooks, READMEs and indexes (all, or only the given modules)
    python _tools/run.py test [N ...]        run the notebooks of the given modules (default: all) in parallel
    python _tools/run.py test 7 --numbers    also list quoted figures that are missing from the output
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(os.path.abspath(__file__)).parent.parent
PY = sys.executable


def modules(numbers: Sequence[str]) -> List[str]:
    """Names of the module folders to work on: all of them, or only those whose number is in `numbers`."""
    found = sorted(p.name for p in (ROOT / "ML_course").glob("module_[0-9][0-9]_*") if p.is_dir())
    if not numbers:
        return found
    return [m for m in found if int(m.split("_")[1]) in {int(n) for n in numbers}]


def env() -> None:
    """Print the Python version, warn about a long path and compare the installed libraries with requirements-lock.txt."""
    print(f"Python {sys.version.split()[0]} | course folder: {ROOT}")
    if len(str(ROOT)) > 150:
        print("  warning: long path. On Windows paths over 260 characters stop the kernels from starting: "
              "move the course to a short path such as C:\practical-ml")
    wanted = {}
    for line in (ROOT / "requirements-lock.txt").read_text().splitlines():
        if "==" in line and not line.startswith("#"):
            name, version = line.split("==")
            wanted[name.strip().lower()] = version.strip()
    import importlib.metadata as md
    for name, version in wanted.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            print(f"  missing {name} (tested with {version})")
            continue
        mark = "ok" if have == version else f"differs from the tested {version}"
        print(f"  {name} {have}: {mark}")
    print("The numbers quoted in the lessons come from the tested versions: other versions may print slightly different values.")


def build(args: Sequence[str]) -> None:
    """Run _tools/build.py with the given arguments and exit with its status."""
    sys.exit(subprocess.call([PY, str(ROOT / "_tools" / "build.py"), *args]))


def test(args: Sequence[str]) -> None:
    """Run check_notebooks.py on the selected modules in parallel and print a summary per module. Exits with 1 if any fails."""
    flags = [a for a in args if a.startswith("--")]
    mods = modules([a for a in args if a.isdigit()])
    procs = [(m, subprocess.Popen([PY, str(ROOT / "_tools" / "check_notebooks.py"), m, *flags],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"))
             for m in mods]
    failed = 0
    for m, p in procs:
        out = p.communicate()[0]
        lines = [l for l in out.splitlines() if re.match(r"(OK  |FAIL|\d+/\d+ notebooks)", l)]
        failed += p.returncode != 0
        print(f"== {m}: {lines[-1] if lines else 'no output'}")
        for l in lines:
            if l.startswith("FAIL") or "quoted numbers" in l:
                print("   " + l[:240])
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    cmd, rest = (sys.argv[1], sys.argv[2:]) if len(sys.argv) > 1 else ("", [])
    {"env": lambda: env(), "build": lambda: build(rest), "test": lambda: test(rest)}.get(cmd, lambda: print(__doc__))()
