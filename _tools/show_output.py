"""Runs a notebook and prints the text output of its cells (to check the numbers quoted in the comments).

Usage:  python _tools/show_output.py path/to/notebook.ipynb
"""
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

p = Path(os.path.abspath(sys.argv[1]))
nb = nbformat.read(str(p), as_version=4)
NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=True, resources={"metadata": {"path": str(p.parent)}}).execute()
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.outputs:
        t = o.get("text") or (o.get("data", {}).get("text/plain") if o.output_type in ("execute_result", "display_data") else None)
        if t:
            print(f"--- cell {i}\n{t.strip()}")
        if o.output_type == "error":
            print(f"--- cell {i} ERROR: {o['ename']}: {o['evalue']}")
