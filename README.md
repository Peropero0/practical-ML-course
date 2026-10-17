# Practical classical ML


While preparing for senior-level job interviews, I wanted to refresh and sharpen my knowledge of classical machine learning.

I started by creating a small learning artifact with the help of Claude, based on my own experience and the topics I wanted to revisit. I then used it as a starting point for my own study: I reviewed the material, expanded it where needed, and corrected mistakes, inaccuracies, and missing details along the way.

After a few weeks, it had evolved into something I found valuable enough to keep using and improving. A few people suggested that I make it public, so here it is.

This course is the result of a combination of my practical experience, my own study and review, and AI-assisted writing and organization. Claude helped me structure and articulate the material, but I reviewed, challenged, corrected, and expanded the content based on my own knowledge and understanding.

If you find it useful and would like to support the project, you can support its continued development here:

**[Support this project → https://ko-fi.com/peropero0]**

**[Read my Medium articles → https://medium.com/@lu.battistoni]**

**[Reach me on LinkedIn → https://www.linkedin.com/in/luigi-battistoni/]**


## About the course

The whole course is built around real-world ML cases and deep dives: leakage that inflates a score, a metric that hides a bad model, a default
that quietly does the wrong thing, a validation that does not look like production, an importance that points at noise. These are the situations
that senior-level interviews probe and that you meet at work, so almost every exercise doubles as interview material.

Every module has short lessons with runnable code and a block of exercises with a dataset, a goal, automatic checks,
a commented solution and a "didactic purpose". The exercises come in six types, with at least two non-trivial exercises per type in
each module (2 to 9; module 10 is made of three long projects). Almost every module also has a **warm-up** exercise
(level ●○○, id `R1`, `R4`...`R9`), guided and short, to do first if it is your first contact with the tool.

**Short on time?** Each module marks a **core path** with ★ (warm-ups and independent exercises, levels ●○○ and ●●○, plus the projects of
module 10): 86 of the 145 exercises, about 60 hours. The level ●●● exercises are the **extended path**: harder, longer, best done after the core.
Every module README lists its prerequisites.

**Where to start:** do the installation below (about 10 minutes, once), then open `ML_course/module_00_start_here/00_start_here.ipynb`
(it checks the environment and the datasets and lets you try a check), then the README of module 1. Two files for review and
interview preparation, described in the next section: `PITFALLS_INDEX.md` and `INTERVIEW_QUESTIONS.md`.


| Type | What you do |
|---|---|
| **A · Implement from scratch** | you write the algorithm with numpy and compare it with scikit-learn |
| **B · Implement with scikit-learn** | a complete problem with the scikit-learn API |
| **C · Break the model** | you build the case where the method fails and measure by how much |
| **D · Interpret** | you read outputs and coefficients and answer questions with a precise answer |
| **E · Find the bug** | code that runs and produces plausible numbers, but is wrong |
| **F · Compare** | a controlled experiment between two or more approaches |

## The modules

31 lessons and 145 exercises (each with its commented solution), for about 110 hours of exercises plus the lessons (the time is the author's estimate per exercise, not measured on a student).

| Module | Topic | Lessons | Exercises | Hours |
|---|---|---|---|---|
| [00](ML_course/module_00_start_here/README.md) | Start here: check your environment | 1 notebook | 0 | 0 |
| [01](ML_course/module_01_linear_regression/README.md) | Linear regression (with Ridge, Lasso, polynomials) | 4 | 19 | 12 |
| [02](ML_course/module_02_preprocessing/README.md) | Preprocessing: scaling, encoding of categoricals, missing data | 3 | 16 | 11 |
| [03](ML_course/module_03_pipeline_and_validation/README.md) | Pipelines, cross-validation and hyperparameter search | 3 | 16 | 11 |
| [04](ML_course/module_04_logistic_regression/README.md) | Logistic regression and classification metrics | 4 | 16 | 12 |
| [05](ML_course/module_05_knn/README.md) | K-nearest neighbors | 3 | 15 | 12 |
| [06](ML_course/module_06_pca/README.md) | PCA: principal component analysis | 3 | 15 | 12 |
| [07](ML_course/module_07_decision_trees/README.md) | Decision trees | 3 | 15 | 12 |
| [08](ML_course/module_08_random_forest/README.md) | Random forest | 3 | 15 | 12 |
| [09](ML_course/module_09_xgboost/README.md) | Gradient boosting and XGBoost | 3 | 15 | 13 |
| [10](ML_course/module_10_projects/README.md) | Final projects (credit scoring, prices with spatial validation, reviewing a notebook) | 2 | 3 | 5 |


**Recommended order:** 0 → 1 → 2 → 3 are the foundation (environment, data, preprocessing, validation) and should be done first and in order.
Modules 4-9 are the models and lean on 2 and 3; module 4 (logistic regression) is needed by anyone tackling 5-9 because it
introduces the classification metrics, the threshold and the calibration. Module 10 requires everything else.

**Interview preparation**: two files cut across all the modules and are meant for interview preparation: every exercise of every module contributes one entry to each of them.

| File | Topic | Entries | Use it for |
|---|---|---|---|
| [`INTERVIEW_QUESTIONS.md`](INTERVIEW_QUESTIONS.md) | One interview question with a model answer per exercise, grouped by module | 145 | Practising answers out loud before an interview |
| [`PITFALLS_INDEX.md`](PITFALLS_INDEX.md) | The typical real-world mistake of each exercise, with a link to the solution that shows it | 145 | A checklist of what can go wrong, and deep dives on demand |


## How to use the course

**Hints.** Each exercise has a hint, but it is not shown up front: it appears by itself under the check message after the
second failed run, so you always get one honest attempt first.

**Where to launch it.** The notebooks find the `common/` folder on their own by walking up the folder tree: you can open them from anywhere
inside the project, without configuring anything. The only requirement is not to move `common/` out of the project root (the folder that contains `ML_course/`).

**Datasets.** California housing is downloaded on first use and kept in `data/`; if the network is missing, the course falls back to
a synthetic version (and the checks that depend on the real numbers say so). The Titanic dataset is also downloaded on first use (from OpenML) with a synthetic fallback.
All the other datasets are included in scikit-learn or generated by `common/datasets.py`, with a fixed seed.

**Exact versions.** `requirements-lock.txt` pins the versions the course was tested with (`python _tools/run.py env` compares your
environment with them). The numbers quoted in the lessons and in "What you learned" come from those versions: with other ones
they may differ slightly, which does not mean your code is wrong.

**Compatibility.** The code is written and verified with Python 3.8, scikit-learn 1.1, pandas 1.5 and xgboost 1.6.
- With scikit-learn 1.4 or later `OneHotEncoder(sparse=False)` no longer exists (it was renamed `sparse_output=False` in 1.2): the notebooks of
  module 2 and the exercises that use it fail with a `TypeError`. Use scikit-learn 1.1 to 1.3, or replace `sparse=` with `sparse_output=`.
- With scikit-learn 1.6 or later the `squared=False` parameter of `mean_squared_error` no longer exists: replace it with
  `root_mean_squared_error`.
- With XGBoost 2.x it has not been tried: the checks of module 9 are written as relations (X is smaller than Y) and not
  as exact values, but small adjustments may be needed (for example to the `fit`/`predict` parameters).
- The time measurements (module 9, F1) depend on the machine: the checks verify the ratios, not the seconds.


## Installation

You need **one** of the two options below. Choose the **virtual environment** if you are comfortable installing Python on your computer;
choose **Docker** if you would rather not install Python and the libraries at all, or if the virtual environment gives you trouble.
Both give you the same result: JupyterLab opens in your browser and you work on the notebooks.

First, get the project onto your computer. If you use git:

```
git clone <the address of this repository>
cd <the folder that was created>
```

If you do not use git, download the repository as a ZIP file (green "Code" button, then "Download ZIP"), unzip it, and open a terminal
**inside the unzipped folder** (the folder that contains `README.md`, `ML_course/`, `common/` and `_tools/`).

> **What is a terminal?** A text window where you type commands. On **Windows** open the Start menu, type `PowerShell` and open
> "Windows PowerShell". On **macOS** open the app "Terminal" (Spotlight: press Cmd+Space, type Terminal). On **Linux** press
> Ctrl+Alt+T. To open it directly in the project folder: on Windows, open the folder in File Explorer, click the address bar, type
> `powershell` and press Enter; on macOS, right-click the folder in Finder and choose "New Terminal at Folder".
> To check where you are, type `cd` (Windows) or `pwd` (macOS/Linux): it prints the current folder. The commands below must be run
> from the project folder.

### Option A · Virtual environment

A virtual environment is a private folder (here called `.venv`) that holds a separate copy of Python's libraries, so the course does not
interfere with anything else on your computer and vice versa. You create it once and "activate" it every time you open a new terminal.

**Step 1 · Install Python 3.8 (or later 3.x).** The course was tested with **Python 3.8**; versions 3.9 to 3.11 normally work too, but
only 3.8 reproduces the numbers in the lessons exactly.
- Check whether you already have it: in the terminal type `python --version` (Windows) or `python3 --version` (macOS/Linux). If it prints
  `Python 3.8.x` or higher you can go to step 2. If you get "not found", or a number below 3.8, install it:
- **Windows:** download the installer from <https://www.python.org/downloads/windows/> (look for "Python 3.8", the 64-bit installer). In the first screen
  of the installer **tick "Add python.exe to PATH"** before clicking "Install Now". Then close and reopen PowerShell.
- **macOS:** download the installer from <https://www.python.org/downloads/macos/> and run it, then reopen Terminal.
- **Linux (Debian/Ubuntu):** `sudo apt update && sudo apt install python3 python3-venv python3-pip`.

**Step 2 · Create the environment** (from the project folder).
- Windows (PowerShell): `python -m venv .venv`
- macOS / Linux: `python3 -m venv .venv`

It takes a few seconds and creates a `.venv` folder. You do this only once.

**Step 3 · Activate it.** Every time you open a new terminal you must activate it again.
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
  - If PowerShell says that "running scripts is disabled", run this once and then repeat the activation command:
    `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`
  - If you use the old Command Prompt (cmd) instead: `.venv\Scripts\activate.bat`
- macOS / Linux: `source .venv/bin/activate`

When it worked, the line in your terminal starts with `(.venv)`. To leave the environment later, type `deactivate`.

**Step 4 · Install the libraries.** With the environment active:

```
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
```

`requirements-lock.txt` installs the **exact versions** the course was tested with (the numbers in the lessons come from them).
This downloads a few hundred megabytes and takes 2 to 10 minutes. Use `requirements.txt` instead if you must use a newer Python or
different library versions (it only sets minimum versions, so some printed numbers may differ slightly from the text). JupyterLab and the
libraries used only to rebuild and test the notebooks (`nbformat`, `nbclient`) are already in the file.

**Step 5 · Start JupyterLab** (from the project folder, environment active):

```
jupyter lab
```

Your browser opens on a page with a file list on the left. If it does not open by itself, copy the address starting with
`http://localhost:8888/` that appears in the terminal into your browser. Do not close the terminal while you work: closing it stops JupyterLab.
In the file list open `ML_course`, then `module_00_start_here`, then double-click `00_start_here.ipynb`.

**Step 6 · Run the first notebook.** Click a code cell, then press Shift+Enter to run it and move on; or use the menu "Run" → "Run All Cells".
If the kernel asks which one to use, pick "Python 3". Every line of the first notebook should show a tick (✔).

*VS Code instead of JupyterLab:* install the "Python" and "Jupyter" extensions, open the project folder, open a notebook, and in the top right
click "Select Kernel" → "Python Environments" → choose the one called `.venv`.

**To stop:** press Ctrl+C twice in the terminal where JupyterLab runs. **Next time:** open a terminal in the project folder, do step 3 and step 5 only.

### Option B · Docker

Docker runs the course inside a **container**: a small, isolated Linux computer that already has Python 3.8 and all the libraries at the right versions.
You install only Docker; nothing else is added to your system. The project folder is **shared** with the container, so what you write in the
notebooks is saved in your normal folder and survives when the container stops.

**Step 1 · Install Docker.**
- **Windows 10/11 and macOS:** install **Docker Desktop** from <https://www.docker.com/products/docker-desktop/>, start it, and wait until its icon says it is
  running. On Windows it may ask you to enable "WSL 2" and to restart the computer: accept. (Docker Desktop is free for personal use and education.)
- **Linux:** follow <https://docs.docker.com/engine/install/> for your distribution, then install the "compose plugin" (`sudo apt install docker-compose-plugin` on Ubuntu/Debian).
- Check it works: in a terminal type `docker --version` and `docker compose version`; both must print a version. If you get an error such as
  "cannot connect to the Docker daemon", Docker Desktop is not running yet: start it and wait.

**Step 2 · Build the image** (once; from the project folder).

```
docker compose build
```

This downloads Python 3.8 and installs every library from `requirements-lock.txt` inside the image. It takes 3 to 10 minutes the first time and needs an internet connection.
Rebuilding later is much faster. You only repeat it if you change `requirements-lock.txt` or `Dockerfile`.

**Step 3 · Start the container.**

```
docker compose up
```

Wait until you see a line with `http://127.0.0.1:8888/lab`. Open that address in your browser: JupyterLab starts, and no password is needed because
the server only accepts connections from your own computer. In the file list open `ML_course`, then `module_00_start_here`, then double-click
`00_start_here.ipynb` and run it (Shift+Enter on each cell, or "Run" → "Run All Cells").

**To stop:** go back to the terminal and press Ctrl+C once, then `docker compose down` to remove the stopped container (your files are not touched).
**Next time:** open Docker Desktop, then run only `docker compose up` in the project folder.

**Running commands inside the container** (for example the test of one module, see below), without opening JupyterLab:

```
docker compose run --rm course python _tools/run.py test 7
```

**If something goes wrong**
- *"port is already allocated"*: another program uses port 8888. Close the other Jupyter, or change `"127.0.0.1:8888:8888"` to `"127.0.0.1:8899:8888"` in `docker-compose.yml` and open `http://127.0.0.1:8899/lab`.
- *On Windows the build is very slow or the notebooks are slow*: keep the project in your normal user folder (for example `C:\Users\you\Projects`), not on a network drive or inside OneDrive.
- *On Linux files created in the container belong to `root`*: run `sudo chown -R $USER:$USER .` in the project folder once after your session.
- *You want a clean start*: `docker compose down --rmi local` removes the container and the image; then `docker compose build` again.
