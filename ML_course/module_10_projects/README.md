# Module 10 · Final projects

2 short lessons and 3 exercises with a commented solution (A: 0, B: 2, C: 0, D: 0, E: 1, F: 0). Estimated time: about 5 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 3 of the 3 exercises (about 5 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** All the previous modules. These are long projects that put the whole course together.

> **This module is different from the others.** It introduces no new tools and it does not have six types of exercise: it has **three long projects** (about an hour and a half each) that put the whole course together. The checks verify the numbers and the decisions, not the way you write the code: no solution is the only possible one.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 10.1 | [The life of a project](lessons/01_the_life_of_a_project.ipynb) | From the problem to the report: closed test set, baseline, pipeline, comparison, tuning, threshold and cost. |
| 10.2 | [Sanity checks](lessons/02_sanity_checks.ipynb) | How to notice a mistake that raises no errors: permuted labels, learning curves, validation that respects the structure of the data. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ P1 | Credit risk scoring: from raw data to the decision | ●●● | 110 min | [exercise](exercises/B_with_sklearn/P1_credit_risk_scoring_from_data_to_decision.ipynb) · [solution](solutions/B_with_sklearn/P1_credit_risk_scoring_from_data_to_decision.ipynb) |
| ★ P2 | House prices: from the prediction to trust in the prediction | ●●● | 100 min | [exercise](exercises/B_with_sklearn/P2_house_prices_from_prediction_to_trust_in_the_prediction.ipynb) · [solution](solutions/B_with_sklearn/P2_house_prices_from_prediction_to_trust_in_the_prediction.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ P3 | Reviewing a notebook: which flaw really matters? | ●●● | 75 min | [exercise](exercises/E_find_the_bug/P3_reviewing_a_notebook_which_flaw_matters.ipynb) · [solution](solutions/E_find_the_bug/P3_reviewing_a_notebook_which_flaw_matters.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 10.1** → [P1](exercises/B_with_sklearn/P1_credit_risk_scoring_from_data_to_decision.ipynb), [P2](exercises/B_with_sklearn/P2_house_prices_from_prediction_to_trust_in_the_prediction.ipynb)
- **Lesson 10.2** → [P3](exercises/E_find_the_bug/P3_reviewing_a_notebook_which_flaw_matters.ipynb)
