# Module 7 · Decision trees

3 short lessons and 15 exercises with a commented solution (A: 2, B: 4, C: 2, D: 2, E: 3, F: 2). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 7 of the 15 exercises (about 5 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 3-4: cross-validation and the classification metrics. Linear models (module 1) for the comparisons.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 7.1 | [How a tree decides](lessons/01_how_a_tree_decides.ipynb) | Splits, Gini impurity and entropy, gain: how a tree is built and how it is read. |
| 7.2 | [Complexity, overfitting and pruning](lessons/02_complexity_and_pruning.ipynb) | Depth, minimum leaf size, `ccp_alpha`: deep trees memorize, and they change a lot with the data. |
| 7.3 | [Trees in practice](lessons/03_trees_in_practice.ipynb) | Regression, importances (and their traps), categoricals, invariance to scale and limits. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| A1 | A classification tree from scratch: Gini, best split, recursion | ●●● | 60 min | [exercise](exercises/A_from_scratch/A1_classification_tree_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_classification_tree_from_scratch.ipynb) |
| A2 | Regression tree and feature importance from scratch | ●●● | 55 min | [exercise](exercises/A_from_scratch/A2_regression_tree_and_importances_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A2_regression_tree_and_importances_from_scratch.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | A tree on the Titanic: overfitting, tuning and readable rules | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B1_titanic_tree_tuning_and_rules.ipynb) · [solution](solutions/B_with_sklearn/B1_titanic_tree_tuning_and_rules.ipynb) |
| ★ B2 | Regression trees on house prices | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B2_regression_tree_on_california.ipynb) · [solution](solutions/B_with_sklearn/B2_regression_tree_on_california.ipynb) |
| B3 | Cost-complexity pruning chosen in cross-validation | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B3_cost_complexity_pruning_in_cv.ipynb) · [solution](solutions/B_with_sklearn/B3_cost_complexity_pruning_in_cv.ipynb) |
| ★ R7 | Warm-up: the first tree and the depth | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R7_warmup_the_first_tree_and_the_depth.ipynb) · [solution](solutions/B_with_sklearn/R7_warmup_the_first_tree_and_the_depth.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Breaking the tree with a trend: flat predictions outside the training range | ●●○ | 45 min | [exercise](exercises/C_break_the_model/C1_the_tree_does_not_extrapolate_the_trend.ipynb) · [solution](solutions/C_break_the_model/C1_the_tree_does_not_extrapolate_the_trend.ipynb) |
| C2 | Tree instability: 30 resamples, 30 different trees | ●●● | 55 min | [exercise](exercises/C_break_the_model/C2_tree_instability_and_the_vote.ipynb) · [solution](solutions/C_break_the_model/C2_tree_instability_and_the_vote.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading a tree from the inside: nodes, paths, probabilities and rules | ●●○ | 50 min | [exercise](exercises/D_interpret/D1_reading_the_structure_of_a_tree.ipynb) · [solution](solutions/D_interpret/D1_reading_the_structure_of_a_tree.ipynb) |
| D2 | Impurity importances reward noise: comparison with permutation | ●●● | 55 min | [exercise](exercises/D_interpret/D2_impurity_importance_vs_permutation.ipynb) · [solution](solutions/D_interpret/D2_impurity_importance_vs_permutation.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | Feature importances with the wrong column names | ●●○ | 40 min | [exercise](exercises/E_find_the_bug/E1_importances_with_the_wrong_column_names.ipynb) · [solution](solutions/E_find_the_bug/E1_importances_with_the_wrong_column_names.ipynb) |
| E2 | A tree with AUC 0.99 on loans: it was the identifier | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E2_the_customer_id_that_predicts_the_outcome.ipynb) · [solution](solutions/E_find_the_bug/E2_the_customer_id_that_predicts_the_outcome.ipynb) |
| E3 | The same pruning prunes nothing: `ccp_alpha` depends on the units of the target | ●●● | 40 min | [exercise](exercises/E_find_the_bug/E3_ccp_alpha_depends_on_the_target_scale.ipynb) · [solution](solutions/E_find_the_bug/E3_ccp_alpha_depends_on_the_target_scale.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ F1 | Gini or entropy? Best split or random split? | ●●○ | 45 min | [exercise](exercises/F_compare/F1_gini_entropy_and_random_splitter.ipynb) · [solution](solutions/F_compare/F1_gini_entropy_and_random_splitter.ipynb) |
| F2 | Tree, logistic and KNN on four boundary shapes (and after a monotonic transformation) | ●●● | 55 min | [exercise](exercises/F_compare/F2_boundary_shapes_and_invariance.ipynb) · [solution](solutions/F_compare/F2_boundary_shapes_and_invariance.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 7.1** → [R7](exercises/B_with_sklearn/R7_warmup_the_first_tree_and_the_depth.ipynb), [A1](exercises/A_from_scratch/A1_classification_tree_from_scratch.ipynb), [D1](exercises/D_interpret/D1_reading_the_structure_of_a_tree.ipynb), [E3](exercises/E_find_the_bug/E3_ccp_alpha_depends_on_the_target_scale.ipynb), [F1](exercises/F_compare/F1_gini_entropy_and_random_splitter.ipynb)
- **Lesson 7.2** → [B1](exercises/B_with_sklearn/B1_titanic_tree_tuning_and_rules.ipynb), [B3](exercises/B_with_sklearn/B3_cost_complexity_pruning_in_cv.ipynb), [C2](exercises/C_break_the_model/C2_tree_instability_and_the_vote.ipynb), [E2](exercises/E_find_the_bug/E2_the_customer_id_that_predicts_the_outcome.ipynb), [A2](exercises/A_from_scratch/A2_regression_tree_and_importances_from_scratch.ipynb)
- **Lesson 7.3** → [B2](exercises/B_with_sklearn/B2_regression_tree_on_california.ipynb), [C1](exercises/C_break_the_model/C1_the_tree_does_not_extrapolate_the_trend.ipynb), [D2](exercises/D_interpret/D2_impurity_importance_vs_permutation.ipynb), [E1](exercises/E_find_the_bug/E1_importances_with_the_wrong_column_names.ipynb), [F2](exercises/F_compare/F2_boundary_shapes_and_invariance.ipynb)
