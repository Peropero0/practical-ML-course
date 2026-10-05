# Module 8 · Random forest

3 short lessons and 15 exercises with a commented solution (A: 2, B: 4, C: 2, D: 2, E: 3, F: 2). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 7 of the 15 exercises (about 5 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Module 7 (a forest is an average of trees) and module 3 (cross-validation). Module 4 for the threshold exercises.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 8.1 | [Bagging and forests](lessons/01_bagging_and_forests.ipynb) | Bootstrap, voting, out-of-bag samples and feature subsets: why averaging trees works. |
| 8.2 | [Hyperparameters and diagnostics](lessons/02_hyperparameters_and_diagnostics.ipynb) | Number of trees, `max_features`, minimum leaf size, OOB error and reliable importances. |
| 8.3 | [Forests in practice](lessons/03_forests_in_practice.ipynb) | Probabilities and calibration, imbalanced classes, regression, cost and limits. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | Bagging from scratch: bootstrap, vote and out-of-bag error | ●●○ | 50 min | [exercise](exercises/A_from_scratch/A1_bagging_and_oob_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_bagging_and_oob_from_scratch.ipynb) |
| A2 | Random forest from scratch: the correlation between trees and the variance of the average | ●●● | 60 min | [exercise](exercises/A_from_scratch/A2_why_max_features_decorrelates_the_trees.ipynb) · [solution](solutions/A_from_scratch/A2_why_max_features_decorrelates_the_trees.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | A forest on the Titanic driven by the OOB error | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B1_forest_on_the_titanic_with_oob.ipynb) · [solution](solutions/B_with_sklearn/B1_forest_on_the_titanic_with_oob.ipynb) |
| ★ B2 | Regression random forest on house prices | ●●○ | 55 min | [exercise](exercises/B_with_sklearn/B2_regression_forest_on_california.ipynb) · [solution](solutions/B_with_sklearn/B2_regression_forest_on_california.ipynb) |
| B3 | Rare classes: weights, `balanced_subsample` and a threshold chosen with the OOB | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B3_imbalanced_forest_threshold_from_oob.ipynb) · [solution](solutions/B_with_sklearn/B3_imbalanced_forest_threshold_from_oob.ipynb) |
| ★ R8 | Warm-up: from the tree to the forest | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R8_warmup_from_the_tree_to_the_forest.ipynb) · [solution](solutions/B_with_sklearn/R8_warmup_from_the_tree_to_the_forest.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| C1 | Duplicate rows: the OOB error (and the cross-validation) say 100% | ●●● | 50 min | [exercise](exercises/C_break_the_model/C1_duplicate_rows_and_optimistic_oob.ipynb) · [solution](solutions/C_break_the_model/C1_duplicate_rows_and_optimistic_oob.ipynb) |
| C2 | Adding copies of a feature makes it look useless | ●●● | 55 min | [exercise](exercises/C_break_the_model/C2_correlated_features_and_permutation_importance.ipynb) · [solution](solutions/C_break_the_model/C2_correlated_features_and_permutation_importance.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | How many trees do you need? Reading OOB curves, test curves and disagreement between forests | ●●○ | 50 min | [exercise](exercises/D_interpret/D1_how_many_trees_do_you_really_need.ipynb) · [solution](solutions/D_interpret/D1_how_many_trees_do_you_really_need.ipynb) |
| D2 | Four ways to measure importance: impurity, permutation, drop-column and group | ●●● | 55 min | [exercise](exercises/D_interpret/D2_four_importance_methods_compared.ipynb) · [solution](solutions/D_interpret/D2_four_importance_methods_compared.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | Comparing two forests with a single seed and 25 trees | ●●○ | 45 min | [exercise](exercises/E_find_the_bug/E1_a_half_point_difference_was_the_seed.ipynb) · [solution](solutions/E_find_the_bug/E1_a_half_point_difference_was_the_seed.ipynb) |
| E2 | The default RandomForestRegressor is not a random forest | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E2_the_default_regression_forest_is_a_bagging.ipynb) · [solution](solutions/E_find_the_bug/E2_the_default_regression_forest_is_a_bagging.ipynb) |
| E3 | Permutation on the training set rewards the noise the forest memorized | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E3_permutation_importance_computed_on_the_training_set.ipynb) · [solution](solutions/E_find_the_bug/E3_permutation_importance_computed_on_the_training_set.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ F1 | From a tree to Extra Trees: how much each ingredient gains | ●●○ | 50 min | [exercise](exercises/F_compare/F1_the_ladder_from_tree_to_extra_trees.ipynb) · [solution](solutions/F_compare/F1_the_ladder_from_tree_to_extra_trees.ipynb) |
| F2 | When the forest loses: linear relationship, trend and cost | ●●● | 60 min | [exercise](exercises/F_compare/F2_when_the_forest_loses.ipynb) · [solution](solutions/F_compare/F2_when_the_forest_loses.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 8.1** → [R8](exercises/B_with_sklearn/R8_warmup_from_the_tree_to_the_forest.ipynb), [A1](exercises/A_from_scratch/A1_bagging_and_oob_from_scratch.ipynb), [A2](exercises/A_from_scratch/A2_why_max_features_decorrelates_the_trees.ipynb), [F1](exercises/F_compare/F1_the_ladder_from_tree_to_extra_trees.ipynb), [E2](exercises/E_find_the_bug/E2_the_default_regression_forest_is_a_bagging.ipynb), [D1](exercises/D_interpret/D1_how_many_trees_do_you_really_need.ipynb)
- **Lesson 8.2** → [B1](exercises/B_with_sklearn/B1_forest_on_the_titanic_with_oob.ipynb), [C1](exercises/C_break_the_model/C1_duplicate_rows_and_optimistic_oob.ipynb), [E1](exercises/E_find_the_bug/E1_a_half_point_difference_was_the_seed.ipynb), [D2](exercises/D_interpret/D2_four_importance_methods_compared.ipynb), [C2](exercises/C_break_the_model/C2_correlated_features_and_permutation_importance.ipynb), [E3](exercises/E_find_the_bug/E3_permutation_importance_computed_on_the_training_set.ipynb)
- **Lesson 8.3** → [B2](exercises/B_with_sklearn/B2_regression_forest_on_california.ipynb), [B3](exercises/B_with_sklearn/B3_imbalanced_forest_threshold_from_oob.ipynb), [F2](exercises/F_compare/F2_when_the_forest_loses.ipynb)
