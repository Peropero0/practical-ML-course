# Module 9 · Gradient boosting and XGBoost

3 short lessons and 15 exercises with a commented solution (A: 2, B: 4, C: 2, D: 2, E: 3, F: 2). Estimated time: about 13 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 7 of the 15 exercises (about 5 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 7-8 (trees and forests) and module 4 (log-loss, probabilities and the threshold).

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 9.1 | [Boosting](lessons/01_boosting.ipynb) | Small trees that correct the errors of the previous ones: residuals, shrinkage and overfitting. |
| 9.2 | [Hyperparameters and early stopping](lessons/02_hyperparameters_and_early_stopping.ipynb) | Learning rate, depth, sampling, regularization and how to stop at the right moment. |
| 9.3 | [XGBoost in practice](lessons/03_xgboost_in_practice.ipynb) | Missing values, categoricals, monotonic constraints, importances and contributions, comparison with the other implementations. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | Gradient boosting for regression from scratch: trees on the residuals | ●●○ | 50 min | [exercise](exercises/A_from_scratch/A1_regression_gradient_boosting_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_regression_gradient_boosting_from_scratch.ipynb) |
| A2 | Gradient boosting for classification: gradient, hessian and Newton step | ●●● | 60 min | [exercise](exercises/A_from_scratch/A2_classification_gradient_boosting_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A2_classification_gradient_boosting_from_scratch.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | XGBoost with early stopping on house prices | ●●○ | 55 min | [exercise](exercises/B_with_sklearn/B1_xgboost_with_early_stopping_on_california.ipynb) · [solution](solutions/B_with_sklearn/B1_xgboost_with_early_stopping_on_california.ipynb) |
| B2 | Missing values: native NaN in XGBoost, imputation and indicators | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B2_native_missing_values_vs_imputation.ipynb) · [solution](solutions/B_with_sklearn/B2_native_missing_values_vs_imputation.ipynb) |
| B3 | Tuning XGBoost with a random search (and what it is worth) | ●●● | 60 min | [exercise](exercises/B_with_sklearn/B3_tuning_with_random_search.ipynb) · [solution](solutions/B_with_sklearn/B3_tuning_with_random_search.ipynb) |
| ★ R9 | Warm-up: the first XGBoost and early stopping | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R9_warmup_the_first_xgboost_and_early_stopping.ipynb) · [solution](solutions/B_with_sklearn/R9_warmup_the_first_xgboost_and_early_stopping.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Too many rounds: the accuracy stays, the probabilities get ruined | ●●○ | 45 min | [exercise](exercises/C_break_the_model/C1_too_many_rounds_probabilities_get_ruined_first.ipynb) · [solution](solutions/C_break_the_model/C1_too_many_rounds_probabilities_get_ruined_first.ipynb) |
| C2 | Wrong labels: boosting chases them, the forest does not | ●●● | 55 min | [exercise](exercises/C_break_the_model/C2_noisy_labels_boosting_chases_them.ipynb) · [solution](solutions/C_break_the_model/C2_noisy_labels_boosting_chases_them.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading the training curves of XGBoost | ●●○ | 50 min | [exercise](exercises/D_interpret/D1_reading_the_training_curves.ipynb) · [solution](solutions/D_interpret/D1_reading_the_training_curves.ipynb) |
| D2 | XGBoost importances: gain, weight, cover, permutation and contributions | ●●● | 55 min | [exercise](exercises/D_interpret/D2_xgboost_importances_gain_weight_and_contributions.ipynb) · [solution](solutions/D_interpret/D2_xgboost_importances_gain_weight_and_contributions.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | The early stopping that never stops | ●●○ | 45 min | [exercise](exercises/E_find_the_bug/E1_the_early_stopping_that_never_stops.ipynb) · [solution](solutions/E_find_the_bug/E1_the_early_stopping_that_never_stops.ipynb) |
| E2 | The production model does not match the validated one | ●●● | 50 min | [exercise](exercises/E_find_the_bug/E2_the_booster_ignores_early_stopping.ipynb) · [solution](solutions/E_find_the_bug/E2_the_booster_ignores_early_stopping.ipynb) |
| E3 | Monotonic constraints ruin the model (or is it the order?) | ●●● | 50 min | [exercise](exercises/E_find_the_bug/E3_monotone_constraints_in_the_wrong_order.ipynb) · [solution](solutions/E_find_the_bug/E3_monotone_constraints_in_the_wrong_order.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ F1 | Five implementations compared: speed and quality as the data grows | ●●○ | 50 min | [exercise](exercises/F_compare/F1_five_tree_implementations_compared.ipynb) · [solution](solutions/F_compare/F1_five_tree_implementations_compared.ipynb) |
| F2 | When boosting does not win: extrapolation, noisy linear signal, interactions | ●●● | 60 min | [exercise](exercises/F_compare/F2_when_boosting_does_not_win.ipynb) · [solution](solutions/F_compare/F2_when_boosting_does_not_win.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 9.1** → [R9](exercises/B_with_sklearn/R9_warmup_the_first_xgboost_and_early_stopping.ipynb), [A1](exercises/A_from_scratch/A1_regression_gradient_boosting_from_scratch.ipynb), [A2](exercises/A_from_scratch/A2_classification_gradient_boosting_from_scratch.ipynb), [C2](exercises/C_break_the_model/C2_noisy_labels_boosting_chases_them.ipynb), [F1](exercises/F_compare/F1_five_tree_implementations_compared.ipynb)
- **Lesson 9.2** → [B1](exercises/B_with_sklearn/B1_xgboost_with_early_stopping_on_california.ipynb), [D1](exercises/D_interpret/D1_reading_the_training_curves.ipynb), [C1](exercises/C_break_the_model/C1_too_many_rounds_probabilities_get_ruined_first.ipynb), [E1](exercises/E_find_the_bug/E1_the_early_stopping_that_never_stops.ipynb), [E2](exercises/E_find_the_bug/E2_the_booster_ignores_early_stopping.ipynb), [B3](exercises/B_with_sklearn/B3_tuning_with_random_search.ipynb)
- **Lesson 9.3** → [B2](exercises/B_with_sklearn/B2_native_missing_values_vs_imputation.ipynb), [D2](exercises/D_interpret/D2_xgboost_importances_gain_weight_and_contributions.ipynb), [E3](exercises/E_find_the_bug/E3_monotone_constraints_in_the_wrong_order.ipynb), [F2](exercises/F_compare/F2_when_boosting_does_not_win.ipynb)
