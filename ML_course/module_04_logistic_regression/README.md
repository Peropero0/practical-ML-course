# Module 4 · Logistic regression and classification metrics

4 short lessons and 16 exercises with a commented solution (A: 2, B: 4, C: 2, D: 3, E: 3, F: 2). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 9 of the 16 exercises (about 6 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 1-3: regression basics, scaling and encoding (module 2), pipelines and cross-validation (module 3).

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 4.1 | [From the line to the probability](lessons/01_from_line_to_probability.ipynb) | Sigmoid, log-odds, log-loss and coefficients as odds ratios: what `LogisticRegression` really does. |
| 4.2 | [Regularization, scaling and multiclass](lessons/02_regularization_scaling_multiclass.ipynb) | `C`, L1 and L2 penalties, why scaling changes the model and how to move to more classes. |
| 4.3 | [Metrics, threshold and imbalance](lessons/03_metrics_and_threshold.ipynb) | Confusion matrix, precision and recall, ROC and PR, `class_weight` and choice of the threshold. |
| 4.4 | [Calibration: are the probabilities true?](lessons/04_calibration.ipynb) | Reliability curves and the Brier score, why `class_weight` and forests give probabilities that lie, and how to recalibrate with `CalibratedClassifierCV`. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | Logistic regression with gradient descent | ●●○ | 50 min | [exercise](exercises/A_from_scratch/A1_logistic_with_gradient_descent.ipynb) · [solution](solutions/A_from_scratch/A1_logistic_with_gradient_descent.ipynb) |
| ★ A2 | Confusion matrix, ROC curve and average precision from scratch | ●●○ | 50 min | [exercise](exercises/A_from_scratch/A2_metrics_and_roc_curve_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A2_metrics_and_roc_curve_from_scratch.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | A complete logistic on the loans: pipeline and ablation study | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B1_loans_pipeline_and_ablation.ipynb) · [solution](solutions/B_with_sklearn/B1_loans_pipeline_and_ablation.ipynb) |
| B2 | Choosing the threshold with the real costs | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B2_optimal_threshold_with_costs.ipynb) · [solution](solutions/B_with_sklearn/B2_optimal_threshold_with_costs.ipynb) |
| ★ B3 | Multiclass classification: handwritten digits | ●●○ | 45 min | [exercise](exercises/B_with_sklearn/B3_multiclass_digits_and_choice_of_c.ipynb) · [solution](solutions/B_with_sklearn/B3_multiclass_digits_and_choice_of_c.ipynb) |
| ★ R4 | Warm-up: the first logistic regression and the threshold | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R4_warmup_the_first_logistic_and_the_threshold.ipynb) · [solution](solutions/B_with_sklearn/R4_warmup_the_first_logistic_and_the_threshold.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Perfect separation: when the logistic becomes very sure and wrong | ●●○ | 40 min | [exercise](exercises/C_break_the_model/C1_perfect_separation_and_extreme_probabilities.ipynb) · [solution](solutions/C_break_the_model/C1_perfect_separation_and_extreme_probabilities.ipynb) |
| C2 | The same model at different prevalences: the AUC stays, the precision collapses | ●●● | 50 min | [exercise](exercises/C_break_the_model/C2_prevalence_and_precision.ipynb) · [solution](solutions/C_break_the_model/C2_prevalence_and_precision.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| D1 | Reading the coefficients: odds ratios, collinearity and omitted variables | ●●● | 55 min | [exercise](exercises/D_interpret/D1_coefficients_odds_ratios_and_omitted_variables.ipynb) · [solution](solutions/D_interpret/D1_coefficients_odds_ratios_and_omitted_variables.ipynb) |
| D2 | Calibration: who tells the truth when they say 30%? | ●●● | 50 min | [exercise](exercises/D_interpret/D2_calibration_of_five_models.ipynb) · [solution](solutions/D_interpret/D2_calibration_of_five_models.ipynb) |
| ★ D3 | Reading a 10x10 confusion matrix with two rare classes | ●●○ | 45 min | [exercise](exercises/D_interpret/D3_multiclass_metrics_with_rare_classes.ipynb) · [solution](solutions/D_interpret/D3_multiclass_metrics_with_rare_classes.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | The income in euros that wipes out the amount/income ratio | ●●○ | 35 min | [exercise](exercises/E_find_the_bug/E1_regularization_without_scaling.ipynb) · [solution](solutions/E_find_the_bug/E1_regularization_without_scaling.ipynb) |
| ★ E2 | `predict_proba[:, 1]` is not always the probability you think | ●●○ | 35 min | [exercise](exercises/E_find_the_bug/E2_predict_proba_columns_and_classes.ipynb) · [solution](solutions/E_find_the_bug/E2_predict_proba_columns_and_classes.ipynb) |
| E3 | 100% recall with a random forest: the rebalancing done before the split | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E3_oversampling_before_the_split.ipynb) · [solution](solutions/E_find_the_bug/E3_oversampling_before_the_split.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | No penalty, L2, L1 and elastic net with few true features and many useless ones | ●●● | 55 min | [exercise](exercises/F_compare/F1_none_l2_l1_elasticnet.ipynb) · [solution](solutions/F_compare/F1_none_l2_l1_elasticnet.ipynb) |
| F2 | Rare classes: weights, undersampling, oversampling or just the threshold? | ●●● | 55 min | [exercise](exercises/F_compare/F2_strategies_for_rare_classes.ipynb) · [solution](solutions/F_compare/F2_strategies_for_rare_classes.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 4.1** → [R4](exercises/B_with_sklearn/R4_warmup_the_first_logistic_and_the_threshold.ipynb), [A1](exercises/A_from_scratch/A1_logistic_with_gradient_descent.ipynb), [B1](exercises/B_with_sklearn/B1_loans_pipeline_and_ablation.ipynb), [C1](exercises/C_break_the_model/C1_perfect_separation_and_extreme_probabilities.ipynb), [D1](exercises/D_interpret/D1_coefficients_odds_ratios_and_omitted_variables.ipynb), [E2](exercises/E_find_the_bug/E2_predict_proba_columns_and_classes.ipynb)
- **Lesson 4.2** → [E1](exercises/E_find_the_bug/E1_regularization_without_scaling.ipynb), [F1](exercises/F_compare/F1_none_l2_l1_elasticnet.ipynb), [D3](exercises/D_interpret/D3_multiclass_metrics_with_rare_classes.ipynb), [B3](exercises/B_with_sklearn/B3_multiclass_digits_and_choice_of_c.ipynb)
- **Lesson 4.3** → [A2](exercises/A_from_scratch/A2_metrics_and_roc_curve_from_scratch.ipynb), [C2](exercises/C_break_the_model/C2_prevalence_and_precision.ipynb), [B2](exercises/B_with_sklearn/B2_optimal_threshold_with_costs.ipynb), [E3](exercises/E_find_the_bug/E3_oversampling_before_the_split.ipynb), [F2](exercises/F_compare/F2_strategies_for_rare_classes.ipynb)
- **Lesson 4.4** → [D2](exercises/D_interpret/D2_calibration_of_five_models.ipynb)
