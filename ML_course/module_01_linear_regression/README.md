# Module 1 · Linear regression

4 short lessons and 19 exercises with a commented solution (A: 3, B: 4, C: 3, D: 3, E: 3, F: 3). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 14 of the 19 exercises (about 8 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** None. Basic Python, numpy and pandas.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 1.1 | [The idea and least squares](lessons/01_idea_and_least_squares.ipynb) | OLS as a matrix product, the normal equation, the `LinearRegression` API. |
| 1.2 | [Evaluating the model](lessons/02_evaluating_the_model.ipynb) | MAE, RMSE, R², baseline, cross-validation, order of the arguments. |
| 1.3 | [Assumptions and diagnostics](lessons/03_assumptions_and_diagnostics.ipynb) | Residual plots, VIF, leverage and Cook's distance. |
| 1.4 | [Regularization and nonlinearity](lessons/04_regularization_and_nonlinearity.ipynb) | Ridge, Lasso, ElasticNet, `PolynomialFeatures`, log target. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | OLS with the normal equation and with the SVD | ●●○ | 30 min | [exercise](exercises/A_from_scratch/A1_ols_normal_equation_and_svd.ipynb) · [solution](solutions/A_from_scratch/A1_ols_normal_equation_and_svd.ipynb) |
| ★ A2 | Gradient descent for linear regression | ●●○ | 35 min | [exercise](exercises/A_from_scratch/A2_gradient_descent.ipynb) · [solution](solutions/A_from_scratch/A2_gradient_descent.ipynb) |
| A3 | Standard errors of the coefficients: closed formula versus bootstrap | ●●● | 40 min | [exercise](exercises/A_from_scratch/A3_standard_errors_formula_and_bootstrap.ipynb) · [solution](solutions/A_from_scratch/A3_standard_errors_formula_and_bootstrap.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | California housing: baseline, linear model and logarithmic target | ●●○ | 35 min | [exercise](exercises/B_with_sklearn/B1_california_baseline_and_log_target.ipynb) · [solution](solutions/B_with_sklearn/B1_california_baseline_and_log_target.ipynb) |
| ★ B2 | Lasso and noise features: automatic selection and scale | ●●○ | 35 min | [exercise](exercises/B_with_sklearn/B2_lasso_and_noise_features.ipynb) · [solution](solutions/B_with_sklearn/B2_lasso_and_noise_features.ipynb) |
| ★ B3 | Polynomial pipeline with Ridge and GridSearchCV | ●●○ | 40 min | [exercise](exercises/B_with_sklearn/B3_polynomial_ridge_gridsearch_pipeline.ipynb) · [solution](solutions/B_with_sklearn/B3_polynomial_ridge_gridsearch_pipeline.ipynb) |
| ★ R1 | Warm-up: the first regression, from data to number | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R1_warmup_the_first_regression.ipynb) · [solution](solutions/B_with_sklearn/R1_warmup_the_first_regression.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Collinearity: unstable coefficients, stable predictions | ●●○ | 35 min | [exercise](exercises/C_break_the_model/C1_collinearity_unstable_coefficients.ipynb) · [solution](solutions/C_break_the_model/C1_collinearity_unstable_coefficients.ipynb) |
| ★ C2 | A single point moves the line: leverage versus residual | ●●○ | 35 min | [exercise](exercises/C_break_the_model/C2_outliers_and_leverage.ipynb) · [solution](solutions/C_break_the_model/C2_outliers_and_leverage.ipynb) |
| C3 | Too many features: when the training R² lies | ●●● | 40 min | [exercise](exercises/C_break_the_model/C3_too_many_features_overfitting.ipynb) · [solution](solutions/C_break_the_model/C3_too_many_features_overfitting.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading the coefficients: units, standardization and the logarithm | ●●○ | 35 min | [exercise](exercises/D_interpret/D1_reading_the_coefficients.ipynb) · [solution](solutions/D_interpret/D1_reading_the_coefficients.ipynb) |
| D2 | Residual diagnostics: four mysteries | ●●● | 45 min | [exercise](exercises/D_interpret/D2_residual_diagnostics_four_mysteries.ipynb) · [solution](solutions/D_interpret/D2_residual_diagnostics_four_mysteries.ipynb) |
| ★ D3 | Anscombe: four datasets, one line, three wrong models | ●●○ | 40 min | [exercise](exercises/D_interpret/D3_anscombe_leverage_and_influence.ipynb) · [solution](solutions/D_interpret/D3_anscombe_leverage_and_influence.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | The model that finds signal in noise | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E1_leakage_in_feature_selection.ipynb) · [solution](solutions/E_find_the_bug/E1_leakage_in_feature_selection.ipynb) |
| ★ E2 | Regression metrics computed wrongly: four bugs | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E2_metrics_computed_wrongly.ipynb) · [solution](solutions/E_find_the_bug/E2_metrics_computed_wrongly.ipynb) |
| ★ E3 | The scaler refitted on the test set | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E3_scaler_refitted_on_test.ipynb) · [solution](solutions/E_find_the_bug/E3_scaler_refitted_on_test.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | OLS, Ridge, Lasso, ElasticNet: which one wins, and when | ●●● | 50 min | [exercise](exercises/F_compare/F1_ols_ridge_lasso_elasticnet_simulation.ipynb) · [solution](solutions/F_compare/F1_ols_ridge_lasso_elasticnet_simulation.ipynb) |
| ★ F2 | Ridge with and without scaling: the same penalty is not the same model | ●●○ | 40 min | [exercise](exercises/F_compare/F2_ridge_with_and_without_scaling.ipynb) · [solution](solutions/F_compare/F2_ridge_with_and_without_scaling.ipynb) |
| F3 | Mean versus median: OLS and quantile regression | ●●● | 45 min | [exercise](exercises/F_compare/F3_mean_versus_median_ols_and_quantile_regression.ipynb) · [solution](solutions/F_compare/F3_mean_versus_median_ols_and_quantile_regression.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 1.1** → [R1](exercises/B_with_sklearn/R1_warmup_the_first_regression.ipynb), [A1](exercises/A_from_scratch/A1_ols_normal_equation_and_svd.ipynb), [A2](exercises/A_from_scratch/A2_gradient_descent.ipynb)
- **Lesson 1.2** → [B1](exercises/B_with_sklearn/B1_california_baseline_and_log_target.ipynb), [E2](exercises/E_find_the_bug/E2_metrics_computed_wrongly.ipynb), [D1](exercises/D_interpret/D1_reading_the_coefficients.ipynb)
- **Lesson 1.3** → [C1](exercises/C_break_the_model/C1_collinearity_unstable_coefficients.ipynb), [C2](exercises/C_break_the_model/C2_outliers_and_leverage.ipynb), [D2](exercises/D_interpret/D2_residual_diagnostics_four_mysteries.ipynb), [D3](exercises/D_interpret/D3_anscombe_leverage_and_influence.ipynb), [A3](exercises/A_from_scratch/A3_standard_errors_formula_and_bootstrap.ipynb)
- **Lesson 1.4** → [B2](exercises/B_with_sklearn/B2_lasso_and_noise_features.ipynb), [B3](exercises/B_with_sklearn/B3_polynomial_ridge_gridsearch_pipeline.ipynb), [C3](exercises/C_break_the_model/C3_too_many_features_overfitting.ipynb), [E1](exercises/E_find_the_bug/E1_leakage_in_feature_selection.ipynb), [E3](exercises/E_find_the_bug/E3_scaler_refitted_on_test.ipynb), [F2](exercises/F_compare/F2_ridge_with_and_without_scaling.ipynb), [F1](exercises/F_compare/F1_ols_ridge_lasso_elasticnet_simulation.ipynb), [F3](exercises/F_compare/F3_mean_versus_median_ols_and_quantile_regression.ipynb)
