# Module 3 · Pipelines, cross-validation and hyperparameter search

3 short lessons and 16 exercises with a commented solution (A: 3, B: 3, C: 3, D: 2, E: 3, F: 2). Estimated time: about 11 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 13 of the 16 exercises (about 8 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 1 and 2: you need the train/test split and the preprocessing steps that a pipeline wraps.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 3.1 | [Pipeline and ColumnTransformer](lessons/01_pipeline_and_columntransformer.ipynb) | A single object for preprocessing and model: how to build it, inspect it and modify it. |
| 3.2 | [Cross-validation and leakage](lessons/02_cross_validation_and_leakage.ipynb) | K-fold, stratified, grouped and temporal: the splitter must mirror the real use of the model. |
| 3.3 | [Hyperparameter search](lessons/03_hyperparameter_search.ipynb) | GridSearchCV, RandomizedSearchCV, `cv_results_` and why `best_score_` is optimistic. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | A Pipeline from scratch: who calls `fit` and who calls `transform` | ●●○ | 35 min | [exercise](exercises/A_from_scratch/A1_pipeline_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_pipeline_from_scratch.ipynb) |
| ★ A2 | K-fold, stratified K-fold and `cross_val_score` from scratch | ●●○ | 40 min | [exercise](exercises/A_from_scratch/A2_kfold_and_cross_val_score_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A2_kfold_and_cross_val_score_from_scratch.ipynb) |
| ★ A3 | GridSearchCV from scratch | ●●○ | 35 min | [exercise](exercises/A_from_scratch/A3_grid_search_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A3_grid_search_from_scratch.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | Titanic: a complete pipeline, three variants compared | ●○○ | 40 min | [exercise](exercises/B_with_sklearn/B1_titanic_complete_pipeline.ipynb) · [solution](solutions/B_with_sklearn/B1_titanic_complete_pipeline.ipynb) |
| ★ B2 | A single GridSearchCV to choose between different models | ●●○ | 45 min | [exercise](exercises/B_with_sklearn/B2_gridsearch_over_several_models.ipynb) · [solution](solutions/B_with_sklearn/B2_gridsearch_over_several_models.ipynb) |
| ★ B3 | Custom features inside the pipeline, and saving the model for production | ●●○ | 40 min | [exercise](exercises/B_with_sklearn/B3_custom_features_and_saving.ipynb) · [solution](solutions/B_with_sklearn/B3_custom_features_and_saving.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Temporal leakage: random K-fold on a time series | ●●○ | 40 min | [exercise](exercises/C_break_the_model/C1_temporal_leakage_kfold_on_series.ipynb) · [solution](solutions/C_break_the_model/C1_temporal_leakage_kfold_on_series.ipynb) |
| ★ C2 | Group leakage: the same patient in training and in validation | ●●○ | 45 min | [exercise](exercises/C_break_the_model/C2_group_leakage_patients.ipynb) · [solution](solutions/C_break_the_model/C2_group_leakage_patients.ipynb) |
| C3 | The optimism of best_score_: measuring it with a simulation | ●●● | 45 min | [exercise](exercises/C_break_the_model/C3_best_score_optimism.ipynb) · [solution](solutions/C_break_the_model/C3_best_score_optimism.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading `cv_results_`: who really wins, and which parameter matters | ●●○ | 40 min | [exercise](exercises/D_interpret/D1_reading_cv_results.ipynb) · [solution](solutions/D_interpret/D1_reading_cv_results.ipynb) |
| ★ D2 | Validation and learning curves: diagnosing a model | ●●○ | 40 min | [exercise](exercises/D_interpret/D2_validation_and_learning_curves.ipynb) · [solution](solutions/D_interpret/D2_validation_and_learning_curves.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | Optimizing and evaluating on the same data | ●●○ | 35 min | [exercise](exercises/E_find_the_bug/E1_search_and_evaluation_on_the_same_data.ipynb) · [solution](solutions/E_find_the_bug/E1_search_and_evaluation_on_the_same_data.ipynb) |
| ★ E2 | The ColumnTransformer that silently drops the most important columns | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E2_columntransformer_drops_columns.ipynb) · [solution](solutions/E_find_the_bug/E2_columntransformer_drops_columns.ipynb) |
| ★ E3 | KFold on data sorted by class and `error_score=0` hiding everything | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E3_kfold_on_sorted_data_and_error_score.ipynb) · [solution](solutions/E_find_the_bug/E3_kfold_on_sorted_data_and_error_score.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | Hold-out, 5-fold, 10-fold, repeated and leave-one-out: bias and variance of the estimate | ●●● | 50 min | [exercise](exercises/F_compare/F1_validation_strategies_bias_and_variance.ipynb) · [solution](solutions/F_compare/F1_validation_strategies_bias_and_variance.ipynb) |
| F2 | GridSearchCV, RandomizedSearchCV and HalvingRandomSearchCV compared | ●●● | 45 min | [exercise](exercises/F_compare/F2_grid_random_halving_compared.ipynb) · [solution](solutions/F_compare/F2_grid_random_halving_compared.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 3.1** → [A1](exercises/A_from_scratch/A1_pipeline_from_scratch.ipynb), [B1](exercises/B_with_sklearn/B1_titanic_complete_pipeline.ipynb), [B3](exercises/B_with_sklearn/B3_custom_features_and_saving.ipynb), [E2](exercises/E_find_the_bug/E2_columntransformer_drops_columns.ipynb)
- **Lesson 3.2** → [A2](exercises/A_from_scratch/A2_kfold_and_cross_val_score_from_scratch.ipynb), [E3](exercises/E_find_the_bug/E3_kfold_on_sorted_data_and_error_score.ipynb), [C1](exercises/C_break_the_model/C1_temporal_leakage_kfold_on_series.ipynb), [C2](exercises/C_break_the_model/C2_group_leakage_patients.ipynb), [F1](exercises/F_compare/F1_validation_strategies_bias_and_variance.ipynb)
- **Lesson 3.3** → [A3](exercises/A_from_scratch/A3_grid_search_from_scratch.ipynb), [B2](exercises/B_with_sklearn/B2_gridsearch_over_several_models.ipynb), [D1](exercises/D_interpret/D1_reading_cv_results.ipynb), [D2](exercises/D_interpret/D2_validation_and_learning_curves.ipynb), [E1](exercises/E_find_the_bug/E1_search_and_evaluation_on_the_same_data.ipynb), [C3](exercises/C_break_the_model/C3_best_score_optimism.ipynb), [F2](exercises/F_compare/F2_grid_random_halving_compared.ipynb)
