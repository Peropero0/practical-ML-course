# Module 2 · Preprocessing: scaling, encoding categoricals, missing data

3 short lessons and 16 exercises with a commented solution (A: 3, B: 3, C: 3, D: 2, E: 3, F: 2). Estimated time: about 11 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 10 of the 16 exercises (about 6 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Module 1 (the regression idea and the train/test split).

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 2.1 | [Scaling and transformations](lessons/01_scaling_and_transformations.ipynb) | StandardScaler, MinMax, Robust, log and PowerTransformer: when they are needed and what they do. |
| 2.2 | [Encoding categoricals](lessons/02_encoding_categoricals.ipynb) | One-hot, ordinal, frequency and target encoding; rare and never-seen categories. |
| 2.3 | [Missing data](lessons/03_missing_data.ipynb) | MCAR, MAR, MNAR; simple, KNN and iterative imputation; the missingness indicator. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | Three scalers from scratch: fit, transform and the constant column case | ●●○ | 35 min | [exercise](exercises/A_from_scratch/A1_scalers_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_scalers_from_scratch.ipynb) |
| ★ A2 | OneHotEncoder from scratch: unknown categories and `drop` | ●●○ | 35 min | [exercise](exercises/A_from_scratch/A2_one_hot_encoder_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A2_one_hot_encoder_from_scratch.ipynb) |
| A3 | Target encoding from scratch: smoothing and out-of-fold encoding | ●●● | 45 min | [exercise](exercises/A_from_scratch/A3_target_encoding_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A3_target_encoding_from_scratch.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | Preprocessing a mixed dataset, one column type at a time | ●○○ | 35 min | [exercise](exercises/B_with_sklearn/B1_manual_preprocessing_mixed_dataset.ipynb) · [solution](solutions/B_with_sklearn/B1_manual_preprocessing_mixed_dataset.ipynb) |
| ★ B2 | Imputers compared: mean, median, KNN and iterative, with known truth | ●●○ | 40 min | [exercise](exercises/B_with_sklearn/B2_imputers_compared.ipynb) · [solution](solutions/B_with_sklearn/B2_imputers_compared.ipynb) |
| ★ B3 | Skewed variables: log, Yeo-Johnson and quantiles, inside and outside the range | ●●○ | 40 min | [exercise](exercises/B_with_sklearn/B3_transformations_of_skewed_variables.ipynb) · [solution](solutions/B_with_sklearn/B3_transformations_of_skewed_variables.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Features in euros sink the KNN: where the distance goes | ●●○ | 35 min | [exercise](exercises/C_break_the_model/C1_scaling_and_knn_distances.ipynb) · [solution](solutions/C_break_the_model/C1_scaling_and_knn_distances.ipynb) |
| ★ C2 | Encoding a nominal with integers: the invented order | ●●○ | 40 min | [exercise](exercises/C_break_the_model/C2_ordinal_on_nominal_variable.ipynb) · [solution](solutions/C_break_the_model/C2_ordinal_on_nominal_variable.ipynb) |
| C3 | Imputation with the mean erases variance and correlation | ●●● | 45 min | [exercise](exercises/C_break_the_model/C3_mean_imputation_distorts.ipynb) · [solution](solutions/C_break_the_model/C3_mean_imputation_distorts.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading a one-hot: names, memory and the dummy variable trap | ●●○ | 35 min | [exercise](exercises/D_interpret/D1_reading_a_one_hot_and_the_dummy_trap.ipynb) · [solution](solutions/D_interpret/D1_reading_a_one_hot_and_the_dummy_trap.ipynb) |
| D2 | Diagnosing missingness mechanisms: what you can and cannot know | ●●● | 45 min | [exercise](exercises/D_interpret/D2_missingness_mechanisms.ipynb) · [solution](solutions/D_interpret/D2_missingness_mechanisms.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | pd.get_dummies on training and test: same size, different columns | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E1_get_dummies_misaligned_columns.ipynb) · [solution](solutions/E_find_the_bug/E1_get_dummies_misaligned_columns.ipynb) |
| ★ E2 | Coefficients with the wrong names: the column order after the ColumnTransformer | ●●○ | 30 min | [exercise](exercises/E_find_the_bug/E2_wrong_coefficient_names.ipynb) · [solution](solutions/E_find_the_bug/E2_wrong_coefficient_names.ipynb) |
| E3 | Target encoding before the cross-validation: a transformer inside the Pipeline | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E3_target_encoding_before_cv.ipynb) · [solution](solutions/E_find_the_bug/E3_target_encoding_before_cv.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | Which scaler for which model, with and without outliers | ●●● | 50 min | [exercise](exercises/F_compare/F1_scaler_per_model_with_and_without_outliers.ipynb) · [solution](solutions/F_compare/F1_scaler_per_model_with_and_without_outliers.ipynb) |
| F2 | Six ways to encode `occupation` (40 levels), with two models | ●●● | 50 min | [exercise](exercises/F_compare/F2_encoders_for_high_cardinality.ipynb) · [solution](solutions/F_compare/F2_encoders_for_high_cardinality.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 2.1** → [A1](exercises/A_from_scratch/A1_scalers_from_scratch.ipynb), [B3](exercises/B_with_sklearn/B3_transformations_of_skewed_variables.ipynb), [C1](exercises/C_break_the_model/C1_scaling_and_knn_distances.ipynb), [F1](exercises/F_compare/F1_scaler_per_model_with_and_without_outliers.ipynb)
- **Lesson 2.2** → [A2](exercises/A_from_scratch/A2_one_hot_encoder_from_scratch.ipynb), [B1](exercises/B_with_sklearn/B1_manual_preprocessing_mixed_dataset.ipynb), [D1](exercises/D_interpret/D1_reading_a_one_hot_and_the_dummy_trap.ipynb), [C2](exercises/C_break_the_model/C2_ordinal_on_nominal_variable.ipynb), [E1](exercises/E_find_the_bug/E1_get_dummies_misaligned_columns.ipynb), [E2](exercises/E_find_the_bug/E2_wrong_coefficient_names.ipynb), [A3](exercises/A_from_scratch/A3_target_encoding_from_scratch.ipynb), [E3](exercises/E_find_the_bug/E3_target_encoding_before_cv.ipynb), [F2](exercises/F_compare/F2_encoders_for_high_cardinality.ipynb)
- **Lesson 2.3** → [B2](exercises/B_with_sklearn/B2_imputers_compared.ipynb), [C3](exercises/C_break_the_model/C3_mean_imputation_distorts.ipynb), [D2](exercises/D_interpret/D2_missingness_mechanisms.ipynb)
