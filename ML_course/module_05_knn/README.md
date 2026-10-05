# Module 5 · K-nearest neighbors

3 short lessons and 15 exercises with a commented solution (A: 2, B: 4, C: 2, D: 2, E: 3, F: 2). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 8 of the 15 exercises (about 6 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 2-4: scaling (KNN is distance based), pipelines and cross-validation, and the classification metrics and threshold from module 4.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 5.1 | [The neighbors model](lessons/01_the_neighbors_model.ipynb) | Predicting with the k nearest points: what `k` means, how the boundary looks and why the model is the dataset. |
| 5.2 | [Scale, distance and the curse of dimensionality](lessons/02_scale_distance_and_dimensions.ipynb) | Why scaling decides everything, how metric and weights change and what happens when the useless features increase. |
| 5.3 | [KNN in practice](lessons/03_knn_in_practice.ipynb) | Prediction cost, choice of `k`, regression with neighbors, KNNImputer and when it makes sense to use it. |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | A KNN in numpy: vectorized distances, voting and weights | ●●○ | 45 min | [exercise](exercises/A_from_scratch/A1_knn_classifier_from_scratch.ipynb) · [solution](solutions/A_from_scratch/A1_knn_classifier_from_scratch.ipynb) |
| A2 | Leave-one-out for all k with a single distance matrix | ●●● | 50 min | [exercise](exercises/A_from_scratch/A2_leave_one_out_for_all_k.ipynb) · [solution](solutions/A_from_scratch/A2_leave_one_out_for_all_k.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | KNN on mixed data: which columns to put in the distance | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B1_knn_on_mixed_data.ipynb) · [solution](solutions/B_with_sklearn/B1_knn_on_mixed_data.ipynb) |
| ★ B2 | The price depends on where you are: KNN regressor on latitude and longitude | ●●○ | 55 min | [exercise](exercises/B_with_sklearn/B2_knn_regressor_and_geography.ipynb) · [solution](solutions/B_with_sklearn/B2_knn_regressor_and_geography.ipynb) |
| B3 | KNNImputer: measuring imputation when you know the true values | ●●● | 50 min | [exercise](exercises/B_with_sklearn/B3_knn_imputer_with_known_truth.ipynb) · [solution](solutions/B_with_sklearn/B3_knn_imputer_with_known_truth.ipynb) |
| ★ R5 | Warm-up: KNN, the value of k and the effect of scale | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R5_warmup_knn_k_and_scale.ipynb) · [solution](solutions/B_with_sklearn/R5_warmup_knn_k_and_scale.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ C1 | Breaking the KNN with noise columns | ●●○ | 45 min | [exercise](exercises/C_break_the_model/C1_curse_of_dimensionality.ipynb) · [solution](solutions/C_break_the_model/C1_curse_of_dimensionality.ipynb) |
| C2 | Wrong labels: where k = 1 collapses and large k holds up | ●●● | 50 min | [exercise](exercises/C_break_the_model/C2_noisy_labels_and_choice_of_k.ipynb) · [solution](solutions/C_break_the_model/C2_noisy_labels_and_choice_of_k.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading the decision boundary: roughness, regions and k | ●●○ | 45 min | [exercise](exercises/D_interpret/D1_decision_boundary_and_k.ipynb) · [solution](solutions/D_interpret/D1_decision_boundary_and_k.ipynb) |
| D2 | What `predict_proba` means in a KNN | ●●● | 50 min | [exercise](exercises/D_interpret/D2_knn_probabilities.ipynb) · [solution](solutions/D_interpret/D2_knn_probabilities.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| E1 | The neighbors feature that contains the label of the point itself | ●●● | 45 min | [exercise](exercises/E_find_the_bug/E1_the_nearest_neighbor_is_you.ipynb) · [solution](solutions/E_find_the_bug/E1_the_nearest_neighbor_is_you.ipynb) |
| ★ E2 | The same columns, in another order: the model does not notice | ●●○ | 40 min | [exercise](exercises/E_find_the_bug/E2_columns_in_a_different_order_in_production.ipynb) · [solution](solutions/E_find_the_bug/E2_columns_in_a_different_order_in_production.ipynb) |
| ★ E3 | Normalizer is not a scaler: it normalizes the rows, not the columns | ●●○ | 35 min | [exercise](exercises/E_find_the_bug/E3_normalizer_instead_of_standardscaler.ipynb) · [solution](solutions/E_find_the_bug/E3_normalizer_instead_of_standardscaler.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | Scaling and metric: what really matters, on three datasets | ●●● | 55 min | [exercise](exercises/F_compare/F1_scaling_and_metric_on_three_datasets.ipynb) · [solution](solutions/F_compare/F1_scaling_and_metric_on_three_datasets.ipynb) |
| F2 | KNN, logistic and random forest on four shapes of data | ●●● | 60 min | [exercise](exercises/F_compare/F2_knn_logistic_forest_on_four_data_shapes.ipynb) · [solution](solutions/F_compare/F2_knn_logistic_forest_on_four_data_shapes.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 5.1** → [R5](exercises/B_with_sklearn/R5_warmup_knn_k_and_scale.ipynb), [A1](exercises/A_from_scratch/A1_knn_classifier_from_scratch.ipynb), [C2](exercises/C_break_the_model/C2_noisy_labels_and_choice_of_k.ipynb), [D1](exercises/D_interpret/D1_decision_boundary_and_k.ipynb), [E1](exercises/E_find_the_bug/E1_the_nearest_neighbor_is_you.ipynb)
- **Lesson 5.2** → [E3](exercises/E_find_the_bug/E3_normalizer_instead_of_standardscaler.ipynb), [F1](exercises/F_compare/F1_scaling_and_metric_on_three_datasets.ipynb), [C1](exercises/C_break_the_model/C1_curse_of_dimensionality.ipynb), [E2](exercises/E_find_the_bug/E2_columns_in_a_different_order_in_production.ipynb), [D2](exercises/D_interpret/D2_knn_probabilities.ipynb)
- **Lesson 5.3** → [A2](exercises/A_from_scratch/A2_leave_one_out_for_all_k.ipynb), [B1](exercises/B_with_sklearn/B1_knn_on_mixed_data.ipynb), [B2](exercises/B_with_sklearn/B2_knn_regressor_and_geography.ipynb), [B3](exercises/B_with_sklearn/B3_knn_imputer_with_known_truth.ipynb), [F2](exercises/F_compare/F2_knn_logistic_forest_on_four_data_shapes.ipynb)
