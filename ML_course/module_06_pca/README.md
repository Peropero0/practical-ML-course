# Module 6 · PCA: principal component analysis

3 short lessons and 15 exercises with a commented solution (A: 2, B: 4, C: 2, D: 2, E: 3, F: 2). Estimated time: about 12 hours of exercises plus a couple of hours of lessons.

**Short on time?** Do the **core path**: 8 of the 15 exercises (about 5 hours), marked ★ in the tables below. The others (level ●●●) are the extended path: harder, longer, and best done after the core.

**Prerequisites:** Modules 2-3: scaling before PCA and pipelines. Module 4 is useful for the classification exercises.

## How to work

1. Read the lesson and **run the cells**: they are not illustrations, they are how the lesson proves things.
2. Open an exercise, read *Objective*, *Dataset* and *Educational purpose*, then the *Task*. Write your code in the cells under "Your work", in place of the `...`.
3. Run the **Check** cell. If a check fails, the message says what is wrong. A hint appears by itself after the second failed run, so you always get one honest attempt first.
4. Open the solution **after** you pass the check (or after you have really tried): the most useful part is *What you learned*, not the code.

## Lessons

| # | Notebook | What it contains |
|---|---|---|
| 6.1 | [The principal components](lessons/01_principal_components.ipynb) | Directions of maximum variance, SVD, scores and reconstruction: what `PCA` really computes. |
| 6.2 | [How many components to keep](lessons/02_how_many_components.ipynb) | Scree plot, cumulative explained variance, choice with cross-validation and reconstruction error. |
| 6.3 | [PCA in practice](lessons/03_pca_in_practice.ipynb) | Inside a pipeline, to visualize, to remove noise, and its limits (nonlinearity, signal in low-variance directions). |

## List of exercises

Level: ●○○ guided, ●●○ independent, ●●● challenge. ★ = core path.

### A · Implement from scratch

*Write the algorithm yourself with numpy, no shortcuts, and then compare it with scikit-learn.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ A1 | PCA in numpy with the SVD (and with the eigenvalues) | ●●○ | 45 min | [exercise](exercises/A_from_scratch/A1_pca_from_scratch_with_svd.ipynb) · [solution](solutions/A_from_scratch/A1_pca_from_scratch_with_svd.ipynb) |
| A2 | The components with the power method: convergence and the gap between eigenvalues | ●●● | 55 min | [exercise](exercises/A_from_scratch/A2_power_iteration_and_spectral_gap.ipynb) · [solution](solutions/A_from_scratch/A2_power_iteration_and_spectral_gap.ipynb) |

### B · Implement with scikit-learn

*A complete problem to solve with the scikit-learn API, the way you would do it at work.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ B1 | How many components? PCA in a pipeline on the digits | ●●○ | 50 min | [exercise](exercises/B_with_sklearn/B1_choosing_k_in_a_pipeline_on_digits.ipynb) · [solution](solutions/B_with_sklearn/B1_choosing_k_in_a_pipeline_on_digits.ipynb) |
| B2 | Removing noise from images with PCA | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B2_pca_to_remove_noise.ipynb) · [solution](solutions/B_with_sklearn/B2_pca_to_remove_noise.ipynb) |
| B3 | Many correlated features: OLS, Ridge, regression on the components and PLS | ●●● | 55 min | [exercise](exercises/B_with_sklearn/B3_regression_on_components_and_pls.ipynb) · [solution](solutions/B_with_sklearn/B3_regression_on_components_and_pls.ipynb) |
| ★ R6 | Warm-up: the first PCA, explained variance and loadings | ●○○ | 25 min | [exercise](exercises/B_with_sklearn/R6_warmup_the_first_pca.ipynb) · [solution](solutions/B_with_sklearn/R6_warmup_the_first_pca.ipynb) |

### C · Break the model

*Build the case where the method fails yourself, and measure how badly it fails.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| C1 | The signal sits in the ninth component: PCR fails | ●●● | 50 min | [exercise](exercises/C_break_the_model/C1_the_signal_in_the_discarded_component.ipynb) · [solution](solutions/C_break_the_model/C1_the_signal_in_the_discarded_component.ipynb) |
| ★ C2 | Three outliers out of 303 points hijack the first component | ●●○ | 40 min | [exercise](exercises/C_break_the_model/C2_three_outliers_hijack_pc1.ipynb) · [solution](solutions/C_break_the_model/C2_three_outliers_hijack_pc1.ipynb) |

### D · Interpret

*Read outputs, coefficients and plots, and answer questions that have a precise answer.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ D1 | Reading a PCA: loadings, explained variance and relationship with the diagnosis | ●●○ | 45 min | [exercise](exercises/D_interpret/D1_reading_loadings_and_variance_on_tumors.ipynb) · [solution](solutions/D_interpret/D1_reading_loadings_and_variance_on_tumors.ipynb) |
| D2 | What PCA sees in the digits: pixels, centroids and lost information | ●●● | 50 min | [exercise](exercises/D_interpret/D2_what_pca_sees_in_the_digits.ipynb) · [solution](solutions/D_interpret/D2_what_pca_sees_in_the_digits.ipynb) |

### E · Find the bug

*Code that runs without errors and produces plausible numbers, but is wrong.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| ★ E1 | PCA redone on the test set: accuracy below chance | ●●○ | 40 min | [exercise](exercises/E_find_the_bug/E1_pca_fitted_on_the_test_set.ipynb) · [solution](solutions/E_find_the_bug/E1_pca_fitted_on_the_test_set.ipynb) |
| ★ E2 | TruncatedSVD does not center: the first component is the mean | ●●○ | 40 min | [exercise](exercises/E_find_the_bug/E2_truncatedsvd_instead_of_pca.ipynb) · [solution](solutions/E_find_the_bug/E2_truncatedsvd_instead_of_pca.ipynb) |
| ★ E3 | A colleague discovers that wines are a one-dimensional problem | ●●○ | 35 min | [exercise](exercises/E_find_the_bug/E3_pca_on_the_raw_wine_data.ipynb) · [solution](solutions/E_find_the_bug/E3_pca_on_the_raw_wine_data.ipynb) |

### F · Compare

*A controlled experiment that compares two or more approaches.*

| ID | Title | Level | Time | |
|---|---|---|---|---|
| F1 | PCA or feature selection? Three data shapes, five strategies | ●●● | 55 min | [exercise](exercises/F_compare/F1_pca_vs_feature_selection.ipynb) · [solution](solutions/F_compare/F1_pca_vs_feature_selection.ipynb) |
| F2 | Linear PCA against KernelPCA on nonlinear data (and what it costs) | ●●● | 55 min | [exercise](exercises/F_compare/F2_linear_pca_vs_kernel_pca.ipynb) · [solution](solutions/F_compare/F2_linear_pca_vs_kernel_pca.ipynb) |

## Suggested path

The order of the types (A-F) is not the order in which it makes sense to do them: the exercises use tools introduced at different points. This order follows the lessons.

- **Lesson 6.1** → [R6](exercises/B_with_sklearn/R6_warmup_the_first_pca.ipynb), [A1](exercises/A_from_scratch/A1_pca_from_scratch_with_svd.ipynb), [D1](exercises/D_interpret/D1_reading_loadings_and_variance_on_tumors.ipynb), [E3](exercises/E_find_the_bug/E3_pca_on_the_raw_wine_data.ipynb), [C2](exercises/C_break_the_model/C2_three_outliers_hijack_pc1.ipynb)
- **Lesson 6.2** → [A2](exercises/A_from_scratch/A2_power_iteration_and_spectral_gap.ipynb), [B1](exercises/B_with_sklearn/B1_choosing_k_in_a_pipeline_on_digits.ipynb), [D2](exercises/D_interpret/D2_what_pca_sees_in_the_digits.ipynb), [E1](exercises/E_find_the_bug/E1_pca_fitted_on_the_test_set.ipynb)
- **Lesson 6.3** → [B2](exercises/B_with_sklearn/B2_pca_to_remove_noise.ipynb), [B3](exercises/B_with_sklearn/B3_regression_on_components_and_pls.ipynb), [C1](exercises/C_break_the_model/C1_the_signal_in_the_discarded_component.ipynb), [E2](exercises/E_find_the_bug/E2_truncatedsvd_instead_of_pca.ipynb), [F1](exercises/F_compare/F1_pca_vs_feature_selection.ipynb), [F2](exercises/F_compare/F2_linear_pca_vs_kernel_pca.ipynb)
