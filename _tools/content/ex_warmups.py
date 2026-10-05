"""Warm-up exercises (level 1, guided): one for modules 1 and 4-9.

They are the only "easy" exercises of the course: they are meant to get you comfortable with the tool of the module before tackling
the non-trivial ones. They have almost complete starter code, with a few `...` to fill in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

# ---------------------------------------------------------------------------------------- R1 (module 1)

R1 = dict(
    id="R1", slug="warmup_the_first_regression", kind="B", level=1, minutes=25, module="module_01_linear_regression",
    title="Warm-up: the first regression, from data to number",
    goal="""
        Go through the minimal cycle of a model on your own: split the data, train a **dumb baseline** (the mean), train a linear regression, compare the error on the training and test sets and read the coefficients.
    """,
    dataset="""
        `ds.diabetes()`: 442 patients, 10 already standardized variables (`age`, `sex`, `bmi`, `bp`, `s1`...`s6`); the target is an index of disease progression after one year.
    """,
    purpose="""
        Get comfortable with `fit`/`predict`/`score`, understand that the R² is read **against the baseline** (the mean is worth 0) and that the number that matters is the one on the test set, not the one on the training set.
    """,
    task=[
        "Split the data: `X_tr, X_te, y_tr, y_te` with `test_size=0.25` and `random_state=0`.",
        "Baseline: `DummyRegressor()` (always predicts the training mean). Save `baseline_r2` (R² on the test set) and `baseline_rmse`.",
        "Model: `LinearRegression()`. Save `train_r2`, `test_r2` and `test_rmse`.",
        "Coefficients: `coef` (`pd.Series` with the column names) and `most_influential` (the variable with the largest coefficient in absolute value).",
    ],
    setup="""
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split

        X, y = ds.diabetes()
        print(X.shape, "| target: mean", round(float(y.mean()), 1), "standard deviation", round(float(y.std()), 1))
    """,
    starter=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(..., ..., test_size=..., random_state=...)
        """),
        C("""
        baseline = DummyRegressor().fit(X_tr, y_tr)
        baseline_r2 = baseline.score(X_te, y_te)
        baseline_rmse = mean_squared_error(y_te, baseline.predict(X_te), squared=False)
        """),
        C("""
        model = LinearRegression().fit(..., ...)
        train_r2 = model.score(..., ...)
        test_r2 = model.score(..., ...)
        test_rmse = mean_squared_error(y_te, model.predict(...), squared=False)
        """),
        C("""
        coef = pd.Series(model.coef_, index=X.columns)
        most_influential = ...
        """),
    ],
    checks="""
        check(len(X_tr) == 331 and len(X_te) == 111, "75/25 split: 331 training rows and 111 test rows", "test_size=0.25, random_state=0")
        check(abs(baseline_r2) < 0.05, "the baseline has R² close to 0: predicting the mean explains nothing", "DummyRegressor().score(X_te, y_te)")
        check(0.3 < test_r2 < 0.5 and train_r2 > test_r2, "test R² between 0.3 and 0.5, and lower than the training one", "model.score(X_te, y_te)")
        check(test_rmse < baseline_rmse - 8, "the error of the model is at least 8 points below that of the baseline", "squared=False for the root")
        check(most_influential == "s5", "the variable with the largest coefficient is s5", "coef.abs().idxmax()")
    """,
    hint="""
        `train_test_split(X, y, test_size=0.25, random_state=0)`; `model.fit(X_tr, y_tr)`, `model.score(X_tr, y_tr)` and `model.score(X_te, y_te)`; `coef.abs().idxmax()`.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        baseline = DummyRegressor().fit(X_tr, y_tr)
        baseline_r2 = baseline.score(X_te, y_te)
        baseline_rmse = mean_squared_error(y_te, baseline.predict(X_te), squared=False)
        print(f"baseline (the mean): test R² {baseline_r2:.3f} | RMSE {baseline_rmse:.1f}")
        """),
        C("""
        model = LinearRegression().fit(X_tr, y_tr)
        train_r2, test_r2 = model.score(X_tr, y_tr), model.score(X_te, y_te)
        test_rmse = mean_squared_error(y_te, model.predict(X_te), squared=False)
        print(f"linear regression: training R² {train_r2:.3f} | test R² {test_r2:.3f} | test RMSE {test_rmse:.1f}")
        """),
        C("""
        coef = pd.Series(model.coef_, index=X.columns)
        most_influential = coef.abs().idxmax()
        print(coef.round(0).sort_values().to_string())
        print("most influential:", most_influential)
        """),
    ],
    learned="""
        - **The baseline is worth R² ≈ 0:** always predicting the mean gives 0 (the test set has a slightly different mean, so 0.000 or a little less). A model is useful only if it beats this number: the linear regression reaches R² 0.36 and reduces the RMSE from 70.5 to 56.4.
        - **Training and test do not coincide:** R² 0.56 on the training set and 0.36 on the test set. The gap is normal with 331 examples and 10 variables, and it is the reason the evaluation is always done on data the model has not seen.
        - **Not all the coefficients are readable:** `s5` and `bmi` have the highest coefficients, but `s1` (−560) and `s2` (+261) are highly correlated with each other and their coefficients compensate each other: alone they mean nothing. You come back to this in module 1 (exercises C1 and D1).
    """,
    pitfall="""
        Reporting the training R² as "the quality of the model": it is always optimistic. And reading an R² without comparing it with the baseline.
    """,
    interview=("What does an R² of 0.36 on the test set mean?", """
        That the model explains 36% of the variance of the target compared with always predicting the training mean (which is worth 0): it is useful but leaves a lot of unexplained variability. The number has to be read on the test set and next to the baseline, and accompanied by an error in the units of the target (RMSE).
        """),
)

# ---------------------------------------------------------------------------------------- R4 (module 4)

R4 = dict(
    id="R4", slug="warmup_the_first_logistic_and_the_threshold", kind="B", level=1, minutes=25, module="module_04_logistic_regression",
    title="Warm-up: the first logistic regression and the threshold",
    goal="""
        Train a logistic regression in a `Pipeline` with scaling, read accuracy and AUC, and see what happens to **precision and recall** when the decision threshold is moved away from 0.5.
    """,
    dataset="""
        `load_breast_cancer`: 569 tumors with 30 measurements; `y = 1` if the tumor is **malignant** (37%). Stratified 75/25 split.
    """,
    purpose="""
        Understand that `predict` uses the 0.5 threshold out of habit and not out of necessity: when missing a malignant case costs more than a false alarm, you lower the threshold and accept losing precision to gain recall.
    """,
    task=[
        "Build `pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))`, train it and save `test_acc` (accuracy) and `test_auc`.",
        "`p_mal = pipe.predict_proba(X_te)[:, 1]`. For the thresholds `[0.5, 0.2, 0.05]` build `table` (index: threshold, columns `recall` and `precision` of the malignant class).",
        "Save `screening_threshold`: the **highest** threshold of the grid `np.linspace(0.01, 0.99, 99)` for which the recall of the malignant cases is at least 0.98.",
        "Without scaler: `acc_without_scaler` and `iter_without` (`n_iter_[0]` of a `LogisticRegression(max_iter=1000)` trained directly on `X_tr`).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, (data.target == 0).astype(int)                       # 1 = malignant
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        print(X_tr.shape, X_te.shape, "| share of malignant cases:", round(float(y.mean()), 3))
    """,
    starter=[
        C("""
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(..., ...)
        test_acc = accuracy_score(y_te, pipe.predict(X_te))
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, ...])
        """),
        C("""
        p_mal = pipe.predict_proba(X_te)[:, 1]
        table = pd.DataFrame(index=[0.5, 0.2, 0.05], columns=["recall", "precision"], dtype=float)
        for threshold in table.index:
            predicted = (p_mal >= threshold).astype(int)
            table.loc[threshold] = [recall_score(y_te, predicted), precision_score(y_te, ...)]
        table.round(3)
        """),
        C("""
        thresholds = np.linspace(0.01, 0.99, 99)
        recalls = np.array([recall_score(y_te, (p_mal >= s).astype(int)) for s in thresholds])
        screening_threshold = ...
        """),
        C("""
        without = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        acc_without_scaler = ...
        iter_without = ...
        """),
    ],
    checks="""
        check(0.93 < test_acc < 0.99 and test_auc > 0.98, "accuracy between 0.93 and 0.99, AUC over 0.98", "auc = roc_auc_score(y_te, p_mal)")
        check(list(table.index) == [0.5, 0.2, 0.05] and table["recall"].is_monotonic_increasing and table["precision"].is_monotonic_decreasing, "lowering the threshold the recall rises and the precision drops", "recall_score and precision_score with (p_mal >= threshold)")
        check(table.loc[0.05, "recall"] >= 0.98 and table.loc[0.05, "precision"] < table.loc[0.5, "precision"] - 0.05, "with threshold 0.05 the recall reaches 0.98 or more, at the expense of precision (over 5 points less)", "compare the rows of the table")
        check(0.02 < screening_threshold < 0.4 and recall_score(y_te, (p_mal >= screening_threshold).astype(int)) >= 0.98, "the screening threshold is low and gives a recall of at least 0.98", "thresholds[recalls >= 0.98].max()")
        check(acc_without_scaler < test_acc and iter_without >= 1000, "without scaler the accuracy is lower and the solver hits the limit of 1000 iterations without converging", "without.score(X_te, y_te), without.n_iter_[0]")
    """,
    hint="""
        `thresholds[recalls >= 0.98].max()`. The recall of a class: `recall_score(y_te, predicted)` with `predicted = (p_mal >= threshold).astype(int)`.
    """,
    solution=[
        C("""
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
        test_acc = accuracy_score(y_te, pipe.predict(X_te))
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        print(f"accuracy {test_acc:.3f} | AUC {test_auc:.3f}")
        """),
        C("""
        p_mal = pipe.predict_proba(X_te)[:, 1]
        table = pd.DataFrame(index=[0.5, 0.2, 0.05], columns=["recall", "precision"], dtype=float)
        for threshold in table.index:
            predicted = (p_mal >= threshold).astype(int)
            table.loc[threshold] = [recall_score(y_te, predicted), precision_score(y_te, predicted)]
        table.round(3)
        """),
        C("""
        thresholds = np.linspace(0.01, 0.99, 99)
        recalls = np.array([recall_score(y_te, (p_mal >= s).astype(int)) for s in thresholds])
        screening_threshold = float(thresholds[recalls >= 0.98].max())
        print(f"highest threshold with recall >= 0.98: {screening_threshold:.2f}")
        """),
        C("""
        without = LogisticRegression(max_iter=1000).fit(X_tr, y_tr)
        acc_without_scaler, iter_without = without.score(X_te, y_te), int(without.n_iter_[0])
        print(f"without scaler: accuracy {acc_without_scaler:.3f}, iterations {iter_without} (with the scaler: {int(pipe[-1].n_iter_[0])})")
        """),
    ],
    learned="""
        - **Accuracy and AUC say different things:** with threshold 0.5 the model has accuracy 0.986 and AUC 0.996. The AUC measures how well the scores **rank** the cases, independently of the threshold.
        - **The threshold is a decision, not a parameter of the model:** lowering it from 0.5 to 0.05 raises the recall of the malignant cases from 0.96 to 1.00 (fewer missed tumors) and lowers the precision from 1.00 to 0.78 (more false alarms). Which trade-off to choose depends on the cost of the two errors, not on the statistics: it is the theme of exercises B2 and D2 of this module.
        - **Scaling is not a detail here either:** without `StandardScaler` the logistic regression does not converge within 1000 iterations (with the scaler 30 are enough) and the accuracy drops from 0.986 to 0.965. It is the subject of lesson 4.2.
    """,
    pitfall="""
        Reporting only the accuracy at threshold 0.5 as if it were "the model": the threshold is a decision choice, and with imbalanced classes or different costs 0.5 is almost never the right one.
    """,
    interview=("Why does lowering the threshold increase the recall and reduce the precision?", """
        Lowering the threshold classifies more cases as positive: more of the true positives are caught (recall up) but more false positives come in too (precision down). The threshold is chosen with the costs of the two errors and validated on data not used to choose it.
        """),
)

# ---------------------------------------------------------------------------------------- R5 (module 5)

R5 = dict(
    id="R5", slug="warmup_knn_k_and_scale", kind="B", level=1, minutes=25, module="module_05_knn",
    title="Warm-up: KNN, the value of k and the effect of scale",
    goal="""
        Train a KNN on 13 chemical measurements of wines with very different scales, compare `k = 1, 5, 15` **without and with** standardization and find which variable dominates the distance.
    """,
    dataset="""
        `load_wine`: 178 wines of 3 cultivars, 13 measurements (`proline` has values in the hundreds or thousands, `hue` around 1). Stratified 70/30 split.
    """,
    purpose="""
        See with your own eyes that a KNN computes **distances**, so the scale of the variables decides which ones count: without standardization a single variable dominates and the result gets much worse, whatever `k` you choose.
    """,
    task=[
        "`table` (index `k` in `[1, 5, 15]`, columns `unscaled` and `scaled`): test accuracy of `KNeighborsClassifier(k)` alone and inside `make_pipeline(StandardScaler(), ...)`.",
        "`mean_gain` (mean over `k` of `scaled − unscaled`).",
        "`std` (`pd.Series`: standard deviation of each column of `X_tr`), `dominant_feature` (column with the largest standard deviation) and `scale_ratio` (largest deviation divided by the smallest).",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_wine(as_frame=True)
        X, y = data.data, data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        print(X_tr.shape, X_te.shape)
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=[1, 5, 15], columns=["unscaled", "scaled"], dtype=float)
        for k in table.index:
            plain = KNeighborsClassifier(k).fit(X_tr, y_tr)
            scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(..., ...)
            table.loc[k] = [plain.score(X_te, y_te), scaled.score(..., ...)]
        table.round(3)
        """),
        C("""
        mean_gain = ...
        """),
        C("""
        std = ...
        dominant_feature = ...
        scale_ratio = ...
        """),
    ],
    checks="""
        check(table.notna().all().all() and (table["scaled"] > 0.93).all(), "with scaling the accuracy is above 0.93 for all the k", "make_pipeline(StandardScaler(), KNeighborsClassifier(k))")
        check((table["unscaled"] < 0.8).all() and (table["scaled"] - table["unscaled"] > 0.2).all(), "without scaling the accuracy stays below 0.8 and the standardization gains over 0.2 for each k", "compare the two columns")
        check(mean_gain > 0.25, "mean gain of the standardization over 0.25", "(table['scaled'] - table['unscaled']).mean()")
        check(dominant_feature == "proline" and scale_ratio > 1000, "proline dominates the distance and the ratio between the largest and the smallest scale exceeds 1000", "X_tr.std().idxmax()")
    """,
    hint="""
        `std = X_tr.std()`; `std.idxmax()`; `std.max() / std.min()`.
    """,
    solution=[
        C("""
        table = pd.DataFrame(index=[1, 5, 15], columns=["unscaled", "scaled"], dtype=float)
        for k in table.index:
            plain = KNeighborsClassifier(k).fit(X_tr, y_tr)
            scaled = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_tr, y_tr)
            table.loc[k] = [plain.score(X_te, y_te), scaled.score(X_te, y_te)]
        print(table.round(3).to_string())
        mean_gain = float((table["scaled"] - table["unscaled"]).mean())
        print(f"mean gain of the standardization: {mean_gain:.3f}")
        """),
        C("""
        std = X_tr.std()
        dominant_feature = std.idxmax()
        scale_ratio = float(std.max() / std.min())
        print(std.round(2).sort_values().tail(4).to_string())
        print(f"dominant: {dominant_feature} | ratio between the largest and the smallest scale: {scale_ratio:.0f}")
        """),
    ],
    learned="""
        - **Without scaling, a single variable decides:** `proline` has a standard deviation of 327, against 0.1-14 for the others (a ratio of 2685 between the largest and the smallest scale): in the Euclidean distance the difference in `proline` crushes all the others. The accuracy without scaling is 0.72, 0.72 and 0.67 (for `k = 1, 5, 15`), that is just above predicting a single cultivar.
        - **With `StandardScaler` the accuracy rises to 0.96-1.0** for the same `k`: the mean gain is about 0.28. It is the difference between an unusable model and an almost perfect one, obtained with one line of code.
        - **The scaler goes inside the `Pipeline`:** that way it is fitted only on the training set and the same treatment is applied to the test set (module 3). Comparing the `k` on a test set of 54 examples, as here, is only an illustration: to choose `k` you use cross-validation (lesson 5.3).
    """,
    pitfall="""
        Applying a distance-based model to variables with different units without standardizing. No error, no warning: just a much worse model.
    """,
    interview=("Why does KNN need standardization and a tree does not?", """
        KNN measures distances: a variable with a large scale dominates the computation. A tree uses thresholds on one variable at a time, so changing the units moves the thresholds but not the order of the examples and the result does not change.
        """),
)

# ---------------------------------------------------------------------------------------- R6 (module 6)

R6 = dict(
    id="R6", slug="warmup_the_first_pca", kind="B", level=1, minutes=25, module="module_06_pca",
    title="Warm-up: the first PCA, explained variance and loadings",
    goal="""
        Apply PCA to 13 measurements of wines: how much variance each component explains, how many are needed for 90%, which variables weigh on each one, and what happens if you forget to standardize.
    """,
    dataset="""
        `load_wine`: 178 wines, 13 measurements with very different scales (`proline` in the hundreds).
    """,
    purpose="""
        Read a PCA: `explained_variance_ratio_` (how much), `components_` (the loadings: which variables) and understand why PCA has to be done on standardized variables: otherwise the first component is just the variable with the largest scale.
    """,
    task=[
        "Standardize (`Z = StandardScaler().fit_transform(X)`) and fit `pca = PCA().fit(Z)`. Save `ev = pca.explained_variance_ratio_`, `ev2` (sum of the first two) and `n90` (minimum number of components for 90% of the variance).",
        "`loadings`: `pd.DataFrame` of the loadings of the first two components (index: variable names, columns `PC1` and `PC2`) and `pc1_variable`, `pc2_variable` (the variable with the largest absolute loading in each).",
        "**Without scaling:** `unscaled_ev1` (variance explained by the first component of `PCA().fit(X)`).",
        "**Use in a model:** `acc_2comp` and `acc_13` = test accuracy of `make_pipeline(StandardScaler(), PCA(2), LogisticRegression())` and of `make_pipeline(StandardScaler(), LogisticRegression())` (`X_tr, X_te, y_tr, y_te` already prepared).",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_wine(as_frame=True)
        X, y = data.data, data.target
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    """,
    starter=[
        C("""
        Z = StandardScaler().fit_transform(X)
        pca = PCA().fit(Z)
        ev = pca.explained_variance_ratio_
        ev2 = ...
        n90 = ...
        """),
        C("""
        loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
        pc1_variable = ...
        pc2_variable = ...
        """),
        C("""
        unscaled_ev1 = ...
        """),
        C("""
        acc_2comp = make_pipeline(StandardScaler(), PCA(2), LogisticRegression()).fit(X_tr, y_tr).score(X_te, y_te)
        acc_13 = ...
        """),
    ],
    checks="""
        check(close(ev.sum(), 1.0) and 0.5 < ev2 < 0.6 and n90 == 8, "the components sum to 1; the first two explain 55%; 8 are needed for 90%", "np.cumsum(ev), np.searchsorted(...)")
        check(pc1_variable == "flavanoids" and pc2_variable == "color_intensity", "PC1 is dominated by flavanoids, PC2 by color_intensity", "loadings['PC1'].abs().idxmax()")
        check(unscaled_ev1 > 0.99, "without standardizing the first component explains over 99%: it is just `proline`", "PCA().fit(X).explained_variance_ratio_[0]")
        check(0.85 < acc_2comp < acc_13 <= 1.0, "with 2 components the accuracy is high (over 0.85) but lower than with all the 13 variables", "make_pipeline(StandardScaler(), LogisticRegression())")
    """,
    hint="""
        `n90 = int(np.searchsorted(np.cumsum(ev), 0.9)) + 1`; `loadings["PC1"].abs().idxmax()`; `PCA().fit(X).explained_variance_ratio_[0]`.
    """,
    solution=[
        C("""
        Z = StandardScaler().fit_transform(X)
        pca = PCA().fit(Z)
        ev = pca.explained_variance_ratio_
        ev2 = float(ev[:2].sum())
        n90 = int(np.searchsorted(np.cumsum(ev), 0.9)) + 1
        print("explained variance (first 4):", ev[:4].round(3), "| first two:", round(ev2, 3), "| components for 90%:", n90)
        """),
        C("""
        loadings = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
        pc1_variable, pc2_variable = loadings["PC1"].abs().idxmax(), loadings["PC2"].abs().idxmax()
        print(loadings.round(2).reindex(loadings["PC1"].abs().sort_values(ascending=False).index).head(4).to_string())
        print("PC1:", pc1_variable, "| PC2:", pc2_variable)
        """),
        C("""
        unscaled_ev1 = float(PCA().fit(X).explained_variance_ratio_[0])
        print(f"first component without standardizing: {unscaled_ev1:.4f} of the variance")
        """),
        C("""
        acc_2comp = make_pipeline(StandardScaler(), PCA(2), LogisticRegression()).fit(X_tr, y_tr).score(X_te, y_te)
        acc_13 = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr).score(X_te, y_te)
        print(f"test accuracy: 2 components {acc_2comp:.3f} | 13 variables {acc_13:.3f}")
        """),
    ],
    learned="""
        - **The explained variance is read cumulatively:** the first component is worth 36%, the first two 55%, and 8 out of 13 are needed to reach 90%. Here the data do not compress much: the variables are informative and only partly redundant.
        - **The loadings say what a component represents:** PC1 is dominated by `flavanoids` (with `total_phenols` and `od280/od315`: the measurements linked to phenols), PC2 by `color_intensity`. A loading that is large in absolute value means that variable weighs a lot on the component.
        - **Without standardizing PCA is useless:** the first component explains 99.8% of the variance, it is simply `proline` (standard deviation 315 against 0.1-14 for the others). PCA maximizes the variance in absolute value, so it follows whoever has the largest scale.
        - **Fewer components, less information:** with 2 components the logistic regression reaches 0.93 accuracy on the test set, with all the 13 variables 1.0. The compression has a cost, which you measure (lesson 6.3).
    """,
    pitfall="""
        Doing PCA on variables with different scales without standardizing: the first component is the variable with the largest scale and the result looks "very informative" (99% of the variance) without being so.
    """,
    interview=("What does it mean that the first two components explain 55% of the variance?", """
        That projecting the data onto the plane of the first two components preserves 55% of the total variance (of the standardized variables): it is a measure of reconstruction fidelity, not of usefulness for predicting something. It is looked at together with the ability of the model to use the components.
        """),
)

# ---------------------------------------------------------------------------------------- R7 (module 7)

R7 = dict(
    id="R7", slug="warmup_the_first_tree_and_the_depth", kind="B", level=1, minutes=25, module="module_07_decision_trees",
    title="Warm-up: the first tree and the depth",
    goal="""
        Train trees of increasing depth, observe how the training accuracy rises up to 1.0 while the test one stalls, and read the **rule at the root** of a small tree.
    """,
    dataset="""
        `load_breast_cancer`: 569 tumors with 30 measurements; `y = 1` if malignant. Stratified 70/30 split.
    """,
    purpose="""
        See the overfitting of a tree: more depth means more leaves and a memorized training set, but after a few splits the test set no longer improves. And learn to read a tree as a set of `if... then` rules.
    """,
    task=[
        "`table` (index: `\"1\"`, `\"2\"`, `\"3\"`, `\"5\"`, `\"no limit\"`; columns `train`, `test`, `leaves`) with `DecisionTreeClassifier(max_depth=..., random_state=0)`.",
        "`gap` (training accuracy minus test accuracy for `max_depth=None`) and `gain_beyond_3` (best test accuracy among the depths beyond 3, that is 5 and no limit, minus that of depth 3).",
        "Tree of depth 2: `root` (name of the variable of the first split, with `tree.tree_.feature[0]`) and `root_threshold` (`tree.tree_.threshold[0]`).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, export_text

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, (data.target == 0).astype(int)                       # 1 = malignant
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    """,
    starter=[
        C("""
        depths = {"1": 1, "2": 2, "3": 3, "5": 5, "no limit": None}
        table = pd.DataFrame(index=list(depths), columns=["train", "test", "leaves"], dtype=float)
        for name, maximum in depths.items():
            tree = DecisionTreeClassifier(max_depth=maximum, random_state=0).fit(X_tr, y_tr)
            table.loc[name] = [tree.score(..., ...), tree.score(..., ...), tree.get_n_leaves()]
        table.round(3)
        """),
        C("""
        gap = ...
        gain_beyond_3 = ...
        """),
        C("""
        tree2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
        print(export_text(tree2, feature_names=list(X.columns)))
        root = ...
        root_threshold = ...
        """),
    ],
    checks="""
        check(table["train"].is_monotonic_increasing and table.loc["1", "leaves"] == 2 and table["train"].iloc[-1] == 1.0, "the training accuracy rises with the depth up to 1.0; depth 1 has 2 leaves", "tree.score(X_tr, y_tr)")
        check(gap > 0.07 and table["test"].max() < 0.95, "the overfitting gap is over 0.07 and the test accuracy does not reach 0.95", "table.loc[None, 'train'] - table.loc[None, 'test']")
        check(gain_beyond_3 < 0.01, "beyond depth 3 the test accuracy does not improve (gain below 0.01): more leaves do not help", "table['test'].iloc[3:].max() - table.loc['3', 'test']")
        check(root in X.columns and any(k in root for k in ["concave", "perimeter", "radius", "area"]) and root_threshold > 0, "the rule at the root is on a shape or size measurement of the nucleus (concave points, perimeter, radius or area)", "X.columns[tree2.tree_.feature[0]]")
    """,
    hint="""
        The last row of the table (no limit) is `table.iloc[-1]`. The name of the root: `X.columns[tree2.tree_.feature[0]]`.
    """,
    solution=[
        C("""
        depths = {"1": 1, "2": 2, "3": 3, "5": 5, "no limit": None}
        table = pd.DataFrame(index=list(depths), columns=["train", "test", "leaves"], dtype=float)
        for name, maximum in depths.items():
            tree = DecisionTreeClassifier(max_depth=maximum, random_state=0).fit(X_tr, y_tr)
            table.loc[name] = [tree.score(X_tr, y_tr), tree.score(X_te, y_te), tree.get_n_leaves()]
        table.round(3)
        """),
        C("""
        gap = float(table["train"].iloc[-1] - table["test"].iloc[-1])
        gain_beyond_3 = float(table["test"].iloc[3:].max() - table.loc["3", "test"])
        print(table.round(3).to_string())
        print(f"\\ntraining-test gap without depth limit: {gap:.3f} | test gain beyond depth 3: {gain_beyond_3:+.3f}")
        """),
        C("""
        tree2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X_tr, y_tr)
        print(export_text(tree2, feature_names=list(X.columns)))
        root = X.columns[tree2.tree_.feature[0]]
        root_threshold = float(tree2.tree_.threshold[0])
        print(f"rule at the root: {root} <= {root_threshold:.3f}")
        """),
    ],
    learned="""
        - **The training accuracy always rises, the test one does not:** from 0.935 to 1.0 on the training set (the tree without limit has 15 leaves and memorizes), while the test accuracy sits between 0.86 and 0.92 and does not follow the training one. The overfitting gap of the tree without limit is about 0.09.
        - **A few rules are enough:** the test accuracy is better with depth 3 (7 leaves, 0.918) than with the tree without limit (15 leaves, 0.906). The first question the tree asks is `mean concave points <= 0.05`: a tumor with many concave points on the contour tends to be malignant. A rule a doctor can read (with another version of scikit-learn the root may be a similar measurement, like `worst perimeter`).
        - **The test set here is small** (171 examples): differences of one or two points between depths are in the noise. To choose the depth you use cross-validation (module 7, exercise B1), not the test set.
    """,
    pitfall="""
        Choosing the depth by looking at the training accuracy: it always rises. Neither the deepest tree nor the "best on the test set" should be used to choose: you need a separate validation.
    """,
    interview=("Why does a tree with no depth limit have accuracy 1.0 on the training set?", """
        Because it keeps splitting until every leaf is pure: it memorizes the training set. It has low bias and high variance, and on the test set it generalizes like a much smaller tree. It is controlled with the depth, the minimum leaf size or cost-complexity pruning.
        """),
)

# ---------------------------------------------------------------------------------------- R8 (module 8)

R8 = dict(
    id="R8", slug="warmup_from_the_tree_to_the_forest", kind="B", level=1, minutes=25, module="module_08_random_forest",
    title="Warm-up: from the tree to the forest",
    goal="""
        Compare in cross-validation a single tree and a forest of 200 trees, check that the **out-of-bag** error estimates the test one well and read the three most important variables.
    """,
    dataset="""
        `load_breast_cancer`: 569 tumors with 30 measurements; `y = 1` if malignant. Stratified 5-fold cross-validation on all the data (`cv`), and a 70/30 split (`X_tr, X_te, y_tr, y_te`).
    """,
    purpose="""
        See the gain of bagging first hand (average of many trees ⇒ less variance), learn that the OOB is a free validation and that impurity importances have to be read with caution.
    """,
    task=[
        "`tree_acc` and `forest_acc`: mean cross-validation accuracy (5 folds, `cv`) of `DecisionTreeClassifier(random_state=0)` and `RandomForestClassifier(200, random_state=0, n_jobs=-1)`, with the standard deviations `tree_sd`, `forest_sd`; `gain` is the difference between the means.",
        "OOB: `forest = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)`; save `oob` (`forest.oob_score_`) and `test_acc` (accuracy on the test set).",
        "`top3`: the three variables with the largest impurity importance (`forest.feature_importances_`), in descending order.",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.tree import DecisionTreeClassifier

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, (data.target == 0).astype(int)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    """,
    starter=[
        C("""
        tree_scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=cv)
        forest_scores = cross_val_score(..., X, y, cv=cv)
        tree_acc, tree_sd = tree_scores.mean(), tree_scores.std()
        forest_acc, forest_sd = ..., ...
        gain = ...
        """),
        C("""
        forest = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        oob = ...
        test_acc = ...
        """),
        C("""
        top3 = ...
        """),
    ],
    checks="""
        check(0.90 < tree_acc < 0.95 and 0.95 < forest_acc < 0.99, "tree between 0.90 and 0.95, forest between 0.95 and 0.99", "cross_val_score(..., cv=cv).mean()")
        check(gain > 0.02 and forest_sd < tree_sd + 0.01, "the forest gains over 2 points and is not more variable than the tree", "forest_acc - tree_acc")
        check(abs(oob - test_acc) < 0.03 and oob > 0.94, "the OOB accuracy (over 0.94) is within 3 points of the test one: a free validation", "forest.oob_score_")
        check(len(top3) == 3 and sum("concave points" in v for v in top3) == 2, "among the three most important variables appear the two `concave points` measurements (`mean` and `worst`)", "np.argsort(-forest.feature_importances_)[:3]")
    """,
    hint="""
        `pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False).index[:3].tolist()`.
    """,
    solution=[
        C("""
        tree_scores = cross_val_score(DecisionTreeClassifier(random_state=0), X, y, cv=cv)
        forest_scores = cross_val_score(RandomForestClassifier(200, random_state=0, n_jobs=-1), X, y, cv=cv)
        tree_acc, tree_sd = float(tree_scores.mean()), float(tree_scores.std())
        forest_acc, forest_sd = float(forest_scores.mean()), float(forest_scores.std())
        gain = forest_acc - tree_acc
        print(f"tree {tree_acc:.3f} +/- {tree_sd:.3f} | forest {forest_acc:.3f} +/- {forest_sd:.3f} | gain {gain:.3f}")
        """),
        C("""
        forest = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        oob, test_acc = float(forest.oob_score_), float(forest.score(X_te, y_te))
        print(f"OOB accuracy {oob:.3f} | test accuracy {test_acc:.3f}")
        """),
        C("""
        importances = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False)
        top3 = importances.index[:3].tolist()
        print(importances.head(5).round(3).to_string())
        """),
    ],
    learned="""
        - **Averaging trees works:** in cross-validation the single tree has accuracy 0.926 and the forest 0.963 (+3.7 points), with a similar or smaller variability across folds (0.022 against 0.017). No optimization: 200 trees with the default parameters.
        - **The OOB is a free validation:** each tree has not seen about a third of the examples, and the average of the "out-of-sample" predictions gives an accuracy (0.957) very close to the one measured on the test set (0.942). With independent data you can avoid keeping a validation set aside; with duplicate or grouped rows you cannot (module 8, exercise C1).
        - **Impurity importances have to be taken with caution:** the highest are `mean concave points` (0.146), `worst concave points` (0.136) and `worst perimeter` (0.107), but right after come `worst radius` (0.105) and `worst area` (0.100), which measure almost the same thing as the perimeter: with highly correlated variables the importance is split among the copies. For a reliable judgment you use the permutation on the test set (lesson 8.2).
    """,
    pitfall="""
        Searching for the "right" number of trees with a grid: more trees never hurt (the error settles), you choose a number high enough and pay in time.
    """,
    interview=("What is the out-of-bag error and when is it not reliable?", """
        It is the error computed, for each example, using only the trees that did not see it in their bootstrap sample: a validation estimate without a separate set. It is not reliable if the examples are not independent (duplicate rows, repeated measurements, time series), because a copy of the example can be in the sample of the tree.
        """),
)

# ---------------------------------------------------------------------------------------- R9 (module 9)

R9 = dict(
    id="R9", slug="warmup_the_first_xgboost_and_early_stopping", kind="B", level=1, minutes=25, module="module_09_xgboost",
    title="Warm-up: the first XGBoost and early stopping",
    goal="""
        Train an `XGBClassifier` with **early stopping** on a validation set, see how many rounds it really uses, compare it with the same model at a fixed 1000 rounds and with a forest.
    """,
    dataset="""
        `make_classification(3000, 12 features, 5 informative, flip_y=0.05)`; 60/20/20 split (`X_tr, X_val, X_te`, with the respective labels).
    """,
    purpose="""
        Learn the basic boosting workflow (low learning rate, many rounds as a ceiling, early stopping on the validation set) and understand that **boosting does not always win**: on noisy data a forest with the defaults can do just as well.
    """,
    task=[
        "Model: `xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric=\"logloss\", early_stopping_rounds=30, n_jobs=-1, random_state=0)`, trained with `eval_set=[(X_val, y_val)]`. Save `rounds_used` (`best_iteration + 1`), `es_auc` and `es_ll` (AUC and log-loss on the test set).",
        "The same model **without** early stopping (1000 rounds, without `early_stopping_rounds` or `eval_set`): `ll_1000` (log-loss on the test set).",
        "Forest `RandomForestClassifier(300, random_state=0, n_jobs=-1)`: `rf_auc`. Save `xgb_wins` (True if `es_auc > rf_auc`).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = make_classification(3000, n_features=12, n_informative=5, class_sep=0.8, flip_y=0.05, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
    """,
    starter=[
        C("""
        model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric="logloss",
                                  early_stopping_rounds=30, n_jobs=-1, random_state=0)
        model.fit(X_tr, y_tr, eval_set=[(..., ...)], verbose=False)
        rounds_used = ...
        p = model.predict_proba(X_te)[:, 1]
        es_auc, es_ll = roc_auc_score(y_te, p), log_loss(y_te, p)
        """),
        C("""
        no_es = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr)
        ll_1000 = ...
        """),
        C("""
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        rf_auc = ...
        xgb_wins = ...
        """),
    ],
    checks="""
        check(50 < rounds_used < 500 and es_auc > 0.9, "early stopping halts between 50 and 500 rounds (out of 1000) and the AUC exceeds 0.9", "model.best_iteration + 1")
        check(es_ll < ll_1000 - 0.01, "with early stopping the test log-loss is better by over 0.01 than at 1000 rounds", "log_loss(y_te, no_es.predict_proba(X_te)[:, 1])")
        check(0.9 < rf_auc < 1.0 and xgb_wins in (True, False), "forest AUC between 0.9 and 1; `xgb_wins` is a boolean", "roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])")
    """,
    hint="""
        `eval_set=[(X_val, y_val)]`; `model.best_iteration + 1`; `log_loss(y_te, no_es.predict_proba(X_te)[:, 1])`.
    """,
    solution=[
        C("""
        model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric="logloss",
                                  early_stopping_rounds=30, n_jobs=-1, random_state=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        rounds_used = int(model.best_iteration) + 1
        p = model.predict_proba(X_te)[:, 1]
        es_auc, es_ll = float(roc_auc_score(y_te, p)), float(log_loss(y_te, p))
        print(f"rounds used: {rounds_used} out of 1000 | test AUC {es_auc:.4f} | test log-loss {es_ll:.4f}")
        """),
        C("""
        no_es = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=3, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr)
        ll_1000 = float(log_loss(y_te, no_es.predict_proba(X_te)[:, 1]))
        print(f"without early stopping (1000 rounds): test log-loss {ll_1000:.4f} (with early stopping {es_ll:.4f})")
        """),
        C("""
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        rf_auc = float(roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1]))
        xgb_wins = bool(es_auc > rf_auc)
        print(f"test AUC: XGBoost {es_auc:.4f} | forest {rf_auc:.4f} -> XGBoost wins: {xgb_wins}")
        """),
    ],
    learned="""
        - **Early stopping chooses the rounds for you:** the model stops at 286 rounds out of 1000. The high ceiling costs nothing because the training is interrupted when the validation log-loss stops dropping.
        - **Not stopping costs:** at 1000 rounds the test log-loss is 0.357 against 0.334 with early stopping (+7%), while the AUC changes very little (0.929 against 0.932): the overfitting of boosting shows up in the probabilities before it shows in the classification (module 9, exercise C1).
        - **Boosting does not always win:** on this data the forest with the defaults has AUC 0.941 against 0.932 for XGBoost. With noisy data (here 5% of random labels) boosting chases the noise and a forest is more robust; to beat the forest you need tuning (module 9, exercises C2 and F2).
    """,
    pitfall="""
        Using the test set as `eval_set` for early stopping: you chose the number of rounds by looking at the test set. The validation set is a separate set, cut from the training data.
    """,
    interview=("Why do you fix a high number of rounds in boosting and use early stopping?", """
        Because the right number of rounds depends on the data and on the learning rate, and with too many rounds the model overfits (unlike a forest). You set a high ceiling, a low learning rate and let the training stop when the validation metric stops improving, then you evaluate once on the test set.
        """),
)

EXERCISES = [R1, R4, R5, R6, R7, R8, R9]
