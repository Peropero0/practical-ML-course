"""Module 5 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="scaling_and_metric_on_three_datasets", kind="F", level=3, minutes=55, module=MODULE,
    title="Scaling and metric: what really matters, on three datasets",
    goal="""
        Systematically compare three treatments of the scales (none, `StandardScaler`, `MinMaxScaler`) and four metrics (Euclidean, Manhattan, Chebyshev, cosine) on three datasets (wines, breast tumors, handwritten digits) with
        a repeated cross-validation, and establish **which choice weighs most** in each case.
    """,
    dataset="""
        `load_wine` (13 features, very different scales), `load_breast_cancer` (30 features, different scales) and `load_digits` (64 pixels, **all on the same scale** from 0 to 16). Model: `KNeighborsClassifier(7, algorithm="brute", metric=...)`.
        Validation: `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)`, accuracy.
    """,
    purpose="""
        Replace the rule "always scale" with a precise understanding: when scaling is decisive (different scales), when it is neutral or harmful (already homogeneous scales, where standardizing amplifies the noise of nearly constant pixels) and how much changing the metric is worth.
    """,
    task=[
        "Build `table`: a `DataFrame` with a two-level index `(dataset, scaling)` (9 rows: 3 datasets × 3 treatments `none`, `standard`, `minmax`) and as columns the four metrics `euclidean`, `manhattan`, `chebyshev`, `cosine` with the mean cross-validated accuracy. "
        "Use a `Pipeline` with the scaler (or without) and the KNN.",
        "For each dataset save in `best` (dictionary) the pair `(scaling, metric)` with the maximum accuracy.",
        "Save `scaling_gain` (dictionary per dataset): best accuracy with a scaler (`standard` or `minmax`) minus the best accuracy without scaling (`none`), over all the metrics.",
        "Save `metric_spread` (dictionary per dataset): for the `standard` scaling, difference between the best and the worst metric. Then `dominant_factor` (dictionary per dataset with value `'scaling'` if `scaling_gain > metric_spread`, otherwise `'metric'`).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer, load_digits, load_wine
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        datasets = {"wine": load_wine(return_X_y=True), "cancer": load_breast_cancer(return_X_y=True), "digits": load_digits(return_X_y=True)}
        scalers = {"none": None, "standard": StandardScaler, "minmax": MinMaxScaler}
        metrics = ["euclidean", "manhattan", "chebyshev", "cosine"]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    """,
    starter=[
        C("""
        index = pd.MultiIndex.from_product([datasets, scalers], names=["dataset", "scaling"])
        table = pd.DataFrame(index=index, columns=metrics, dtype=float)
        for (name, sc) in table.index:
            X, y = datasets[name]
            for metric in metrics:
                ...
        table.round(3)
        """),
        C("""
        best = {}
        scaling_gain = {}
        metric_spread = {}
        dominant_factor = {}
        """),
    ],
    checks="""
        check(table.shape == (9, 4) and table.notna().all().all() and list(table.columns) == metrics, "9 rows (3 datasets x 3 scalings) and 4 metrics, with no gaps", "one accuracy for every combination")
        check(table.loc[("wine", "none")].max() < 0.82 and table.loc[("wine", "standard")].min() > 0.92, "wine: without scaling below 0.82; with StandardScaler above 0.92 with any metric", "the scales of the wines are very different")
        check(scaling_gain["wine"] > 0.15 and 0.005 < scaling_gain["cancer"] < 0.08, "gain of scaling: over 0.15 on the wines, between 0.005 and 0.08 on the cancer data", "best with scaler minus best without")
        check(scaling_gain["digits"] < 0.005 and table.loc[("digits", "standard"), "euclidean"] < table.loc[("digits", "none"), "euclidean"], "on the digits scaling does not help (gain below 0.005) and StandardScaler worsens the Euclidean", "the pixels already have the same scale")
        check(best["wine"][0] in ("standard", "minmax") and best["digits"][0] in ("none", "minmax"), "the best treatment: a scaler on the wines; none (or minmax) on the digits", "idxmax of the table")
        check(metric_spread["wine"] < 0.06 and scaling_gain["wine"] > 4 * metric_spread["wine"], "on the wines the metric changes less than 0.06 and scaling weighs over 4 times more", "spread = max - min across metrics with StandardScaler")
        check(dominant_factor["wine"] == "scaling" and dominant_factor["digits"] == "metric", "the dominant factor: scaling for the wines, metric for the digits (on the cancer data the two weigh about the same)", "compare scaling_gain and metric_spread")
    """,
    hint="""
        `steps = [] if sc == "none" else [scalers[sc]()]`; `m = make_pipeline(*steps, KNeighborsClassifier(7, algorithm="brute", metric=metric))`; `cross_val_score(m, X, y, cv=cv).mean()`. Best: `table.loc[name].stack().idxmax()`.
        Digits: the table has `none`, `standard`, `minmax`: compare with `table.loc[name]`.
    """,
    solution=[
        C("""
        index = pd.MultiIndex.from_product([datasets, scalers], names=["dataset", "scaling"])
        table = pd.DataFrame(index=index, columns=metrics, dtype=float)
        for (name, sc) in table.index:
            X, y = datasets[name]
            steps = [] if sc == "none" else [scalers[sc]()]
            for metric in metrics:
                model = make_pipeline(*steps, KNeighborsClassifier(7, algorithm="brute", metric=metric))
                table.loc[(name, sc), metric] = cross_val_score(model, X, y, cv=cv).mean()
        table.round(3)
        """),
        C("""
        best, scaling_gain, metric_spread, dominant_factor = {}, {}, {}, {}
        for name in datasets:
            t = table.loc[name]
            best[name] = t.stack().idxmax()
            scaling_gain[name] = t.loc[["standard", "minmax"]].to_numpy().max() - t.loc["none"].max()
            metric_spread[name] = t.loc["standard"].max() - t.loc["standard"].min()
            dominant_factor[name] = "scaling" if scaling_gain[name] > metric_spread[name] else "metric"
        print(pd.DataFrame({"best": best, "scaling gain": scaling_gain, "metric spread": metric_spread, "dominant": dominant_factor}).round(3))
        """),
    ],
    learned="""
        - **On the wines scaling is everything**: without a scaler the accuracy sits between 0.67 and 0.80 (`proline` decides alone), with `StandardScaler` or `MinMaxScaler` it goes up to 0.94-0.97 with any metric. Changing the metric moves it by 3 points, changing the scaling by 27.
        - **On the cancer data scaling helps only a little** (about 4 points) because the most variable features (areas) are also informative; the metric counts almost as much (2-3 points).
        - **On the digits scaling is not needed and `StandardScaler` makes it worse** (0.972 against 0.983): the 64 pixels are already on the same scale (0-16), and standardizing them magnifies the small variations of the almost always dark pixels, which are noise. Here `minmax`
          (a constant factor) coincides with not scaling, and the cosine does not get worse with it (0.984 both with and without scaling). **The rule is not "always scale": it is "bring the columns to a comparable weight for the problem".**
        - Manhattan is often a bit better than Euclidean with many features and outliers; Chebyshev (only the maximum gap) is almost always worse; the cosine works well where the direction matters (digits), badly on unscaled wines... and even better after standardization.
          The metric is a second-order hyperparameter: scales first, then the rest.
    """,
    pitfall="""
        Applying `StandardScaler` to everything out of habit. With already homogeneous data (pixels, counts, percentages) it can make things worse, and the signal is not an error but a lower number that nobody compares with the baseline without scaling.
    """,
    interview=(
        "Do I always have to scale the features before a KNN?",
        """
        Almost always, because the distance sums the differences and different scales decide by themselves. But it is not a rule: if the features already have the same meaning and the same scale (pixels, histograms, normalized counts) standardizing can amplify the noise
        of the nearly constant columns. I always compare with and without a scaler in cross-validation and also look at the metric, which usually counts less.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="knn_logistic_forest_on_four_data_shapes", kind="F", level=3, minutes=60, module=MODULE,
    title="KNN, logistic and random forest on four shapes of data",
    goal="""
        Compare three models (scaled KNN with `k=15`, logistic regression, random forest of 100 trees) on **four datasets of different shapes**: nonlinear boundary in 2D, linear boundary in 40 dimensions, nonlinear boundary with 30 noise columns, and a
        large dataset. Measure accuracy, training time and prediction time, and summarize when each model is the right choice.
    """,
    dataset="""
        `datasets()` (given): dictionary name → `(X, y)` with `moons 2D` (3000 points, `noise=0.3`), `linear 40D` (3000 points, 40 features, linear boundary), `moons + 30 noise` (the moons with 30 noise columns), `large` (40,000 points, 20 features). 70/30 split.
    """,
    purpose="""
        Build the habit of **choosing the model by looking at the shape of the data** and not at the fashionable name: every model has a case where it wins and one where it loses, and the cost (training and prediction) is part of the choice.
    """,
    task=[
        "For each of the four datasets and each of the three models (`knn`, `logistic`, `forest`) measure: `acc` (test accuracy), `fit_time` (seconds for `fit`) and `pred_time` (seconds for `predict` on the test set). Save `table` with index `(dataset, model)` and the three columns.",
        "Save `best` (dictionary dataset → model with maximum accuracy) and `worst` (dictionary dataset → model with minimum accuracy).",
        "Save `knn_fastest_fit` (True if on the `large` dataset the KNN's `fit_time` is the smallest of the three) and `pred_ratio` (on the `large` dataset: `pred_time` of the KNN divided by `pred_time` of the logistic).",
        "Answer with a string chosen among `'knn'`, `'logistic'`, `'forest'`: `for_low_latency` (the model best suited to serving thousands of predictions per second on a large dataset) and `for_noise` (the most robust to many useless features, among the three).",
    ],
    setup="""
        import time
        from sklearn.datasets import make_classification, make_moons
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def datasets():
            rng = np.random.default_rng(0)
            moons, y_moons = make_moons(3000, noise=0.3, random_state=0)
            Xl = rng.normal(size=(3000, 40))
            w = np.r_[rng.normal(size=10), np.zeros(30)]                     # only 10 features out of 40 matter, linearly
            yl = rng.binomial(1, 1 / (1 + np.exp(-3 * Xl @ w / np.linalg.norm(w))))
            large = make_classification(40000, n_features=20, n_informative=8, n_redundant=0, class_sep=1.0, random_state=0)
            return {"moons 2D": (moons, y_moons), "linear 40D": (Xl, yl), "moons + 30 noise": (np.c_[moons, rng.normal(size=(3000, 30))], y_moons), "large": large}


        models = {
            "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "forest": lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1),
        }
    """,
    starter=[
        C("""
        rows = {}
        for name, (X, y) in datasets().items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
            for model_name, make in models.items():
                ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        best = {}
        worst = {}
        knn_fastest_fit = None
        pred_ratio = None
        for_low_latency = None
        for_noise = None
        """),
    ],
    checks="""
        check(table.shape == (12, 3) and list(table.columns) == ["acc", "fit_time", "pred_time"], "12 rows (4 datasets x 3 models) and three columns", "acc, fit_time, pred_time")
        check(best["linear 40D"] == "logistic" and worst["linear 40D"] == "knn", "on the linear boundary in 40 dimensions the logistic wins and the KNN loses", "compare the accuracies")
        check(worst["moons + 30 noise"] == "knn" and best["moons + 30 noise"] == "forest", "with 30 noise columns the forest wins and the KNN is last", "compare the accuracies")
        check(table.loc[("moons 2D", "logistic"), "acc"] < table.loc[("moons 2D", "knn"), "acc"] - 0.03 and worst["moons 2D"] == "logistic", "on the 2D moons the logistic (linear boundary) loses to the KNN by over 3 points", "the boundary is not linear")
        check(knn_fastest_fit is True and pred_ratio > 20, "on the large dataset the KNN has the fastest fit and prediction over 20 times slower than the logistic", "KNN predict computes the distances from the whole training set")
        check(for_low_latency == "logistic" and for_noise == "forest", "low latency: logistic; robustness to noise: forest", "summary of the comparison")
    """,
    hint="""
        For each model: `m = make(); t0 = time.time(); m.fit(X_tr, y_tr); fit_time = time.time() - t0; t0 = time.time(); acc = m.score(X_te, y_te); pred_time = time.time() - t0`. Row: `rows[(name, model_name)] = {"acc": acc, "fit_time": fit_time, "pred_time": pred_time}`.
        Best: `table["acc"].unstack().idxmax(axis=1)` (with `unstack`, the indices become rows/columns).
    """,
    solution=[
        C("""
        rows = {}
        for name, (X, y) in datasets().items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
            for model_name, make in models.items():
                m = make()
                t0 = time.time(); m.fit(X_tr, y_tr); fit_time = time.time() - t0
                t0 = time.time(); acc = m.score(X_te, y_te); pred_time = time.time() - t0
                rows[(name, model_name)] = {"acc": acc, "fit_time": fit_time, "pred_time": pred_time}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        a = table["acc"].unstack()
        best, worst = a.idxmax(axis=1).to_dict(), a.idxmin(axis=1).to_dict()
        knn_fastest_fit = bool(table.loc["large", "fit_time"].idxmin() == "knn")
        pred_ratio = table.loc[("large", "knn"), "pred_time"] / table.loc[("large", "logistic"), "pred_time"]
        for_low_latency = "logistic"
        for_noise = "forest"
        print("best:", best, "\\nworst:", worst, f"\\nKNN on the large dataset: fastest fit = {knn_fastest_fit}, prediction {pred_ratio:.0f} times slower than the logistic")
        """),
    ],
    learned="""
        - **No model wins everywhere.** On the moons (nonlinear boundary in 2 dimensions) the KNN is first (0.91) and the logistic last (0.85: a linear boundary cannot follow a curve). On the **linear** boundary in 40 dimensions the logistic wins (0.82) and the KNN is last (0.74, distances in
          many dimensions with 30 useless columns). With 30 noise columns added to the moons the forest stays at 0.89 while the KNN drops to 0.80 (below the logistic, 0.85). The forest is never last: it is the model most robust to different shapes, at the price of training time and a larger model.
        - **Different costs.** The KNN has almost no fit (0.02 s on the 28,000 examples) but pays at prediction: on the large dataset it is about 140 times slower than the logistic (1 s against 7 ms for 12,000 points). The forest has the most expensive fit (2.4 s against 0.08 s for the logistic) but a
          fast prediction. The logistic has both minimal. To serve thousands of requests per second on a large dataset, the "pure" KNN is out of the question without an approximate index.
        - The comparison has value because it is made **on the same data and with the same split**; but it is still a single split: differences of one or two points (2D moons: KNN 0.91 against forest 0.90; linear: logistic 0.82 against forest 0.81) are not significant. To decide you need repeated
          cross-validation (F1 of module 3).
        - A procedure follows: look at the shape of the data (dimension, linearity, noise, number of rows), **always** try a linear baseline and a tree one, and use KNN if and only if the proximity structure is the heart of the problem.
    """,
    pitfall="""
        Concluding from a comparison on a single dataset (the moons, which are the case KNN likes best) that "KNN is better than the logistic", or from the large dataset that "KNN is unusable". The answer depends on the shape of the data, and the comparison must be repeated on yours.
    """,
    interview=(
        "Tabular data, 50 features, 200,000 rows, classification goal: where do you start?",
        """
        From a linear baseline (regularized logistic) and a tree model (random forest or gradient boosting): they are robust, scale well and at dimension 50 the KNN would have problems both of distance and of prediction cost. I compare with cross-validation, look at the calibration and
        the required latency. I reserve KNN for problems where proximity has a direct meaning (recommendations, similarity search, geographic data) and in that case with approximate indexes.
        """),
)

EXERCISES = [F1, F2]
