"""Module 3 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="temporal_leakage_kfold_on_series", kind="C", level=2, minutes=40, module=MODULE,
    title="Temporal leakage: random K-fold on a time series",
    goal="""
        Show that on a time series a random K-fold gives an R² close to 1 for a model that actually cannot predict the
        future, compare it with `TimeSeriesSplit` and with a true out-of-sample forecast, and compare the model with a seasonal baseline.
    """,
    dataset="""
        `ds.time_series()`: 600 months. `y` = random walk (trend) + annual seasonality + noise. The features are `t` (month
        index) and the season (`s`, `c`). A `RandomForestRegressor` with these features can memorize "what `y` is around month t"
        but does not know what the trend will do in the following months.
    """,
    purpose="""
        Get a hands-on feel for why the splitter must mirror the real use: predicting the future. In a random K-fold every validation month
        has its adjacent months in the training set and the model **interpolates**; in production it will have to **extrapolate**.
    """,
    task=[
        "Compute `r2_kfold` with `cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring=\"r2\").mean()`.",
        "Compute `r2_ts` with `TimeSeriesSplit(5)`.",
        "Compute `r2_future`: train `rf` on the first 480 months and evaluate the R² on the last 120 (the real use situation).",
        "Compute `r2_seasonal`: R² on the last 120 months of the naive forecast \"same month of the previous year\" (`y.shift(12)`), which does not "
        "require any model.",
        "Compute `kfold_distance = |r2_kfold − r2_future|` and `ts_distance = |r2_ts − r2_future|`.",
    ],
    setup="""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import r2_score
        from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score

        s = ds.time_series()
        X, y = s[["t", "s", "c"]], s["y"]
        rf = RandomForestRegressor(100, random_state=0)
        cut = 480
    """,
    starter=[
        C("""
        r2_kfold = ...
        r2_ts = ...
        print(f"R2 with random KFold: {r2_kfold:.3f} | with TimeSeriesSplit: {r2_ts:.3f}")
        """),
        C("""
        r2_future = ...
        r2_seasonal = ...
        kfold_distance = ...
        ts_distance = ...
        print(f"true R2 on the last 120 months: {r2_future:.3f} | forecast 'same month of the previous year': {r2_seasonal:.3f}")
        """),
    ],
    checks="""
        check(r2_kfold > 0.9, "the random K-fold gives a very high R2 (above 0.9)", "with random KFold the model interpolates between adjacent months")
        check(r2_ts < 0.6, "with TimeSeriesSplit the R2 collapses (below 0.6)", "TimeSeriesSplit trains on the past and evaluates on the future")
        check(close(kfold_distance, abs(r2_kfold - r2_future)) and close(ts_distance, abs(r2_ts - r2_future)), "distances computed correctly",
              "distance = |estimated R2 - true R2 on the future|")
        check(ts_distance < kfold_distance / 2, "the temporal estimate is much closer to the truth than the random K-fold one",
              "TimeSeriesSplit must get much closer to the true R2 on the future than the random KFold")
        check(r2_seasonal > r2_future + 0.3, "the baseline 'same month of the previous year' beats the random forest on the future (by more than 0.3)",
              "the forest with t as a feature cannot extrapolate the trend: the seasonal baseline requires no model")
    """,
    hint="""
        `r2_future`: `rf.fit(X.iloc[:cut], y.iloc[:cut]); r2_score(y.iloc[cut:], rf.predict(X.iloc[cut:]))`. Baseline:
        `r2_score(y.iloc[cut:], y.shift(12).iloc[cut:])` (the only rows with NaN are in the first 12 months, not in the last block).
    """,
    solution=[
        C("""
        r2_kfold = cross_val_score(rf, X, y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
        r2_ts = cross_val_score(rf, X, y, cv=TimeSeriesSplit(5), scoring="r2").mean()
        print(f"R2 with random KFold: {r2_kfold:.3f} | with TimeSeriesSplit: {r2_ts:.3f}")
        """),
        C("""
        rf.fit(X.iloc[:cut], y.iloc[:cut])
        r2_future = r2_score(y.iloc[cut:], rf.predict(X.iloc[cut:]))
        r2_seasonal = r2_score(y.iloc[cut:], y.shift(12).iloc[cut:])
        kfold_distance = abs(r2_kfold - r2_future)
        ts_distance = abs(r2_ts - r2_future)
        print(f"true R2 on the last 120 months: {r2_future:.3f} | forecast 'same month of the previous year': {r2_seasonal:.3f}")
        print(f"error of the estimate: KFold {kfold_distance:.3f} | TimeSeriesSplit {ts_distance:.3f}")
        """),
    ],
    learned="""
        - With a random K-fold the random forest looks almost perfect (R² about 0.97): for every validation month it has the nearby months in the
          training set and just interpolates the trend. On the true future its R² is much lower: it does not know what the random walk will do.
        - `TimeSeriesSplit` (always trains on the past) gives an estimate much closer to the truth: the splitter is the one that imitates
          the real use of the model.
        - A model that does not beat a trivial baseline (here "same month of the previous year") has no value, even with a high R² in
          cross-validation. Comparing with a baseline is the first thing to do, and it is free.
    """,
    pitfall="""
        Adding `t` (or the date) as a feature to a tree model and validating randomly: the tree learns to "remember" the periods
        already seen. In production `t` takes never-seen values and the model has nothing left to remember: a tree does not extrapolate.
    """,
    interview=(
        "Your sales forecasting model has R² 0.95 in cross-validation but in production it is very wrong. Where do you start?",
        """
        From how I validated: if I used a random K-fold on temporal data, the validation is an interpolation, not a forecast of the
        future. I redo the evaluation with `TimeSeriesSplit` (or a chronological hold-out), compare with naive baselines (last value, same
        period of the previous year) and check that no feature contains information from the future. If the model does not beat the baseline,
        it is simply not ready.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="group_leakage_patients", kind="C", level=2, minutes=45, module=MODULE,
    title="Group leakage: the same patient in training and in validation",
    goal="""
        Measure the effect of repeated measurements on the performance estimate: compare `KFold` and `GroupKFold` for a flexible model
        and for a rigid one, and show that the model "recognizes" the patient from their measurements.
    """,
    dataset="""
        `ds.patient_measurements()`: 60 patients × 10 measurements = 600 rows, columns `patient`, `m1`, `m2`, `m3`, `outcome`. The measurements are
        almost constant per patient (a fingerprint) and the outcome depends mostly on an individual characteristic that the measurements do not explain.
        In production the model will be used on **never-seen** patients.
    """,
    purpose="""
        Understand that leakage depends on the combination of **data structure** (non-independent rows) and **model capacity**
        (memorizing): a random forest suffers from it enormously, a linear regression hardly at all.
    """,
    task=[
        "For `RandomForestRegressor(100, random_state=0)` and `LinearRegression()` compute the mean R² with `KFold(5, shuffle=True, "
        "random_state=0)` and with `GroupKFold(5)` (with `groups=g`). Build `table` (rows: models; columns: `random`, `group`, `difference`).",
        "With `KFold(5, shuffle=True, random_state=0)` compute `kfold_shared_share`: the fraction of validation rows (mean over the folds) whose "
        "patient also appears in the training set. Compute the same for `GroupKFold` in `group_shared_share`.",
        "Compute `patient_accuracy`: with `KNeighborsClassifier(1)` and random `KFold`, how well the **patient** (the id) is recognized from the "
        "three measurements alone. It is the proof that the measurements are a fingerprint.",
    ],
    setup="""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GroupKFold, KFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier

        p = ds.patient_measurements()
        X, y, g = p[["m1", "m2", "m3"]], p["outcome"], p["patient"]
        kf = KFold(5, shuffle=True, random_state=0)
        gkf = GroupKFold(5)
    """,
    starter=[
        C("""
        models = {"random forest": RandomForestRegressor(100, random_state=0), "linear regression": LinearRegression()}
        rows = {}
        for name, m in models.items():
            ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        def shared_share(splitter, groups=None):
            ...

        kfold_shared_share = ...
        group_shared_share = ...
        patient_accuracy = ...
        """),
    ],
    checks="""
        check(list(table.columns) == ["random", "group", "difference"] and set(table.index) == {"random forest", "linear regression"},
              "table with the two models and the three columns", "columns: random, group, difference")
        check(table.loc["random forest", "random"] > 0.6 and table.loc["random forest", "group"] < 0,
              "the random forest looks great with random K-fold and becomes worse than the mean with GroupKFold", "random > 0.6, group < 0 for the random forest")
        check(table.loc["random forest", "difference"] > 4 * table.loc["linear regression", "difference"] > 0,
              "the effect of the leakage is much larger for the flexible model (over 4 times that of the linear model)",
              "the forest must have a random-group difference much larger than that of the linear regression")
        check(kfold_shared_share > 0.99 and group_shared_share == 0, "K-fold: almost all the validation patients are also in the training set; GroupKFold: never",
              "with KFold the patient appears in the training set; with GroupKFold the groups are disjoint")
        check(patient_accuracy > 0.8, "the measurements identify the patient with over 80% accuracy (against 1.7% by chance)", "the three measurements are a fingerprint of the patient")
    """,
    hint="""
        `cross_val_score(m, X, y, cv=kf, scoring="r2").mean()` and `cross_val_score(m, X, y, cv=gkf, groups=g, scoring="r2").mean()`. For the share:
        `for tr, va in splitter.split(X, y, groups): np.isin(g.iloc[va], g.iloc[tr]).mean()`, then the mean. Recognition: `cross_val_score(KNeighborsClassifier(1), X, g, cv=kf)`.
    """,
    solution=[
        C("""
        models = {"random forest": RandomForestRegressor(100, random_state=0), "linear regression": LinearRegression()}
        rows = {}
        for name, m in models.items():
            random_r2 = cross_val_score(m, X, y, cv=kf, scoring="r2").mean()
            group_r2 = cross_val_score(m, X, y, cv=gkf, groups=g, scoring="r2").mean()
            rows[name] = {"random": random_r2, "group": group_r2, "difference": random_r2 - group_r2}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        def shared_share(splitter, groups=None):
            shares = [np.isin(g.iloc[va], g.iloc[tr]).mean() for tr, va in splitter.split(X, y, groups)]
            return float(np.mean(shares))


        kfold_shared_share = shared_share(kf)
        group_shared_share = shared_share(gkf, g)
        patient_accuracy = cross_val_score(KNeighborsClassifier(1), X, g, cv=kf).mean()
        print(f"validation rows whose patient is in the training set: KFold {kfold_shared_share:.2f} | GroupKFold {group_shared_share:.2f}")
        print(f"recognition of the patient from the measurements: {patient_accuracy:.3f}")
        """),
    ],
    learned="""
        - With random `KFold` every validation row has 8 "twins" (the same patient) in the training set. A random forest recognizes them from the
          measurements and remembers their outcome: R² about 0.75. With `GroupKFold` the validation patients are new and the R² is negative: the model
          has learned nothing transferable. It is the correct estimate if new patients arrive in production.
        - The linear regression loses much less (from 0.06 to −0.20, a difference of 0.26 against 1.52): it does not have the capacity to memorize, so
          the leakage helps it little. **Group leakage inflates mostly the flexible models**: a comparison between models done with the
          wrong splitter picks the model that memorizes best, not the one that generalizes best.
        - The measurements are a fingerprint: a simple "nearest neighbor" rule recognizes the patient in 88% of the cases (against 1.7% by chance). Every time the rows
          have an implicit identifier (customer, patient, device) the right question is: "will the model see new entities?".
    """,
    pitfall="""
        Not having a `patient` column ("it is not needed as a feature") and so not thinking about grouping. The group must not be a
        feature: it is needed only by the splitter. If you don't pass it, the validation is silently optimistic.
    """,
    interview=(
        "You have 5,000 transactions from 200 customers and you want to evaluate a fraud model that will have to work on new customers. How do you validate?",
        """
        With `GroupKFold` (or `StratifiedGroupKFold` if the positive class is rare) grouping by customer, so that all the transactions of a
        customer are in the same fold. A random K-fold would put transactions of the same customer in training and in validation and the
        model could recognize the customer, inflating the estimate. I also check that the estimate with the groups is consistent with what I see
        on truly new customers in production.
        """),
)

# ---------------------------------------------------------------------------------------- C3

C3 = dict(
    id="C3", slug="best_score_optimism", kind="C", level=3, minutes=45, module=MODULE,
    title="The optimism of best_score_: measuring it with a simulation",
    goal="""
        Measure, on 16 simulated datasets with a weak signal, by how much the `best_score_` of a `GridSearchCV` exceeds the true performance of the
        chosen model, and verify that a nested cross-validation estimates it without bias.
    """,
    dataset="""
        `generate(seed)` produces, with `make_classification` (2 informative features out of 15, noise in the labels), a **development** dataset of 120
        rows and a **truth** one of 2000 rows never used to choose anything. Grid of 24 combinations for a KNN (number of neighbors, weights, metric).
    """,
    purpose="""
        Turn "best_score_ is optimistic" from a sentence into a number. You will see that the estimate grows with the number of combinations tried and that
        nested cross-validation corrects it, at the cost of many trainings.
    """,
    task=[
        "Write `experiment(seed)`: generate the data, run `GridSearchCV(estimator, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed), "
        "scoring=\"roc_auc\")` on the development set and return the triple `(best_score, true_auc, nested_estimate)`. `true_auc` is the AUC of the `best_estimator_` "
        "on the truth dataset; `nested_estimate` is `cross_val_score(GridSearchCV(..., cv=3 folds), X_dev, y_dev, cv=5 outer folds, scoring=\"roc_auc\").mean()`.",
        "Run `experiment` for `seed` from 0 to 15 and build `results` (DataFrame, columns `best_score`, `true_auc`, `nested`).",
        "Compute `best_optimism = (best_score − true_auc).mean()` and `nested_error = (nested − true_auc).mean()`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        estimator = make_pipeline(StandardScaler(), KNeighborsClassifier())
        grid = {"kneighborsclassifier__n_neighbors": [1, 5, 11, 17, 25, 35],
                "kneighborsclassifier__weights": ["uniform", "distance"],
                "kneighborsclassifier__p": [1, 2]}                       # 6 x 2 x 2 = 24 combinations


        def generate(seed):
            X, y = make_classification(n_samples=2120, n_features=15, n_informative=2, n_redundant=0, n_clusters_per_class=1,
                                       class_sep=0.35, flip_y=0.2, random_state=seed)
            return X[:120], y[:120], X[120:], y[120:]                       # development (120) and truth (2000)
    """,
    starter=[
        C("""
        def experiment(seed):
            X_dev, y_dev, X_true, y_true = generate(seed)
            search = ...
            best_score = ...
            true_auc = ...
            nested_estimate = ...
            return best_score, true_auc, nested_estimate
        """),
        C("""
        results = pd.DataFrame([experiment(s) for s in range(16)], columns=["best_score", "true_auc", "nested"])
        best_optimism = ...
        nested_error = ...
        print(results.mean().round(3).to_dict())
        """),
    ],
    checks="""
        check(results.shape == (16, 3), "16 experiments", "one row per seed")
        check((results["best_score"] > results["true_auc"]).mean() > 0.8, "best_score exceeds the true performance in the great majority of the datasets",
              "in over 80% of the datasets best_score_ should exceed the AUC on the truth dataset")
        check(close(best_optimism, (results.best_score - results.true_auc).mean()) and best_optimism > 0.04,
              "the mean optimism of best_score_ exceeds 0.04 of AUC", "optimism = mean(best_score - true_auc)")
        check(close(nested_error, (results.nested - results.true_auc).mean()) and abs(nested_error) < best_optimism / 2,
              "the nested estimate is much less biased than best_score_ (less than half of the optimism)", "error = mean(nested - true_auc)")
    """,
    hint="""
        `search = GridSearchCV(estimator, grid, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").fit(X_dev, y_dev)`;
        `roc_auc_score(y_true, search.predict_proba(X_true)[:, 1])`. Nested: `cross_val_score(GridSearchCV(estimator, grid, cv=3, scoring="roc_auc"),
        X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True, random_state=seed), scoring="roc_auc").mean()`.
    """,
    solution=[
        C("""
        def experiment(seed):
            X_dev, y_dev, X_true, y_true = generate(seed)
            cv = StratifiedKFold(5, shuffle=True, random_state=seed)
            search = GridSearchCV(estimator, grid, cv=cv, scoring="roc_auc").fit(X_dev, y_dev)
            best_score = search.best_score_                                                      # the estimate "of the search"
            true_auc = roc_auc_score(y_true, search.predict_proba(X_true)[:, 1])                 # the performance on never-seen data
            inner = GridSearchCV(estimator, grid, cv=3, scoring="roc_auc")                       # the search becomes part of the model
            nested_estimate = cross_val_score(inner, X_dev, y_dev, cv=cv, scoring="roc_auc").mean()
            return best_score, true_auc, nested_estimate
        """),
        C("""
        results = pd.DataFrame([experiment(s) for s in range(16)], columns=["best_score", "true_auc", "nested"])
        best_optimism = (results["best_score"] - results["true_auc"]).mean()
        nested_error = (results["nested"] - results["true_auc"]).mean()
        print(results.mean().round(3).to_dict())
        print(f"optimism of best_score_: {best_optimism:+.3f} | error of the nested estimate: {nested_error:+.3f}")
        """),
    ],
    learned="""
        - With only 24 candidates and 5 folds, `best_score_` overestimates the true performance by about 0.04 of AUC (from 0.681 to 0.638 on average) in almost
          every dataset. It is the maximum of 24 noisy estimates: choosing the winner also selects its luck.
        - The optimism grows with the number of combinations tried and with the smallness of the dataset: here 120 rows and a weak signal are
          the worst case, but the mechanism is always there.
        - **Nested cross-validation** (the search inside every outer fold) estimates the performance of the *whole procedure* "search and train"
          almost without bias: here 0.628 against 0.638 (an error of −0.01, slightly pessimistic because the inner search uses less
          data). It costs outer_folds × trainings of the search: here 5 × 72 = 360 trainings instead of 120.
    """,
    pitfall="""
        Reporting `best_score_` in a report, or choosing between two tuning procedures by looking at their `best_score_`: you compare scores
        inflated in different ways. To compare procedures you need the nested estimate (or an untouched test set).
    """,
    interview=(
        "When do you really need a nested cross-validation, and when is a test set enough?",
        """
        It is needed when the data are few and I cannot afford a separate test set of sufficient size, or when I have to compare different
        tuning procedures and report an unbiased estimate (for example in a publication or in a regulatory review). With plenty of data a test set held out and touched once is enough: it costs much less and does not require
        repeating the search in every fold.
        """),
)

EXERCISES = [C1, C2, C3]
