"""Module 9 - type B exercises: implement with XGBoost and scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="xgboost_with_early_stopping_on_california", kind="B", level=2, minutes=55, module=MODULE,
    title="XGBoost with early stopping on house prices",
    goal="""
        Train an `XGBRegressor` on California with **early stopping** on a validation set, compare three learning rates (0.3, 0.1, 0.03) for chosen rounds, RMSE and time, and place the result against linear regression, tree and random forest (previous modules). Check that halving the learning rate requires more rounds.
    """,
    dataset="""
        `ds.california()` split into training (60%), **validation** (20%, used only for early stopping) and test (20%, touched once): `X_tr, X_val, X_te` and `y_tr, y_val, y_te`. RMSE with `squared=False`.
    """,
    purpose="""
        Learn the standard workflow: low learning rate, many rounds allowed, `early_stopping_rounds`, best round read from `best_iteration`, test set only at the end. And understand the `learning_rate` ↔ number of rounds trade-off.
    """,
    task=[
        "For `eta` in `[0.3, 0.1, 0.03]` train `XGBRegressor(n_estimators=4000, learning_rate=eta, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method=\"hist\", early_stopping_rounds=50, eval_metric=\"rmse\", n_jobs=-1, random_state=0)` with `eval_set=[(X_val, y_val)]`. "
        "Save `table` (index `eta`, columns `best_round` (`best_iteration + 1`), `val_rmse` (`best_score`), `test_rmse` and `time_s`).",
        "Save `eta_round_product` (`pd.Series`: `eta * best_round` for each learning rate) and `round_ratio_003_over_03` (rounds at 0.03 divided by rounds at 0.3).",
        "Save `reference_rmse` (dictionary on the test set): `linear` (`LinearRegression`), `tree` (`DecisionTreeRegressor(min_samples_leaf=20)`), `forest` (`RandomForestRegressor(200, max_features=0.33, random_state=0, n_jobs=-1)`) and `xgb` (the best model of the three by `val_rmse`).",
        "Save `val_test_gap` (`test_rmse − val_rmse` for the best model) and comment on it: it is small because the validation set was used only to stop.",
    ],
    setup="""
        import time
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        print(X_tr.shape, X_val.shape, X_te.shape)
    """,
    starter=[
        C("""
        rows = {}
        models = {}
        for eta in [0.3, 0.1, 0.03]:
            ...
        table = pd.DataFrame(rows).T
        table.round(4)
        """),
        C("""
        eta_round_product = None
        round_ratio_003_over_03 = None
        """),
        C("""
        reference_rmse = {}
        val_test_gap = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["best_round", "val_rmse", "test_rmse", "time_s"] and list(table.index) == [0.3, 0.1, 0.03], "table with three learning rates and four columns", "best_round, val_rmse, test_rmse, time_s")
        check(table["best_round"].is_monotonic_increasing and table.loc[0.03, "best_round"] > 5 * table.loc[0.3, "best_round"], "lower learning rate = more rounds: at 0.03 over 5 times those at 0.3", "best_iteration + 1")
        check(table["test_rmse"].max() < 0.5 and table["test_rmse"].min() > 0.4 and table.loc[0.03, "test_rmse"] <= table.loc[0.3, "test_rmse"] + 0.005, "test RMSE between 0.40 and 0.50; the low learning rate does not get worse", "XGBRegressor with early stopping")
        check(close(eta_round_product.to_numpy(), table.index.to_numpy() * table["best_round"].to_numpy()) and eta_round_product.max() / eta_round_product.min() < 6 and round_ratio_003_over_03 > 5, "the eta x round product varies by less than 6 times across the learning rates (the rounds scale almost like 1/eta); round ratio over 5", "eta * best_round")
        check(reference_rmse["xgb"] < reference_rmse["forest"] < reference_rmse["tree"] < reference_rmse["linear"], "order: xgb < forest < tree < linear", "compare the test RMSEs")
        check(0 <= val_test_gap < 0.03, "the gap between test and validation RMSE is small (less than 0.03)", "the validation set only decided the number of rounds")
    """,
    hint="""
        `m = xgb.XGBRegressor(...); m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)`; `m.best_iteration`, `m.best_score`; test RMSE: `mean_squared_error(y_te, m.predict(X_te), squared=False)` (with `early_stopping_rounds` the prediction already uses the best round).
    """,
    solution=[
        C("""
        rows, models = {}, {}
        for eta in [0.3, 0.1, 0.03]:
            m = xgb.XGBRegressor(n_estimators=4000, learning_rate=eta, max_depth=5, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                 early_stopping_rounds=50, eval_metric="rmse", n_jobs=-1, random_state=0)
            t0 = time.time()
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            rows[eta] = {"best_round": m.best_iteration + 1, "val_rmse": m.best_score, "test_rmse": mean_squared_error(y_te, m.predict(X_te), squared=False), "time_s": time.time() - t0}
            models[eta] = m
        table = pd.DataFrame(rows).T
        table.round(4)
        """),
        C("""
        eta_round_product = pd.Series({eta: eta * table.loc[eta, "best_round"] for eta in table.index})
        round_ratio_003_over_03 = table.loc[0.03, "best_round"] / table.loc[0.3, "best_round"]
        print(eta_round_product.round(1).to_string(), f"\\nratio of the rounds between eta 0.03 and 0.3: {round_ratio_003_over_03:.1f}")
        """),
        C("""
        best = table["val_rmse"].idxmin()
        error = lambda m: mean_squared_error(y_te, m.predict(X_te), squared=False)
        reference_rmse = {"linear": error(LinearRegression().fit(X_tr, y_tr)), "tree": error(DecisionTreeRegressor(min_samples_leaf=20, random_state=0).fit(X_tr, y_tr)),
                          "forest": error(RandomForestRegressor(200, max_features=0.33, random_state=0, n_jobs=-1).fit(X_tr, y_tr)), "xgb": table.loc[best, "test_rmse"]}
        val_test_gap = float(table.loc[best, "test_rmse"] - table.loc[best, "val_rmse"])
        print({k: round(v, 3) for k, v in reference_rmse.items()}, f"| test - validation gap: {val_test_gap:+.4f} (best learning rate: {best})")
        """),
    ],
    learned="""
        - **The standard workflow:** low learning rate, high `n_estimators` as a ceiling, `early_stopping_rounds`, best round from `best_iteration`, test set only at the end. You do not need to guess the number of rounds: 4000 is a ceiling that early stopping never touches (the real rounds are 149, 843 and 1985 for eta 0.3, 0.1 and 0.03).
        - **`learning_rate` and rounds are linked:** with eta 0.3 tens of rounds are enough, with 0.03 you need hundreds or thousands (about 1/eta). The `eta × round` product stays of the same order of magnitude: the "total amount of correction". With a small eta the generalization is better (slightly) and the time grows.
        - **The ranking of the models is the expected one:** linear (0.734) > optimized tree (0.613) > forest (0.505) > XGBoost (0.457). XGBoost reduces the error by about a tenth compared with the forest: that is why it is the reference model on tabular data.
        - **The validation-test gap is small** (0.009 of RMSE): the validation set only decided *when to stop* (one parameter), not the weights of the tree. If you had used the test set as validation the estimate would have been optimistic by this same small amount: a good reason not to, and to put the validation set **last** in `eval_set` (exercise E1).
    """,
    pitfall="""
        Using `n_estimators` as a hyperparameter to search on a grid together with `learning_rate`: the two things compensate each other. Fix the learning rate, keep `n_estimators` high and let early stopping decide.
    """,
    interview=(
        "How do you choose the number of trees in XGBoost?",
        """
        I fix a low learning rate (0.03-0.1), allow many trees (thousands) and use early stopping on a separate validation set: the training stops when the metric does not improve for k rounds and the model uses the best one. The number of rounds is therefore a result, not a hyperparameter to search; for the final estimate I use a test set never seen,
        or a cross-validation with early stopping inside each fold.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="native_missing_values_vs_imputation", kind="B", level=3, minutes=55, module=MODULE,
    title="Missing values: native NaN in XGBoost, imputation and indicators",
    goal="""
        Compare in repeated cross-validation three ways of handling missing values with XGBoost on the loans (where the missingness of the income is linked to the outcome): **native NaN**, median imputation, median plus missingness indicators; and with a random forest (which requires imputation). Measure the differences with their standard
        error: many are small.
    """,
    dataset="""
        `ds.loans(n=3000)` with missing values in `age`, `income`, `job_tenure` (the missingness of `income` is more frequent among defaulters). Features: `ratio` (amount/income, NaN where income is missing), `age`, `income`, `job_tenure`, `amount`. Validation `RepeatedStratifiedKFold(5, n_repeats=4, random_state=0)`, AUC.
    """,
    purpose="""
        Understand how XGBoost "uses" NaN (the default direction of each split) and how much it matters compared with the standard step of imputation + indicator of module 4. See that the difference is modest but systematic, and that the estimate has an uncertainty to compute.
    """,
    task=[
        "Build `models` (dictionary name → `Pipeline` or estimator): `xgb native` (`XGBClassifier` directly, without imputation), `xgb median` (`SimpleImputer(strategy=\"median\")` + XGB), `xgb median+indicators` (`SimpleImputer(strategy=\"median\", add_indicator=True)` + XGB), `rf median+indicators` (same imputer + `RandomForestClassifier(200, min_samples_leaf=5)`). "
        "XGB with `n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric=\"logloss\", n_jobs=-1, random_state=0`.",
        "Compute with `cross_val_score(..., scoring=\"roc_auc\", cv=cv)` (20 folds): `table` (index: models; columns `auc` (mean), `sd` (standard deviation across folds)).",
        "Paired comparison: for each model other than `xgb native` compute `diff_vs_native` = mean of the fold-by-fold differences `auc(native) − auc(model)`; and `standard_error` (`sd of the differences / √20`). Save `diff` (`pd.DataFrame` with the two columns).",
        "Save `native_better_than_median` (True if the mean difference against `xgb median` is positive and exceeds two standard errors) and `indicators_close_gap` (True if `xgb median+indicators` is within one standard error of the native).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline

        df = ds.loans(n=3000)
        df["ratio"] = df["amount"] / df["income"]                        # NaN where the income is missing
        X, y = df[["ratio", "age", "income", "job_tenure", "amount"]], df["defaulted"]
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
        print("share of missing values:", X.isna().mean().round(3).to_dict())


        def xgb_(): return xgb.XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=0)
    """,
    starter=[
        C("""
        models = {
            "xgb native": xgb_(),
        }
        """),
        C("""
        scores = {}
        for name, m in models.items():
            ...
        table = pd.DataFrame({n: {"auc": s.mean(), "sd": s.std()} for n, s in scores.items()}).T
        table.round(4)
        """),
        C("""
        diff = None
        native_better_than_median = None
        indicators_close_gap = None
        """),
    ],
    checks="""
        check(list(models) == ["xgb native", "xgb median", "xgb median+indicators", "rf median+indicators"] and len(scores) == 4 and all(len(s) == 20 for s in scores.values()), "four models, twenty scores each", "cross_val_score with RepeatedStratifiedKFold(5, 4)")
        check(table["auc"].min() > 0.72 and table["auc"].max() < 0.86, "AUC between 0.72 and 0.86 for all", "the problem has a realistic AUC")
        check(list(diff.index) == ["xgb median", "xgb median+indicators", "rf median+indicators"] and list(diff.columns) == ["diff_vs_native", "standard_error"], "paired differences for the three comparisons", "sd of the differences / sqrt(20)")
        check(diff.loc["xgb median", "diff_vs_native"] > 0 and native_better_than_median is True, "native NaN beat the median alone (by over two standard errors)", "the missingness information is lost with imputation alone")
        check(indicators_close_gap is True and abs(diff.loc["xgb median+indicators", "diff_vs_native"]) < 0.01, "with the missingness indicator the difference from the native disappears (within one standard error, less than one point)", "median + add_indicator is equivalent to the native")
    """,
    hint="""
        `scores[name] = cross_val_score(m, X, y, cv=cv, scoring="roc_auc")`. Paired differences: `d = scores["xgb native"] - scores[name]`; `d.mean()`, `d.std() / np.sqrt(len(d))`. Pipeline: `make_pipeline(SimpleImputer(strategy="median", add_indicator=True), xgb_())`.
    """,
    solution=[
        C("""
        models = {
            "xgb native": xgb_(),
            "xgb median": make_pipeline(SimpleImputer(strategy="median"), xgb_()),
            "xgb median+indicators": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), xgb_()),
            "rf median+indicators": make_pipeline(SimpleImputer(strategy="median", add_indicator=True), RandomForestClassifier(200, min_samples_leaf=5, random_state=0, n_jobs=-1)),
        }
        """),
        C("""
        scores = {name: cross_val_score(m, X, y, cv=cv, scoring="roc_auc") for name, m in models.items()}
        table = pd.DataFrame({n: {"auc": s.mean(), "sd": s.std()} for n, s in scores.items()}).T
        table.round(4)
        """),
        C("""
        rows = {}
        for name in ["xgb median", "xgb median+indicators", "rf median+indicators"]:
            d = scores["xgb native"] - scores[name]                                  # paired comparison fold by fold
            rows[name] = {"diff_vs_native": d.mean(), "standard_error": d.std() / np.sqrt(len(d))}
        diff = pd.DataFrame(rows).T
        native_better_than_median = bool(diff.loc["xgb median", "diff_vs_native"] > 2 * diff.loc["xgb median", "standard_error"])
        indicators_close_gap = bool(abs(diff.loc["xgb median+indicators", "diff_vs_native"]) <= diff.loc["xgb median+indicators", "standard_error"])
        print(diff.round(4))
        print("native better than the median alone:", native_better_than_median, "| with the indicators the difference disappears:", indicators_close_gap)
        """),
    ],
    learned="""
        - **XGBoost uses missingness as information:** at each split it chooses the direction in which to send the NaN that reduces the loss the most, so it learns that "missing income" indicates a higher default risk. With the median alone this information is **lost**: the AUC drops by about 0.6-1 points systematically (more than two standard errors).
        - **The missingness indicator closes the gap:** median plus `add_indicator=True` makes the missingness explicit and brings the model back to the level of the native one (difference within one standard error). It is the result of module 4 (B1) seen from another model: the signal is in **the fact that it is missing**.
        - **The differences are small:** less than one point of AUC, with a standard deviation across folds of 2-3 points. The paired comparison (fold-by-fold difference, with its standard error) is the way to see effects this small: a comparison between two means would not distinguish them from noise.
        - The random forest (which does not accept NaN) requires imputation and with the indicators reaches comparable levels: the choice of XGBoost for missing values is a **convenience** with a small advantage, not a quality leap.
    """,
    pitfall="""
        Imputing "to be safe" the data of a model that handles NaN: if the missingness is informative (data not missing at random) imputation without an indicator erases it. Try native and imputation+indicators and compare in cross-validation with the standard error.
    """,
    interview=(
        "How does XGBoost handle missing values?",
        """
        At each split it learns a default direction for the NaN (left or right) choosing the one that minimizes the loss on the training data; so the missing values become a signal and no imputation or indicators are needed. If the missingness is informative it is an advantage, but with other models you get the same effect with imputation plus missingness indicators;
        I always compare the two routes in cross-validation.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="tuning_with_random_search", kind="B", level=3, minutes=60, module=MODULE,
    title="Tuning XGBoost with a random search (and what it is worth)",
    goal="""
        Search the XGBoost hyperparameters with `RandomizedSearchCV` (20 combinations over `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `gamma`) in a 3-fold cross-validation, then fix the rounds with **early stopping** on a validation set and evaluate once on the test set.
        Compare with the default parameters and measure how much the optimization is worth.
    """,
    dataset="""
        `ds.loans(n=5000)` with missing values (XGBoost handles them), features `ratio, age, income, job_tenure, amount` plus dummies of `contract_type`. Split: 60% training, 20% validation, 20% test. `learning_rate=0.05` fixed.
    """,
    purpose="""
        Learn to set up a realistic tuning and to **measure its gain honestly**: random search over XGBoost hyperparameters is effective, but the advantage over the defaults is usually small (1-2 points of AUC), and it has to be compared with the uncertainty of the estimate.
    """,
    task=[
        "Compute `default_auc` on the test set: `XGBClassifier` with defaults and `learning_rate=0.05`, `n_estimators` chosen with early stopping on the validation set (`early_stopping_rounds=30`).",
        "`RandomizedSearchCV` (20 iterations, `cv=StratifiedKFold(3, shuffle=True, random_state=0)`, `scoring=\"roc_auc\"`, `random_state=0`, `n_jobs=1`) on the **training set only** with `n_estimators=200`, `learning_rate=0.05` fixed and the parameter distribution given in `space`. Save `best` (`best_params_`), `cv_auc` (`best_score_`) and `n_fits` (number of trainings: 20 combinations × 3 folds).",
        "With the best parameters retrain with early stopping on the validation set and save `tuned_auc` (AUC on the test set) and `final_rounds`.",
        "Save `gain = tuned_auc − default_auc` and `within_noise` (True if `gain` is smaller than `2 * 0.01`: the typical uncertainty of an AUC on 1000 examples is about 1 point).",
    ],
    setup="""
        import xgboost as xgb
        from scipy.stats import randint, uniform
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split

        df = ds.loans(n=5000)
        df["ratio"] = df["amount"] / df["income"]
        X = pd.concat([df[["ratio", "age", "income", "job_tenure", "amount"]], pd.get_dummies(df["contract_type"], dtype=float)], axis=1)
        y = df["defaulted"]
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, stratify=y_rest, random_state=0)
        space = {"max_depth": randint(2, 7), "min_child_weight": randint(1, 30), "subsample": uniform(0.5, 0.5), "colsample_bytree": uniform(0.5, 0.5),
                 "reg_lambda": [0.1, 1, 5, 20], "gamma": [0, 0.5, 2, 5]}
        base = dict(learning_rate=0.05, eval_metric="auc", n_jobs=-1, random_state=0)
    """,
    starter=[
        C("""
        default_auc = None
        """),
        C("""
        search = None
        best = None
        cv_auc = None
        n_fits = None
        """),
        C("""
        tuned_auc = None
        final_rounds = None
        gain = None
        within_noise = None
        """),
    ],
    checks="""
        check(0.7 < default_auc < 0.87, "test AUC with the defaults between 0.70 and 0.87", "XGBClassifier(learning_rate=0.05, ...) with early stopping")
        check(set(best) == set(space) and n_fits == 60 and 0.72 < cv_auc < 0.87, "the search returns the six parameters; 60 trainings; cross-validation AUC between 0.72 and 0.87", "RandomizedSearchCV(n_iter=20, cv=3)")
        check(best["max_depth"] <= 4 and best["min_child_weight"] >= 5, "the signal is weak and noisy: the search prefers shallow trees (depth 4 or less) and a high min_child_weight (5 or more)", "conservative hyperparameters")
        check(0.7 < tuned_auc < 0.87 and final_rounds >= 20, "test AUC of the optimized model between 0.70 and 0.87 with at least 20 rounds", "early stopping on the validation set")
        check(close(gain, tuned_auc - default_auc) and abs(gain) < 0.04 and within_noise is True, "the gain over the defaults is small (less than 4 points, within the noise)", "gain = tuned_auc - default_auc")
    """,
    hint="""
        `search = RandomizedSearchCV(xgb.XGBClassifier(n_estimators=200, **base), space, n_iter=20, cv=StratifiedKFold(3, shuffle=True, random_state=0), scoring="roc_auc", random_state=0, n_jobs=1).fit(X_tr, y_tr)`; `search.best_params_`. Final rounds: `m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **base, **best).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)`; `m.best_iteration + 1`.
    """,
    solution=[
        C("""
        def with_early_stopping(**params):
            m = xgb.XGBClassifier(n_estimators=2000, early_stopping_rounds=30, **base, **params)
            m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            return m


        m0 = with_early_stopping()
        default_auc = roc_auc_score(y_te, m0.predict_proba(X_te)[:, 1])
        print(f"defaults with early stopping: {m0.best_iteration + 1} rounds | test AUC {default_auc:.4f}")
        """),
        C("""
        search = RandomizedSearchCV(xgb.XGBClassifier(n_estimators=200, **base), space, n_iter=20, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                                    scoring="roc_auc", random_state=0, n_jobs=1).fit(X_tr, y_tr)
        best = search.best_params_
        cv_auc = search.best_score_
        n_fits = len(search.cv_results_["params"]) * 3
        print("best parameters:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in best.items()}, f"| CV AUC {cv_auc:.4f} | trainings: {n_fits}")
        """),
        C("""
        m1 = with_early_stopping(**best)
        tuned_auc = roc_auc_score(y_te, m1.predict_proba(X_te)[:, 1])
        final_rounds = m1.best_iteration + 1
        gain = tuned_auc - default_auc
        within_noise = bool(abs(gain) < 0.02)
        print(f"optimized: {final_rounds} rounds | test AUC {tuned_auc:.4f} | gain over the defaults: {gain:+.4f} | within the noise (less than 2 points): {within_noise}")
        """),
    ],
    learned="""
        - **A random search with 20 combinations does the job:** with 6 hyperparameters a grid would be huge, but 20 random points almost always find a good region (Bergstra and Bengio, M3 F2). It costs 60 trainings (20 × 3 folds), a few seconds with `n_estimators=200`.
        - **The search confirms what the problem asks for:** with a weak, noisy signal (AUC around 0.80) it prefers **shallow**, conservative trees (`max_depth` 2-3, high `min_child_weight`): deep boosting would memorize the noise.
        - **The gain on the test set is modest:** +0.019 of AUC (from 0.770 to 0.789). With 1000 test examples the uncertainty of an AUC is about a point and a half, so the advantage is at the edge of the noise. It is a common result: the XGBoost defaults + early stopping are already very close to the optimum. The order of priority: first early stopping and learning rate, then depth and `min_child_weight`, then the rest.
        - **Order of the steps:** search the hyperparameters with `n_estimators` fixed and learning rate fixed, then fix the rounds with early stopping on a validation set, then evaluate on the test set. Mixing the two phases (early stopping inside the search, with a different validation set per fold) is possible but more expensive.
    """,
    pitfall="""
        Reporting the cross-validation AUC of the best point of the search as the estimate of the performance: it is the **maximum** of 20 noisy estimates (selection optimism, M3 C3). The number to report is the one on the test set, touched once.
    """,
    interview=(
        "How do you set up the tuning of XGBoost in practice?",
        """
        I start from sensible defaults (learning rate 0.05, max_depth 4, subsample and colsample 0.8), choose the rounds with early stopping on a validation set, then run a random search (20-50 combinations) over max_depth, min_child_weight, subsample, colsample_bytree, reg_lambda and gamma with cross-validation, and retrain with early stopping. I evaluate once on the test set.
        I expect a modest gain over the defaults and compare it with the uncertainty of the estimate.
        """),
)

EXERCISES = [B1, B2, B3]
