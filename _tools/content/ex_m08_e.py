"""Module 8 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="a_half_point_difference_was_the_seed", kind="E", level=2, minutes=45, module=MODULE, start="must_fail",
    title="Comparing two forests with a single seed and 25 trees",
    goal="""
        Take apart a conclusion based on a single run: a colleague compared `max_features=4` and `max_features=6` with 25 trees and a single `random_state`, and declared "no difference" (or, with another seed, "6 is much better"). Measure the **variability due to the seed** by repeating with 30 seeds, and establish
        what can really be concluded.
    """,
    dataset="""
        `make_classification(800, 16 features, 5 informative, 3 redundant, flip_y=0.1)`. Validation: `StratifiedKFold(5, shuffle=True, random_state=0)` (fixed) on accuracy. The colleague uses `n_estimators=25`.
    """,
    purpose="""
        Learn that a forest is a **stochastic** model: two trainings with different seeds give different results. With few trees the variability between seeds can be larger than the difference you want to measure, and a single-seed comparison picks the winner at random.
    """,
    task=[
        "Run the colleague's cell: `diff_seed0` (accuracy with `max_features=6` minus `max_features=4`, seed 0, 25 trees) and the conclusion drawn from it.",
        "Repeat for **30 seeds** (`random_state = s` for both models): save `acc4`, `acc6` (`np.array` of 30 accuracies) and `differences = acc6 − acc4`.",
        "Compute `seed_sd` (standard deviation of `acc4` across seeds), `mean_diff`, `diff_sd`, `standard_error = diff_sd / √30`, `wins6` (in how many seeds `max_features=6` wins) and `sign_changes` (True if the differences have both positive and negative values).",
        "Redo the comparison with **300 trees** on 10 seeds (`seed_sd_300` = standard deviation of `acc4` at 300 trees) and save `sd_reduction = seed_sd / seed_sd_300`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X, y = make_classification(800, n_features=16, n_informative=5, n_redundant=3, class_sep=0.7, flip_y=0.1, random_state=5)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)


        def acc(mf, seed, n=25):
            return cross_val_score(RandomForestClassifier(n, max_features=mf, random_state=seed, n_jobs=-1), X, y, cv=cv).mean()
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        diff_seed0 = acc(6, 0) - acc(4, 0)
        print(f"difference (max_features=6 minus 4) with a single seed and 25 trees: {diff_seed0:+.4f} -> 'there is no difference, let's keep sqrt'")
        """),
        M("### Your diagnosis"),
        C("""
        acc4 = None
        acc6 = None
        differences = None
        """),
        C("""
        seed_sd = None
        mean_diff = None
        diff_sd = None
        standard_error = None
        wins6 = None
        sign_changes = None
        """),
        C("""
        seed_sd_300 = None
        sd_reduction = None
        """),
    ],
    checks="""
        check(abs(diff_seed0) < 0.01, "the colleague sees a tiny difference (less than one point) with seed 0", "don't modify the colleague's code")
        check(acc4 is not None and len(acc4) == 30 and len(acc6) == 30 and close(differences, np.asarray(acc6) - np.asarray(acc4)), "30 accuracies for each configuration and their differences", "acc(4, s), acc(6, s) for s = 0..29")
        check(0.004 < seed_sd < 0.012 and diff_sd > 0.005, "the standard deviation across seeds is 0.4-1.2 points; that of the differences over half a point", "np.std")
        check(sign_changes is True and wins6 >= 18 and 0.002 < mean_diff < 0.01 and close(standard_error, diff_sd / np.sqrt(30)), "the difference changes sign across seeds; max_features=6 wins in at least 18 seeds out of 30; mean between 0.2 and 1 point", "mean and standard error of the differences")
        check(mean_diff > 2 * standard_error, "the mean difference exceeds two standard errors: it is probably real, but small", "mean_diff / standard_error")
        check(seed_sd_300 is not None and sd_reduction > 1.6, "with 300 trees the variability across seeds drops by over 1.6 times", "sd of acc(4, s, n=300)")
    """,
    hint="""
        `acc4 = np.array([acc(4, s) for s in range(30)])`; `acc6 = np.array([acc(6, s) for s in range(30)])`. With 300 trees: `np.std([acc(4, s, 300) for s in range(10)])` (slower: it takes a few tens of seconds).
    """,
    solution=[
        C("""
        diff_seed0 = acc(6, 0) - acc(4, 0)
        print(f"difference (max_features=6 minus 4) with a single seed and 25 trees: {diff_seed0:+.4f} -> 'there is no difference, let's keep sqrt'")
        """),
        M("""
        **Where the bug is.** The comparison is made on **a single run**, but a forest with 25 trees is a stochastic model: the variability across seeds of the same configuration (about 0.7 points) is larger than the difference the colleague wants to measure (about half a point). With seed 0 the
        difference is +0.1 points (no difference), with other seeds it can be +2.6 or −0.9: the conclusion depends on the chosen seed.
        """),
        C("""
        acc4 = np.array([acc(4, s) for s in range(30)])
        acc6 = np.array([acc(6, s) for s in range(30)])
        differences = acc6 - acc4
        print("first differences (max_features 6 - 4):", differences[:8].round(3))
        """),
        C("""
        seed_sd = float(acc4.std())
        mean_diff, diff_sd = float(differences.mean()), float(differences.std())
        standard_error = diff_sd / np.sqrt(30)
        wins6 = int((differences > 0).sum())
        sign_changes = bool((differences > 0).any() and (differences < 0).any())
        print(f"sd across seeds (max_features=4): {seed_sd:.4f} | mean difference {mean_diff:+.4f} +/- {standard_error:.4f} (sd {diff_sd:.4f}) | max_features=6 wins in {wins6}/30 seeds | the sign changes: {sign_changes}")
        """),
        C("""
        seed_sd_300 = float(np.std([acc(4, s, 300) for s in range(10)]))
        sd_reduction = seed_sd / seed_sd_300
        print(f"sd across seeds at 25 trees: {seed_sd:.4f} | at 300 trees: {seed_sd_300:.4f} (reduction {sd_reduction:.1f}x)")
        """),
    ],
    learned="""
        - **A forest is stochastic:** with 25 trees, the same configuration with different seeds varies by about 0.7 points (standard deviation) and the difference between two configurations varies by about 0.9 points from one seed to another. The colleague saw +0.1 with seed 0; with other seeds they would have seen +2.6 or −0.9.
        - **The true difference exists but is small:** over 30 seeds `max_features=6` wins 23 times and the mean difference is half a point (3 standard errors). The honest conclusion is "6 is probably a bit better, by about half a point": neither "no difference" nor "much better".
        - **With more trees the variability drops** (0.3 points at 300 trees): the seed noise shrinks roughly like `1/√n`, so to compare configurations **you need a high number of trees** (and/or many seeds). With 25 trees the noise of the comparison is as large as the effect you are looking for.
        - **Protocol:** fix the same seed for the compared models (**paired** comparison), use 200+ trees, repeat with at least 10 seeds or with a repeated cross-validation, and report the difference with its standard error. And write `random_state` in the code: without it you cannot reproduce anything.
    """,
    pitfall="""
        Picking the "winner" between two configurations after a single run, maybe with different `random_state` values. With stochastic models it is a random choice, and in a tuning with dozens of configurations you systematically pick the lucky one.
    """,
    interview=(
        "You compared two random forests and one does better by 0.5%. Do you trust it?",
        """
        First I measure the variability due to the seed: I repeat with many seeds (same seed for both, paired comparison) and with a high number of trees, and I report the mean difference with the standard error. If the difference is smaller than or of the same order as the noise, I do not consider it a result; if it is distinguishable but small, I
        weigh it against the cost (time, complexity).
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="the_default_regression_forest_is_a_bagging", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="The default RandomForestRegressor is not a random forest",
    goal="""
        Find out why a colleague gets the same RMSE (0.5216) with `RandomForestRegressor` and with `BaggingRegressor(DecisionTreeRegressor())` and concludes "the random forest adds nothing to bagging": in the regressor `max_features` is **1.0** (all the features), so there is no feature subsampling and the forest is a bagging. Fix it and measure the gain.
    """,
    dataset="""
        `ds.california()`, 75/25 split (`random_state=0`), RMSE with `squared=False`. Models with 100 trees, `random_state=0`, `n_jobs=-1`.
    """,
    purpose="""
        Recognize a tricky default: class names do not guarantee the algorithm. In the classifier `max_features="sqrt"`, in the regressor `1.0` (in recent versions; `"auto"` = all in older ones): for regression the default is a bagging of trees. It is always worth trying smaller values.
    """,
    task=[
        "Run the colleague's cell: `rf_default_rmse` and `bagging_rmse`, and their difference.",
        "Check the default: `default_max_features` (`RandomForestRegressor().max_features`) and `same_prediction` (True if the predictions of `rf_default` and `explicit_bagging` = `RandomForestRegressor(max_features=1.0)` match exactly on the test set).",
        "Try `max_features` in `[0.2, 0.33, 0.5, 0.75, 1.0]`: save `mf_curve` (`pd.Series`, test RMSE) and `best_mf`.",
        "Save `relative_gain = 1 − mf_curve[best_mf] / mf_curve[1.0]` and `prediction_corr` (correlation between the bagging predictions and those of the forest with `best_mf`: the trees are more different but the final predictions stay very similar).",
    ],
    setup="""
        from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        rmse = lambda p: mean_squared_error(y_te, p, squared=False)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        rf_default = RandomForestRegressor(100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        bagging = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        rf_default_rmse, bagging_rmse = rmse(rf_default.predict(X_te)), rmse(bagging.predict(X_te))
        print(f"RMSE: random forest {rf_default_rmse:.4f} | bagging {bagging_rmse:.4f} -> 'the random forest adds nothing to bagging'")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        default_max_features = None
        same_prediction = None
        """),
        C("""
        mf_curve = pd.Series(dtype=float)
        best_mf = None
        relative_gain = None
        prediction_corr = None
        """),
    ],
    checks="""
        check(abs(rf_default_rmse - bagging_rmse) < 0.002, "the colleague sees almost equal RMSEs (difference below 0.002)", "don't modify the colleague's code")
        check(default_max_features in (1.0, 1, "auto") and same_prediction is True, "the default is 1.0 (all the features) and the predictions match RandomForestRegressor(max_features=1.0)", "RandomForestRegressor().max_features")
        check(list(mf_curve.index) == [0.2, 0.33, 0.5, 0.75, 1.0] and best_mf in (0.33, 0.5), "the best is 0.33 or 0.5 (among the five values tried)", "idxmin of the curve")
        check(mf_curve[best_mf] < mf_curve[1.0] - 0.015 and relative_gain > 0.03, "the RMSE with the best value is over 0.015 below the default; relative gain over 3%", "1 - RMSE(best) / RMSE(1.0)")
        check(0.95 < prediction_corr < 1.0, "the predictions of the two forests are highly correlated (0.95-1.0) but not identical", "np.corrcoef of the predictions on the test set")
    """,
    hint="""
        `mf_curve = pd.Series({mf: rmse(RandomForestRegressor(100, max_features=mf, random_state=0, n_jobs=-1).fit(X_tr, y_tr).predict(X_te)) for mf in [0.2, 0.33, 0.5, 0.75, 1.0]})`. Correlation: `np.corrcoef(p_bagging, p_forest)[0, 1]`.
    """,
    solution=[
        C("""
        rf_default = RandomForestRegressor(100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        bagging = BaggingRegressor(DecisionTreeRegressor(), n_estimators=100, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        rf_default_rmse, bagging_rmse = rmse(rf_default.predict(X_te)), rmse(bagging.predict(X_te))
        print(f"RMSE: random forest {rf_default_rmse:.4f} | bagging {bagging_rmse:.4f} -> 'the random forest adds nothing to bagging'")
        """),
        M("""
        **Where the bug is.** In `RandomForestClassifier` the default is `max_features="sqrt"` (at each split you choose among sqrt(p) features), in `RandomForestRegressor` it is **1.0**: all the features are considered at each split. Without the random subsampling of features a "random forest" is a
        bagging of trees: same RMSE to the fourth decimal.
        """),
        C("""
        default_max_features = RandomForestRegressor().max_features
        explicit = RandomForestRegressor(100, max_features=1.0, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        same_prediction = bool(np.allclose(rf_default.predict(X_te), explicit.predict(X_te)))
        print("default max_features of the regressor:", default_max_features, "| predictions identical to max_features=1.0:", same_prediction)
        """),
        C("""
        predictions = {mf: RandomForestRegressor(100, max_features=mf, random_state=0, n_jobs=-1).fit(X_tr, y_tr).predict(X_te) for mf in [0.2, 0.33, 0.5, 0.75, 1.0]}
        mf_curve = pd.Series({mf: rmse(p) for mf, p in predictions.items()})
        best_mf = mf_curve.idxmin()
        relative_gain = 1 - mf_curve[best_mf] / mf_curve[1.0]
        prediction_corr = float(np.corrcoef(bagging.predict(X_te), predictions[best_mf])[0, 1])
        print(mf_curve.round(4).to_string())
        print(f"best: max_features={best_mf} | gain over the default: {relative_gain:.1%} | correlation of the predictions with the bagging: {prediction_corr:.4f}")
        """),
    ],
    learned="""
        - **The default `RandomForestRegressor` is a bagging:** `max_features=1.0` means all the features at each split, so the trees are correlated exactly as in bagging and the RMSE is the same (0.5216). The classifier has `"sqrt"`; the regression does not. For a real random forest you have to set it: with `0.33` the RMSE drops to 0.498 (−4.6%).
        - **The gain is moderate but free:** one line of code and one parameter. With an already low test error, 4-5% less is an improvement that with other methods (tuning, ensembles) would take much more work. It is the kind of "default that is not optimal" to check first.
        - **Do not trust the names:** always check the default parameter values (`estimator.get_params()`), even for classes with names like "RandomForest". They change between classification and regression and between scikit-learn versions (`"auto"` used to be all for the regressor and `sqrt` for the classifier, now they are `1.0` and `"sqrt"`).
        - The predictions of the bagging and of the forest with `max_features=0.33` are highly correlated (0.99): the detail changes, not the structure of the prediction. That is why the gain is a few percentage points and not a factor.
    """,
    pitfall="""
        Drawing a conclusion from a comparison between "two algorithms" with defaults that are equivalent. Before comparing two models check that the parameters really do what the name promises.
    """,
    interview=(
        "What distinguishes a random forest from a bagging of trees in scikit-learn?",
        """
        Only the random subsampling of the features at each split (`max_features`). In the classifier the default is sqrt(p), so RandomForestClassifier really is a random forest; in the regressor the default is 1.0 (all the features), so without changing it it is equivalent to a bagging of trees. For regression I try values like
        0.33 or 0.5 and choose them with the OOB or cross-validation.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="permutation_importance_computed_on_the_training_set", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="Permutation on the training set rewards the noise the forest memorized",
    goal="""
        Find out why a report "corrected with the permutation" puts `ticket_code` (a random integer) ahead of `age` and `siblings_spouses`: the permutation was computed on the **training set**, where the forest has memorized the noise too. Redo the computation on the test set and measure the difference.
    """,
    dataset="""
        `ds.titanic()` with two useless columns added (`random_number`, `ticket_code`); 70/30 split. Pipeline: median, mode + `OrdinalEncoder`, `RandomForestClassifier(300, random_state=0)`; `pipe` is already trained. Scoring: AUC.
    """,
    purpose="""
        Remember that **every importance measure must be evaluated on data the model has not seen**. A forest has AUC 1.0 on the training set: shuffling any column, even a noise one, makes it get worse because it has learned to use it to recognize the single examples.
    """,
    task=[
        "Run the colleague's cell: `train_perm` (`permutation_importance` computed on `X_tr, y_tr`, `n_repeats=10`, `random_state=0`, scoring `roc_auc`, as a `pd.Series` indexed by the columns) and `train_top5`.",
        "Compute `test_perm` (same computation on `X_te, y_te`) and `test_top5`. Save `train_auc` and `test_auc` of the model: the difference explains the phenomenon.",
        "Save `train_noise` and `test_noise` (sum of the positive importances of `random_number` and `ticket_code` for the two computations) and `train_code_rank` / `test_code_rank` (position, from 1, of `ticket_code` in the descending order).",
        "**Stability:** compute the importances on the test set on 3 random subsamples of half of the test set (`idx = np.random.default_rng(s).choice(len(X_te), len(X_te) // 2, replace=False)`, `s = 0, 1, 2`, `n_repeats=5`) and save `stable` (True if `ticket_code` has an importance below 0.02 in all three).",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder

        df = ds.titanic()
        X, y = df.drop(columns="survived").copy(), df["survived"]
        r = np.random.default_rng(0)
        X["random_number"] = r.uniform(size=len(X))
        X["ticket_code"] = r.integers(0, 100000, len(X))
        NUM = ["pclass", "age", "siblings_spouses", "parents_children", "fare", "random_number", "ticket_code"]
        CAT = ["sex", "port", "deck", "title"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM), ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder()), CAT)])
        pipe = make_pipeline(prep, RandomForestClassifier(300, random_state=0, n_jobs=-1)).fit(X_tr, y_tr)
        NOISE = ["random_number", "ticket_code"]


        def importances(A, b, n_repeats=10, seed=0):
            r = permutation_importance(pipe, A, b, n_repeats=n_repeats, random_state=seed, scoring="roc_auc")
            return pd.Series(r.importances_mean, index=A.columns)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        train_perm = importances(X_tr, y_tr)
        train_top5 = list(train_perm.nlargest(5).index)
        print("top 5 by permutation (computed on the training set):", train_top5)
        print(train_perm.round(3).sort_values(ascending=False).head(7).to_string())
        """),
        M("### Your diagnosis and the fix"),
        C("""
        test_perm = None
        test_top5 = None
        train_auc = None
        test_auc = None
        """),
        C("""
        train_noise = None
        test_noise = None
        train_code_rank = None
        test_code_rank = None
        """),
        C("""
        stable = None
        """),
    ],
    checks="""
        check("ticket_code" in train_top5, "the colleague's report puts `ticket_code` in the top five", "don't modify the colleague's code")
        check(train_auc is not None and train_auc > 0.99 and test_auc < 0.9, "training AUC 1.0 (memorization); test below 0.9", "pipe.score(...) is not the AUC: use roc_auc_score with predict_proba")
        check(test_perm is not None and "ticket_code" not in test_top5 and "random_number" not in test_top5 and test_top5[0] == "sex", "on the test set the two noise columns are not in the top five and `sex` is first", "importances(X_te, y_te)")
        check(train_noise > 0.03 and test_noise < 0.02 and train_code_rank < test_code_rank, "the noise is worth over 0.03 on the training set and less than 0.02 on the test set; the ticket code drops in the ranking", "sum of the positive importances of the two noise columns")
        check(stable is True, "on three subsamples of the test set `ticket_code` always stays below 0.02", "n_repeats=5, random subsamples of the test set")
    """,
    hint="""
        `test_perm = importances(X_te, y_te)`. AUC: `roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])`. Rank: `perm.rank(ascending=False)["ticket_code"]`. Subsamples: `idx = np.random.default_rng(s).choice(len(X_te), len(X_te) // 2, replace=False)`; `importances(X_te.iloc[idx], y_te.iloc[idx], 5)`.
    """,
    solution=[
        C("""
        train_perm = importances(X_tr, y_tr)
        train_top5 = list(train_perm.nlargest(5).index)
        print("top 5 by permutation (computed on the training set):", train_top5)
        print(train_perm.round(3).sort_values(ascending=False).head(7).to_string())
        """),
        M("""
        **Where the bug is.** The permutation measures how much the model **gets worse** when a column is shuffled. On the training set the forest has AUC 1.0: it has memorized every row, including the value of the random `ticket_code` (which identifies the passenger). Shuffling that column breaks the identification and the AUC drops,
        even though for a new passenger the code says nothing. The same measure on the test set, where memorization does not help, brings it to zero.
        """),
        C("""
        test_perm = importances(X_te, y_te)
        test_top5 = list(test_perm.nlargest(5).index)
        train_auc = roc_auc_score(y_tr, pipe.predict_proba(X_tr)[:, 1])
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        print("top 5 by permutation (on the test set):", test_top5, f"| training AUC {train_auc:.3f}, test {test_auc:.3f}")
        """),
        C("""
        train_noise = float(train_perm[NOISE].clip(lower=0).sum())
        test_noise = float(test_perm[NOISE].clip(lower=0).sum())
        train_code_rank = int(train_perm.rank(ascending=False)["ticket_code"])
        test_code_rank = int(test_perm.rank(ascending=False)["ticket_code"])
        print(f"importance of the two noise columns: training {train_noise:.3f} | test {test_noise:.3f} | position of ticket_code: {train_code_rank} -> {test_code_rank}")
        """),
        C("""
        stable = True
        for s in range(3):
            idx = np.random.default_rng(s).choice(len(X_te), len(X_te) // 2, replace=False)
            imp = importances(X_te.iloc[idx], y_te.iloc[idx], n_repeats=5, seed=s)
            stable = stable and bool(imp["ticket_code"] < 0.02)
        print("ticket_code stays below 0.02 in all the subsamples of the test set:", stable)
        """),
    ],
    learned="""
        - **A model that memorizes the training set makes any column "important":** on the training set the forest has AUC 1.0 and the permutation of `ticket_code` (a random integer) removes 0.02 of AUC, more than `age`; on the test set the same column is worth 0.013 and the ranking changes (and `random_number` drops to −0.003).
        - **Any permutation importance must be computed on unseen data** (test set, validation, or on the OOB with implementations that support it): the training set contains the memory of the model, not its ability to generalize. It holds for the permutation, for the drop-column, for SHAP: compute them on validation data.
        - With a "large" deep forest the phenomenon is stronger the more the model memorizes: it is another reason to use a higher `min_samples_leaf`, which makes the training set less perfect (and the importances on the training set less distorted).
        - **A cheap check:** add two random columns (a continuous one, one with many integer values) to your dataset and check where they end up in the ranking. If a real feature is **below** the random columns, its importance cannot be distinguished from noise.
    """,
    pitfall="""
        Computing the importance on the same set you trained on "because it is more convenient". With flexible models (forests, boosting) it is a method that rewards the memorized noise.
    """,
    interview=(
        "Why must the permutation importance be computed on a validation set?",
        """
        Because it measures how much the performance gets worse by shuffling a column: on the training set a flexible model has almost perfect performance and also uses noise columns to memorize, so their permutation makes it worse and they look important. On unseen data the performance reflects only what the model
        learned that generalizes. A practical test is to add random columns and check that they come out at the bottom of the ranking.
        """),
)

EXERCISES = [E1, E2, E3]
