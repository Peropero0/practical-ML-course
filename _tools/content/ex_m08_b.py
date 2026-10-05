"""Module 8 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="forest_on_the_titanic_with_oob", kind="B", level=2, minutes=50, module=MODULE,
    title="A forest on the Titanic driven by the OOB error",
    goal="""
        Build a pipeline with a `RandomForestClassifier` on the Titanic and use the **out-of-bag** error instead of cross-validation to: see how many trees are needed (OOB curve with `warm_start`), choose `max_features` and `min_samples_leaf`, and compare the OOB estimate with the cross-validation one and with the test set.
    """,
    dataset="""
        `ds.titanic()`, stratified 75/25 split: `X_tr, X_te, y_tr, y_te`. Preprocessing (given in `build_prep()`): numeric `pclass, age, siblings_spouses, parents_children, fare` (median) and categorical `sex, port, deck, title` (most frequent + `OrdinalEncoder`).
    """,
    purpose="""
        Learn to use the OOB as a working tool: it is free (no retraining per fold), it is honest (it uses only trees that have not seen the example) and it lets you iterate quickly on the hyperparameters.
    """,
    task=[
        "With `warm_start=True` train the pipeline increasing `n_estimators` in `[5, 10, 20, 50, 100, 200, 400]` and save `oob_curve` (`pd.Series` of the OOB **error** = `1 − oob_score_`, index: number of trees) and `stable_n`: the minimum number of trees for which the OOB error is within **0.005** of the final value (at 400).",
        "With `n_estimators = 300` compute the OOB error for `max_features` in `[1, 2, 3, \"sqrt\", 5, 9]` (`mf_grid`, `pd.Series` indexed by the label as a string) and for `min_samples_leaf` in `[1, 2, 5, 10]` (`leaf_grid`); save `best_mf` and `best_leaf`.",
        "Train the model with the best values: save `oob_err` (OOB error), `cv_err` (error in `StratifiedKFold(5, shuffle=True, random_state=0)` on the training set) and `test_err` (error on the test set). Compare them: which one is the most optimistic?",
        "Save `train_acc` (accuracy on the training set) and show how misleading it is.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder

        df = ds.titanic()
        X, y = df.drop(columns="survived"), df["survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)


        def build_prep():
            return ColumnTransformer([("num", SimpleImputer(strategy="median"), ["pclass", "age", "siblings_spouses", "parents_children", "fare"]),
                                      ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), ["sex", "port", "deck", "title"])])


        def model(**kw):
            return make_pipeline(build_prep(), RandomForestClassifier(oob_score=True, random_state=0, n_jobs=-1, **kw))
    """,
    starter=[
        C("""
        oob_curve = pd.Series(dtype=float)
        stable_n = None
        """),
        C("""
        mf_grid = pd.Series(dtype=float)
        leaf_grid = pd.Series(dtype=float)
        best_mf = None
        best_leaf = None
        """),
        C("""
        oob_err = None
        cv_err = None
        test_err = None
        train_acc = None
        """),
    ],
    checks="""
        check(list(oob_curve.index) == [5, 10, 20, 50, 100, 200, 400] and oob_curve[5] > oob_curve[100] + 0.02 and 20 <= stable_n <= 200, "the OOB error drops from 5 to 100 trees by over 2 points and settles between 20 and 200 trees", "warm_start=True and set_params(randomforestclassifier__n_estimators=n)")
        check(len(mf_grid) == 6 and len(leaf_grid) == 4 and str(best_mf) in mf_grid.index and best_leaf in (1, 2, 5, 10), "six values of max_features and four of min_samples_leaf", "one OOB error for each value")
        check(mf_grid["1"] > mf_grid.min() + 0.01, "with max_features=1 the error is worse than the best by over 1 point (trees too weak)", "compare the OOB errors")
        check(0.12 < oob_err < 0.25 and 0.12 < cv_err < 0.25 and 0.12 < test_err < 0.25, "OOB, cross-validation and test errors between 0.12 and 0.25 (accuracies between 75 and 88%)", "1 - accuracy")
        check(abs(oob_err - cv_err) < 0.03 and abs(oob_err - test_err) < 0.05, "the OOB error is close to the cross-validation one (within 3 points) and to the test one (within 5)", "OOB as an honest estimate")
        check(train_acc > 0.85 and 1 - train_acc < oob_err * 0.85, "on the training set the accuracy goes over 0.85: misleading (error clearly below the OOB one)", "model.score(X_tr, y_tr)")
    """,
    hint="""
        `m = model(warm_start=True)`; in the loop `m.set_params(randomforestclassifier__n_estimators=n).fit(X_tr, y_tr)`; `1 - m[-1].oob_score_`. For `max_features`: `model(n_estimators=300, max_features=mf)`. CV: `cross_val_score(model(...), X_tr, y_tr, cv=cv).mean()`.
    """,
    solution=[
        C("""
        m = model(warm_start=True)
        rows = {}
        for n in [5, 10, 20, 50, 100, 200, 400]:
            m.set_params(randomforestclassifier__n_estimators=n).fit(X_tr, y_tr)
            rows[n] = 1 - m[-1].oob_score_
        oob_curve = pd.Series(rows)
        stable_n = int(next(n for n in oob_curve.index if abs(oob_curve[n] - oob_curve[400]) <= 0.005))
        print(oob_curve.round(4).to_string(), "\\ntrees to settle the OOB error within 0.005:", stable_n)
        """),
        C("""
        def oob_error(**kw):
            return 1 - model(n_estimators=300, **kw).fit(X_tr, y_tr)[-1].oob_score_


        mf_grid = pd.Series({str(mf): oob_error(max_features=mf) for mf in [1, 2, 3, "sqrt", 5, 9]})
        leaf_grid = pd.Series({l: oob_error(min_samples_leaf=l) for l in [1, 2, 5, 10]})
        best_mf = mf_grid.idxmin()
        best_leaf = leaf_grid.idxmin()
        print("OOB error by max_features:", mf_grid.round(4).to_dict())
        print("OOB error by min_samples_leaf:", leaf_grid.round(4).to_dict())
        """),
        C("""
        mf = int(best_mf) if best_mf.isdigit() else best_mf
        final = model(n_estimators=300, max_features=mf, min_samples_leaf=best_leaf).fit(X_tr, y_tr)
        oob_err = 1 - final[-1].oob_score_
        cv_err = 1 - cross_val_score(model(n_estimators=300, max_features=mf, min_samples_leaf=best_leaf), X_tr, y_tr, cv=cv).mean()
        test_err = 1 - final.score(X_te, y_te)
        train_acc = final.score(X_tr, y_tr)
        print(f"best: max_features={best_mf}, min_samples_leaf={best_leaf}")
        print(f"error: OOB {oob_err:.4f} | cross-validation {cv_err:.4f} | test {test_err:.4f} | (training accuracy: {train_acc:.3f})")
        """),
    ],
    learned="""
        - **The OOB error settles early:** it drops quickly up to about 20 trees (0.27 with 5, 0.22 with 20) and then stays flat (0.223 → 0.214 between 100 and 400 trees, differences of half a point: noise): a number like 200-300 is more than enough. Adding trees never overfits, but after the settling it only costs time and memory.
        - **The OOB replaces cross-validation for choosing the hyperparameters:** each value of `max_features` or `min_samples_leaf` requires **one** training, not five. With `max_features=1` the trees are too weak (0.223); the minimum is with 5 features out of 9 (0.205); the best leaf is 5-10 (0.189) against 0.212 with single-example leaves.
        - **OOB, cross-validation and test give almost the same number** (error 0.180, 0.184 and 0.183: accuracy 82%): the OOB is an honest estimate of the generalization. **The training accuracy is 0.87**: it is the only one of the four figures never to report.
        - The limits of the OOB (correlated or duplicate rows, C1) and the fact that by default it measures accuracy (for AUC, F1 or costs you need to use `oob_decision_function_` as in B3) are the two reasons cross-validation remains the standard for the final comparison.
    """,
    pitfall="""
        Reporting the `model.score(X_tr, y_tr)` of a random forest: it is almost 1.0 and says nothing. The OOB error (`1 - oob_score_`) is the internal estimate to use when you do not have a validation set.
    """,
    interview=(
        "How do you choose the hyperparameters of a random forest without cross-validation?",
        """
        With the out-of-bag error: each configuration requires a single training and the OOB is an honest estimate (each example is predicted only by the trees that have not seen it). I look at the curve as a function of n_estimators for stability, then search max_features and min_samples_leaf on a small grid.
        I use cross-validation or a test set only for the final confirmation and when the rows are not independent.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="regression_forest_on_california", kind="B", level=2, minutes=55, module=MODULE,
    title="Regression random forest on house prices",
    goal="""
        Compare on California the linear regression, the geographic KNN (M5), an optimized tree (M7) and the random forest, looking at how the error changes with `n_estimators`, `max_features` and `min_samples_leaf`. Then measure the feature importance by permutation and the cost in time and memory.
    """,
    dataset="""
        `ds.california()` (20,640 districts), 75/25 split with `random_state=0`; RMSE with `squared=False`; the internal estimate is the OOB (`RandomForestRegressor(oob_score=True)` gives the R²).
    """,
    purpose="""
        See how much a forest gains over a tree (and over simpler models) on a real problem with strong interactions, and understand which parameters really matter: `max_features` and `min_samples_leaf`, not `n_estimators` beyond a certain point.
    """,
    task=[
        "Save `rmse` (dictionary) on the test set of: `linear`, `geo knn` (`KNeighborsRegressor(5)` on `Latitude`, `Longitude`), `tree` (`DecisionTreeRegressor(min_samples_leaf=20)`) and `forest` (`RandomForestRegressor(200, max_features=0.33, min_samples_leaf=3, random_state=0, n_jobs=-1)`).",
        "With `RandomForestRegressor(100, oob_score=True, ...)` compute `oob_r2` for `max_features` in `[0.2, 0.33, 0.5, 1.0]` and `min_samples_leaf` in `[1, 3, 10]` (`oob_table`, DataFrame with `max_features` rows, `min_samples_leaf` columns); save `best_mf` and `best_leaf` (the argmax).",
        "Train the forest with the best values and save `forest_rmse` on the test set, `t_fit` and `t_predict` (seconds for `fit` and for `predict` on the test set) and `size_mb` (size in MB of `pickle.dumps(model)`).",
        "Compute `perm` (`pd.Series`, permutation importance on the test set with `n_repeats=5`, `random_state=0`, scoring `neg_root_mean_squared_error`; sorted in descending order) and `top3` (the names of the first three features).",
    ],
    setup="""
        import pickle
        import time
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        error = lambda m, cols=None: mean_squared_error(y_te, m.fit(X_tr if cols is None else X_tr[cols], y_tr).predict(X_te if cols is None else X_te[cols]), squared=False)
    """,
    starter=[
        C("""
        rmse = {}
        rmse
        """),
        C("""
        oob_table = pd.DataFrame(index=[0.2, 0.33, 0.5, 1.0], columns=[1, 3, 10], dtype=float)
        best_mf = None
        best_leaf = None
        """),
        C("""
        forest_rmse = None
        t_fit = None
        t_predict = None
        size_mb = None
        """),
        C("""
        perm = None
        top3 = None
        """),
    ],
    checks="""
        check(set(rmse) == {"linear", "geo knn", "tree", "forest"} and rmse["forest"] < rmse["tree"] - 0.1 and rmse["forest"] < rmse["geo knn"], "the forest beats the tree by over 0.1 and the geographic KNN", "compare the RMSEs")
        check(rmse["forest"] < 0.55 and rmse["forest"] < rmse["linear"] - 0.15, "forest RMSE below 0.55 and over 0.15 better than the linear one", "RandomForestRegressor(200, max_features=0.33, min_samples_leaf=3)")
        check(oob_table.notna().all().all() and best_mf in (0.33, 0.5) and oob_table.loc[1.0].max() < oob_table.loc[0.33].max(), "the best is 0.33 or 0.5; with all the features (bagging) the OOB R2 is worse", "oob_score_ (R2) of RandomForestRegressor")
        check(best_leaf in (1, 3), "with 100 trees the best leaf is small (1 or 3)", "argmax of the table")
        check(forest_rmse < 0.52 and t_fit > t_predict and 20 < size_mb < 2000, "final RMSE below 0.52; training slower than prediction; model of tens or hundreds of MB", "pickle.dumps(model)")
        check(top3[0] == "MedInc" and set(top3) >= {"MedInc"} and len(perm) == 8 and (perm.diff().dropna() <= 1e-12).all(), "the median income is the most important feature; sorted series", "permutation_importance with scoring neg_root_mean_squared_error")
    """,
    hint="""
        Permutation: `r = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=0, scoring="neg_root_mean_squared_error")`; `pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)`.
        Size: `len(pickle.dumps(model)) / 1e6`. Times: `time.time()` before and after `fit` and `predict`.
    """,
    solution=[
        C("""
        rmse = {
            "linear": error(LinearRegression()),
            "geo knn": error(KNeighborsRegressor(5), ["Latitude", "Longitude"]),
            "tree": error(DecisionTreeRegressor(min_samples_leaf=20, random_state=0)),
            "forest": error(RandomForestRegressor(200, max_features=0.33, min_samples_leaf=3, random_state=0, n_jobs=-1)),
        }
        print({k: round(v, 3) for k, v in rmse.items()})
        """),
        C("""
        oob_table = pd.DataFrame(index=[0.2, 0.33, 0.5, 1.0], columns=[1, 3, 10], dtype=float)
        for mf in oob_table.index:
            for leaf in oob_table.columns:
                f = RandomForestRegressor(100, max_features=mf, min_samples_leaf=leaf, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
                oob_table.loc[mf, leaf] = f.oob_score_
        pos = np.unravel_index(oob_table.to_numpy().argmax(), oob_table.shape)
        best_mf, best_leaf = oob_table.index[pos[0]], oob_table.columns[pos[1]]
        print(oob_table.round(4)); print("best:", best_mf, best_leaf)
        """),
        C("""
        final = RandomForestRegressor(200, max_features=best_mf, min_samples_leaf=best_leaf, random_state=0, n_jobs=-1)
        t0 = time.time(); final.fit(X_tr, y_tr); t_fit = time.time() - t0
        t0 = time.time(); pred = final.predict(X_te); t_predict = time.time() - t0
        forest_rmse = mean_squared_error(y_te, pred, squared=False)
        size_mb = len(pickle.dumps(final)) / 1e6
        print(f"test RMSE {forest_rmse:.3f} | fit {t_fit:.1f} s | predict {t_predict:.2f} s | size {size_mb:.0f} MB")
        """),
        C("""
        result = permutation_importance(final, X_te, y_te, n_repeats=5, random_state=0, scoring="neg_root_mean_squared_error")
        perm = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
        top3 = list(perm.index[:3])
        print(perm.round(3).to_string())
        """),
    ],
    learned="""
        - **The forest is the quality leap over simple models:** RMSE 0.49-0.50 against 0.62 of the optimized tree (M7), 0.53 of the geographic KNN (M5), 0.735 of the linear one. A reduction of about 20% of the error over the tree just by averaging many trees.
        - **`max_features` matters, the leaf much less:** with all the features (bagging) the OOB R² is lower; the best is between 0.33 and 0.5. With 100 deep trees (leaf 1-3) the OOB is almost indifferent to the minimum leaf size: the trees are already decorrelated and the average does the rest.
        - **The cost is real:** training takes seconds and the model weighs hundreds of MB (200 trees with thousands of nodes each); prediction on 5000 examples takes a fraction of a second but to serve single low-latency requests the depth and the number of trees matter.
        - **Permutation importance:** `MedInc` (the income) is by far the most important, followed by the geographic position (`Latitude`, `Longitude`) and `AveOccup`; the others are worth little. It is a reading that does not depend on the units or on the cardinality of the features, unlike the impurity importance.
    """,
    pitfall="""
        Comparing the forest with an unregularized tree and concluding "the forest is much better": the honest comparison is with the **optimized** tree (here 0.62): the real gain is a fifth of the error, not a third.
    """,
    interview=(
        "What are the most important hyperparameters of a random forest?",
        """
        max_features (it controls the correlation between trees and the bias of each) and min_samples_leaf (it regularizes the leaves), then n_estimators as a budget to choose high enough by looking at the OOB error. The maximum depth is rarely needed. For regression the default of max_features
        in scikit-learn is 1.0 (all the features: bagging), so it is almost always worth trying smaller values like 0.33.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="imbalanced_forest_threshold_from_oob", kind="B", level=3, minutes=55, module=MODULE,
    title="Rare classes: weights, `balanced_subsample` and a threshold chosen with the OOB",
    goal="""
        On a problem with 3% of positives compare four forests (base, `class_weight=\"balanced\"`, `\"balanced_subsample\"`, minimum leaf 10) with average precision and F1; choose the **threshold** from the `oob_decision_function_` (without cross-validation and without touching the test set) and check that it improves the F1 over 0.5;
        measure the effect on the mean probabilities.
    """,
    dataset="""
        `make_classification(12000, 12 features, 5 informative, about 3% positives)`, stratified 50/50 split: `X_tr, y_tr, X_te, y_te` (204 positives in each).
    """,
    purpose="""
        Exploit the peculiarity of the forest: the **out-of-bag probabilities** are out-of-sample predictions for the whole training set, so a threshold can be chosen on them without cross-validation and without contaminating the test set. And see what class weights really do in a forest.
    """,
    task=[
        "For each of the four configurations (`RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1, ...)`: `base`, `balanced`, `balanced_subsample`, `leaf10` = `min_samples_leaf=10`) save on the test set: `ap` (average precision), `auc`, `f1_05` (F1 with threshold 0.5) and `mean_p` (mean predicted probability). Build `table` (index: configurations).",
        "Write `f1_threshold(y, p)` (grid `np.linspace(0.02, 0.9, 89)`, threshold with maximum F1). For each configuration compute `oob_threshold` from the OOB probabilities (`oob_decision_function_[:, 1]`) and `oob_f1` (F1 on the **test set** with that threshold). Add the two columns to `table`.",
        "Save `best_f1` (configuration with the highest `oob_f1`) and `threshold_gain` = `oob_f1 − f1_05` for `base`.",
        "Save `threshold_far_from_05` (True if the `oob_threshold` of `base` is smaller than 0.4).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = make_classification(12000, n_features=12, n_informative=5, n_redundant=0, weights=[0.97], class_sep=0.9, flip_y=0.01, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
        configs = {"base": {}, "balanced": {"class_weight": "balanced"}, "balanced_subsample": {"class_weight": "balanced_subsample"}, "leaf10": {"min_samples_leaf": 10}}
        print("positives: training", int(y_tr.sum()), "| test", int(y_te.sum()))
    """,
    starter=[
        C("""
        def f1_threshold(y, p):
            ...
        """),
        C("""
        rows = {}
        for name, kw in configs.items():
            f = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1, **kw).fit(X_tr, y_tr)
            ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        best_f1 = None
        threshold_gain = None
        threshold_far_from_05 = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["ap", "auc", "f1_05", "mean_p", "oob_threshold", "oob_f1"] and list(table.index) == list(configs), "table with six columns and four configurations", "ap, auc, f1_05, mean_p, oob_threshold, oob_f1")
        check(table["auc"].min() > 0.83 and table["ap"].min() > 0.5 and table["ap"].max() - table["ap"].min() < 0.06, "AUC over 0.83 and average precision 0.5-0.65 for all: the configurations rank almost the same way", "the forest already handles the imbalance well")
        check((table["oob_f1"] > table["f1_05"]).all(), "with the OOB threshold the F1 on the test set improves over threshold 0.5 for all the configurations", "oob_decision_function_[:, 1]")
        check(table.loc["base", "oob_threshold"] < 0.4 and threshold_far_from_05 is True and threshold_gain > 0.04, "the OOB threshold for the base forest is below 0.4 and gains over 0.04 of F1", "oob_f1 - f1_05 for the base configuration")
        check(best_f1 in configs and table.loc[best_f1, "oob_f1"] == table["oob_f1"].max(), "configuration with the highest F1", "idxmax of oob_f1")
        check(table.loc["base", "f1_05"] > table.loc["balanced", "f1_05"], "at threshold 0.5 the 'balanced' weights worsen the F1 (the base model has a much higher precision)", "compare f1_05")
    """,
    hint="""
        `p_oob = f.oob_decision_function_[:, 1]`; `threshold = f1_threshold(y_tr, p_oob)`; `f1_score(y_te, p_test >= threshold)`. With `oob_score=True` the attribute `oob_decision_function_` exists after `fit`.
    """,
    solution=[
        C("""
        def f1_threshold(y, p):
            ts = np.linspace(0.02, 0.9, 89)
            return float(ts[int(np.argmax([f1_score(y, p >= t) for t in ts]))])
        """),
        C("""
        rows = {}
        for name, kw in configs.items():
            f = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1, **kw).fit(X_tr, y_tr)
            p = f.predict_proba(X_te)[:, 1]
            threshold = f1_threshold(y_tr, f.oob_decision_function_[:, 1])                  # threshold chosen on the out-of-bag probabilities of the TRAINING set
            rows[name] = {"ap": average_precision_score(y_te, p), "auc": roc_auc_score(y_te, p), "f1_05": f1_score(y_te, p >= 0.5), "mean_p": p.mean(),
                          "oob_threshold": threshold, "oob_f1": f1_score(y_te, p >= threshold)}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        best_f1 = table["oob_f1"].idxmax()
        threshold_gain = table.loc["base", "oob_f1"] - table.loc["base", "f1_05"]
        threshold_far_from_05 = bool(table.loc["base", "oob_threshold"] < 0.4)
        print(f"best F1 with the OOB threshold: {best_f1} | gain of the threshold for the base forest: {threshold_gain:+.3f} (threshold {table.loc['base', 'oob_threshold']:.2f})")
        """),
    ],
    learned="""
        - **The base forest is already robust to imbalance:** AUC 0.86 and average precision 0.59, and the class weights change little (AP 0.60 with `balanced`). But **at threshold 0.5** a rare positive rarely receives a fraction of votes above 0.5: recall of a third and F1 0.50.
        - **A threshold from the OOB improves the F1 without touching the test set:** 0.29 on the out-of-bag probabilities (out-of-sample predictions of the whole training set) takes the F1 from 0.50 to 0.58 on the test set. It is a direct application of the idea of M4 (choose the threshold on the training set with out-of-sample probabilities), but without `cross_val_predict`: the OOB is already there.
        - **The class weights (`balanced`) do not help** with the optimized threshold and worsen the F1 at 0.5 (more false positives): with a forest it is almost always better to leave the probabilities free and tune the threshold. It is the same result as M4 F2 with another model.
        - With a small number of positives (204) the estimate of the threshold is noisy: here it is fine because the OOB has a large number of examples, but with few positives a coarser grid of thresholds is better and do not "optimize" to the third decimal.
    """,
    pitfall="""
        Rebalancing the classes (`class_weight`, oversampling) as a first move with a forest. The forest already ranks the cases well (high AUC): the problem is where to cut, that is the threshold, and that is solved without distorting the probabilities.
    """,
    interview=(
        "How do you choose the threshold of a random forest on an imbalanced problem?",
        """
        I use the out-of-bag probabilities of the training set (oob_decision_function_), which are out-of-sample predictions: I choose the threshold on them based on costs or F1 and then validate it once on the test set. I avoid rebalancing at the start: the forest already ranks well, and the weights distort the probabilities. If I need calibrated
        probabilities I recalibrate them (sigmoid or isotonic) with data not used for training.
        """),
)

EXERCISES = [B1, B2, B3]
