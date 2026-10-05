"""Module 9 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_the_training_curves", kind="D", level=2, minutes=50, module=MODULE,
    title="Reading the training curves of XGBoost",
    goal="""
        Read `evals_result()` of four configurations (depth 2, 4, 6 and 6 with `min_child_weight=20`) trained for 600 rounds without early stopping: where the validation log-loss minimum is, how large the training-validation gap is at the minimum and at the end, and which configuration is the best and why. Then
        predict the early stopping round by looking at the curve.
    """,
    dataset="""
        `make_classification(4000, 12 features, 5 informative, flip_y=0.08)`, 60% training, 20% validation, 20% test split. Models: `XGBClassifier(n_estimators=600, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")`, `evals_result` already computed in `curves[name]` with keys `validation_0` (training) and `validation_1` (validation).
    """,
    purpose="""
        Learn to read the two curves and recognize what they say: the validation minimum is the "right round"; the gap with the training curve is a measure of overfitting; depth and `min_child_weight` change both the minimum and the gap.
    """,
    task=[
        "For each configuration (`curves` is a dictionary name → result of `evals_result()`), build `table` (index: configurations; columns `min_round` (1-based index of the validation minimum), `val_min` (minimum value), `train_at_min`, `gap_at_min` (`val_min − train_at_min`), `final_val` and `final_gap`).",
        "Save `best` (configuration with the lowest `val_min`) and `most_overfit` (configuration with the largest `final_gap`).",
        "Save `degradation` (`pd.Series`, `final_val − val_min` per configuration): how much the validation gets worse if you do not stop at the minimum.",
        "Simulate early stopping with patience 30: `es_round` (dictionary configuration → first round `t`, from 1, such that the minimum of `val[:t]` was reached 30 rounds earlier, that is `argmin(val[:t]) + 1 + 30 == t`) and `es_cost` (`pd.Series`: the `val_min` you would find by stopping at `es_round`, that is `min(val[:es_round])`, minus the `val_min` of the table). Note that the stop triggers at the **first** plateau: it can come before the global minimum.",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split

        X, y = make_classification(4000, n_features=12, n_informative=5, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        configs = {"depth 2": dict(max_depth=2), "depth 4": dict(max_depth=4), "depth 6": dict(max_depth=6), "depth 6, min_child_weight 20": dict(max_depth=6, min_child_weight=20)}
        curves = {}
        for name, kw in configs.items():
            m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", n_jobs=-1, random_state=0, **kw)
            m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
            curves[name] = m.evals_result()
        print(list(curves))
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=list(curves), columns=["min_round", "val_min", "train_at_min", "gap_at_min", "final_val", "final_gap"], dtype=float)
        for name, r in curves.items():
            train, val = np.array(r["validation_0"]["logloss"]), np.array(r["validation_1"]["logloss"])
            ...
        table.round(3)
        """),
        C("""
        best = None
        most_overfit = None
        degradation = None
        es_round = {}
        es_cost = None
        """),
    ],
    checks="""
        check(table.notna().all().all() and list(table.columns) == ["min_round", "val_min", "train_at_min", "gap_at_min", "final_val", "final_gap"], "complete table", "one row per configuration")
        check(table["min_round"].is_monotonic_decreasing is False and table.loc["depth 2", "min_round"] > table.loc["depth 6", "min_round"] + 100, "shallow trees reach the minimum much later (over 100 rounds after the deep ones)", "min_round")
        check(best in table.index and table.loc[best, "val_min"] == table["val_min"].min() and table["val_min"].max() - table["val_min"].min() < 0.05, "the four configurations have a similar validation minimum (difference below 0.05): depth matters less than the right round", "val_min")
        check(most_overfit == "depth 6" and table.loc["depth 6", "final_gap"] > 2 * table.loc["depth 2", "final_gap"], "depth 6 without constraints has the largest final gap (over double that of depth 2)", "final_gap")
        check(degradation["depth 6"] > 0.03 and degradation["depth 2"] < 0.01 and degradation["depth 6, min_child_weight 20"] < degradation["depth 6"] / 3, "without early stopping depth 6 loses over 0.03 of log-loss; depth 2 less than 0.01; min_child_weight cuts the effect to less than a third", "final_val - val_min")
        check(len(es_round) == 4 and all(30 < es_round[n] <= table.loc[n, "min_round"] + 30 for n in table.index), "the stop triggers at the first plateau: never after the global minimum plus 30 rounds, often before", "argmin(val[:t]) + 1 + 30 == t")
        check(es_cost is not None and (es_cost > -1e-9).all() and (es_cost < 0.01).all(), "stopping at the first plateau costs less than 0.01 of validation log-loss compared with the global minimum", "min(val[:es_round]) - val_min")
    """,
    hint="""
        `min_round = int(val.argmin()) + 1`; `train_at_min = train[val.argmin()]`. Early stopping: for `t` from 1 to 600 the first `t` for which `val[:t].argmin() + 1 + 30 == t`; then `val[:t].min() - val.min()`.
    """,
    solution=[
        C("""
        table = pd.DataFrame(index=list(curves), columns=["min_round", "val_min", "train_at_min", "gap_at_min", "final_val", "final_gap"], dtype=float)
        for name, r in curves.items():
            train, val = np.array(r["validation_0"]["logloss"]), np.array(r["validation_1"]["logloss"])
            k = int(val.argmin())
            table.loc[name] = [k + 1, val[k], train[k], val[k] - train[k], val[-1], val[-1] - train[-1]]
        table.round(3)
        """),
        C("""
        best = table["val_min"].idxmin()
        most_overfit = table["final_gap"].idxmax()
        degradation = table["final_val"] - table["val_min"]
        es_round = {}
        for name, r in curves.items():
            val = np.array(r["validation_1"]["logloss"])
            for t in range(1, len(val) + 1):
                if int(val[:t].argmin()) + 1 + 30 == t:                    # 30 rounds without improving on the minimum
                    es_round[name] = t
                    break
        print("best:", best, "| most overfit:", most_overfit)
        print("validation degradation if you do not stop:", degradation.round(3).to_dict())
        es_cost = pd.Series({name: np.array(curves[name]["validation_1"]["logloss"])[:t].min() - table.loc[name, "val_min"] for name, t in es_round.items()})
        print("early stopping rounds:", es_round)
        print("cost of stopping at the first plateau (log-loss):", es_cost.round(4).to_dict())
        """),
    ],
    learned="""
        - **The curves say three things:** where the validation minimum is (the right round), how large the training-validation gap is (the overfitting) and how the validation degrades after the minimum (the cost of not stopping). With `learning_rate=0.1` deep trees have the minimum early (round 116 with depth 6, 153 with 4) and shallow trees much later (566 with depth 2).
        - **The minima are almost equal** (log-loss 0.307-0.334 for all four): depth changes *when* you reach the minimum more than *how low* it is. That is why the tuning of `max_depth` has a second-order effect if the early stopping is done well: the configuration with `min_child_weight=20` is the best, but by 0.005 over depth 4.
        - **The final gap is a measure of overfitting** (depth 6: 0.36; depth 2: 0.15): it grows with depth and with the rounds. The cost of not stopping shows in the validation: after the minimum depth 6 loses 0.053 of log-loss, depth 2 only 0.002.
        - **`min_child_weight` is a strong regularization:** with depth 6 and `min_child_weight=20` the degradation after the minimum drops from 0.053 to 0.008 and the minimum moves from 116 to 265 rounds: it prevents isolating a few examples, which is the road of deep boosting to memorization.
        - **Early stopping halts at the first plateau, not at the global minimum:** with patience 30 depth 2 stops at 470 rounds, before the global minimum (566), and the others at 146-199 rounds. The cost in log-loss is negligible (less than 0.01): the curve is flat around the minimum.
    """,
    pitfall="""
        Comparing configurations after a **fixed** number of rounds (for example 200 for all): the deep ones are already past the minimum and look worse only because you did not stop them. Compare each one at its own best round.
    """,
    interview=(
        "How do you read the training curve of a boosting model?",
        """
        The training log-loss always drops; the validation one drops, touches a minimum and then climbs back. The minimum is the right number of rounds, the gap between the two curves measures the overfitting, and the climb back is the cost of not stopping. Configurations are compared each at its own minimum (early stopping),
        not at a fixed number of rounds.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="xgboost_importances_gain_weight_and_contributions", kind="D", level=3, minutes=55, module=MODULE,
    title="XGBoost importances: gain, weight, cover, permutation and contributions",
    goal="""
        On a dataset with a **very informative** feature, a moderately informative one, two correlated ones and two noise ones (one continuous and one binary) compare five XGBoost importance measures: `weight` (how many times used), `gain` (average gain), `cover`, permutation on the test set and the mean of the **per-example contributions** (`pred_contribs`).
        Check that the contributions **sum to the raw margin** and explain a single prediction.
    """,
    dataset="""
        `generate()` (given): 4000 examples with `strong`, `medium`, `corr_a`, `corr_b` (two noisy versions of the same factor, both informative), `noise_c` (continuous) and `noise_b` (binary). `X_tr, X_te, y_tr, y_te`; model `XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4)` already trained (`model`).
    """,
    purpose="""
        Know what each "importance" measures: `weight` rewards continuous features (many thresholds), `gain` is more sensible but depends on the training set, the permutation on new data answers the question "is it really needed?", the per-example contributions are a **local explanation** that sums to the margin.
    """,
    task=[
        "Build `table` (index: the six features; columns `weight`, `gain`, `cover`, `permutation`, `contributions`) with `booster.get_score(importance_type=...)` for the first three (0 if the feature does not appear), the permutation on the test set (AUC, `n_repeats=10`, `random_state=0`) and the mean of the absolute values of `pred_contribs` on the test set. Normalize each column to sum 1.",
        "Answer: `weight_rewards_noise` (True if `noise_c` has a share of at least 10% in `weight` while having less than 2% in the permutation), `gain_relegates_noise` (True if in `gain` `noise_c` has less than half the share of `medium`) and `top_by_method` (dictionary method → feature with the largest share).",
        "**Additivity:** for `X_te` compute `contribs = booster.predict(DMatrix(X_te), pred_contribs=True)` and `margin = booster.predict(DMatrix(X_te), output_margin=True)`; save `sum_ok` (True if `contribs.sum(1)` matches `margin` with tolerance 1e-4) and `rebuilt_prob` (sigmoid of the margin) compared with `predict_proba` (`prob_ok`).",
        "**Explain an example:** for example `i = 0` of the test set save `explanation` (`pd.Series` of the contributions of the six features sorted by absolute value, plus `bias`) and `decisive_feature` (name of the feature with the largest absolute contribution).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split


        def generate(n=4000, seed=0):
            rng = np.random.default_rng(seed)
            strong, medium, z = rng.normal(size=(3, n))
            corr_a, corr_b = z + rng.normal(0, 0.3, n), z + rng.normal(0, 0.3, n)
            y = (2 * strong + 0.7 * medium + 1.2 * z + rng.normal(0, 1, n) > 0).astype(int)
            X = pd.DataFrame({"strong": strong, "medium": medium, "corr_a": corr_a, "corr_b": corr_b, "noise_c": rng.normal(size=n), "noise_b": rng.integers(0, 2, n).astype(float)})
            return X, y


        X, y = generate()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr)
        booster = model.get_booster()
        names = list(X.columns)
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=names, columns=["weight", "gain", "cover", "permutation", "contributions"], dtype=float)
        table
        """),
        C("""
        weight_rewards_noise = None
        gain_relegates_noise = None
        top_by_method = None
        """),
        C("""
        sum_ok = None
        prob_ok = None
        """),
        C("""
        explanation = None
        decisive_feature = None
        """),
    ],
    checks="""
        check(table.notna().all().all() and close(table.sum().to_numpy(), np.ones(5)), "complete table with columns normalized to sum 1", "divide each column by its sum")
        check(weight_rewards_noise is True and gain_relegates_noise is True, "the continuous noise weighs over 10% for weight, less than half of `medium` for gain and less than 2% for permutation", "weight counts the cuts: many distinct values, many cuts")
        check(top_by_method["gain"] == "strong" and top_by_method["permutation"] == "strong" and top_by_method["contributions"] == "strong", "gain, permutation and contributions put `strong` first", "idxmax per column")
        check(sum_ok is True and prob_ok is True, "the contributions sum to the raw margin; the sigmoid of the margin reproduces predict_proba", "pred_contribs has a final 'bias' column")
        check(len(explanation) == 7 and explanation.index[-1] == "bias" or "bias" in explanation.index, "the explanation includes six features and the constant term", "one column per feature and one for the bias")
        check(decisive_feature in names and abs(explanation[decisive_feature]) == explanation.drop("bias").abs().max(), "the decisive feature is the one with the largest absolute contribution", "argmax of the absolute contributions")
    """,
    hint="""
        `scores = booster.get_score(importance_type="gain")` returns a dictionary only for the features used: `pd.Series(scores).reindex(names).fillna(0)`. Contributions: `xgb.DMatrix(X_te)` (with the column names of the DataFrame); the last column is the bias. Sigmoid: `1 / (1 + np.exp(-margin))`.
        Permutation: `permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0, scoring="roc_auc").importances_mean`.
    """,
    solution=[
        C("""
        dm = xgb.DMatrix(X_te)
        contribs = booster.predict(dm, pred_contribs=True)                       # (examples, features + bias)
        table = pd.DataFrame(index=names, columns=["weight", "gain", "cover", "permutation", "contributions"], dtype=float)
        for kind in ["weight", "gain", "cover"]:
            table[kind] = pd.Series(booster.get_score(importance_type=kind)).reindex(names).fillna(0)
        table["permutation"] = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=0, scoring="roc_auc").importances_mean
        table["contributions"] = np.abs(contribs[:, :-1]).mean(0)
        table = table.clip(lower=0)
        table = table / table.sum()
        table.round(3)
        """),
        C("""
        weight_rewards_noise = bool(table.loc["noise_c", "weight"] >= 0.1 and table.loc["noise_c", "permutation"] < 0.02)
        gain_relegates_noise = bool(table.loc["noise_c", "gain"] < 0.5 * table.loc["medium", "gain"])
        top_by_method = table.idxmax().to_dict()
        print("weight rewards the continuous noise:", weight_rewards_noise, "| gain relegates it:", gain_relegates_noise)
        print("top feature by method:", top_by_method)
        """),
        C("""
        margin = booster.predict(dm, output_margin=True)
        sum_ok = bool(np.allclose(contribs.sum(1), margin, atol=1e-4))
        prob_ok = bool(np.allclose(1 / (1 + np.exp(-margin)), model.predict_proba(X_te)[:, 1], atol=1e-4))
        print("the sum of the contributions matches the raw margin:", sum_ok, "| the sigmoid of the margin is predict_proba:", prob_ok)
        """),
        C("""
        c0 = pd.Series(contribs[0], index=names + ["bias"])
        explanation = c0.reindex(c0.drop("bias").abs().sort_values(ascending=False).index.tolist() + ["bias"])
        decisive_feature = explanation.drop("bias").abs().idxmax()
        print(explanation.round(3).to_string())
        print(f"margin = sum = {explanation.sum():.3f} -> probability {1 / (1 + np.exp(-explanation.sum())):.3f} | decisive feature: {decisive_feature}")
        """),
    ],
    learned="""
        - **`weight` counts the cuts, not the usefulness:** a continuous pure-noise feature (`noise_c`) is used in 13% of the splits (almost as much as the two correlated features, 17-19%); its `gain` share is 3.6% and that of permutation and contributions practically zero. `cover` (10.9%) does not relegate it: it counts how many examples go through its cuts, not how useful they are. The binary noise feature (`noise_b`) has only one possible threshold and gets 1% of `weight`. Like the impurity importance of a tree (M7 D2), `weight` rewards many distinct values.
        - **`gain` is the most sensible default** (and the value of `feature_importances_` of the scikit-learn wrapper), but it is computed on the training set and depends on the structure of the trees: here it assigns 46% to `strong` and the permutation gives it 76%. The permutation on unseen data remains the reference; with the correlated `corr_a` and `corr_b` the importance is split between the two copies (M8 C2).
        - **The per-example contributions sum exactly to the margin:** `contribs.sum(1) == margin` and the sigmoid of the margin is `predict_proba`. It is an **exact decomposition** of the prediction: each feature "pushes" the log-odds by a certain value relative to the mean prediction (`bias`). The mean of their absolute values is a global importance consistent with the permutation.
        - **A local explanation is a table, and it can contradict the global one:** for the first test example the decisive feature is `medium` (+2.29), countered by `corr_b` (−1.17) and `corr_a` (−0.41); `strong`, the most important on average, weighs only +0.17. Sum with the bias: margin 1.00, probability 0.73. It is the foundation of SHAP (`shap.TreeExplainer` computes values of the same kind with a more general method).
    """,
    pitfall="""
        Reporting `weight` as the "importance of the variables" in a report: it rewards features with many distinct values and makes you overestimate the continuous noise. For a business question use the permutation on validation data and for the single case the contributions.
    """,
    interview=(
        "How do you explain the prediction of an XGBoost model?",
        """
        At the global level with the permutation importance on a validation set (avoiding `weight`, which rewards continuous features), at the local level with the per-example contributions (`pred_contribs` or SHAP TreeExplainer): each feature contributes to the log-odds with a signed value, and the sum of the contributions plus the bias is the raw
        margin. With correlated features I group the variables before interpreting.
        """),
)

EXERCISES = [D1, D2]
