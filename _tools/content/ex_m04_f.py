"""Module 4 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="none_l2_l1_elasticnet", kind="F", level=3, minutes=55, module=MODULE,
    title="No penalty, L2, L1 and elastic net with few true features and many useless ones",
    goal="""
        Compare four regularizations of the logistic (none, L2, L1, elastic net) on 5 simulated datasets with 100 features of which only **6 informative** and 200 training examples, choosing `C` with
        cross-validation **for each model**. Measure log-loss and AUC on 2000 new examples, how many features each model keeps and how many of the six true ones it finds again.
    """,
    dataset="""
        `generate(seed)` (given): 2200 examples, 100 features (the first 6 informative, the other 94 noise), 3% of labels swapped; training = the first 200, test = the other 2000. The features are standardized inside the `Pipeline`.
    """,
    purpose="""
        Answer with numbers "when do you need L1 and when L2?". With few true features among many useless ones the **sparse** penalty (L1, elastic net) has an advantage, not only in interpretability but in prediction
        error; with no penalty at all the model is unusable.
    """,
    task=[
        "Build `models`, a dictionary name → `Pipeline` (`StandardScaler` + `LogisticRegression`): `none` (`C=1e6, max_iter=20000`), `L2` (`solver=\"lbfgs\"`), `L1` (`penalty=\"l1\", solver=\"liblinear\"`) and "
        "`elasticnet` (`penalty=\"elasticnet\", solver=\"saga\", l1_ratio=0.5`). All with `max_iter=5000` except `none`.",
        "For `L2`, `L1` and `elasticnet` wrap the pipeline in `GridSearchCV` over `C` in `np.logspace(-3, 1, 9)` with `StratifiedKFold(5, shuffle=True, random_state=0)` and `scoring=\"neg_log_loss\"`. "
        "`none` stays without a search.",
        "For each seed from 0 to 4: generate the data, train the four models and record on the test set `logloss` (with probabilities clipped to `[1e-15, 1−1e-15]`), `auc`, `n_coef` (coefficients with absolute value > 1e-8) and "
        "`n_true` (how many of these have an index below 6, that is are among the informative features). Save in `table` the **mean over the 5 seeds** (rows: models).",
        "Save `sparse_win` (True if the log-loss of `L1` **and** `elasticnet` are lower than that of `L2`) and `l1_true` (mean of the true features found by L1).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def generate(seed):
            X, y = make_classification(n_samples=2200, n_features=100, n_informative=6, n_redundant=0, class_sep=1.0, flip_y=0.03, shuffle=False, random_state=seed)
            o = np.random.default_rng(seed).permutation(len(y))         # shuffle the rows (the informative columns stay the first 6)
            X, y = X[o], y[o]
            return X[:200], y[:200], X[200:], y[200:]


        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        Cs = np.logspace(-3, 1, 9)
    """,
    starter=[
        C("""
        def new_models():
            \"\"\"Returns a dictionary name -> estimator, to be recreated for every seed.\"\"\"
            ...
        """),
        C("""
        rows = []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = generate(seed)
            for name, m in new_models().items():
                ...
        table = pd.DataFrame(rows).groupby("model").mean().drop(columns="seed")
        table.round(3)
        """),
        C("""
        sparse_win = None
        l1_true = None
        """),
    ],
    checks="""
        check(set(table.index) == {"none", "L2", "L1", "elasticnet"} and list(table.columns) == ["logloss", "auc", "n_coef", "n_true"], "table with four models and four metrics", "columns: logloss, auc, n_coef, n_true")
        check(table.loc["none", "logloss"] > 3 and table.loc["none", "n_coef"] == 100, "without penalty the test log-loss exceeds 3 and all 100 coefficients are nonzero", "C = 1e6")
        check(table.loc["L2", "n_coef"] == 100 and table.loc["L1", "n_coef"] < 50 and table.loc["elasticnet", "n_coef"] < 60, "L2 zeroes nothing; L1 and elastic net keep fewer than half of the features", "count |coef| > 1e-8")
        check(sparse_win is True and table.loc["L1", "auc"] > table.loc["L2", "auc"] + 0.03, "L1 and elastic net beat L2 on log-loss, and L1 by over 0.03 of AUC", "compare the logloss and auc columns")
        check(table.loc["L2", "logloss"] < table.loc["none", "logloss"] / 5, "L2 reduces the test log-loss by over 5 times compared to no penalty", "compare logloss")
        check(l1_true == table.loc["L1", "n_true"] and l1_true >= 4, "L1 finds on average at least 4 of the 6 informative features", "n_true of the L1 row")
        check(table.loc["L1", "n_coef"] > 12, "but it keeps many useless features (over 12): the selection is not clean", "n_coef of the L1 row")
    """,
    hint="""
        `elasticnet`: `LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, max_iter=5000)`. Search: `GridSearchCV(pipe, {"logisticregression__C": Cs}, cv=cv, scoring="neg_log_loss")`; after `fit` the best model is
        `search.best_estimator_[-1].coef_[0]`. `none` has no `best_estimator_`: use `m[-1].coef_[0]`.
    """,
    solution=[
        C("""
        def new_models():
            base = lambda **kw: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, **kw))
            grid = {"logisticregression__C": Cs}
            search = lambda pipe: GridSearchCV(pipe, grid, cv=cv, scoring="neg_log_loss")
            return {
                "none": make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=20000)),
                "L2": search(base(solver="lbfgs")),
                "L1": search(base(penalty="l1", solver="liblinear")),
                "elasticnet": search(base(penalty="elasticnet", solver="saga", l1_ratio=0.5)),
            }
        """),
        C("""
        rows = []
        for seed in range(5):
            X_tr, y_tr, X_te, y_te = generate(seed)
            for name, m in new_models().items():
                m.fit(X_tr, y_tr)
                p = np.clip(m.predict_proba(X_te)[:, 1], 1e-15, 1 - 1e-15)
                w = (m.best_estimator_ if hasattr(m, "best_estimator_") else m)[-1].coef_[0]
                active = np.flatnonzero(np.abs(w) > 1e-8)
                rows.append({"seed": seed, "model": name, "logloss": log_loss(y_te, p), "auc": roc_auc_score(y_te, p),
                             "n_coef": len(active), "n_true": int((active < 6).sum())})
        table = pd.DataFrame(rows).groupby("model").mean().drop(columns="seed")
        table = table.loc[["none", "L2", "L1", "elasticnet"]]
        table.round(3)
        """),
        C("""
        sparse_win = bool(table.loc["L1", "logloss"] < table.loc["L2", "logloss"] and table.loc["elasticnet", "logloss"] < table.loc["L2", "logloss"])
        l1_true = table.loc["L1", "n_true"]
        print(f"the sparse models beat L2: {sparse_win} | true features found by L1: {l1_true:.1f} out of 6 (with {table.loc['L1', 'n_coef']:.0f} active features in total)")
        """),
    ],
    learned="""
        - **No penalty:** with 200 examples and 100 features the model is separable and overfits completely (test log-loss 5.8, AUC 0.76): it is C1 in a realistic version.
        - **L2 shrinks but does not select**: it keeps all the 100 features with small weights and "spreads" the signal over the useless features (log-loss 0.58, AUC 0.78). It works when almost all the features carry a little information.
        - **L1 and elastic net win here** (log-loss 0.48, AUC 0.86) because the truth is **sparse**: few features matter and the rest is noise. L1 keeps on average 39 features, not 6: it finds 5.0 of the 6 true ones but
          adds many useless ones. L1's selection is **not reliable** as a variable selection procedure (for that you would need stability and repetition).
        - Elastic net (with `l1_ratio=0.5`) does almost as well as L1 and is generally more stable when the features are correlated, but requires the `saga` solver and more time. No rule always holds: if you know the signal is diffuse, L2;
          if it is sparse, L1 or elastic net. `C` and `l1_ratio` are chosen in cross-validation, not by eye.
    """,
    pitfall="""
        Concluding from a single dataset that "L1 is better than L2" (or vice versa). The answer depends on how the signal is made (sparse or diffuse) and on how many examples you have: here the difference is large because six features out of a hundred really matter. Try
        redoing the experiment with 30 informative features out of 100 and the result shrinks or flips.
    """,
    interview=(
        "When would you use L1 instead of L2 for a logistic?",
        """
        When I think few variables really matter and I want a sparse, interpretable model, or when I have many more features than examples. L2 when the signal is diffuse or the features are highly correlated (L1 picks one of the group
        unstably). Elastic net is the compromise. In all cases I standardize the features and choose `C` (and `l1_ratio`) in cross-validation, remembering that the selection done by L1 must be checked for stability.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="strategies_for_rare_classes", kind="F", level=3, minutes=55, module=MODULE,
    title="Rare classes: weights, undersampling, oversampling or just the threshold?",
    goal="""
        Compare five strategies for a rare event (about 3.5% positives): base model with threshold 0.5, `class_weight="balanced"`, random undersampling and oversampling (on the training set only) and base model with **threshold chosen
        in cross-validation**. Measure average precision, AUC, F1 and Brier score on 20,000 new examples, averaged over 10 datasets.
    """,
    dataset="""
        `generate(seed)` (given): 23,000 examples, 12 features (5 informative), about 3% positives; training = the first 3000 (about 105 positives), test = the other 20,000.
    """,
    purpose="""
        Replace intuition ("with imbalanced classes you must rebalance") with a measurement. You will find that the rebalancing techniques **shift the effective threshold**, they don't improve the ordering, and they **ruin** the probabilities.
    """,
    task=[
        "Write `best_threshold(y, p)`: the threshold (grid `np.linspace(0.02, 0.9, 89)`) that maximizes F1.",
        "For each seed from 0 to 9 and for each of the five strategies (`base`, `balanced`, `undersampling`, `oversampling`, `cv threshold`) compute on the test set: `ap`, `auc`, `f1`, `recall`, `precision` (threshold 0.5 except `cv threshold`), `brier` "
        "and `mean_p`. Undersampling: all the positives + the same number of random negatives; oversampling: all the negatives + positives repeated at random until they match (`rng = np.random.default_rng(seed)`). "
        "`cv threshold`: base model with the threshold chosen with `best_threshold` on the out-of-sample probabilities of `cross_val_predict` (5 stratified folds) on the training set.",
        "Save `table` (mean over the seeds, rows: strategies) and the answers: `best_f1` (strategy with maximum mean F1), `best_brier` (minimum Brier) and `balanced_mean_p` (mean probability predicted by the `balanced` strategy).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def generate(seed):
            X, y = make_classification(n_samples=23000, n_features=12, n_informative=5, n_redundant=0, weights=[0.97], class_sep=0.9, flip_y=0.01, random_state=seed)
            return X[:3000], y[:3000], X[3000:], y[3000:]


        def model(**kw):
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, **kw))
    """,
    starter=[
        C("""
        def best_threshold(y, p):
            ...
        """),
        C("""
        rows = []
        for seed in range(10):
            X_tr, y_tr, X_te, y_te = generate(seed)
            rng = np.random.default_rng(seed)
            ...
        table = pd.DataFrame(rows).groupby("strategy").mean().drop(columns="seed")
        table.round(3)
        """),
        C("""
        best_f1 = None
        best_brier = None
        balanced_mean_p = None
        """),
    ],
    checks="""
        check(set(table.index) == {"base", "balanced", "undersampling", "oversampling", "cv threshold"} and list(table.columns) == ["ap", "auc", "f1", "recall", "precision", "brier", "mean_p"], "five strategies and seven metrics", "columns: ap, auc, f1, recall, precision, brier, mean_p")
        check(best_f1 == "cv threshold" and table.loc["cv threshold", "f1"] > table.loc["base", "f1"] + 0.08, "the threshold chosen in cross-validation has the best F1 of all (and by over 0.08 compared to base)", "compare the f1 column")
        check(table.loc[["balanced", "undersampling", "oversampling"], "recall"].min() > 0.6 and table.loc[["balanced", "undersampling", "oversampling"], "precision"].max() < 0.2, "the three rebalancings have high recall (above 0.6) and low precision (below 0.2)", "they are shifts of the effective threshold")
        check(abs(table["auc"].max() - table["auc"].min()) < 0.03, "the AUC is practically the same (difference below 0.03)", "rebalancing does not change the ordering much")
        check(table.loc["base", "ap"] > table.loc[["balanced", "undersampling", "oversampling"], "ap"].max() + 0.05, "the average precision of the base model is better by 0.05 than any rebalancing", "AP: base model against rebalanced ones")
        check(best_brier in ("base", "cv threshold") and balanced_mean_p > 0.25 and table.loc["base", "mean_p"] < 0.05, "the best Brier is that of the base model; `balanced` predicts on average over 0.25 against a prevalence of 3.5%", "minimum brier and mean_p")
    """,
    hint="""
        `best_threshold`: `ts = np.linspace(0.02, 0.9, 89); ts[np.argmax([f1_score(y, p >= t) for t in ts])]`. Undersampling: `pos = np.where(y_tr == 1)[0]; neg = np.where(y_tr == 0)[0]; idx = np.r_[pos, rng.choice(neg, len(pos), replace=False)]`.
        Oversampling: `idx = np.r_[neg, rng.choice(pos, len(neg))]`. Metrics: a function `row(name, m, X_fit, y_fit, threshold=0.5)` avoids repeating the code.
    """,
    solution=[
        C("""
        def best_threshold(y, p):
            ts = np.linspace(0.02, 0.9, 89)
            return ts[int(np.argmax([f1_score(y, p >= t) for t in ts]))]
        """),
        C("""
        rows = []
        for seed in range(10):
            X_tr, y_tr, X_te, y_te = generate(seed)
            rng = np.random.default_rng(seed)
            pos, neg = np.where(y_tr == 1)[0], np.where(y_tr == 0)[0]

            def row(name, m, X_fit, y_fit, threshold=0.5):
                p = m.fit(X_fit, y_fit).predict_proba(X_te)[:, 1]
                rows.append({"seed": seed, "strategy": name, "ap": average_precision_score(y_te, p), "auc": roc_auc_score(y_te, p),
                             "f1": f1_score(y_te, p >= threshold), "recall": recall_score(y_te, p >= threshold),
                             "precision": precision_score(y_te, p >= threshold, zero_division=0), "brier": brier_score_loss(y_te, p), "mean_p": p.mean()})

            row("base", model(), X_tr, y_tr)
            row("balanced", model(class_weight="balanced"), X_tr, y_tr)
            u = np.r_[pos, rng.choice(neg, len(pos), replace=False)]
            row("undersampling", model(), X_tr[u], y_tr[u])
            o = np.r_[neg, rng.choice(pos, len(neg))]
            row("oversampling", model(), X_tr[o], y_tr[o])
            oof = cross_val_predict(model(), X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
            row("cv threshold", model(), X_tr, y_tr, threshold=best_threshold(y_tr, oof))
        table = pd.DataFrame(rows).groupby("strategy").mean().drop(columns="seed").loc[["base", "balanced", "undersampling", "oversampling", "cv threshold"]]
        table.round(3)
        """),
        C("""
        best_f1 = table["f1"].idxmax()
        best_brier = table["brier"].idxmin()
        balanced_mean_p = table.loc["balanced", "mean_p"]
        print(f"best F1: {best_f1} | best Brier: {best_brier} | mean probability predicted by 'balanced': {balanced_mean_p:.3f} (real prevalence about 0.035)")
        """),
    ],
    learned="""
        - **No rebalancing improves the ordering:** the AUC is almost equal (0.81 against 0.82) and the **average precision gets worse** (0.39 against 0.25-0.29), because weights and samplings reduce the information
          (undersampling throws away 90% of the negatives) or duplicate it (oversampling).
        - The three rebalancings are, in practice, **a threshold change in disguise**: recall 0.73, precision 0.10-0.11 and low F1 (0.17-0.20) at threshold 0.5. They move the working point along the curve without changing it, and in the wrong direction for the F1.
        - **The probabilities get ruined**: the mean predicted by `balanced` is 0.33 against a prevalence of 3.5%, and the Brier is six times worse (0.153 against 0.026). If you need probabilities (costs, ranking, recalibration) it is damage.
        - The best strategy for F1 is the simplest one: **base model + threshold chosen in cross-validation** (F1 0.43 against 0.29 for the base at threshold 0.5 and 0.17-0.20 for the rebalancings), without distorting the probabilities. Rebalancing
          can help with models that ignore the rare class (very deep trees, unregulated boosting) but it must be measured, not taken for granted.
    """,
    pitfall="""
        Applying undersampling or oversampling "because the classes are imbalanced" and judging the result with accuracy or with the recall at threshold 0.5. Before rebalancing, try the threshold: if the base model ranks well (high AUC),
        moving the threshold is almost always enough and costs nothing.
    """,
    interview=(
        "You have a dataset with 1% positives. Where do you start?",
        """
        From the base model without rebalancing, evaluated with average precision and precision-recall curves (not accuracy), and from the threshold chosen on the costs or in cross-validation. Only if the results are insufficient do I try class weights
        or sampling, always inside the validation (only on the training part of every fold) and comparing the probabilities and the calibration, not just the recall. With tree models I also consider the hyperparameters that control
        depth and minimum leaves, which often matter more than rebalancing.
        """),
)

EXERCISES = [F1, F2]
