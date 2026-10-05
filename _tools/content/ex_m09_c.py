"""Module 9 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="too_many_rounds_probabilities_get_ruined_first", kind="C", level=2, minutes=45, module=MODULE,
    title="Too many rounds: the accuracy stays, the probabilities get ruined",
    goal="""
        Train XGBoost with a high learning rate and deep trees for 400 rounds and observe, on the test set, **accuracy, log-loss and calibration** as a function of the rounds: the log-loss touches its minimum around 25 rounds and then rises by 45%, while the accuracy stays almost still. The training log-loss goes to almost zero.
    """,
    dataset="""
        `make_classification(6000, 12 features, 5 informative, flip_y=0.1)`, split 50% training, 30% test, 20% validation (`X_tr, X_te, X_val`); the model is `XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=6)`.
    """,
    purpose="""
        Understand that the overfitting of boosting shows up **in the probabilities first** and then in the classes: with `predict` (threshold 0.5) you may not notice it, but if you use the probabilities (risk, costs, ranking) they have become extreme and wrong. It is why early stopping is done on the log-loss and not on the classification error.
    """,
    task=[
        "With `m.predict_proba(X, iteration_range=(0, k))` compute for `k` in `[10, 25, 50, 100, 200, 400]` the training log-loss (`train_ll`), the test log-loss (`test_ll`) and the test accuracy (`test_acc`) in a `pd.DataFrame` `table` (index `k`).",
        "Save `k_min_ll` (the `k` with the minimum test log-loss), `min_ll`, `ll_400` and `ll_increase` (`ll_400 / min_ll − 1`), and `acc_change` (maximum `test_acc` minus `test_acc` at 400).",
        "Save `extreme_share` (for `k = 400` and for `k = 25`, the fraction of test probabilities below 0.02 or above 0.98, in a dictionary) and `extreme_err` (for `k = 400`, among the extreme probabilities, the fraction that gets the class wrong).",
        "**Cure:** train with `early_stopping_rounds=30` on `X_val, y_val` with `eval_metric=\"logloss\"` and save `early_test_ll` and `early_round`.",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        X, y = make_classification(6000, n_features=12, n_informative=5, class_sep=0.7, flip_y=0.1, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.6, random_state=0)
        m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=6, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr)
        ks = [10, 25, 50, 100, 200, 400]
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=ks, columns=["train_ll", "test_ll", "test_acc"], dtype=float)
        for k in ks:
            ...
        table.round(3)
        """),
        C("""
        k_min_ll = None
        min_ll = None
        ll_400 = None
        ll_increase = None
        acc_change = None
        """),
        C("""
        extreme_share = {}
        extreme_err = None
        """),
        C("""
        early_test_ll = None
        early_round = None
        """),
    ],
    checks="""
        check(table.notna().all().all() and table["train_ll"].is_monotonic_decreasing and table.loc[400, "train_ll"] < 0.02, "the training log-loss always drops, down to under 0.02", "iteration_range=(0, k)")
        check(k_min_ll in (10, 25, 50) and ll_increase > 0.35 and ll_400 > 0.65, "the test log-loss has its minimum between 10 and 50 rounds and at 400 it is over 35% higher (over 0.65)", "compare the rows of the table")
        check(abs(acc_change) < 0.02 and table["test_acc"].min() > 0.78, "the test accuracy varies by less than 2 points: the problem is not visible with predict", "maximum test_acc minus test_acc at 400")
        check(extreme_share[400] > 0.4 and extreme_share[25] < 0.1 and extreme_err > 0.05, "at 400 rounds over 40% of the probabilities are extreme (below 0.02 or above 0.98), at 25 less than 10%; among the extreme ones over 5% are wrong", "(p < 0.02) | (p > 0.98)")
        check(early_round < 100 and early_test_ll < 0.5 and early_test_ll < ll_400 - 0.15, "early stopping halts before 100 rounds and gives a log-loss below 0.5", "early_stopping_rounds=30 on X_val")
    """,
    hint="""
        `p = m.predict_proba(X_te, iteration_range=(0, k))[:, 1]`; `log_loss(y_te, p)`; `((p > 0.5) == y_te).mean()`. Early stopping: `xgb.XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=6, eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1, random_state=0).fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)`; `best_iteration + 1`.
    """,
    solution=[
        C("""
        table = pd.DataFrame(index=ks, columns=["train_ll", "test_ll", "test_acc"], dtype=float)
        for k in ks:
            p_tr = m.predict_proba(X_tr, iteration_range=(0, k))[:, 1]
            p_te = m.predict_proba(X_te, iteration_range=(0, k))[:, 1]
            table.loc[k] = [log_loss(y_tr, p_tr), log_loss(y_te, p_te), ((p_te > 0.5) == y_te).mean()]
        table.round(3)
        """),
        C("""
        k_min_ll = int(table["test_ll"].idxmin())
        min_ll, ll_400 = float(table.loc[k_min_ll, "test_ll"]), float(table.loc[400, "test_ll"])
        ll_increase = ll_400 / min_ll - 1
        acc_change = float(table["test_acc"].max() - table.loc[400, "test_acc"])
        print(f"test log-loss: minimum {min_ll:.3f} at {k_min_ll} rounds, at 400 rounds {ll_400:.3f} ({ll_increase:+.0%}) | accuracy: maximum change {acc_change:.3f}")
        """),
        C("""
        def share(k):
            p = m.predict_proba(X_te, iteration_range=(0, k))[:, 1]
            extreme = (p < 0.02) | (p > 0.98)
            return float(extreme.mean()), float((((p > 0.5) != y_te) & extreme).sum() / max(extreme.sum(), 1))


        extreme_share = {k: share(k)[0] for k in (25, 400)}
        extreme_err = share(400)[1]
        print(f"extreme probabilities (below 0.02 or above 0.98): {extreme_share} | at 400 rounds, among the extreme ones {extreme_err:.0%} get the class wrong")
        """),
        C("""
        m_es = xgb.XGBClassifier(n_estimators=400, learning_rate=0.3, max_depth=6, eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1, random_state=0)
        m_es.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        early_round = m_es.best_iteration + 1
        early_test_ll = log_loss(y_te, m_es.predict_proba(X_te)[:, 1])
        print(f"early stopping: {early_round} rounds | test log-loss {early_test_ll:.3f} (at 400 rounds: {ll_400:.3f})")
        """),
    ],
    learned="""
        - **The training log-loss goes to zero, the test one rises by 45%:** from 0.473 at 25 rounds to 0.683 at 400. The model is more and more sure of itself (at 400 rounds 52% of the test probabilities are below 0.02 or above 0.98, at 25 rounds 1.5%) and it is wrong with more and more confidence: among the extreme probabilities 12% are on the wrong side. Every wrong prediction with probability 0.99 costs the log-loss a great deal.
        - **The accuracy does not notice:** it stays at 0.79-0.80 all the way: the class is decided with the 0.5 threshold and changes only when a probability crosses 0.5, while the overfitting makes the probabilities more extreme without moving them to the other side. Anyone who looks only at `score()` may never notice the problem.
        - **It matters if you use the probabilities:** for an expected cost (M4 B2), a ranking of customers or a calibration, those extreme probabilities are wrong. That is why early stopping is done on the **log-loss** (or on a probabilistic metric) and not on the classification error.
        - **Early stopping fixes it:** 30 rounds of patience after the minimum stop the model at 15 rounds with a test log-loss of 0.466: like the absolute minimum of the table (0.473) and without looking at the test set.
    """,
    pitfall="""
        Evaluating a boosting model with `score()` (accuracy) after increasing the rounds. Accuracy is the last metric to notice the overfitting: look at the log-loss and the calibration.
    """,
    interview=(
        "Why is early stopping done on a probabilistic metric like the log-loss?",
        """
        Because the overfitting of boosting shows up first in the probabilities (which become extreme and wrong) than in the thresholded classification: the accuracy can stay almost constant while the validation log-loss is already rising a lot. A probabilistic metric picks up the deterioration right away; if the business metric is another
        (AUC, costs), you monitor it together.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="noisy_labels_boosting_chases_them", kind="C", level=3, minutes=55, module=MODULE,
    title="Wrong labels: boosting chases them, the forest does not",
    goal="""
        Corrupt 20% of the training labels and measure the accuracy on a **clean** test set as a function of the XGBoost rounds (learning rate 0.1, `max_depth=5`): it rises up to about 25 rounds and then drops by **about 6 points** while a random forest stays stable. Then show that regularization (`min_child_weight`,
        `reg_lambda`, `subsample`, shallow trees) limits the damage.
    """,
    dataset="""
        `make_classification(3000, 10 features, 5 informative, flip_y=0)` clean, 60/40 split (`X_tr, X_te, y_tr, y_te`). `y_noisy` = training labels with 20% swapped at random (`rng = np.random.default_rng(0)`). The test set stays clean.
    """,
    purpose="""
        Understand the difference in temperament between forest and boosting: the forest averages independent trees and the label errors cancel out; boosting **always corrects the errors of the previous ones**, so with enough rounds it tries to "explain" the wrong labels too.
    """,
    task=[
        "Train `XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric=\"logloss\", n_jobs=-1, random_state=0)` on `X_tr, y_noisy` and compute with `iteration_range=(0, k)` the accuracy on the clean test set for `k` in `[10, 25, 50, 100, 250, 500]`: `xgb_acc` (`pd.Series`).",
        "Repeat with `XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=3, min_child_weight=20, reg_lambda=10, subsample=0.7, ...)`: `reg_xgb_acc`. Also train `RandomForestClassifier(300, random_state=0, n_jobs=-1)` on `y_noisy` and save `rf_acc`, and the same accuracy with clean labels `clean_xgb_acc` (base XGB at 500 rounds on `y_tr`).",
        "Save `xgb_peak` (`xgb_acc.max()`), `peak_k`, `xgb_loss = xgb_peak − xgb_acc[500]` and `reg_loss` (same for `reg_xgb_acc`).",
        "Save `xgb_beaten_by_rf` (True if `rf_acc` is larger than `xgb_acc[500]`) and `rf_beats_xgb_at_peak` (True if `rf_acc` is larger than `xgb_peak`).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X, y = make_classification(3000, n_features=10, n_informative=5, class_sep=1.2, flip_y=0, random_state=2)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        rng = np.random.default_rng(0)
        y_noisy = np.where(rng.random(len(y_tr)) < 0.2, 1 - y_tr, y_tr)
        ks = [10, 25, 50, 100, 250, 500]
    """,
    starter=[
        C("""
        xgb_acc = pd.Series(dtype=float)
        reg_xgb_acc = pd.Series(dtype=float)
        rf_acc = None
        clean_xgb_acc = None
        """),
        C("""
        xgb_peak = None
        peak_k = None
        xgb_loss = None
        reg_loss = None
        xgb_beaten_by_rf = None
        rf_beats_xgb_at_peak = None
        """),
    ],
    checks="""
        check(list(xgb_acc.index) == ks and list(reg_xgb_acc.index) == ks, "curves over six values of k", "iteration_range=(0, k)")
        check(clean_xgb_acc > 0.93, "with clean labels XGBoost reaches over 0.93", "y_tr instead of y_noisy")
        check(peak_k in (25, 50, 100) and xgb_peak > 0.89 and xgb_loss > 0.04 and xgb_acc[500] < xgb_acc[peak_k] - 0.04, "with noisy labels XGBoost has a peak at 25-100 rounds and then loses over 4 points", "xgb_acc.max() and xgb_acc[500]")
        check(reg_loss < xgb_loss - 0.02 and reg_xgb_acc.max() > 0.9, "the regularized model loses less than the unregularized one (by over 2 points) and goes over 0.9 at the peak", "min_child_weight, reg_lambda, subsample, depth 3")
        check(rf_acc > 0.9 and xgb_beaten_by_rf is True, "the forest (over 0.9) beats the unregularized XGBoost at 500 rounds", "rf_acc > xgb_acc[500]")
        check(rf_beats_xgb_at_peak in (True, False), "boolean answer (the forest and the XGBoost peak are close)", "compare rf_acc and xgb_peak")
    """,
    hint="""
        `m = xgb.XGBClassifier(...).fit(X_tr, y_noisy)`; `xgb_acc[k] = (m.predict(X_te, iteration_range=(0, k)) == y_te).mean()`. Forest: `RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_noisy).score(X_te, y_te)`.
    """,
    solution=[
        C("""
        def curve(**kw):
            m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, eval_metric="logloss", n_jobs=-1, random_state=0, **kw).fit(X_tr, y_noisy)
            return pd.Series({k: float((m.predict(X_te, iteration_range=(0, k)) == y_te).mean()) for k in ks})


        xgb_acc = curve(max_depth=5)
        reg_xgb_acc = curve(max_depth=3, min_child_weight=20, reg_lambda=10, subsample=0.7)
        rf_acc = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_noisy).score(X_te, y_te)
        clean_xgb_acc = float((xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr).predict(X_te) == y_te).mean())
        print(pd.DataFrame({"xgb": xgb_acc, "regularized xgb": reg_xgb_acc}).round(3).T)
        print(f"random forest: {rf_acc:.3f} | XGBoost with clean labels at 500 rounds: {clean_xgb_acc:.3f}")
        """),
        C("""
        xgb_peak, peak_k = float(xgb_acc.max()), int(xgb_acc.idxmax())
        xgb_loss = xgb_peak - xgb_acc[500]
        reg_loss = float(reg_xgb_acc.max() - reg_xgb_acc[500])
        xgb_beaten_by_rf = bool(rf_acc > xgb_acc[500])
        rf_beats_xgb_at_peak = bool(rf_acc > xgb_peak)
        print(f"XGBoost: peak {xgb_peak:.3f} at {peak_k} rounds, then loses {xgb_loss:.3f} | regularized: loses {reg_loss:.3f} | the forest ({rf_acc:.3f}) beats XGBoost at 500 rounds: {xgb_beaten_by_rf}; at the peak: {rf_beats_xgb_at_peak}")
        """),
    ],
    learned="""
        - **Boosting corrects the errors of the previous ones, even those that are not errors:** with 20% of the labels swapped the XGBoost accuracy rises up to 0.90 (25 rounds) and then drops to 0.85 at 500: the later rounds learn the wrong labels. With clean labels the same model reaches 0.95.
        - **The forest is much more robust:** 0.915, above the boosting peak (0.904), **without** having to choose anything. Averaging independent trees is a good antidote to label noise; boosting needs a brake (early stopping, regularization).
        - **Regularization helps a lot:** `max_depth=3`, `min_child_weight=20`, `reg_lambda=10`, `subsample=0.7` bring the peak to 0.92 and the loss at 500 rounds from 5.7 to 3.7 points. A high `min_child_weight` prevents isolating a few noisy examples; `subsample` decorrelates.
        - **Warning sign in practice:** if your boosting model does **better with few rounds and shallow trees** than with many, suspect noise in the labels. A check: compare with a forest (M8): if the forest beats the "pushed" boosting, the signal is noisy.
    """,
    pitfall="""
        Believing that "boosting is always the best" and pushing rounds and depth until the training is fine. With noisy data boosting is the model most at risk of overfitting; a forest or a heavily regularized boosting are safer choices.
    """,
    interview=(
        "How does boosting behave with noisy labels?",
        """
        Worse than the forest, if not regularized: each round corrects the errors of the previous ones, so with enough rounds it tries to explain the wrong labels too and the error on clean data climbs back. It is mitigated with early stopping, shallow trees, high min_child_weight and reg_lambda, subsample and a low learning rate; and it is worth comparing it with a forest,
        which is naturally robust to label noise.
        """),
)

EXERCISES = [C1, C2]
