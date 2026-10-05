"""Module 4 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="perfect_separation_and_extreme_probabilities", kind="C", level=2, minutes=40, module=MODULE,
    title="Perfect separation: when the logistic becomes very sure and wrong",
    goal="""
        Build a case in which the classes are **perfectly separable on the training set** (few examples, many features), observe that without regularization the weights explode and the probabilities
        become extreme, and show that the **AUC does not notice** while the log-loss does. Then find `C` with cross-validation.
    """,
    dataset="""
        `generate()` (given): 60 training examples and 5000 test ones with 60 features of which only 5 informative (`make_classification`, with 5% of the labels swapped).
        With more features than examples there is always a hyperplane that separates the training set, even if the features say almost nothing.
    """,
    purpose="""
        See live three things that often happen in real data: (1) with `p > n` (or small classes) the training set is separable and the unregularized logistic **has no
        minimum**; (2) its "confidence" is an artifact: predictions at 99% and wrong; (3) the ranking metrics (AUC) are blind to all this.
    """,
    task=[
        "For `C` in `Cs = np.logspace(-4, 6, 11)` train `StandardScaler` + `LogisticRegression(C=C, max_iter=20000)` on the training set and build `table` (index `C`, columns "
        "`train_acc`, `l1_norm` (sum of the absolute values of the coefficients), `train_logloss`, `test_logloss`, `test_auc`, `confident_share` (fraction of test predictions with "
        "`p < 0.01` or `p > 0.99`), `confident_errors` (of these, how many are **wrong**, using 0.5 as threshold).",
        "Save `best_test_C` (the `C` with the minimum test log-loss: the absolute best, which you usually don't know).",
        "Choose `C` **without** looking at the test set: `GridSearchCV` with `scoring=\"neg_log_loss\"` and `StratifiedKFold(5, shuffle=True, random_state=0)` on the training set, on the same pipeline and on the `Cs`. "
        "Save `cv_C` (float) and check that the corresponding test log-loss is much better than the one without regularization.",
        "Save `auc_range = table['test_auc'].max() - table['test_auc'].min()` and `logloss_range` (the same for the test log-loss).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def generate():
            X, y = make_classification(n_samples=5060, n_features=60, n_informative=5, n_redundant=0, class_sep=1.0, flip_y=0.05, random_state=1)
            return X[:60], y[:60], X[60:], y[60:]


        X_tr, y_tr, X_te, y_te = generate()
        Cs = np.logspace(-4, 6, 11)
    """,
    starter=[
        C("""
        rows = []
        for c in Cs:
            ...
        table = pd.DataFrame(rows, index=Cs)
        table.round(3)
        """),
        C("""
        best_test_C = None
        cv_C = None
        auc_range = None
        logloss_range = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["train_acc", "l1_norm", "train_logloss", "test_logloss", "test_auc", "confident_share", "confident_errors"], "table with the seven columns", "check names and order")
        check(table.loc[Cs[-1], "train_acc"] == 1.0 and table.loc[Cs[0], "train_acc"] < 0.7, "without regularization the training set is perfectly separated", "train_acc with C = 1e6 must be 1")
        check(table["l1_norm"].is_monotonic_increasing, "the norm of the weights always grows with C", "more C, less penalty, more weights")
        check(table.loc[Cs[-1], "train_logloss"] < 1e-3 and table.loc[Cs[-1], "test_logloss"] > 5, "training: log-loss almost 0; test: over 5", "train_logloss and test_logloss at the row C = 1e6")
        check(table.loc[Cs[-1], "confident_errors"] > 1000, "over 1000 examples out of 5000 are wrong with probability below 1% or above 99%", "count (p < 0.01 or p > 0.99) and (predicted class != y)")
        check(best_test_C == table["test_logloss"].idxmin() and best_test_C <= 0.1, "the best C on the test set is small (strong regularization)", "idxmin of the test log-loss")
        check(isinstance(cv_C, float) and cv_C <= 0.1 and table.loc[cv_C, "test_logloss"] < 0.75, "the C chosen in cross-validation clearly beats the unpenalized logistic on the test set", "GridSearchCV with scoring='neg_log_loss'")
        check(auc_range < 0.06 and logloss_range > 6, "the test AUC varies little (less than 0.06), the log-loss by more than 6", "compare the two ranges")
    """,
    hint="""
        For every `c`: `m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=20000)).fit(X_tr, y_tr)`; `p = np.clip(m.predict_proba(X_te)[:, 1], 1e-15, 1 - 1e-15)`;
        `confident = (p < 0.01) | (p > 0.99)`; `errors = (p > 0.5) != y_te`. `cv_C = float(GridSearchCV(m, {"logisticregression__C": Cs}, ...).fit(X_tr, y_tr).best_params_["logisticregression__C"])`.
    """,
    solution=[
        C("""
        rows = []
        for c in Cs:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=20000)).fit(X_tr, y_tr)
            p_te = np.clip(m.predict_proba(X_te)[:, 1], 1e-15, 1 - 1e-15)
            confident = (p_te < 0.01) | (p_te > 0.99)
            wrong = (p_te > 0.5) != y_te
            rows.append({"train_acc": m.score(X_tr, y_tr), "l1_norm": np.abs(m[-1].coef_).sum(),
                         "train_logloss": log_loss(y_tr, m.predict_proba(X_tr)[:, 1], labels=[0, 1]),
                         "test_logloss": log_loss(y_te, p_te), "test_auc": roc_auc_score(y_te, p_te),
                         "confident_share": confident.mean(), "confident_errors": int((confident & wrong).sum())})
        table = pd.DataFrame(rows, index=Cs)
        table.round(3)
        """),
        C("""
        best_test_C = float(table["test_logloss"].idxmin())
        search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=20000)), {"logisticregression__C": Cs},
                              scoring="neg_log_loss", cv=StratifiedKFold(5, shuffle=True, random_state=0)).fit(X_tr, y_tr)
        cv_C = float(search.best_params_["logisticregression__C"])
        auc_range = table["test_auc"].max() - table["test_auc"].min()
        logloss_range = table["test_logloss"].max() - table["test_logloss"].min()
        print(f"best C on the test set: {best_test_C} | C chosen in cross-validation: {cv_C} (test log-loss {table.loc[cv_C, 'test_logloss']:.3f} against {table.loc[Cs[-1], 'test_logloss']:.3f} without penalty)")
        print(f"range of the test AUC: {auc_range:.3f} | range of the test log-loss: {logloss_range:.2f}")
        """),
    ],
    learned="""
        - With more features than examples the training set is **always** separable: `train_acc = 1` is not a success, it is a property of the geometry. The training log-loss goes to zero and the
          weights grow without limit: the logistic without penalty has no solution. That is why scikit-learn's `LogisticRegression` is regularized by default.
        - The "confident" predictions are an artifact: with `C = 1e6` over 1700 test examples out of 5000 are wrong with a probability below 1% or above 99%. **The test log-loss
          goes from 0.67 (the best model) to 7**: worse than a model that always says 50% (which is 0.693).
        - **The AUC varies only from 0.62 to 0.66.** The ordering stays poorly informative in any case, and does not notice that the probabilities have become absurd. If the model feeds decisions
          with costs (exercise B2) or another estimate, look at the log-loss and the calibration too.
        - `C` is chosen with cross-validation on a **probabilistic** metric (`neg_log_loss`): with the AUC as criterion the choice would have been almost indifferent and you would not have seen the
          problem. With 60 examples the choice is noisy: the value found is reasonable, not "the right one".
    """,
    pitfall="""
        Raising `C` "to make the model converge better" or removing the regularization to "have pure coefficients". With small data or separable classes it produces enormous, meaningless
        coefficients, and the solver stops because of the iteration limit, not because it found the minimum.
    """,
    interview=(
        "What happens to an unregularized logistic regression if the classes are linearly separable?",
        """
        The likelihood has no finite maximum: as the norm of the weights increases the loss keeps decreasing toward zero, so the coefficients diverge and the solver stops only because of
        tolerance or the maximum number of iterations. The probabilities become 0 or 1, the standard errors of the coefficients are nonsensical. It is solved with a penalty (L2, or L1 if I also want to
        select) or with a Bayesian regularization, and the strength is chosen in cross-validation.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="prevalence_and_precision", kind="C", level=3, minutes=50, module=MODULE,
    title="The same model at different prevalences: the AUC stays, the precision collapses",
    goal="""
        Measure what happens to the metrics of a **fixed** classifier when the fraction of positives in the test data drops from 50% to 0.5%. Derive the formula of the
        precision as a function of recall, false positive rate and prevalence, and correct the probabilities for the change of prevalence.
    """,
    dataset="""
        `generate_test(prevalence)` (given) draws from a large test pool a sample with the desired prevalence (20,000 negatives and a proportional number of positives). The model `m`,
        a logistic, is trained **once** on 3000 positives and 3000 negatives (prevalence 50%).
    """,
    purpose="""
        Understand why a model that "works" in the lab (balanced data) can be unusable in production (rare event): AUC, recall and FPR are properties of the model, the
        precision is **not**. And understand how a probability is corrected when the prevalence changes.
    """,
    task=[
        "For `prevalences = [0.5, 0.2, 0.05, 0.01, 0.005]` build `table` (index: prevalence) with: `auc`, `avg_precision`, `accuracy` (threshold 0.5), `baseline` (accuracy of someone who always says negative), "
        "`precision`, `recall`, `mean_p` (mean of the predicted probabilities), `logloss`.",
        "From the false positive rate and the recall measured at prevalence 0.5 compute `theoretical_precision[π] = R·π / (R·π + FPR·(1−π))` for every prevalence (`pd.Series`, same index) "
        "and check that it matches the measured precision.",
        "**Prior correction.** The probabilities of the model assume prevalence 0.5. For a prevalence `π` the odds get multiplied by $\\frac{\\pi/(1-\\pi)}{0.5/0.5}$. "
        "Add to `table` the columns `corrected_mean_p` and `corrected_logloss` computed with the corrected probabilities.",
        "After the correction threshold 0.5 means something else entirely: add `corrected_recall` (recall with threshold 0.5 on the corrected probabilities) and answer with `correction_cost = True` if at prevalence 0.005 the corrected recall is below 0.1.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, average_precision_score, log_loss, precision_score, recall_score, roc_auc_score

        X, y = make_classification(n_samples=80000, n_features=8, n_informative=5, n_redundant=0, class_sep=0.8, flip_y=0.02, random_state=0)
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        m = LogisticRegression(max_iter=1000).fit(X[np.r_[pos[:3000], neg[:3000]]], y[np.r_[pos[:3000], neg[:3000]]])
        pos_te, neg_te = pos[3000:], neg[3000:23000]                    # 20,000 test negatives


        def generate_test(prevalence, seed=0):
            n_pos = int(round(prevalence / (1 - prevalence) * len(neg_te)))
            idx = np.r_[neg_te, np.random.default_rng(seed).choice(pos_te, n_pos, replace=False)]
            return X[idx], y[idx]


        prevalences = [0.5, 0.2, 0.05, 0.01, 0.005]
    """,
    starter=[
        C("""
        rows = {}
        for pi in prevalences:
            X_te, y_te = generate_test(pi)
            p = m.predict_proba(X_te)[:, 1]
            rows[pi] = ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        theoretical_precision = None
        """),
        C("""
        correction_cost = None
        """),
    ],
    checks="""
        check(list(table.columns[:8]) == ["auc", "avg_precision", "accuracy", "baseline", "precision", "recall", "mean_p", "logloss"], "first eight columns correct", "check names and order")
        check(table["auc"].max() - table["auc"].min() < 0.04 and table["recall"].max() - table["recall"].min() < 0.1, "AUC and recall are (almost) constant as the prevalence varies (they vary only by sampling noise)", "they are properties of the model")
        check(table.loc[0.01, "precision"] < 0.1 and table.loc[0.005, "precision"] < 0.05, "with prevalence 1% the precision is below 10%, with 0.5% below 5%", "precision at threshold 0.5")
        check((table.loc[[0.05, 0.01, 0.005], "accuracy"] < table.loc[[0.05, 0.01, 0.005], "baseline"]).all(), "below 5% prevalence the model is less accurate than someone who always says 'negative'", "accuracy against baseline")
        check(theoretical_precision is not None and close(theoretical_precision.reindex(table.index).to_numpy(), table["precision"].to_numpy(), atol=0.03), "the precision formula matches the measured one (gap < 0.03)", "R*pi / (R*pi + FPR*(1-pi)) with R and FPR measured at prevalence 0.5")
        check("corrected_mean_p" in table and close(table["corrected_mean_p"].to_numpy(), np.array(table.index), rtol=0.25), "after the correction the mean predicted probability is close to the prevalence (within 25%)", "odds * k / (1 + odds * k), with k = (pi/(1-pi))")
        check((table["corrected_logloss"].iloc[1:] < table["logloss"].iloc[1:]).all() and table.loc[0.005, "corrected_logloss"] < 0.1, "the log-loss improves at every prevalence below 50% (at 0.5% below 0.1)", "compare logloss and corrected_logloss")
        check(correction_cost is True and table.loc[0.005, "corrected_recall"] < 0.1 and table.loc[0.005, "corrected_recall"] < table.loc[0.005, "recall"], "threshold 0.5 on the corrected probabilities finds almost no positive: the threshold must be revised", "corrected_recall at prevalence 0.005")
    """,
    hint="""
        Correction: `odds = p / (1 - p); k = (pi / (1 - pi)) / 1.0; p_c = odds * k / (1 + odds * k)`. FPR at prevalence 0.5: `((p >= 0.5) & (y == 0)).sum() / (y == 0).sum()`. Precision:
        `R * pi / (R * pi + FPR * (1 - pi))`.
    """,
    solution=[
        C("""
        rows = {}
        for pi in prevalences:
            X_te, y_te = generate_test(pi)
            p = m.predict_proba(X_te)[:, 1]
            k = (pi / (1 - pi)) / (0.5 / 0.5)                                       # ratio between the odds of the new prevalence and those of training
            p_c = (p / (1 - p) * k) / (1 + p / (1 - p) * k)
            rows[pi] = {"auc": roc_auc_score(y_te, p), "avg_precision": average_precision_score(y_te, p),
                        "accuracy": accuracy_score(y_te, p >= 0.5), "baseline": 1 - y_te.mean(),
                        "precision": precision_score(y_te, p >= 0.5), "recall": recall_score(y_te, p >= 0.5),
                        "mean_p": p.mean(), "logloss": log_loss(y_te, p),
                        "corrected_mean_p": p_c.mean(), "corrected_logloss": log_loss(y_te, p_c), "corrected_recall": recall_score(y_te, p_c >= 0.5)}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        X_b, y_b = generate_test(0.5)
        p_b = m.predict_proba(X_b)[:, 1]
        recall_b = ((p_b >= 0.5) & (y_b == 1)).sum() / (y_b == 1).sum()
        fpr_b = ((p_b >= 0.5) & (y_b == 0)).sum() / (y_b == 0).sum()
        theoretical_precision = pd.Series({pi: recall_b * pi / (recall_b * pi + fpr_b * (1 - pi)) for pi in prevalences})
        print(pd.DataFrame({"theoretical": theoretical_precision, "measured": table["precision"]}).round(3).T.to_string())
        """),
        C("""
        correction_cost = bool(table.loc[0.005, "corrected_recall"] < 0.1)
        print("recall at prevalence 0.005 with threshold 0.5:", table.loc[0.005, "recall"].round(3), "-> after the prior correction:", table.loc[0.005, "corrected_recall"].round(3))
        """),
    ],
    learned="""
        - **AUC and recall do not change** (they are conditional on the true class: they use only positives or only negatives), and the same goes for the false positive rate. The **precision changes
          enormously**: from 0.86 at 50% to 0.06 at 1%, because $P = \\frac{R\\pi}{R\\pi + \\mathrm{FPR}(1-\\pi)}$ and with small $\\pi$ the second term (false positives over a huge mass of negatives) dominates.
          It is the reason why with rare events you look at precision, average precision and the number of alarms per day, not the AUC.
        - Below 5% prevalence the model is less "accurate" than someone who always says "no": accuracy is useless and must be replaced with metrics on the rare class.
        - A model trained with balanced data (for convenience or with `class_weight`) gives probabilities that are **too high** in production: the predicted mean stays about 0.23, the true frequency is 0.005. The
          prior correction brings the probabilities to the right values (log-loss from 0.34 to 0.02) but **requires a new threshold**: with 0.5 on the corrected probabilities the recall is almost zero. Correcting is not
          an end in itself: you correct so that you can then choose the threshold with the real costs (B2).
        - The sampling with which you build the test set is part of the evaluation model: a balanced test set "for convenience" describes a world that does not exist.
    """,
    pitfall="""
        Evaluating a rare-event model (fraud, failures, diseases) on a balanced or rebalanced test set and reporting precision and accuracy as if they held in production. AUC and recall are
        preserved, precision and accuracy are not: they must be computed on the **real prevalence**.
    """,
    interview=(
        "Your fraud model has AUC 0.92 but the antifraud team says 95% of the alarms are false. Is the model wrong?",
        """
        Probably not: with a prevalence of 0.5% even an excellent classifier has low precision, because the precision depends on the prevalence (P = R·π / (R·π + FPR·(1−π))).
        I check the precision on the real stream, compare it with the formula, and set the threshold based on how many alarms the team can work and the cost of an undetected fraud. The AUC says
        the ordering is good; the precision says how much it costs to act on the top of the list.
        """),
)

EXERCISES = [C1, C2]
