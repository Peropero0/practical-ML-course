"""Module 4 lessons: logistic regression and classification metrics."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_04_logistic_regression"
TITLE = "Logistic regression and classification metrics"
LESSONS = [
    ("01_from_line_to_probability.ipynb", "4.1", "From the line to the probability",
     "Sigmoid, log-odds, log-loss and coefficients as odds ratios: what `LogisticRegression` really does."),
    ("02_regularization_scaling_multiclass.ipynb", "4.2", "Regularization, scaling and multiclass",
     "`C`, L1 and L2 penalties, why scaling changes the model and how to move to more classes."),
    ("03_metrics_and_threshold.ipynb", "4.3", "Metrics, threshold and imbalance",
     "Confusion matrix, precision and recall, ROC and PR, calibration and choice of the threshold."),
]
PATH = [
    ("Lesson 4.1", ["A1", "B1", "C1", "D1", "E2"]),
    ("Lesson 4.2", ["E1", "F1", "D3", "B3"]),
    ("Lesson 4.3", ["A2", "C2", "D2", "B2", "E3", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_4_1() -> List[Cell]:
    """Cells of lesson 4.1."""
    return [
        M("""
        # Lesson 4.1 · From the line to the probability

        In module 1 you predicted a number with a line. Now the target is a **class** (default yes/no) and you want a
        **probability**. Logistic regression is the linear model that answers this question: it is the baseline of almost every
        classification problem, and it is also the only model that will give you, besides the prediction, interpretable coefficients.

        **What you will learn**

        - Why linear regression is not suitable for a probability, and what the sigmoid adds.
        - To read the coefficients as **log-odds** and **odds ratios**.
        - What `fit` minimizes: the **log-loss**, and why the squared error is not used.
        """),
        _setup("""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]                          # how much the loan weighs on the income
        print(df[["ratio", "age", "defaulted"]].describe().round(2).loc[["mean", "std", "min", "max"]])
        print("default rate:", round(df["defaulted"].mean(), 3))
        """),
        M("""
        ## Why linear regression is not enough

        Let's try the method from module 1 on the 0/1 target using a single feature, the amount/income ratio.
        """),
        C("""
        x = df[["ratio"]].to_numpy()
        y = df["defaulted"].to_numpy()
        lin = LinearRegression().fit(x, y)
        log = LogisticRegression(C=1e6, max_iter=1000).fit(x, y)                 # huge C = almost no regularization

        grid = np.linspace(0, x.max(), 300).reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(7, 3.6))
        ax.scatter(x, y + np.random.default_rng(0).uniform(-0.03, 0.03, len(y)), s=6, alpha=0.3, color="gray")
        ax.plot(grid, lin.predict(grid), label="line (linear regression)")
        ax.plot(grid, log.predict_proba(grid)[:, 1], label="sigmoid (logistic regression)")
        ax.axhline(0, color="k", lw=0.5); ax.axhline(1, color="k", lw=0.5)
        ax.set_xlabel("amount / income"); ax.set_ylabel("P(default)"); ax.legend(); plt.show()
        print("the line predicts values outside [0, 1]:", (lin.predict(x) < 0).sum(), "rows below 0,", (lin.predict(x) > 1).sum(), "above 1")
        """),
        M("""
        The line produces negative "probabilities" and has no probabilistic interpretation. Logistic regression applies to the linear model a
        function that squashes every real number into (0, 1): the **sigmoid** $\\sigma(z) = 1/(1+e^{-z})$.

        $$P(y=1\\mid x) = \\sigma(w_0 + w_1 x_1 + \\dots + w_p x_p)$$

        **Intuition.** The model stays linear, but on a different scale: $z = w_0 + \\mathbf{w}^\\top \\mathbf{x}$ is the **log-odds** (the logarithm
        of $p/(1-p)$). Every extra unit of $x_j$ adds $w_j$ to the log-odds, that is **multiplies the odds by $e^{w_j}$**.
        """),
        M("""
        ## Coefficients as odds ratios

        Let's standardize the features (scaling has a precise reason, you will see it in lesson 4.2) and read the coefficients.
        """),
        C("""
        cols = ["ratio", "age", "job_tenure"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.25, stratify=df["defaulted"], random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

        w = model[-1].coef_[0]
        table = pd.DataFrame({"coef (log-odds per 1 std dev)": w, "odds ratio": np.exp(w)}, index=cols)
        print(table.round(3))
        print("intercept:", model[-1].intercept_.round(3), "-> baseline probability of an average person:", round(1 / (1 + np.exp(-model[-1].intercept_[0])), 3))
        print("test AUC:", round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))
        """),
        M("""
        An odds ratio of 3.3 for `ratio` reads: "all other features being equal, an increase of one standard deviation of the
        amount/income ratio multiplies the default **odds** by 3.3". Not the probability: the odds. With small probabilities
        the two are close, but the difference grows with p.

        > **Common mistake.** Reading a coefficient as the "pure" effect of the feature. The coefficient holds **all else being equal**: if
        > two features are correlated (here `age` and `job_tenure`), the effect is split between the two in an unstable way. You will see it in exercise D1.

        ## What `fit` minimizes: the log-loss

        For every example with predicted probability $\\hat p$ and true class $y$, the loss is $-[y\\log \\hat p + (1-y)\\log(1-\\hat p)]$: small if
        you assign a high probability to the right class, enormous if you are sure and wrong. The mean over the training set is the **log-loss** (or cross
        entropy). The squared error is not used because with the sigmoid it becomes nonconvex with vanishing gradients; the log-loss is
        convex: a single minimum, found by an iterative solver (`lbfgs` by default).
        """),
        C("""
        p = model.predict_proba(X_te)[:, 1]
        manual = -np.mean(y_te * np.log(p) + (1 - y_te) * np.log(1 - p))
        print("manual log-loss:", round(manual, 4), "| sklearn:", round(log_loss(y_te, p), 4))
        print("log-loss of someone who always predicts the mean rate:", round(log_loss(y_te, np.full(len(y_te), y_tr.mean())), 4))
        """),
        M("""
        The second number is the baseline: a model that uses no feature. The log-loss of the model must be **below** that figure.

        ## Check your understanding

        """ + details(
            "What is the difference between `predict` and `predict_proba`?",
            "`predict_proba` returns the probabilities (one column per class, in `classes_` order). `predict` applies the 0.5 threshold and "
            "returns the labels. If you need another threshold (lesson 4.3) you must start from `predict_proba`.") + "\n\n" + details(
            "If a coefficient is 0.7, by how much does the probability change with +1 in the feature?",
            "There is no single answer: the log-odds change by 0.7 (the odds are multiplied by e^0.7 ≈ 2.01), but the change in probability "
            "depends on the starting point. Near p = 0.5 the slope is maximal (about w/4); near 0 or 1 the sigmoid is almost flat.")),
    ]


def lesson_4_2() -> List[Cell]:
    """Cells of lesson 4.2."""
    return [
        M("""
        # Lesson 4.2 · Regularization, scaling and multiclass

        scikit-learn's `LogisticRegression` is **regularized by default**: what it hands over is not the minimum of the log-loss but the minimum of the
        log-loss plus a penalty on the size of the coefficients. This changes three things that the module 1 theory had already given you for
        linear regression and that carry over here.

        **What you will learn**

        - The role of `C` (careful: it is the **inverse** of the regularization) and the difference between L2 and L1.
        - Why scaling changes the model.
        - How to go from 2 to K classes (softmax, one-vs-rest) and which metrics to use.
        """),
        _setup("""
        from sklearn.datasets import load_breast_cancer, load_digits
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target                                   # 30 nodule measurements, y = 1 benign
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        print(X.shape, "| benign:", round(y.mean(), 3))
        """),
        M("""
        ## `C` and the penalty

        The minimized function is $C \\cdot \\text{log-loss} + \\text{penalty}(\\mathbf{w})$, with penalty $\\lVert\\mathbf{w}\\rVert_2^2/2$ (L2, default)
        or $\\lVert\\mathbf{w}\\rVert_1$ (L1). **Small `C` = strong regularization.** L2 shrinks the coefficients; L1 zeroes some of them
        (feature selection), but requires a solver that supports it (`liblinear` or `saga`).
        """),
        C("""
        rows = []
        for c in [0.001, 0.01, 0.1, 1, 10, 100]:
            for pen in ["l2", "l1"]:
                m = make_pipeline(StandardScaler(), LogisticRegression(C=c, penalty=pen, solver="liblinear", max_iter=2000)).fit(X_tr, y_tr)
                rows.append({"C": c, "penalty": pen, "nonzero coefs": int((m[-1].coef_ != 0).sum()),
                             "coef norm": np.abs(m[-1].coef_).sum().round(2), "test AUC": roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]).round(4)})
        pd.DataFrame(rows).pivot(index="C", columns="penalty")
        """),
        M("""
        Read the table: with large `C` the coefficients grow and the AUC is almost unchanged; with `C = 0.001` L1 zeroes **all** the coefficients (AUC 0.5, that is chance)
        and L2 shrinks them a lot but the model still works. With intermediate `C` L1 keeps only 7-13 features out of 30 (with `C = 0.1` and `C = 1`) with an AUC almost equal to that of L2 which uses them all; L2 never zeroes anything. `C` must be chosen in cross-validation, like any
        hyperparameter (`LogisticRegressionCV` does it for you).

        ## Why scaling changes the model

        The penalty sums the coefficients of **all** the features with the same weight. If a feature is measured in large units (meters instead of
        millimeters) its coefficient needs to be small to produce the same effect, and so it is penalized less: changing the unit of
        measure changes the solution. Without regularization scaling would not change the predictions, but it would make the solver converge more slowly.
        """),
        C("""
        X_mm = X_tr.copy(); X_mm["mean area"] = X_mm["mean area"] / 1000            # same information, another unit (area / 1000)
        X_mm_te = X_te.copy(); X_mm_te["mean area"] = X_mm_te["mean area"] / 1000

        for name, (a, b) in {"original unit": (X_tr, X_te), "area / 1000": (X_mm, X_mm_te)}.items():
            without = LogisticRegression(C=0.1, max_iter=5000).fit(a, y_tr)
            with_scaling = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(a, y_tr)
            print(f"{name:18s} | without scaling: AUC {roc_auc_score(y_te, without.predict_proba(b)[:, 1]):.4f} | with scaling: AUC {roc_auc_score(y_te, with_scaling.predict_proba(b)[:, 1]):.4f}")
        """),
        M("""
        With scaling the two cases coincide to the fourth decimal (after standardization the features are the same, whatever the unit of measure). Without scaling the result changes with the units: here only a little, because the dataset is easy, but the penalty hits features with a small scale harder.
        The rule of thumb: **always standardize the features before a regularized logistic** (inside the `Pipeline`, never before the split).

        ## More than two classes

        With K classes the logistic becomes **multinomial** (softmax): a vector of coefficients for every class and probabilities that sum to 1
        (`multi_class="multinomial"`, the default with `lbfgs`). Alternatively `OneVsRestClassifier` trains K independent binary models.
        The coefficient is therefore a `(K, p)` matrix.
        """),
        C("""
        digits = load_digits()
        Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(digits.data, digits.target, test_size=0.3, stratify=digits.target, random_state=0)
        multi = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(Xd_tr, yd_tr)
        print("shape of coef_:", multi[-1].coef_.shape, "| accuracy:", round(multi.score(Xd_te, yd_te), 4))
        print("the probabilities of a digit sum to", multi.predict_proba(Xd_te[:1]).sum().round(6))

        from sklearn.metrics import ConfusionMatrixDisplay
        ConfusionMatrixDisplay.from_estimator(multi, Xd_te, yd_te, cmap="Blues", colorbar=False); plt.show()
        """),
        M("""
        With more classes the confusion matrix is 10×10: the errors concentrate in a few pairs (digits that look alike). To summarize in
        a single number you need averages: **macro** (mean of the per-class metrics, every class weighs the same), **micro** (global counts,
        it coincides with the accuracy) or **weighted**. You choose by looking at what you care about: the rare classes (macro) or the total (micro).

        > **Common mistake.** Comparing metrics of models with the L1 and L2 penalty without redoing the choice of `C`: the best `C` for one penalty is not the best for the other.

        ## Check your understanding

        """ + details(
            "Why is `C=1e10` a way to get the logistic \"without regularization\"?",
            "Because `C` multiplies the log-loss: with a huge `C` the penalty weighs almost nothing. It is the portable way across scikit-learn versions "
            "(`penalty=None` or `\"none\"` change name and are removed depending on the version).") + "\n\n" + details(
            "Do I also need to scale the one-hot dummies before a regularized logistic?",
            "It is a choice. Scaling them brings them to the same scale as the numerics (fair penalty), but makes them less interpretable. Leaving them at 0/1 is common "
            "and works well; the important thing is to scale the numerics with very different scales.")),
    ]


def lesson_4_3() -> List[Cell]:
    """Cells of lesson 4.3."""
    return [
        M("""
        # Lesson 4.3 · Metrics, threshold and imbalance

        A classification model returns scores: turning them into decisions requires a **threshold**, and evaluating them requires a **metric**
        suited to the cost of the errors. This lesson is the one you need to avoid being fooled by an accuracy of 95%.

        **What you will learn**

        - Confusion matrix, precision, recall and F1: what each one measures.
        - ROC and precision–recall curves, and why the ROC AUC is not enough when the positive class is rare.
        - How to choose the threshold (and why it is not 0.5) and what `class_weight` is for.
        """),
        _setup("""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve, precision_score,
                                     recall_score, roc_auc_score, roc_curve)
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.3, stratify=df["defaulted"], random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]
        print("defaulters in the test set:", int(y_te.sum()), "out of", len(y_te), f"({y_te.mean():.1%})")
        """),
        M("""
        ## The confusion matrix and the derived metrics

        With threshold 0.5 every example falls into one of four cells: true positives (TP), false positives (FP), false negatives (FN), true negatives (TN).

        | metric | formula | answers |
        |---|---|---|
        | accuracy | (TP+TN) / all | how many did I get right? |
        | **precision** | TP / (TP+FP) | among those I flagged, how many are really positive? |
        | **recall** (sensitivity) | TP / (TP+FN) | among the true positives, how many did I find? |
        | F1 | harmonic mean of precision and recall | a compromise in a single number |
        """),
        C("""
        pred = (scores >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"accuracy {(tp + tn) / len(y_te):.3f} | precision {precision_score(y_te, pred):.3f} | recall {recall_score(y_te, pred):.3f} | F1 {f1_score(y_te, pred):.3f}")
        print(f"a model that always says 'no default' has accuracy {1 - y_te.mean():.3f}")
        """),
        M("""
        The model is only slightly less wrong than the trivial one, and finds a small fraction of the defaulters: with imbalanced classes **accuracy is
        almost useless**. With threshold 0.5 the model is cautious: it declares default only whoever has a default probability above 50%, but the base rate is about 20%.

        ## The threshold is a business choice

        Lowering the threshold finds more defaulters (recall goes up) but also flags more innocents (precision goes down). There is no right threshold: there is
        the one that minimizes the cost, which must be chosen with the real costs (an unpaid loan costs much more than a customer wrongly turned down).
        """),
        C("""
        rows = []
        for s in [0.1, 0.2, 0.3, 0.4, 0.5]:
            p = (scores >= s).astype(int)
            rows.append({"threshold": s, "flagged": int(p.sum()), "precision": precision_score(y_te, p), "recall": recall_score(y_te, p), "F1": f1_score(y_te, p)})
        pd.DataFrame(rows).round(3).set_index("threshold")
        """),
        M("""
        ## ROC and precision–recall: all the thresholds together

        The **ROC curve** shows recall (true positive rate) against false positive rate as the threshold varies; its area (AUC) is the probability that
        a random positive gets a higher score than a random negative (0.5 = chance, 1 = perfect). The **precision–recall curve** shows how
        the precision falls when you try to increase the recall: with rare classes it is more informative, because the ROC uses the false positive rate
        (divided by the many negatives) and stays optimistic.
        """),
        C("""
        fpr, tpr, _ = roc_curve(y_te, scores)
        prec, rec, _ = precision_recall_curve(y_te, scores)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        ax[0].plot(fpr, tpr); ax[0].plot([0, 1], [0, 1], "k:"); ax[0].set(title=f"ROC (AUC = {roc_auc_score(y_te, scores):.3f})", xlabel="false positive rate", ylabel="recall")
        ax[1].plot(rec, prec); ax[1].axhline(y_te.mean(), color="k", ls=":"); ax[1].set(title=f"PR (AP = {average_precision_score(y_te, scores):.3f})", xlabel="recall", ylabel="precision")
        plt.tight_layout(); plt.show()
        """),
        M("""
        The dotted line in the PR plot is the precision of chance (the prevalence, about 0.2): it is the baseline to beat. In ROC the baseline is the diagonal.

        ## `class_weight` and calibration

        `class_weight="balanced"` weighs the errors on the rare class more: it moves the model toward recall, it is almost equivalent to lowering the threshold.
        **But the output probabilities are no longer calibrated**: "0.6" no longer means "6 out of 10 are positive". If you need true probabilities (to
        compute an expected cost, for example) keep the model without weights and move the threshold, or recalibrate with `CalibratedClassifierCV`.
        """),
        C("""
        from sklearn.calibration import calibration_curve
        balanced = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_tr, y_tr)
        for name, m in {"no weights": model, "balanced": balanced}.items():
            p = m.predict_proba(X_te)[:, 1]
            print(f"{name:10s} AUC {roc_auc_score(y_te, p):.3f} | recall at threshold 0.5: {recall_score(y_te, p >= 0.5):.3f} | mean predicted probability {p.mean():.3f} (true rate {y_te.mean():.3f})")
        """),
        M("""
        The AUC is practically identical (the ordering of the cases is the same), but with `balanced` the mean predicted probability is much higher than the real rate: the model is deliberately skewed.

        > **Common mistake.** Optimizing the threshold on the test set. The threshold is a hyperparameter: it must be chosen with cross-validation on the training set (`cross_val_predict` with `method="predict_proba"`) and then applied to the test set once.

        ## Check your understanding

        """ + details(
            "I have 1 positive in 1000: which metric do I report?",
            "Precision and recall at the chosen threshold, and the average precision (area under the PR curve). Accuracy is useless (99.9% by always saying no) and "
            "the ROC AUC can look high even when the precision is very low, because the false positives are divided by an enormous number of negatives.") + "\n\n" + details(
            "Why does the AUC (almost) not change with `class_weight='balanced'` but the recall does?",
            "The AUC depends only on the ordering of the scores; the weights change the scale of the scores (the probabilities shift upward) more than their order. "
            "The recall at threshold 0.5 depends instead on how many scores exceed 0.5, and that changes a lot.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_from_line_to_probability.ipynb", lesson_4_1),
                    ("02_regularization_scaling_multiclass.ipynb", lesson_4_2),
                    ("03_metrics_and_threshold.ipynb", lesson_4_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
