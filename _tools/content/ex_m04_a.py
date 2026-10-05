"""Module 4 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="logistic_with_gradient_descent", kind="A", level=2, minutes=50, module=MODULE,
    title="Logistic regression with gradient descent",
    goal="""
        Implement with numpy an L2-regularized logistic regression (loss function, gradient, gradient descent), check the gradient with finite
        differences and show that the coefficients match those of `LogisticRegression`. Then show **why** without regularization the method does not converge
        on nearly separable data.
    """,
    dataset="""
        `load_breast_cancer` (569 nodules, 30 measurements, target 1 = benign) split 70/30 and standardized: `Z_tr, y_tr, Z_te, y_te`.
    """,
    purpose="""
        Understand what `fit` does: an iterative solver that minimizes a convex log-loss. Above all: understand the **relationship between `C` and the penalty** (`C` multiplies the
        log-loss, not the penalty) and see perfect separation in action, which you will need in exercise C1.
    """,
    task=[
        "Write `sigmoid(z)`, `loss(w, b, Z, y, C)` and `gradient(w, b, Z, y, C)` (returns `(grad_w, grad_b)`). The scikit-learn objective "
        "$C\\sum_i \\ell_i + \\tfrac12\\lVert w\\rVert^2$ divided by $C n$ is the mean loss $\\frac1n\\sum_i \\ell_i + \\frac{1}{2Cn}\\lVert w\\rVert^2$ "
        "(the intercept is **not** penalized).",
        "Check the gradient with finite differences: save in `gradient_error` the maximum absolute gap between analytic and numeric gradient "
        "(step `1e-6`, central difference) over all the components of `w` and over `b`, computed at a random point `w0` (given).",
        "Write `train(Z, y, C, lr, n_iter)`: gradient descent from `w = 0, b = 0`; return `w, b, history` (the loss at every iteration).",
        "Train with `C=1, lr=0.5, n_iter=5000` and compare with `LogisticRegression(C=1, tol=1e-10, max_iter=100000)`: save `coef_gap` (maximum absolute gap on the coefficients).",
        "**Without regularization:** train with `C=1e6` for 500 and for 5000 iterations and save the Euclidean norms of the weights in `norms = (norm_500, norm_5000)`. "
        "The weights keep growing: why?",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer()
        X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, stratify=data.target, random_state=0)
        scaler = StandardScaler().fit(X_tr)
        Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
        w0 = np.random.default_rng(0).normal(0, 0.3, Z_tr.shape[1])         # point at which to check the gradient
    """,
    starter=[
        M("### 1. Loss and gradient"),
        C("""
        def sigmoid(z):
            ...


        def loss(w, b, Z, y, C):
            ...


        def gradient(w, b, Z, y, C):
            ...
        """),
        M("### 2. Check with finite differences"),
        C("""
        gradient_error = None
        """),
        M("### 3. Gradient descent"),
        C("""
        def train(Z, y, C, lr, n_iter):
            ...
        """),
        C("""
        w, b, history = train(Z_tr, y_tr, C=1, lr=0.5, n_iter=5000)
        sk = LogisticRegression(C=1, tol=1e-10, max_iter=100000).fit(Z_tr, y_tr)
        coef_gap = None
        """),
        M("### 4. Without regularization"),
        C("""
        norms = None
        """),
    ],
    checks="""
        check(close(sigmoid(np.array([-50.0, 0.0, 50.0])), [0.0, 0.5, 1.0], atol=1e-12), "sigmoid correct at the extremes (and without overflow)", "sigmoid(-50) = 0, sigmoid(0) = 0.5, sigmoid(50) = 1")
        check(close(loss(np.zeros(30), 0.0, Z_tr, y_tr, 1.0), np.log(2)), "with zero weights the loss is ln 2", "with w = 0 and b = 0 every example costs ln(2)")
        check(gradient_error is not None and gradient_error < 1e-6, "analytic gradient equal to the numeric one (gap < 1e-6)", "check the formula: Z.T @ (p - y) / n + w / (C n) for w, and the mean of (p - y) for b")
        check(len(history) == 5000 and all(np.diff(history) <= 1e-12), "the loss never goes up", "with a small enough step the loss must go down at every iteration")
        check(coef_gap is not None and coef_gap < 1e-3, "the coefficients match those of scikit-learn (gap < 1e-3)", "did you penalize the intercept? used 1/C instead of 1/(C n)?")
        check(norms is not None and norms[1] > 1.5 * norms[0] and norms[1] > 5, "without regularization the norm of the weights keeps growing", "norm of the weights after 5000 iterations against 500, with C = 1e6")
        check(np.all(((Z_te @ w + b) > 0).astype(int) == sk.predict(Z_te)), "the predictions on the test set match those of scikit-learn", "same rule: class 1 if z > 0")
    """,
    hint="""
        `p = sigmoid(Z @ w + b)`; `grad_w = Z.T @ (p - y) / n + w / (C * n)`; `grad_b = (p - y).mean()`. Stable sigmoid: `1 / (1 + np.exp(-np.clip(z, -500, 500)))`.
        Finite differences: for every component `(loss(w + h e_j) - loss(w - h e_j)) / (2h)`.
    """,
    solution=[
        C("""
        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


        def loss(w, b, Z, y, C):
            n = len(y)
            p = np.clip(sigmoid(Z @ w + b), 1e-15, 1 - 1e-15)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) + w @ w / (2 * C * n)


        def gradient(w, b, Z, y, C):
            n = len(y)
            p = sigmoid(Z @ w + b)
            return Z.T @ (p - y) / n + w / (C * n), np.mean(p - y)
        """),
        M("""
        **Where `1/(2Cn)` comes from.** scikit-learn minimizes $C\\sum_i \\ell_i + \\tfrac12\\lVert w\\rVert^2$. Dividing by $Cn$ gives the mean loss
        $\\frac1n\\sum_i\\ell_i + \\frac{1}{2Cn}\\lVert w\\rVert^2$, which has the same minimum and is more convenient for the gradient. Large `C` = a penalty that
        weighs little.
        """),
        C("""
        gw, gb = gradient(w0, 0.3, Z_tr, y_tr, 1.0)
        h = 1e-6
        num_w = np.array([(loss(w0 + h * e, 0.3, Z_tr, y_tr, 1.0) - loss(w0 - h * e, 0.3, Z_tr, y_tr, 1.0)) / (2 * h) for e in np.eye(30)])
        num_b = (loss(w0, 0.3 + h, Z_tr, y_tr, 1.0) - loss(w0, 0.3 - h, Z_tr, y_tr, 1.0)) / (2 * h)
        gradient_error = max(np.abs(gw - num_w).max(), abs(gb - num_b))
        print("maximum gap between analytic and numeric gradient:", gradient_error)
        """),
        C("""
        def train(Z, y, C, lr, n_iter):
            w, b = np.zeros(Z.shape[1]), 0.0
            history = []
            for _ in range(n_iter):
                gw, gb = gradient(w, b, Z, y, C)
                w, b = w - lr * gw, b - lr * gb
                history.append(loss(w, b, Z, y, C))
            return w, b, history
        """),
        C("""
        w, b, history = train(Z_tr, y_tr, C=1, lr=0.5, n_iter=5000)
        sk = LogisticRegression(C=1, tol=1e-10, max_iter=100000).fit(Z_tr, y_tr)
        coef_gap = np.abs(w - sk.coef_[0]).max()
        print(f"maximum gap on the coefficients: {coef_gap:.2e} | gap on the intercept: {abs(b - sk.intercept_[0]):.2e}")
        """),
        C("""
        norms = []
        for n_iter in (500, 5000):
            w_nr, _, _ = train(Z_tr, y_tr, C=1e6, lr=0.5, n_iter=n_iter)
            norms.append(np.linalg.norm(w_nr))
        norms = tuple(norms)
        sk_nr = LogisticRegression(C=1e6, tol=1e-10, max_iter=100000).fit(Z_tr, y_tr)
        print(f"norm of the weights without regularization: {norms[0]:.2f} after 500 iterations, {norms[1]:.2f} after 5000 | scikit-learn (which stops at the tolerance): {np.linalg.norm(sk_nr.coef_):.1f}")
        """),
        M("""
        **Why the norm grows.** On this data the two classes are nearly separable by a hyperplane: the larger the weights, the more the
        probabilities approach 0 and 1 on the well-classified examples and the more the log-loss drops. There is no finite minimum: the gradient is always nonzero and the weights
        keep growing. L2 regularization adds a term that grows with the norm and **creates** a minimum.
        """),
    ],
    learned="""
        - `fit` is a descent (with a better solver, `lbfgs`) on a convex loss: the mean log-loss plus $\\frac{1}{2Cn}\\lVert w\\rVert^2$. Writing the gradient by hand and
          checking it with finite differences is the standard test for any model you implement: if the gap is 1e-6 or less, the formula is right.
        - **`C` is the inverse of the regularization strength** and multiplies the log-loss: the penalty per example is $1/(Cn)$, so with larger datasets
          the same `C` regularizes less.
        - The intercept is not penalized: penalizing it would mean depending on the choice of the origin of the variable.
        - With separable (or nearly) classes and no penalty, the coefficients diverge: it is a **mathematical fact**, not a defect of the solver. `LogisticRegression` without
          regularization on separable data stops only because the iterations run out or the tolerance is reached. You come back to it with C1.
    """,
    pitfall="""
        Implementing the logistic gradient with the wrong sign or without dividing by `n`: the loss jumps or does not go down, and you can't tell whether
        it is the step's fault or the formula's. The check with finite differences should always be the **first** test.
    """,
    interview=(
        "Why the log-loss and not the squared error for logistic regression?",
        """
        Because with the sigmoid the squared error is not convex (several local minima) and has tiny gradients when the model is sure and wrong, so it
        learns slowly exactly where it matters. The log-loss is the negative log-likelihood of a Bernoulli model (maximum likelihood estimation):
        it is convex, its gradient is simply `(p - y) x` and it heavily penalizes whoever is sure and wrong.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="metrics_and_roc_curve_from_scratch", kind="A", level=2, minutes=50, module=MODULE,
    title="Confusion matrix, ROC curve and average precision from scratch",
    goal="""
        Implement with numpy the metrics of lesson 4.3: confusion matrix, precision/recall/F1, ROC curve and AUC (in two ways), average precision and the
        search for the threshold that maximizes F1, correctly handling **tied scores**. Compare everything with scikit-learn.
    """,
    dataset="""
        `scores` and `y_te`: the probabilities of a logistic on the test set of `ds.loans` (450 rows, about 19% positives), **rounded to 2 decimals** on purpose,
        so that many examples share the same score.
    """,
    purpose="""
        The metrics are not formulas to memorize: they are counts on a moving threshold. Implementing them shows you that the AUC is the probability that a positive
        beats a negative and, above all, that **ties** must be handled with care (it is the most common mistake when writing an ROC by hand).
    """,
    task=[
        "`confusion_matrix_(y, pred)` → `(tn, fp, fn, tp)` and `metrics(y, pred)` → dictionary with `accuracy`, `precision`, `recall`, `f1` (0 if the denominator is 0).",
        "`roc_curve_(y, s)` → `(fpr, tpr)`: sort the scores in decreasing order, accumulate true and false positives and **keep a single point for each distinct value** of the "
        "score (the point (0, 0) at the start). It must match `roc_curve(..., drop_intermediate=False)`.",
        "Compute the AUC in two ways: `auc_trapezoids(fpr, tpr)` (area under the curve) and `auc_pairs(y, s)` (fraction of positive-negative pairs ordered correctly, "
        "with ties counting 1/2). They must match each other and `roc_auc_score`.",
        "`average_precision(y, s)` = $\\sum_k (R_k - R_{k-1}) P_k$ over the distinct values of the scores only (from high threshold to low); like `average_precision_score`.",
        "`best_f1_threshold(y, s)` → `(threshold, f1)`: the threshold (among the distinct scores, with the rule `s >= threshold`) that maximizes F1.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.3, stratify=df["defaulted"], random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
        y_te = y_te.to_numpy()
        scores = np.round(model.predict_proba(X_te)[:, 1], 2)
        print(len(scores), "scores,", len(np.unique(scores)), "distinct values")
    """,
    starter=[
        C("""
        def confusion_matrix_(y, pred):
            ...


        def metrics(y, pred):
            ...
        """),
        C("""
        def roc_curve_(y, s):
            ...


        def auc_trapezoids(fpr, tpr):
            ...


        def auc_pairs(y, s):
            ...
        """),
        C("""
        def average_precision(y, s):
            ...


        def best_f1_threshold(y, s):
            ...
        """),
    ],
    checks="""
        pred = (scores >= 0.3).astype(int)
        check(confusion_matrix_(y_te, pred) == tuple(confusion_matrix(y_te, pred).ravel()), "correct confusion matrix", "tn, fp, fn, tp: mind the order")
        m = metrics(y_te, pred)
        check(close([m["precision"], m["recall"], m["f1"]], [precision_score(y_te, pred), recall_score(y_te, pred), f1_score(y_te, pred)]), "correct precision, recall and F1", "precision = TP/(TP+FP), recall = TP/(TP+FN), F1 = harmonic mean")
        check(m["accuracy"] == (pred == y_te).mean() and metrics(y_te, np.zeros_like(y_te))["precision"] == 0.0, "correct accuracy and no division by zero", "if you flag nothing the precision is 0 by convention")
        fpr, tpr = roc_curve_(y_te, scores)
        fpr_sk, tpr_sk, _ = roc_curve(y_te, scores, drop_intermediate=False)
        check(len(fpr) == len(np.unique(scores)) + 1, "one ROC point for every distinct value of the score (plus the origin)", "with ties you can't have a point for every example")
        check(close(fpr, fpr_sk) and close(tpr, tpr_sk), "the ROC curve matches scikit-learn's", "accumulate tp and fp only on the last indices of every group of equal scores")
        auc_sk = roc_auc_score(y_te, scores)
        check(close(auc_trapezoids(fpr, tpr), auc_sk) and close(auc_pairs(y_te, scores), auc_sk), "the AUC matches roc_auc_score in both ways", "pairs: (pos > neg) + 0.5 (pos == neg), divided by the number of pairs")
        check(close(average_precision(y_te, scores), average_precision_score(y_te, scores)), "correct average precision", "sum of (R_k - R_{k-1}) * P_k over the distinct values")
        s_best, f1_best = best_f1_threshold(y_te, scores)
        f1_brute_force = max(f1_score(y_te, (scores >= t).astype(int)) for t in np.unique(scores))
        check(close(f1_best, f1_brute_force) and close(f1_score(y_te, (scores >= s_best).astype(int)), f1_best), "the threshold found really maximizes F1", "try every distinct value of the score as a threshold")
    """,
    hint="""
        `order = np.argsort(-s, kind="stable")`; `tps = np.cumsum(y[order])`; `fps = np.cumsum(1 - y[order])`; the indices of the **last** elements of each group of
        equal scores: `np.r_[np.where(np.diff(s[order]) != 0)[0], len(s) - 1]`. Pairs: `diff = pos[:, None] - neg[None, :]`.
    """,
    solution=[
        C("""
        def confusion_matrix_(y, pred):
            y, pred = np.asarray(y), np.asarray(pred)
            tn = int(((y == 0) & (pred == 0)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
            fn = int(((y == 1) & (pred == 0)).sum()); tp = int(((y == 1) & (pred == 1)).sum())
            return tn, fp, fn, tp


        def metrics(y, pred):
            tn, fp, fn, tp = confusion_matrix_(y, pred)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            return {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall, "f1": f1}
        """),
        C("""
        def _cumulate(y, s):
            \"\"\"Cumulative true and false positives for decreasing threshold, one value per distinct score.\"\"\"
            y, s = np.asarray(y), np.asarray(s)
            order = np.argsort(-s, kind="stable")
            s_o, y_o = s[order], y[order]
            last = np.r_[np.where(np.diff(s_o) != 0)[0], len(s_o) - 1]        # one point for every group of ties
            return np.cumsum(y_o)[last], np.cumsum(1 - y_o)[last]


        def roc_curve_(y, s):
            tps, fps = _cumulate(y, s)
            return np.r_[0, fps / fps[-1]], np.r_[0, tps / tps[-1]]


        def auc_trapezoids(fpr, tpr):
            return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


        def auc_pairs(y, s):
            y, s = np.asarray(y), np.asarray(s)
            differences = s[y == 1][:, None] - s[y == 0][None, :]                  # all the (positive, negative) pairs
            return float(((differences > 0).sum() + 0.5 * (differences == 0).sum()) / differences.size)
        """),
        C("""
        def average_precision(y, s):
            tps, fps = _cumulate(y, s)
            precision, recall = tps / (tps + fps), tps / tps[-1]
            return float(np.sum(np.diff(np.r_[0, recall]) * precision))


        def best_f1_threshold(y, s):
            y, s = np.asarray(y), np.asarray(s)
            candidates = np.unique(s)
            f1 = [metrics(y, (s >= t).astype(int))["f1"] for t in candidates]
            k = int(np.argmax(f1))
            return float(candidates[k]), float(f1[k])
        """),
        C("""
        fpr, tpr = roc_curve_(y_te, scores)
        print("ROC points:", len(fpr), "| AUC (trapezoids):", round(auc_trapezoids(fpr, tpr), 4), "| AUC (pairs):", round(auc_pairs(y_te, scores), 4), "| scikit-learn:", round(roc_auc_score(y_te, scores), 4))
        print("average precision:", round(average_precision(y_te, scores), 4), "| best threshold for F1:", best_f1_threshold(y_te, scores))
        """),
    ],
    learned="""
        - The ROC curve is a list of cumulative counts: you sort the scores from the highest, and every extra example adds a true or a false positive. The **AUC is a probability**:
          pick a positive and a negative at random, how often does the positive have the higher score (ties count 1/2)? The two definitions (area and pairs) coincide.
        - **Ties** must be grouped: if you move from one threshold to the next one example at a time inside a group of equal scores, you draw a diagonal that corresponds to
          no real threshold and the AUC changes. It is a bug that gives no errors.
        - `average_precision` is the area of the PR curve **as a step sum** (not with trapezoids: interpolating linearly between PR points is optimistic).
        - The threshold that maximizes F1 on the test set is an optimistic estimate: here you search for it only as an exercise, the real threshold must be chosen on the training set (lesson 4.3).
    """,
    pitfall="""
        Computing the ROC by sorting the scores and taking a point for every **example** instead of every **distinct value**: with rounded or discrete scores (trees!)
        the curve comes out different from scikit-learn's and the AUC slightly wrong, with no error.
    """,
    interview=(
        "How do you interpret an AUC of 0.78?",
        """
        If I pick a defaulting customer and a non-defaulting one at random, in 78% of the cases the model gives the higher score to the defaulter. It says nothing about the
        calibration of the probabilities nor about which threshold to use: it measures only the ordering. That is why with very imbalanced classes I pair it with precision and recall
        at the chosen threshold or with the average precision.
        """),
)

EXERCISES = [A1, A2]
