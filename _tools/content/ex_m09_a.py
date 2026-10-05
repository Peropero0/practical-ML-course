"""Module 9 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="regression_gradient_boosting_from_scratch", kind="A", level=2, minutes=50, module=MODULE,
    title="Gradient boosting for regression from scratch: trees on the residuals",
    goal="""
        Implement gradient boosting with the squared loss using only `DecisionTreeRegressor` as a building block: constant initial prediction, small trees on the residuals, update with shrinkage. Check that it matches `GradientBoostingRegressor` in the first rounds, measure the error curve as a function of the
        rounds and find the round at which the test error starts to get worse.
    """,
    dataset="""
        `generate()` (given): 600 points with 3 uniform features in [−3, 3] and `y = sin(x0) + 0.09 x1² + noise(0.3)`, 60/40 split: `X_tr, y_tr, X_te, y_te`.
    """,
    purpose="""
        Understand that boosting is a four-line loop: residuals → small tree → update with step `eta`. See the two central parameters (`eta` and the number of rounds) in action and the difference from the forest: here the test error **rises** if you go on too long.
    """,
    task=[
        "`fit_gb(X, y, n_rounds, eta, depth)`: `f0 = y.mean()`; for each round fit `DecisionTreeRegressor(max_depth=depth, criterion=\"squared_error\", random_state=0)` to the residuals `y − F` and update `F += eta * tree.predict(X)`. Return `(f0, trees)`.",
        "`predict_gb(model, X, eta)` and `error_curve(model, X, y, eta)`: the MSE after each round (list of `n_rounds` values; the partial predictions are built by adding one tree at a time).",
        "Train `model = fit_gb(X_tr, y_tr, 300, 0.3, 2)` and save `train_curve` and `test_curve` (MSE per round), `best_round` (1-based index of the minimum of `test_curve`) and `best_mse`.",
        "Compare with `GradientBoostingRegressor(n_estimators=..., learning_rate=0.1, max_depth=3, criterion=\"squared_error\", random_state=0)`: save `diff_5_rounds` (maximum absolute difference between the test predictions after **5 rounds** with `eta=0.1, depth=3`) and `diff_100_rounds` (after 100 rounds: small differences accumulate because of the tie-breaking and the 32-bit precision of scikit-learn).",
    ],
    setup="""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor


        def generate():
            rng = np.random.default_rng(0)
            X = rng.uniform(-3, 3, (600, 3))
            y = np.sin(X[:, 0]) + 0.09 * X[:, 1] ** 2 + rng.normal(0, 0.3, 600)
            return train_test_split(X, y, test_size=0.4, random_state=0)


        X_tr, X_te, y_tr, y_te = generate()
    """,
    starter=[
        C("""
        def fit_gb(X, y, n_rounds, eta, depth):
            ...


        def predict_gb(model, X, eta):
            ...


        def error_curve(model, X, y, eta):
            ...
        """),
        C("""
        model = fit_gb(X_tr, y_tr, 300, 0.3, 2)
        train_curve = None
        test_curve = None
        best_round = None
        best_mse = None
        """),
        C("""
        diff_5_rounds = None
        diff_100_rounds = None
        """),
    ],
    checks="""
        check(len(model[1]) == 300 and close(model[0], y_tr.mean()), "300 trees and initial prediction equal to the mean", "f0 = y.mean()")
        check(len(train_curve) == 300 and all(np.diff(train_curve) <= 1e-9), "the training error never goes up", "each tree reduces (or leaves equal) the training error")
        check(close(test_curve[-1], np.mean((predict_gb(model, X_te, 0.3) - y_te) ** 2)) and close(train_curve[0], np.mean((y_tr - (y_tr.mean() + 0.3 * model[1][0].predict(X_tr))) ** 2)), "curves consistent with the full predictions", "MSE after each round")
        check(train_curve[-1] < 0.05 and test_curve[-1] > test_curve[best_round - 1] + 0.02 and 5 <= best_round <= 100, "the training error goes below 0.05; the test error has its minimum between round 5 and 100 and then climbs back by over 0.02", "overfitting: with eta=0.3 the test error climbs back")
        check(close(best_mse, min(test_curve)) and best_mse < 0.15, "minimum test MSE below 0.15", "min(test_curve)")
        check(diff_5_rounds < 1e-9 and diff_100_rounds < 0.25, "after 5 rounds the predictions match scikit-learn (1e-9); after 100 they differ by at most 0.25", "criterion='squared_error', random_state=0")
    """,
    hint="""
        Loop: `F = np.full(len(y), y.mean()); for _ in range(n_rounds): t = DecisionTreeRegressor(...).fit(X, y - F); F += eta * t.predict(X); trees.append(t)`. Curve on the test set: `F = f0; for t in trees: F = F + eta * t.predict(X_te); curve.append(mean((y_te - F) ** 2))`.
        Comparison: `GradientBoostingRegressor(n_estimators=5, learning_rate=0.1, max_depth=3, criterion="squared_error", random_state=0).fit(X_tr, y_tr).predict(X_te)`.
    """,
    solution=[
        C("""
        def fit_gb(X, y, n_rounds, eta, depth):
            f0 = y.mean()
            F = np.full(len(y), f0)
            trees = []
            for _ in range(n_rounds):
                tree = DecisionTreeRegressor(max_depth=depth, criterion="squared_error", random_state=0).fit(X, y - F)      # small tree on the residuals
                F = F + eta * tree.predict(X)
                trees.append(tree)
            return f0, trees


        def predict_gb(model, X, eta):
            f0, trees = model
            return f0 + eta * sum(t.predict(X) for t in trees)


        def error_curve(model, X, y, eta):
            f0, trees = model
            F = np.full(len(X), f0)
            curve = []
            for t in trees:
                F = F + eta * t.predict(X)
                curve.append(np.mean((y - F) ** 2))
            return curve
        """),
        C("""
        model = fit_gb(X_tr, y_tr, 300, 0.3, 2)
        train_curve = error_curve(model, X_tr, y_tr, 0.3)
        test_curve = error_curve(model, X_te, y_te, 0.3)
        best_round = int(np.argmin(test_curve)) + 1
        best_mse = float(min(test_curve))
        print(f"minimum test MSE {best_mse:.4f} at round {best_round} | after 300 rounds: test {test_curve[-1]:.4f}, training {train_curve[-1]:.4f}")
        """),
        C("""
        def difference(B):
            mine = predict_gb(fit_gb(X_tr, y_tr, B, 0.1, 3), X_te, 0.1)
            sk = GradientBoostingRegressor(n_estimators=B, learning_rate=0.1, max_depth=3, criterion="squared_error", random_state=0).fit(X_tr, y_tr).predict(X_te)
            return float(np.abs(mine - sk).max())


        diff_5_rounds, diff_100_rounds = difference(5), difference(100)
        print(f"maximum difference from scikit-learn: after 5 rounds {diff_5_rounds:.1e} | after 100 rounds {diff_100_rounds:.3f}")
        """),
    ],
    learned="""
        - **Boosting is a four-line loop:** initial prediction (the mean), residuals, small tree, update with step `eta`. With 5 rounds the by-hand implementation matches scikit-learn's to 1e-9 (the predictions are identical to machine precision).
        - **The training error always drops, the test error does not:** with `eta = 0.3` and trees of 4 leaves the training error gets below 0.01 while the test error touches its minimum (0.109) around 20 rounds and then climbs back (0.139 at 300 rounds). It is the sign of the overfitting of boosting: more rounds = more complexity, and the later rounds learn the noise of the residuals.
        - **After 100 rounds small differences with scikit-learn appear** (up to about 0.1 in absolute value): scikit-learn converts the features to `float32` and resolves ties between splits differently, and the differences accumulate. It is not a mistake: two correct implementations of the same algorithm can diverge in the numerical details.
        - Compare with the forest (M8): the forest averages **deep** trees and never climbs back; boosting sums **small** trees and can climb back. That is why the number of rounds is a hyperparameter to choose with validation.
    """,
    pitfall="""
        Increasing the rounds as long as the training error drops. In boosting the training error always drops (there is always something left to learn in the residuals): the only measure of the "right point" is the error on unseen data.
    """,
    interview=(
        "How does gradient boosting work with the squared loss?",
        """
        It starts from a constant prediction (the mean of the target); at each round it computes the residuals of the current model, trains a small tree to predict the residuals and adds its prediction multiplied by a learning rate. With other losses the residuals become the negative gradient of the loss.
        The training error always drops, so the number of rounds is chosen with a validation set (early stopping).
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="classification_gradient_boosting_from_scratch", kind="A", level=3, minutes=60, module=MODULE,
    title="Gradient boosting for classification: gradient, hessian and Newton step",
    goal="""
        Implement boosting with the logistic loss: initial prediction as log-odds, **gradient** ($y - p$) as the target of the tree and leaf values with the **Newton step** ($\\sum r / \\sum p(1-p)$, that is gradient divided by hessian). Check that it matches `GradientBoostingClassifier` in the first rounds and measure how much
        a plain gradient step (without the Newton factor) departs.
    """,
    dataset="""
        `make_classification(1500, 8 features, 4 informative, flip_y=0.05)`, 60/40 split: `X_tr, y_tr, X_te, y_te`.
    """,
    purpose="""
        Understand where the math of XGBoost comes from: it uses the gradient **and the hessian** of the loss (a second-order expansion). The optimal value of a leaf is the ratio between the sum of the gradients and the sum of the hessians: it is the Newton step for the problem of that leaf.
    """,
    task=[
        "`sigmoid(z)`. `fit_gbc(X, y, n_rounds, eta, depth)`: `F0 = log(p0 / (1 − p0))` with `p0 = y.mean()`; for each round: `p = sigmoid(F)`, residuals `r = y − p`, tree `DecisionTreeRegressor(max_depth=depth, criterion=\"squared_error\", random_state=0)` trained on `r`; for each **leaf** (`tree.apply(X)`) the value `Σ r / Σ p(1−p)` (over the examples of the leaf) and update `F += eta * leaf_value`. Return `(F0, [(tree, leaf→value dictionary)])`.",
        "`predict_proba(model, X, eta)`: probability of class 1 (start from `F0`, add `eta * value` for each tree, apply the sigmoid; leaves not seen in training: it does not happen, the structure is the same).",
        "Compare with `GradientBoostingClassifier(n_estimators=B, learning_rate=0.1, max_depth=3, criterion=\"squared_error\", random_state=0)`: save `diff_3` and `diff_50` (maximum absolute difference between the test probabilities after 3 and 50 rounds) and `mean_diff_50` (mean absolute difference after 50 rounds). Save `class_agreement_50` (fraction of test examples with the same predicted class).",
        "**Without the Newton factor:** repeat with leaf value = **mean of the residuals** (gradient only, without dividing by the hessian) and save `acc_without_newton` and `acc_with_newton` at 50 rounds with `eta=0.1`; and `logloss_without_newton`, `logloss_with_newton`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = make_classification(1500, n_features=8, n_informative=4, class_sep=0.8, flip_y=0.05, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
    """,
    starter=[
        C("""
        def sigmoid(z):
            ...


        def fit_gbc(X, y, n_rounds, eta, depth, newton=True):
            ...


        def predict_proba(model, X, eta):
            ...
        """),
        C("""
        diff_3 = None
        diff_50 = None
        mean_diff_50 = None
        class_agreement_50 = None
        """),
        C("""
        acc_without_newton = None
        acc_with_newton = None
        logloss_without_newton = None
        logloss_with_newton = None
        """),
    ],
    checks="""
        check(close(sigmoid(np.array([0.0, 30.0, -30.0])), [0.5, 1.0, 0.0], atol=1e-9), "correct sigmoid", "1 / (1 + exp(-z))")
        m = fit_gbc(X_tr, y_tr, 1, 0.1, 3)
        check(len(m[1]) == 1 and close(m[0], np.log(y_tr.mean() / (1 - y_tr.mean()))), "a single round, F0 = log-odds of the prevalence", "F0 = log(p0 / (1 - p0))")
        check(diff_3 < 1e-9, "after 3 rounds the probabilities match scikit-learn (1e-9)", "leaf values: sum of the residuals / sum of p (1 - p)")
        check(mean_diff_50 < 0.005 and class_agreement_50 > 0.99, "after 50 rounds the mean difference is tiny (below 0.005) and the classes match for over 99%", "differences from float32 and tie-breaking")
        check(acc_without_newton > 0.7 and abs(acc_with_newton - acc_without_newton) < 0.05 and logloss_without_newton > logloss_with_newton + 0.05, "without the Newton factor the log-loss is worse by over 0.05 (the steps are too small), the accuracy almost the same", "compare the two versions")
    """,
    hint="""
        Leaf values: `leaves = tree.apply(X)`; `{l: r[leaves == l].sum() / (p[leaves == l] * (1 - p[leaves == l])).sum() for l in np.unique(leaves)}`; update: `F += eta * np.array([values[l] for l in leaves])`. Without Newton: `r[leaves == l].mean()`. Prediction: for a new example `tree.apply(X)` returns the leaf.
    """,
    solution=[
        C("""
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))


        def fit_gbc(X, y, n_rounds, eta, depth, newton=True):
            p0 = y.mean()
            F0 = np.log(p0 / (1 - p0))                                              # log-odds of the prevalence
            F = np.full(len(y), F0)
            trees = []
            for _ in range(n_rounds):
                p = sigmoid(F)
                r = y - p                                                            # negative gradient of the log-loss
                tree = DecisionTreeRegressor(max_depth=depth, criterion="squared_error", random_state=0).fit(X, r)
                leaves = tree.apply(X)
                values = {}
                for l in np.unique(leaves):
                    m = leaves == l
                    hessian = (p[m] * (1 - p[m])).sum()
                    values[l] = r[m].sum() / max(hessian, 1e-12) if newton else r[m].mean()      # Newton step: gradient / hessian
                F = F + eta * np.array([values[l] for l in leaves])
                trees.append((tree, values))
            return F0, trees


        def predict_proba(model, X, eta):
            F0, trees = model
            F = np.full(len(X), F0)
            for tree, values in trees:
                F = F + eta * np.array([values[l] for l in tree.apply(X)])
            return sigmoid(F)
        """),
        C("""
        def attempt(B):
            mine = predict_proba(fit_gbc(X_tr, y_tr, B, 0.1, 3), X_te, 0.1)
            sk = GradientBoostingClassifier(n_estimators=B, learning_rate=0.1, max_depth=3, criterion="squared_error", random_state=0).fit(X_tr, y_tr)
            p_sk = sk.predict_proba(X_te)[:, 1]
            return mine, p_sk


        m3, s3 = attempt(3)
        m50, s50 = attempt(50)
        diff_3 = float(np.abs(m3 - s3).max())
        diff_50 = float(np.abs(m50 - s50).max())
        mean_diff_50 = float(np.abs(m50 - s50).mean())
        class_agreement_50 = float(((m50 > 0.5) == (s50 > 0.5)).mean())
        print(f"maximum difference: 3 rounds {diff_3:.1e}, 50 rounds {diff_50:.3f} | mean after 50 rounds {mean_diff_50:.5f} | same classes: {class_agreement_50:.1%}")
        """),
        C("""
        p_newton = predict_proba(fit_gbc(X_tr, y_tr, 50, 0.1, 3, newton=True), X_te, 0.1)
        p_grad = predict_proba(fit_gbc(X_tr, y_tr, 50, 0.1, 3, newton=False), X_te, 0.1)
        acc_with_newton, acc_without_newton = float(((p_newton > 0.5) == y_te).mean()), float(((p_grad > 0.5) == y_te).mean())
        logloss_with_newton, logloss_without_newton = log_loss(y_te, p_newton), log_loss(y_te, p_grad)
        print(f"with the Newton step: accuracy {acc_with_newton:.3f}, log-loss {logloss_with_newton:.3f} | gradient only: accuracy {acc_without_newton:.3f}, log-loss {logloss_without_newton:.3f}")
        """),
    ],
    learned="""
        - **Logistic boosting uses the gradient to choose the structure and the hessian for the leaf value.** The tree is trained on the residuals $y - p$; the value of a leaf is $\\sum r/\\sum p(1-p)$: the Newton step to minimize the log-loss in that leaf. It matches scikit-learn in the first rounds to 1e-9; after 50 rounds the differences are on average 0.0006 (at most 0.14 on a single example) and the predicted classes match 100%.
        - **The hessian $p(1-p)$ is small for "safe" examples** (p close to 0 or 1) and large for uncertain ones: the leaf "trusts" the examples near the boundary more. It is the foundation of `min_child_weight` in XGBoost: a minimum for the **sum of the hessians** in a leaf, not for the number of examples.
        - **Without the Newton factor** (value = mean of the residuals) the steps are too small (the residuals $y - p$ are in [−1, 1], but the right step on the log-odds scale is much larger): after the same number of rounds the log-loss is much worse (0.48 against 0.37) and the accuracy drops too (0.80 against 0.84). It is the lesson: the class reads fine even with an undertrained model, the **probabilities** do not.
        - XGBoost extends the idea: it uses a second-order expansion of the loss **plus a penalty on the leaf weights** ($\\lambda$): the optimal leaf value becomes $-\\sum g/(\\sum h + \\lambda)$ and the gain of a split is a formula in $G$ and $H$.
    """,
    pitfall="""
        Interpreting the raw predictions of a boosting model as probabilities without the sigmoid, or using a "simplified" implementation (value = mean of the residuals) and looking only at the accuracy: the probabilities come out squashed toward the prevalence.
    """,
    interview=(
        "Why does XGBoost also use the hessian?",
        """
        Because it approximates the loss with a second-order expansion: the optimal value of a leaf is -G/(H+λ), where G is the sum of the gradients and H that of the hessians; it is the Newton step, much more efficient than a plain gradient step, and it lets you compute the gain of every split in closed form.
        The hessian also gives the meaning of min_child_weight: a minimum weight per leaf, where very confident examples count less.
        """),
)

EXERCISES = [A1, A2]
