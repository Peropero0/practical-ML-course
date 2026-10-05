"""Module 9 lessons: gradient boosting and XGBoost."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_09_xgboost"
TITLE = "Gradient boosting and XGBoost"
LESSONS = [
    ("01_boosting.ipynb", "9.1", "Boosting",
     "Small trees that correct the errors of the previous ones: residuals, shrinkage and overfitting."),
    ("02_hyperparameters_and_early_stopping.ipynb", "9.2", "Hyperparameters and early stopping",
     "Learning rate, depth, sampling, regularization and how to stop at the right moment."),
    ("03_xgboost_in_practice.ipynb", "9.3", "XGBoost in practice",
     "Missing values, categoricals, monotonic constraints, importances and contributions, comparison with the other implementations."),
]
PATH = [
    ("Lesson 9.1", ["A1", "A2", "C2", "F1"]),
    ("Lesson 9.2", ["B1", "D1", "C1", "E1", "E2", "B3"]),
    ("Lesson 9.3", ["B2", "D2", "E3", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_9_1() -> List[Cell]:
    """Cells of lesson 9.1."""
    return [
        M("""
        # Lesson 9.1 · Boosting

        Random forest reduces the variance by **averaging** many deep trees in parallel. **Boosting** does the opposite: it builds **small, weak trees, one after the other**, and each tree corrects the errors of the previous ones. A sum of many small steps in the right direction:
        it is the algorithm that wins most competitions on tabular data, and it is the one behind XGBoost, LightGBM and CatBoost.

        **What you will learn**

        - The idea of gradient boosting: fit each tree to the **residuals** of the current model.
        - The role of **shrinkage** (`learning_rate`) and of the rounds.
        - Why boosting can overfit (unlike the forest) and how to see it.
        """),
        _setup("""
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, (600, 1))
        y = np.sin(X[:, 0]) + 0.3 * X[:, 0] + rng.normal(0, 0.3, 600)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        """),
        M("""
        ## Trees on the residuals

        With the squared loss, boosting is very simple:

        1. Start from a constant prediction (the mean of `y`): $F_0$.
        2. Compute the **residuals** $r = y - F_{m-1}(x)$: what the current model gets wrong.
        3. Train a **small** tree (depth 2-6) to predict the residuals.
        4. Update $F_m = F_{m-1} + \\eta\\,h_m(x)$, where $\\eta$ is the *learning rate* (shrinkage, typically 0.01-0.3).
        5. Repeat.

        For other losses (logistic, quantile) the "residuals" become the **negative gradient** of the loss with respect to the prediction, hence the name *gradient* boosting.
        """),
        C("""
        eta, rounds = 0.3, 5
        F = np.full(len(y_tr), y_tr.mean())                         # F0: the mean
        grid = np.linspace(-3, 3, 200).reshape(-1, 1)
        Fg = np.full(len(grid), y_tr.mean())
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
        for m in range(1, rounds + 1):
            tree = DecisionTreeRegressor(max_depth=2).fit(X_tr, y_tr - F)          # SMALL tree on the residuals
            F += eta * tree.predict(X_tr); Fg += eta * tree.predict(grid)
            if m in (1, 3, 5):
                a = ax[[1, 3, 5].index(m)]
                a.scatter(X_tr, y_tr, s=5, color="gray", alpha=0.5); a.plot(grid, Fg, "r"); a.set_title(f"after {m} trees | train error {np.mean((y_tr - F) ** 2):.3f}")
        plt.tight_layout(); plt.show()
        """),
        M("""
        Each tree alone is very little (2 levels, 4 leaves), but the sum of a few steps already follows the shape of the relationship. It is the opposite of the forest: **high bias, low variance per tree**, and the complexity grows at every round.

        ## Shrinkage and rounds: a trade-off

        A small `learning_rate` takes small steps: you need **more rounds** but the generalization is better (each tree weighs little and cannot "memorize"). With a large `learning_rate` a few rounds are enough and the risk of overfitting is high. That is why `learning_rate` and `n_estimators` have to be chosen
        together: a practical rule is a low `learning_rate` (0.03-0.1) and the number of rounds chosen with **early stopping** (lesson 9.2).
        """),
        C("""
        for eta in [0.5, 0.1, 0.02]:
            gb = GradientBoostingRegressor(n_estimators=400, learning_rate=eta, max_depth=2, random_state=0).fit(X_tr, y_tr)
            curve = [mean_squared_error(y_te, p) for p in gb.staged_predict(X_te)]           # test error after each round
            print(f"learning_rate {eta:4}: minimum test error {min(curve):.4f} at round {int(np.argmin(curve)) + 1:3d} | after 400 rounds {curve[-1]:.4f}")
        """),
        M("""
        With `learning_rate = 0.5` the minimum arrives immediately and then the error **climbs back** (overfitting); with 0.02 the curve drops slowly and after 400 rounds it is still close to the minimum. **Unlike the forest, boosting overfits as the rounds increase**: you have to stop.

        > **Common mistake.** Fixing `n_estimators=1000` "to be safe" without early stopping: with a medium learning rate and a noisy dataset you are just memorizing the training set.

        ## Check your understanding

        """ + details(
            "Why are small trees used in boosting?",
            "Because each tree must be a weak corrector: the sum of many small steps in directions guided by the gradient builds a flexible function in a controlled way, with the bias dropping at every round. Deep trees would memorize the residuals already at the first step.") + "\n\n" + details(
            "What is the difference between bagging and boosting?",
            "Bagging trains many deep trees **in parallel** on different samples and averages them (it reduces the variance). Boosting trains small trees **in sequence**, each one on the errors of the previous one (it reduces the bias, and with too many rounds it increases the variance).")),
    ]


def lesson_9_2() -> List[Cell]:
    """Cells of lesson 9.2."""
    return [
        M("""
        # Lesson 9.2 · Hyperparameters and early stopping

        XGBoost has many hyperparameters, but few make a difference. This lesson gives you the map and shows how to choose them efficiently with **early stopping**: stopping the training when the error on a validation set stops dropping.

        **What you will learn**

        - The groups of hyperparameters: step (`learning_rate`), tree complexity, sampling, regularization.
        - How to use `eval_set` and `early_stopping_rounds`, and what never to do.
        - Why the choice of the rounds is made on a validation set **different from the test**.
        """),
        _setup("""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split

        X, y = make_classification(4000, n_features=12, n_informative=5, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        print("training", X_tr.shape, "| validation", X_val.shape, "| test", X_te.shape)
        """),
        M("""
        ## The map of the hyperparameters

        | group | parameters (XGBoost) | what they do |
        |---|---|---|
        | step | `learning_rate` (eta), `n_estimators` | how much each tree weighs and how many are needed |
        | complexity | `max_depth`, `min_child_weight`, `gamma` | depth and minimum "weight" (for the logistic: sum of the hessians) to split; `gamma` = minimum gain |
        | sampling | `subsample`, `colsample_bytree` | fraction of rows and columns for each tree: they decorrelate and regularize (as in the forest) |
        | regularization | `reg_lambda` (L2), `reg_alpha` (L1) | penalty on the leaf values |

        A robust starting rule: `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, and the rounds chosen with early stopping.

        ## Early stopping

        You pass an `eval_set` (validation) and `early_stopping_rounds=k`: the training stops if the validation metric does not improve for `k` consecutive rounds, and the model uses the **best round**.
        """),
        C("""
        model = xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                                  eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1, random_state=0)
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        curves = model.evals_result()
        print(f"rounds run: {len(curves['validation_1']['logloss'])} | best round: {model.best_iteration} | validation log-loss at the best: {model.best_score:.4f}")
        print(f"training log-loss at the best round: {curves['validation_0']['logloss'][model.best_iteration]:.4f} | at the end: {curves['validation_0']['logloss'][-1]:.4f}")
        print(f"AUC on the test: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.4f}")
        """),
        M("""
        The **training** log-loss keeps dropping for all the rounds, the **validation** one reaches a minimum and climbs back: that is the overfitting gap. Early stopping stops 30 rounds after the minimum and restores the best round.

        **Golden rule:** the set used for early stopping is a **validation** set (part of the training data), not the test set: otherwise you chose the number of rounds by looking at the test and the generalization estimate is optimistic.

        **Two details that raise no errors and ruin the model** (they are the bugs of exercises E1 and E2):

        - With several sets in `eval_set`, the stopping is based on the **last** one. Here the training is first and the validation is last, which is the right order: putting the training last makes its log-loss always drop and the model never stops (E1).
        - `best_iteration` does not truncate the booster: after `fit` it contains **all** the trained trees, including the 30 of "patience". `predict` and `predict_proba` of the wrapper use only the best ones; the native API (`Booster.predict`, `pred_contribs`) uses everything if you do not pass `iteration_range` (E2).
        """),
        C("""
        booster = model.get_booster()
        print(f"best_iteration = {model.best_iteration} -> good trees {model.best_iteration + 1}, trees in the booster {booster.num_boosted_rounds()}")
        p_wrapper = model.predict_proba(X_te)[:, 1]
        p_native = booster.predict(xgb.DMatrix(X_te))                                                            # all the trees
        p_right = booster.predict(xgb.DMatrix(X_te), iteration_range=(0, model.best_iteration + 1))              # only the best ones
        print(f"maximum gap from the wrapper: booster without iteration_range {np.abs(p_wrapper - p_native).max():.4f} | with iteration_range {np.abs(p_wrapper - p_right).max():.1e}")
        """),
        M("""
        > **Common mistake.** Changing the `learning_rate` without recalibrating the rounds. Halving eta requires about twice the rounds to get the same complexity: with a fixed `n_estimators` the model becomes undertrained. The curves of exercise D1 show how the round of the minimum changes as the configuration changes.

        ## Check your understanding

        """ + details(
            "Why is a small learning rate usually better?",
            "Because each tree corrects little and the sum of many small steps is smoother and less prone to memorizing: the generalization improves (up to a limit) at the price of more rounds and more time. In practice you choose 0.02-0.1 and let early stopping decide the rounds.") + "\n\n" + details(
            "What does `min_child_weight` do?",
            "It prevents creating leaves with little 'weight' (the sum of the hessians: for regression it is the number of examples, for the logistic it is Σ p(1−p), so very confident examples count less). Higher values = more conservative trees, regularization against overfitting on few examples.")),
    ]


def lesson_9_3() -> List[Cell]:
    """Cells of lesson 9.3."""
    return [
        M("""
        # Lesson 9.3 · XGBoost in practice

        Practical features that make XGBoost the reference model on tabular data: it handles **missing values** without imputation, it accepts **monotonic constraints**, it provides per-example contributions (`pred_contribs`) and it has a very fast implementation (`tree_method="hist"`).

        **What you will learn**

        - How XGBoost treats NaN and what changes compared with imputation.
        - What the different importances (`gain`, `weight`, `cover`) and the per-example contributions mean.
        - How XGBoost chooses a split (gradient, hessian, $\\lambda$, $\\gamma$) and where the hyperparameters come in.
        - Monotonic constraints and imbalanced classes.
        - How it compares with scikit-learn's `GradientBoostingClassifier` and `HistGradientBoostingClassifier`.
        """),
        _setup("""
        import time
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X, y = make_classification(12000, n_features=20, n_informative=8, class_sep=0.8, flip_y=0.05, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        """),
        M("""
        ## Native missing values

        At every split XGBoost learns **which way to send the NaN** (left or right) by choosing the direction that reduces the loss the most: missingness becomes information, like the missingness indicator of module 4 but without having to build it. You do not need an imputer (exercise B2).

        ## Per-example contributions

        `predict(..., pred_contribs=True)` (on the `Booster` object) returns, for each example, the contribution of every feature to the prediction (on the log-odds scale) and a constant term (`bias`): the **sum of the contributions is the raw prediction**. It is a local explanation in the style of SHAP computed exactly on the trees.
        """),
        C("""
        model = xgb.XGBClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, eval_metric="logloss", n_jobs=-1, random_state=0).fit(X_tr, y_tr)
        booster = model.get_booster()
        contribs = booster.predict(xgb.DMatrix(X_te[:3]), pred_contribs=True)            # (examples, features + bias)
        margin = booster.predict(xgb.DMatrix(X_te[:3]), output_margin=True)
        print("shape of the contributions:", contribs.shape, "| sum of the contributions = raw margin:", np.allclose(contribs.sum(1), margin, atol=1e-5))
        print("the three largest contributions for the first example (column index):", np.argsort(-np.abs(contribs[0, :-1]))[:3], np.round(contribs[0, np.argsort(-np.abs(contribs[0, :-1]))[:3]], 3))
        """),
        M("""
        ## The importances: `weight`, `gain`, `cover`

        `booster.get_score(importance_type=...)` counts three different things for each feature: `weight` = how many times it appears in a split, `gain` = average gain of its splits, `cover` = how many examples (in terms of hessian) go through its splits. They are three answers to three questions, and they do not coincide.
        """),
        C("""
        table = pd.DataFrame({kind: pd.Series(booster.get_score(importance_type=kind)) for kind in ["weight", "gain", "cover"]}).fillna(0)
        (table / table.sum()).sort_values("gain", ascending=False).head(6).round(3)
        """),
        M("""
        `gain` **concentrates** the importance: the first feature (`f0`) is worth 22% of the gain but only 11% of the splits; at the opposite extreme `f16` appears in 7% of the splits and is worth 5% of the gain. `weight` counts the cuts, not how useful they are, and it rewards continuous features with many distinct values. To decide which features are really needed, the permutation on validation data remains the reference (exercise D2).

        ## How XGBoost chooses a split

        For a generic loss XGBoost approximates the error with its second-order expansion: for each example it computes the **gradient** $g_i$ and the **hessian** $h_i$ of the loss. In a leaf with sums $G = \\sum g_i$ and $H = \\sum h_i$ the optimal value is $w^* = -G/(H + \\lambda)$, and a left/right split improves the loss by

        $$\\text{gain} = \\tfrac12\\left[\\frac{G_L^2}{H_L+\\lambda} + \\frac{G_R^2}{H_R+\\lambda} - \\frac{(G_L+G_R)^2}{H_L+H_R+\\lambda}\\right] - \\gamma .$$

        All the hyperparameters of lesson 9.2 are in here: $\\lambda$ (`reg_lambda`) shrinks the leaf values, $\\gamma$ (`gamma`) is the minimum gain to split, `min_child_weight` is a minimum for $H$. In exercise A2 you rebuild the Newton step by hand (leaf value $\\sum r/\\sum p(1-p)$).

        ## Monotonic constraints and imbalanced classes

        - **Monotonic constraints:** `monotone_constraints={"income": 1, "debt": -1}` forces the prediction to be nondecreasing in `income` and nonincreasing in `debt`. In credit, rents or insurance it is often a domain requirement, and it costs little if the constraint is consistent with the data. Prefer the dictionary by name over the positional tuple (exercise E3).
        - **Imbalanced classes:** `scale_pos_weight` weights the positive class. It rebalances the weights and therefore **distorts the probabilities**: if you need probabilities or a decision with costs, it is safer not to reweight and to choose the threshold (modules 4 and 10).

        ## Three implementations

        - `GradientBoostingClassifier` (scikit-learn): the simplest, very slow on large data.
        - `HistGradientBoostingClassifier` (scikit-learn): uses feature histograms and supports NaN; much faster.
        - `XGBClassifier` with `tree_method="hist"`: histograms, parallelism, explicit regularization, monotonic constraints; the fastest and the richest in features.
        """),
        C("""
        for name, m in {"GradientBoosting (sklearn)": GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=0),
                        "HistGradientBoosting (sklearn)": HistGradientBoostingClassifier(max_iter=60, max_depth=3, random_state=0),
                        "XGBoost hist": xgb.XGBClassifier(n_estimators=60, max_depth=3, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=0)}.items():
            t0 = time.time(); m.fit(X_tr, y_tr); tf = time.time() - t0
            print(f"{name:32s} fit {tf:5.1f} s | accuracy {m.score(X_te, y_te):.3f}")
        """),
        M("""
        > **Common mistake.** Using the `weight` importance (number of times a feature is used in a split) as "importance": it rewards continuous features with many thresholds, like the impurity of a tree. `gain` (average gain) and the per-example contributions are more informative; for reliable judgments the permutation on validation data remains the choice.

        ## Check your understanding

        """ + details(
            "When do you prefer a random forest to XGBoost?",
            "When you want a robust model with few hyperparameters and little sensitivity to how they are chosen, with free OOB estimates and easy parallel training; with very noisy data (boosting chases the wrong labels: exercise C2). XGBoost usually wins in accuracy if you tune it carefully.") + "\n\n" + details(
            "Does XGBoost extrapolate?",
            "No: it sums trees, so outside the training range it is flat like any tree-based model. For trends you need a linear model as a base (`booster=\"gblinear\"` or a hybrid) or to transform the target (differences).")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_boosting.ipynb", lesson_9_1),
                    ("02_hyperparameters_and_early_stopping.ipynb", lesson_9_2),
                    ("03_xgboost_in_practice.ipynb", lesson_9_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
