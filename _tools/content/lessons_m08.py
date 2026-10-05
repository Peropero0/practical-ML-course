"""Module 8 lessons: random forest."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_08_random_forest"
TITLE = "Random forest"
LESSONS = [
    ("01_bagging_and_forests.ipynb", "8.1", "Bagging and forests",
     "Bootstrap, voting, out-of-bag samples and feature subsets: why averaging trees works."),
    ("02_hyperparameters_and_diagnostics.ipynb", "8.2", "Hyperparameters and diagnostics",
     "Number of trees, `max_features`, minimum leaf size, OOB error and reliable importances."),
    ("03_forests_in_practice.ipynb", "8.3", "Forests in practice",
     "Probabilities and calibration, imbalanced classes, regression, cost and limits."),
]
PATH = [
    ("Lesson 8.1", ["A1", "A2", "F1", "E2", "D1"]),
    ("Lesson 8.2", ["B1", "C1", "E1", "D2", "C2", "E3"]),
    ("Lesson 8.3", ["B2", "B3", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_8_1() -> List[Cell]:
    """Cells of lesson 8.1."""
    return [
        M("""
        # Lesson 8.1 · Bagging and forests

        In module 7 you saw that a deep tree has **low bias and high variance**: it changes a lot if the sample changes a little. A random forest exploits this flaw: it trains many deep trees on different samples and **averages them**. The average of estimators
        that are variable but little biased has a much smaller variance, without increasing the bias.

        **What you will learn**

        - **Bagging** (bootstrap aggregating): sampling with replacement and voting.
        - What the random forest adds: a random subset of features at each split (`max_features`), which **decorrelates** the trees.
        - The **out-of-bag** sample and the OOB error: a free validation estimate.
        """),
        _setup("""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(2000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
        bagging = BaggingClassifier(DecisionTreeClassifier(), n_estimators=200, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        forest = RandomForestClassifier(200, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        for name, m in {"one tree": tree, "bagging of 200 trees": bagging, "random forest of 200 trees": forest}.items():
            print(f"{name:28s} train {m.score(X_tr, y_tr):.3f} | test {m.score(X_te, y_te):.3f}")
        """),
        M("""
        ## The bootstrap and the vote

        A **bootstrap** sample is a sample with replacement of the same size as the training set: it contains about **63%** of the distinct examples (each one appears zero, one or more times). Each tree is trained on a different sample; the final prediction is the **majority vote**
        (in scikit-learn, the average of the leaf probabilities). By averaging, the "random" errors of each tree (those due to its sample) cancel out, and the systematic ones remain.
        """),
        C("""
        rng = np.random.default_rng(0)
        n = 2000
        share = np.mean([len(np.unique(rng.integers(0, n, n))) / n for _ in range(20)])
        print(f"average share of distinct examples in a bootstrap sample: {share:.3f} (theoretical 1 - 1/e = {1 - 1 / np.e:.3f})")
        """),
        M("""
        ## What the random forest adds

        Bagging has a limit: the trees are **correlated** (if a feature is very strong, they all choose it at the root and look alike). The variance of the average of $B$ variables with variance $\\sigma^2$ and correlation $\\rho$ is

        $$\\rho\\,\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$$

        Adding trees only reduces the second term: the first, $\\rho\\sigma^2$, stays. The forest reduces $\\rho$ by choosing, at **every split**, the best among a random subset of features (`max_features`, by default $\\sqrt{p}$ in classification): more diverse trees, a better average.

        > **Common mistake.** Thinking that "more trees = more overfitting". Adding trees to a forest does not increase overfitting: the variance drops and levels off. Overfitting depends on the complexity of the individual trees and on their correlation.

        ## The out-of-bag error

        Each tree does not see about 37% of the examples (the ones outside its bootstrap). Predicting each example **only with the trees that did not see it** gives an honest prediction for the whole training set: the **OOB error**, a free estimate of generalization, with no validation set.
        """),
        C("""
        oob = RandomForestClassifier(300, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        print(f"OOB accuracy: {oob.oob_score_:.3f} | accuracy on the test set: {oob.score(X_te, y_te):.3f} | (on the training set: {oob.score(X_tr, y_tr):.3f})")
        """),
        M("""
        The OOB accuracy is close to the test one, while the training one is 1.0 (each tree has seen 63% of the examples and the forest memorizes them). **Never evaluate a forest on the training set**: the OOB is the correct internal estimate.

        ## Check your understanding

        """ + details(
            "Why does the bootstrap sample contain about 63% of distinct examples?",
            "Because the probability that an example is never drawn in n draws with replacement is (1 − 1/n)^n → 1/e ≈ 0.368: 36.8% stays out (out-of-bag) and 63.2% is present at least once.") + "\n\n" + details(
            "Why does a small max_features improve the forest?",
            "Because it reduces the correlation between the trees: if every split can only use some of the features, the trees do not all choose the same dominant feature at the root, and the average of less correlated trees has a smaller variance. If it is too small, though, each tree is weak (more bias): it is a hyperparameter to choose.")),
    ]


def lesson_8_2() -> List[Cell]:
    """Cells of lesson 8.2."""
    return [
        M("""
        # Lesson 8.2 · Hyperparameters and diagnostics

        A forest has few hyperparameters and almost all of them are robust: that is why it is an excellent "first attempt" model. This lesson says which ones matter, which do not, and how to diagnose the forest with the OOB error.

        **What you will learn**

        - `n_estimators`: it is a budget, not a hyperparameter to optimize.
        - `max_features`, `min_samples_leaf`, `max_depth`: where the gains are.
        - Importances: impurity (with its flaws), permutation, and how they behave with correlated features.
        """),
        _setup("""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X, y = make_classification(2000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        """),
        M("""
        ## Number of trees: a budget

        With `warm_start=True` you can add trees to an already trained forest and watch how the OOB error settles: it drops quickly in the first 50-100 trees and then stays flat. It is not "optimized": you choose a number high enough (100-500) and pay in time and memory.
        """),
        C("""
        f = RandomForestClassifier(warm_start=True, oob_score=True, random_state=0, n_jobs=-1)
        rows = []
        for n in [5, 10, 20, 50, 100, 200, 400]:
            f.set_params(n_estimators=n).fit(X_tr, y_tr)
            rows.append({"trees": n, "OOB error": 1 - f.oob_score_, "test error": 1 - f.score(X_te, y_te)})
        pd.DataFrame(rows).set_index("trees").round(4).T
        """),
        M("""
        ## `max_features` and the minimum leaf size

        - **`max_features`**: `"sqrt"` (default for classification), `1.0` = all the features (equivalent to **bagging**), or a fraction. It is the hyperparameter with the clearest effect on the correlation between trees: search it on a small grid.
        - **`min_samples_leaf`**: larger leaves = smoother trees, less variance. With many noisy rows it can help; forests tolerate deep trees well, so the gain is usually modest.
        - **`max_depth`**: it is rarely needed to limit it; look at `min_samples_leaf` first.
        """),
        C("""
        for mf in [1, 2, "sqrt", 6, 12]:
            f = RandomForestClassifier(200, max_features=mf, oob_score=True, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
            print(f"max_features={str(mf):5s} | OOB error {1 - f.oob_score_:.4f} | test error {1 - f.score(X_te, y_te):.4f}")
        """),
        M("""
        ## Feature importances

        `feature_importances_` is the average of the impurity importances of the individual trees: it has the flaws seen in module 7 (it rewards features with many distinct values, and it is computed on the training set). The alternative is the **permutation importance** on data not used for training
        (test set, or the OOB). With **correlated features** both split the importance among the copies: when one copy is shuffled the model can use the other, and the permutation **underestimates** each of them (exercise C2).
        """),
        C("""
        f = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
        perm = permutation_importance(f, X_te, y_te, n_repeats=10, random_state=0)
        pd.DataFrame({"impurity": f.feature_importances_, "permutation (test)": perm.importances_mean}, index=[f"x{i}" for i in range(12)]).round(3).T
        """),
        M("""
        > **Common mistake.** Trusting the defaults and a single experiment. In `RandomForestRegressor` `max_features` is 1.0 (all the features): without changing it, it is bagging, not a random forest (exercise E2). And a forest is a stochastic model: comparing two configurations with a single `random_state` and few trees
        > often means comparing the noise of the seed (exercise E1).

        ## Check your understanding

        """ + details(
            "Do I have to optimize n_estimators with cross-validation?",
            "No: more trees never worsen the generalization (the error drops and levels off). Choose a sufficient number (usually 200-500) checking that the OOB error is flat, and accept the cost in time and memory. Optimizing it with a grid is almost always a waste of computation.") + "\n\n" + details(
            "When is the OOB error less reliable than cross-validation?",
            "With duplicate or correlated rows (same customer, repeated measurements, time series): the OOB assumes that an example outside the sample of a tree is independent from those inside, but if a copy of it is in the sample the tree has already 'seen' it and the OOB is optimistic (exercise C1). "
            "In those cases you need validation by group or by time.")),
    ]


def lesson_8_3() -> List[Cell]:
    """Cells of lesson 8.3."""
    return [
        M("""
        # Lesson 8.3 · Forests in practice

        The random forest is a model that almost always works "as it is". It has features worth knowing, though: its probabilities are not calibrated, with imbalanced classes you need to pay attention to the threshold, outside the range of the data it is flat like a tree, and it costs memory and prediction time.

        **What you will learn**

        - How the probabilities of a forest behave and how to correct them.
        - Regression with forests and what they cannot do (extrapolate).
        - The costs: model size and latency.
        """),
        _setup("""
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        import time, pickle
        """),
        M("""
        ## Probabilities of a forest

        `predict_proba` is the average of the leaf fractions of the trees. With deep trees and almost pure leaves the probabilities are extreme in a single tree, but the **average** softens them: a forest gives reasonable probabilities, but they tend to be **underconfident** at the margins (rarely 0 or 1) and not calibrated.
        Post-hoc calibration (`CalibratedClassifierCV`) corrects them.
        """),
        C("""
        X, y = make_classification(6000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.7, flip_y=0.08, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "forest": RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
            "calibrated forest": CalibratedClassifierCV(RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1), method="isotonic", cv=3),
        }
        for name, m in models.items():
            p = m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
            print(f"{name:18s} AUC {roc_auc_score(y_te, p):.3f} | Brier {brier_score_loss(y_te, p):.4f} | log-loss {log_loss(y_te, p):.3f}")
        """),
        M("""
        ## Regression and limits

        `RandomForestRegressor` predicts the average of the leaf means: robust, no scaling, it handles interactions. But it is a **piecewise constant** function and outside the range of the data **it is flat** (like the single tree, M7 C1).

        ## Cost

        A forest with hundreds of deep trees takes up memory (here 11 MB for 100 trees on 15,000 rows, and it grows with the rows: hundreds of MB on large datasets) and prediction requires going through all the trees. To serve the model in real time the number of trees, the depth and the minimum leaf size are cost levers, as well as accuracy levers: with a minimum leaf of 20 the model weighs 3 MB and loses a single point (0.955 → 0.942).
        """),
        C("""
        Xr, yr = make_classification(20000, n_features=20, n_informative=8, random_state=0)
        for name, kw in {"100 trees, leaf 1": dict(n_estimators=100), "100 trees, leaf 20": dict(n_estimators=100, min_samples_leaf=20), "30 trees, leaf 20": dict(n_estimators=30, min_samples_leaf=20)}.items():
            f = RandomForestClassifier(random_state=0, n_jobs=-1, **kw).fit(Xr[:15000], yr[:15000])
            t0 = time.time(); f.predict(Xr[15000:]); tp = time.time() - t0
            print(f"{name:24s} size {len(pickle.dumps(f)) / 1e6:7.1f} MB | prediction on 5000 examples {tp:.2f} s | test {f.score(Xr[15000:], yr[15000:]):.3f}")
        """),
        M("""
        > **Common mistake.** Treating the probabilities of a forest as calibrated risks (to compute an expected cost, for example) without checking them: use a calibration curve or `CalibratedClassifierCV` (module 4).

        ## Check your understanding

        """ + details(
            "When do you prefer a forest to a boosting?",
            "When you want a robust model, not very sensitive to hyperparameters, with free OOB estimates and easy parallel training. Boosting (module 9) usually reaches higher accuracy but requires more care in choosing the learning rate, the depth and the number of trees.") + "\n\n" + details(
            "Can a forest overfit?",
            "Not by adding trees. It can memorize the training set with deep trees (training accuracy 1.0) without the test getting much worse: that is why you always look at the OOB or the test, never the training. With very noisy data or few examples, a larger minimum leaf helps.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_bagging_and_forests.ipynb", lesson_8_1),
                    ("02_hyperparameters_and_diagnostics.ipynb", lesson_8_2),
                    ("03_forests_in_practice.ipynb", lesson_8_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
