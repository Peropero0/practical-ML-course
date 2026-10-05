"""Module 7 lessons: decision trees."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_07_decision_trees"
TITLE = "Decision trees"
LESSONS = [
    ("01_how_a_tree_decides.ipynb", "7.1", "How a tree decides",
     "Splits, Gini impurity and entropy, gain: how a tree is built and how it is read."),
    ("02_complexity_and_pruning.ipynb", "7.2", "Complexity, overfitting and pruning",
     "Depth, minimum leaf size, `ccp_alpha`: deep trees memorize, and they change a lot with the data."),
    ("03_trees_in_practice.ipynb", "7.3", "Trees in practice",
     "Regression, importances (and their traps), categoricals, invariance to scale and limits."),
]
PATH = [
    ("Lesson 7.1", ["A1", "D1", "E3", "F1"]),
    ("Lesson 7.2", ["B1", "B3", "C2", "E2", "A2"]),
    ("Lesson 7.3", ["B2", "C1", "D2", "E1", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_7_1() -> List[Cell]:
    """Cells of lesson 7.1."""
    return [
        M("""
        # Lesson 7.1 · How a tree decides

        A decision tree is a **sequence of yes/no questions** about the features ("is the income above 24,000?") that splits the data into groups that are more and more homogeneous. At the end each group (a **leaf**) gives a prediction: the most frequent class, or the mean of the target.
        It is the easiest model to read and to explain, and it is the building block of random forest and XGBoost (modules 8 and 9).

        **What you will learn**

        - How the best question is chosen at each step (Gini impurity, entropy, gain).
        - How to read a tree: `export_text`, the path of an example, leaf probability.
        - What `fit` does and why a tree does not need scaling.
        """),
        _setup("""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

        data = load_breast_cancer()
        X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, stratify=data.target, random_state=0)
        tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_tr, y_tr)
        print(export_text(tree, feature_names=list(data.feature_names)))
        print("accuracy: train", round(tree.score(X_tr, y_tr), 3), "| test", round(tree.score(X_te, y_te), 3))
        """),
        M("""
        ## How a question is chosen

        At each node the algorithm tries **every feature and every possible threshold** (the midpoints between consecutive values) and picks the split that makes the two child groups **purer**. Purity is measured with the **Gini impurity**

        $$G = 1 - \\sum_k p_k^2$$

        where $p_k$ is the fraction of class $k$ in the node: it is 0 if the node is pure and 0.5 (with two classes) if it is half and half. The **gain** of a split is the impurity of the parent minus the weighted average (by number of examples) of the impurities of the children. The alternative is
        the **entropy** $-\\sum_k p_k\\log_2 p_k$: the two measures almost always give the same trees (exercise F1).
        """),
        C("""
        def gini(y):
            p = np.bincount(y) / len(y)
            return 1 - (p ** 2).sum()

        # the best threshold of the feature 'worst perimeter' at the root, searched by hand
        x = X_tr[:, list(data.feature_names).index("worst perimeter")]
        thresholds = np.unique(x)
        thresholds = (thresholds[:-1] + thresholds[1:]) / 2
        def gain(s):
            left, right = y_tr[x <= s], y_tr[x > s]
            return gini(y_tr) - (len(left) * gini(left) + len(right) * gini(right)) / len(y_tr)
        g = np.array([gain(s) for s in thresholds])
        print(f"impurity of the root: {gini(y_tr):.3f} | best threshold for 'worst perimeter': {thresholds[g.argmax()]:.2f} (gain {g.max():.3f})")
        print("threshold chosen by the tree at the root:", round(tree.tree_.threshold[0], 2), "on", data.feature_names[tree.tree_.feature[0]])
        """),
        M("""
        The tree picked exactly `worst perimeter`, with the same threshold found by hand (106.1). After choosing, it repeats the same procedure **in every child group** (a *greedy* algorithm: it optimizes one step at a time, without looking ahead) until a stopping criterion halts it:
        maximum depth, minimum number of examples in a leaf, pure node. Without limits it goes on until every leaf is pure, that is it **memorizes** the training set.

        ## Reading the prediction

        Each leaf holds the fractions of the training classes that ended up there: `predict_proba` returns those fractions and `predict` the most frequent class. With pure leaves the probabilities are 0 and 1 (like KNN with `k = 1`, lesson 5.1).
        """),
        C("""
        x0 = X_te[:1]
        path = tree.decision_path(x0).indices                       # the nodes the example goes through
        print("nodes visited:", path, "| leaf:", tree.apply(x0))
        for n in path[:-1]:
            f, s = tree.tree_.feature[n], tree.tree_.threshold[n]
            print(f"  node {n}: {data.feature_names[f]} = {x0[0, f]:.2f} {'<=' if x0[0, f] <= s else '>'} {s:.2f}")
        print("class proportions in the leaf:", tree.predict_proba(x0).round(3), "-> prediction", tree.predict(x0))
        """),
        M("""
        ## Trees and scales

        Every question is a threshold on **a single** feature, so a tree is not affected by the scale or by increasing monotonic transformations (logarithm, square root...): changing the units moves the thresholds but not the order of the examples. **There is no need to standardize** (exercise F2).

        > **Common mistake.** Standardizing the features "for a tree" or using the log transformation to "help it": it does not change the predictions. It is wasted time (and, if done badly, a source of leakage with no benefit).

        ## Check your understanding

        """ + details(
            "Why is a tree a greedy algorithm?",
            "Because at each node it chooses the best split **for that node**, without considering whether a worse split now would allow better splits later. Finding the optimal tree is an NP-hard problem, and the greedy approach gives very good trees in time linear in the depth.") + "\n\n" + details(
            "What does it mean that a tree's `predict_proba` is a leaf fraction?",
            "That the probability for an example is the proportion of training examples of that class in the leaf where it lands. With small or pure leaves it is very noisy (0 or 1 with few examples) and it is not calibrated.")),
    ]


def lesson_7_2() -> List[Cell]:
    """Cells of lesson 7.2."""
    return [
        M("""
        # Lesson 7.2 · Complexity, overfitting and pruning

        A tree without limits can bring the training accuracy to 1.0 on any dataset without duplicates with different labels: it is a model of **unlimited complexity**. This lesson shows how to control it and why trees are **high-variance** models.

        **What you will learn**

        - The three ways to limit a tree: `max_depth`, `min_samples_leaf`, `ccp_alpha` (cost-complexity pruning).
        - How overfitting shows up in a validation curve.
        - Why small changes in the data change the tree (instability).
        """),
        _setup("""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, validation_curve
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(1500, n_features=10, n_informative=4, n_redundant=0, class_sep=0.8, flip_y=0.1, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        depths = [1, 2, 3, 4, 5, 6, 8, 10, 15, None]
        tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X_tr, y_tr, param_name="max_depth", param_range=depths, cv=cv)
        tab = pd.DataFrame({"train": tr.mean(1), "validation": va.mean(1)}, index=[str(p) for p in depths])
        print(tab.round(3).T)
        full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
        print("tree without limits: depth", full.get_depth(), "| leaves", full.get_n_leaves(), "| train", full.score(X_tr, y_tr), "| test", round(full.score(X_te, y_te), 3))
        """),
        M("""
        ## Three knobs

        - **`max_depth`**: limits the number of questions in a row. Simple and readable, but arbitrary (the right depth depends on the branch).
        - **`min_samples_leaf`** (and `min_samples_split`): prevents leaves with few examples. It directly controls the **stability** of the leaf predictions: with 20 examples per leaf a probability is much less noisy.
        - **`ccp_alpha`**: *cost-complexity pruning*. The full tree is built and then the branches that are not "worth" their complexity are removed: each leaf costs $\\alpha$, and a branch is cut if the impurity gain it brings is smaller than the cost of its leaves.
        """),
        C("""
        full_tree = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
        path = full_tree.cost_complexity_pruning_path(X_tr, y_tr)
        print("candidate alpha values:", len(path.ccp_alphas))
        chosen = path.ccp_alphas[::max(1, len(path.ccp_alphas) // 8)]
        for a in chosen:
            t = DecisionTreeClassifier(random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
            print(f"alpha {a:.5f} | leaves {t.get_n_leaves():4d} | train {t.score(X_tr, y_tr):.3f} | test {t.score(X_te, y_te):.3f}")
        """),
        M("""
        With `alpha = 0` the tree is complete; as `alpha` grows the leaves shrink in number and the gap between train and test closes, down to a tiny tree. Here too the best value is chosen with cross-validation, and the rule "the simplest within one standard error of the best" is worth using.

        ## Unstable trees

        A tree makes cascading choices: if the split at the root changes (because two features have almost the same gain), **everything else changes**. Resampling the data slightly, the tree can have a different structure and different predictions on many examples. Here the root is stable (the same feature in 29 resamples out of 30), but two bootstrap trees of depth 5 still disagree on 15% of the test examples. This is the **high variance**: the reason forests exist.
        """),
        C("""
        rng = np.random.default_rng(0)
        roots, predictions = [], []
        for _ in range(30):
            i = rng.integers(0, len(X_tr), len(X_tr))                          # bootstrap resample
            t = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X_tr[i], y_tr[i])
            roots.append(t.tree_.feature[0]); predictions.append(t.predict(X_te))
        predictions = np.array(predictions)
        print("feature chosen at the root in the 30 resamples:", np.bincount(roots, minlength=10))
        print("average fraction of test examples on which two bootstrap trees disagree:", round(np.mean([(predictions[i] != predictions[j]).mean() for i in range(10) for j in range(i + 1, 10)]), 3))
        """),
        M("""
        > **Common mistake.** Choosing `max_depth` by hand by looking at a single tree: it is an unstable model, and the same choice on another sample can give a different structure. Always validate with cross-validation and do not interpret the details of a single tree (precise thresholds, order of the questions) as "truth".

        ## Check your understanding

        """ + details(
            "What changes between `max_depth` and `min_samples_leaf` as a way of regularizing?",
            "`max_depth` limits the complexity uniformly (all the branches reach at most that depth); `min_samples_leaf` limits it adaptively: a branch in a dense region can go down a lot, one in a sparse region stops early. Usually `min_samples_leaf` is more effective because it ties the "
            "complexity to the amount of data.") + "\n\n" + details(
            "Why is cost-complexity pruning preferable to stopping the growth early?",
            "Because stopping early (pre-pruning) is shortsighted: a split that is of little use now can enable very useful ones later. Post-pruning builds the full tree and cuts the branches that are not worth it, looking at the overall effect of the branch.")),
    ]


def lesson_7_3() -> List[Cell]:
    """Cells of lesson 7.3."""
    return [
        M("""
        # Lesson 7.3 · Trees in practice

        Regression, feature importances and their pitfalls, categoricals and missing values, and the limits of a model with axis-parallel splits.

        **What you will learn**

        - Regression trees: piecewise constant predictions, and why they do not extrapolate.
        - Importances: the impurity-based one (`feature_importances_`) and the permutation one, with their biases.
        - What trees do not do well.
        """),
        _setup("""
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        """),
        M("""
        ## Regression: piecewise constant

        `DecisionTreeRegressor` splits by minimizing the squared error and predicts the **mean of the target** in the leaf. The result is a step function: it can follow nonlinear relationships, but with few examples per leaf it is noisy and **outside the training range it is flat**.
        """),
        C("""
        rng = np.random.default_rng(0)
        x = np.sort(rng.uniform(0, 6, 200)); y = np.sin(x) + 0.3 * x + rng.normal(0, 0.25, 200)
        grid = np.linspace(-1, 8, 400).reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.scatter(x, y, s=10, color="gray")
        for d in [2, 4, None]:
            ax.step(grid, DecisionTreeRegressor(max_depth=d, min_samples_leaf=5, random_state=0).fit(x.reshape(-1, 1), y).predict(grid), where="mid", label=f"max_depth={d}")
        ax.axvspan(6, 8, color="orange", alpha=0.1); ax.legend(); plt.show()
        """),
        M("""
        Beyond the value 6 the prediction is a **constant**: the tree does not continue the trend (like KNN, lesson 5.3, and unlike a linear regression). It is a serious limit with data that change over time (exercise C1).

        ## Feature importance

        Every split reduces the impurity: summing the reductions per feature (weighted by the number of examples) gives `feature_importances_`. It is free and convenient, but it has a flaw: it **favors features with many distinct values**, even useless ones, because they have more thresholds to try and more chances of finding a lucky one.
        """),
        C("""
        rng = np.random.default_rng(1)
        n = 800
        signal = rng.normal(size=(n, 2))
        y = (signal[:, 0] + 0.5 * signal[:, 1] + rng.normal(0, 1, n) > 0).astype(int)
        continuous_noise = rng.normal(size=n)                         # many distinct values, no information
        binary_noise = rng.integers(0, 2, n)                          # 2 values, no information
        X = np.c_[signal, continuous_noise, binary_noise]
        names = ["signal 1", "signal 2", "continuous noise", "binary noise"]
        Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.4, random_state=0)
        t = DecisionTreeClassifier(min_samples_leaf=3, random_state=0).fit(Xa, ya)
        perm = permutation_importance(t, Xb, yb, n_repeats=20, random_state=0)
        print(pd.DataFrame({"importance (impurity)": t.feature_importances_, "permutation (test)": perm.importances_mean}, index=names).round(3))
        """),
        M("""
        The impurity importance gives the **continuous noise** 24% (more than the second signal, 21%), while the permutation on the test set (which measures how much the accuracy gets worse by shuffling that column) brings it to about zero. Rule: to judge the usefulness of a feature use the permutation importance **on data not used
        for training** (exercise D2).

        ## Limits

        - **Axis-parallel splits:** a diagonal boundary (`x1 > x2`) needs a staircase; a logistic regression describes it with a single coefficient (exercise F2).
        - **Instability**: see above.
        - **No extrapolation** in regression problems.
        - **Missing values:** in scikit-learn 1.1 trees do **not** accept NaN (from 1.3 they do): you need an imputer (with a missingness indicator, as in module 4).
        - **Categoricals:** one-hot works but fragments; the ordinal encoding (`OrdinalEncoder`) lets the tree treat the categories as numbers and split them with a threshold: with many levels it is often better.

        > **Common mistake.** Reading `feature_importances_` as "the effect" of a variable: it is a measure of how much the tree used it, not of how much the variable matters in the world, and with correlated features the importance is split arbitrarily.

        ## Check your understanding

        """ + details(
            "Why does a tree not need scaling but logistic regression does?",
            "Because the tree only uses the order of the values of each feature (thresholds), which does not change under increasing monotonic transformations. Logistic regression multiplies the values by coefficients and sums them: the scale determines the relative weight (and the penalty), and it changes the model.") + "\n\n" + details(
            "When do you prefer a single tree to a forest?",
            "When explainability matters most (written rules, audits) or you need a very small model that is fast to implement. For performance a forest or a boosting almost always wins, at the price of losing the direct reading of the rules.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_how_a_tree_decides.ipynb", lesson_7_1),
                    ("02_complexity_and_pruning.ipynb", lesson_7_2),
                    ("03_trees_in_practice.ipynb", lesson_7_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
