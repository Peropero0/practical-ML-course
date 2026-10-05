"""Module 8 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="bagging_and_oob_from_scratch", kind="A", level=2, minutes=50, module=MODULE,
    title="Bagging from scratch: bootstrap, vote and out-of-bag error",
    goal="""
        Implement bagging of trees with numpy and scikit-learn (only `DecisionTreeClassifier` as a building block): bootstrap samples, majority vote and **out-of-bag error** computed by hand, checking that it matches the estimate of `BaggingClassifier`. Also measure that a bootstrap sample contains 63.2% of the distinct examples.
    """,
    dataset="""
        `make_classification(2000, 12 features, 5 informative, 2 redundant, flip_y=0.08)`, 60/40 split: `X_tr, y_tr, X_te, y_te`.
    """,
    purpose="""
        Take the mystery out of the random forest: it is a loop of bootstraps, trees and votes. Understand how the OOB error gives a validation estimate **without a separate set**, and why the vote of 100 trees beats each single tree by a wide margin.
    """,
    task=[
        "`bagging(X, y, B, seed)`: for `b` in `range(B)` draw bootstrap indices (`rng.integers(0, n, n)`), train `DecisionTreeClassifier(random_state=b)` and return the list `trees` and the list `indices` (the bootstrap indices).",
        "`vote_predict(trees, X)`: predictions by majority vote (mean of the tree predictions > 0.5).",
        "`oob(trees, indices, X, y)`: for each training example collect the votes **only of the trees whose bootstrap does not contain it**; return `(oob_predictions, covered)` where `covered` is the mask of the examples with at least one OOB tree. Compute `oob_acc` on the covered set.",
        "Save `single_acc` (mean test accuracy of the 100 trees), `vote_acc` (test accuracy of the vote), `distinct_share` (mean fraction of distinct examples in the bootstraps) and `covered_share` (fraction of examples with at least one OOB tree: with 100 trees it is almost 1). Compare `oob_acc` with `BaggingClassifier(DecisionTreeClassifier(), 100, oob_score=True, random_state=0).oob_score_`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import BaggingClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(2000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
    """,
    starter=[
        C("""
        def bagging(X, y, B, seed=0):
            ...


        def vote_predict(trees, X):
            ...


        def oob(trees, indices, X, y):
            ...
        """),
        C("""
        trees, indices = bagging(X_tr, y_tr, 100)
        single_acc = None
        vote_acc = None
        distinct_share = None
        """),
        C("""
        oob_predictions, covered = oob(trees, indices, X_tr, y_tr)
        oob_acc = None
        covered_share = None
        sklearn_acc = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True, random_state=0).fit(X_tr, y_tr).oob_score_
        """),
    ],
    checks="""
        check(len(trees) == 100 and len(indices) == 100 and all(len(i) == len(y_tr) for i in indices), "100 trees and 100 bootstrap samples of the same size as the training set", "rng.integers(0, n, n)")
        check(0.62 < distinct_share < 0.645, "a bootstrap contains on average 63.2% of distinct examples", "len(np.unique(i)) / n")
        check(vote_acc > single_acc + 0.06 and single_acc < 0.8 and vote_acc > 0.82, "the vote (over 0.82) beats the single tree (below 0.8) by over 6 points", "mean of the predictions > 0.5")
        check(covered_share > 0.99 and oob_predictions.shape == y_tr.shape, "with 100 trees almost all the examples have at least one OOB tree", "an example is covered if it is missing from the bootstrap of at least one tree")
        check(abs(oob_acc - vote_acc) < 0.05 and abs(oob_acc - sklearn_acc) < 0.02, "the OOB accuracy is close to the test one (within 5 points) and to the estimate of BaggingClassifier (within 2)", "only trees that have not seen the example")
        check(oob_acc < 0.999, "the OOB is not 1.0: it is not the estimate on the training set (which is almost perfect)", "an example never votes with the trees that have seen it")
    """,
    hint="""
        OOB votes: `votes = np.zeros(n); n_votes = np.zeros(n)`; for each tree `outside = np.setdiff1d(np.arange(n), indices[b])`; `votes[outside] += tree.predict(X[outside]); n_votes[outside] += 1`; prediction `votes / n_votes > 0.5` where `n_votes > 0`.
    """,
    solution=[
        C("""
        def bagging(X, y, B, seed=0):
            rng = np.random.default_rng(seed)
            trees, indices = [], []
            for b in range(B):
                i = rng.integers(0, len(X), len(X))                                  # bootstrap sample: with replacement
                trees.append(DecisionTreeClassifier(random_state=b).fit(X[i], y[i]))
                indices.append(i)
            return trees, indices


        def vote_predict(trees, X):
            return (np.mean([t.predict(X) for t in trees], axis=0) > 0.5).astype(int)


        def oob(trees, indices, X, y):
            n = len(X)
            votes, n_votes = np.zeros(n), np.zeros(n)
            for tree, i in zip(trees, indices):
                outside = np.setdiff1d(np.arange(n), i)                              # examples this tree has not seen
                votes[outside] += tree.predict(X[outside])
                n_votes[outside] += 1
            covered = n_votes > 0
            predictions = np.where(covered, votes / np.maximum(n_votes, 1) > 0.5, 0).astype(int)
            return predictions, covered
        """),
        C("""
        trees, indices = bagging(X_tr, y_tr, 100)
        single_acc = float(np.mean([t.score(X_te, y_te) for t in trees]))
        vote_acc = float((vote_predict(trees, X_te) == y_te).mean())
        distinct_share = float(np.mean([len(np.unique(i)) / len(y_tr) for i in indices]))
        print(f"mean accuracy of a single tree: {single_acc:.3f} | vote of 100 trees: {vote_acc:.3f} | share of distinct examples in a bootstrap: {distinct_share:.3f}")
        """),
        C("""
        oob_predictions, covered = oob(trees, indices, X_tr, y_tr)
        oob_acc = float((oob_predictions[covered] == y_tr[covered]).mean())
        covered_share = float(covered.mean())
        sklearn_acc = BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, oob_score=True, random_state=0).fit(X_tr, y_tr).oob_score_
        print(f"OOB accuracy (by hand): {oob_acc:.3f} | BaggingClassifier: {sklearn_acc:.3f} | test accuracy of the vote: {vote_acc:.3f} | covered examples: {covered_share:.1%}")
        """),
    ],
    learned="""
        - **Bagging is a three-line loop:** sample with replacement, tree, vote. A single tree without limits gets 0.775; the vote of 100 trees 0.845: 7 points gained **without changing the tree algorithm**, just by averaging its variance.
        - **A bootstrap contains 63.2% of distinct examples** ($1 - 1/e$): the remaining 36.8% is the out-of-bag set of that tree. With 100 trees the probability that an example is "inside" in all of them is tiny, so almost all of them receive at least one OOB prediction.
        - **The OOB error is a free honest estimate:** 0.86 against 0.85 on the test set, much closer to reality than the training accuracy (almost 1.0). Every OOB prediction uses only the trees that have never seen that example, so it is a piece of validation built into the training.
        - The random forest adds only one thing to bagging: at **every split** the tree can choose only among a random subset of features. It serves to decorrelate the trees (A2).
    """,
    pitfall="""
        Computing the accuracy of a forest or of a bagging on the **training set** and reporting it: it is almost 1.0 because every example has been seen by about 63% of the trees. The estimate to report is the OOB or the one on a test set.
    """,
    interview=(
        "What is the out-of-bag error and when can you trust it?",
        """
        Each tree is trained on a bootstrap that leaves out about 37% of the examples: predicting each example only with the trees that have not seen it gives me an honest prediction for the whole training set, and I compute its error, without a validation set. It is a good estimate when the examples are
        independent; with duplicate or correlated rows (same customer, time series) it is optimistic and you need a validation by group or by time.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="why_max_features_decorrelates_the_trees", kind="A", level=3, minutes=60, module=MODULE,
    title="Random forest from scratch: the correlation between trees and the variance of the average",
    goal="""
        Build a forest by hand (trees with `max_features`) and measure, for `max_features` = all, `sqrt` and 2, the **mean correlation between the tree predictions**, the variance of a single tree and of the average, and the accuracy. Check the formula $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$ and understand
        why a forest of **worse** trees (with limited features) does better than bagging.
    """,
    dataset="""
        The same dataset as A1 (`make_classification`, 12 features): 60/40 split; the predictions are class 1 probabilities (`predict_proba`) of the trees on the test set.
    """,
    purpose="""
        Understand the real reason the random forest exists: it is not a better tree, it is a set of trees that are **more different from each other**. Each tree with a small `max_features` is less accurate, but the average gains more than each tree loses.
    """,
    task=[
        "`forest(X, y, B, max_features, seed)`: `B` trees on bootstraps, each `DecisionTreeClassifier(max_features=max_features, random_state=b)` (with `max_features=None` = all the features). Return the list of trees.",
        "For `B = 60` and `max_features` in `[None, \"sqrt\", 2]` build the matrix `P` (trees × test examples) of the predicted probabilities of class 1 and compute: `rho` (mean correlation between different pairs of trees: mean of the **off-diagonal** coefficients of `np.corrcoef(P)`), "
        "`single_var` (`P.var(axis=0).mean()`: variance over the trees of a prediction, averaged over the examples), `single_acc` (mean accuracy of a tree: threshold 0.5 on the rows of `P`) and `forest_acc` (accuracy of the mean `P.mean(0)` with threshold 0.5).",
        "**The formula:** the variance of the average of `B` trees with variance `single_var` and correlation `rho` is `rho * single_var + (1 - rho) * single_var / B`: compute it and save it as `mean_var_formula`. Put everything in `table` (index `max_features` as a string: `\"None\"`, `\"sqrt\"`, `\"2\"`; columns `rho`, `single_acc`, `forest_acc`, `single_var`, `mean_var_formula`).",
        "Save `best` (the `max_features` with the highest `forest_acc`) and `residual_variance_rho = rho * single_var` for that `max_features`: the term that **does not drop** when adding trees.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(2000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.8, flip_y=0.08, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        B = 60
        options = [None, "sqrt", 2]
    """,
    starter=[
        C("""
        def forest(X, y, B, max_features, seed=0):
            ...
        """),
        C("""
        rows = {}
        for mf in options:
            trees = forest(X_tr, y_tr, B, mf)
            P = np.array([t.predict_proba(X_te)[:, 1] for t in trees])         # (trees, examples)
            rows[str(mf)] = ...
        table = pd.DataFrame(rows).T
        table.round(4)
        """),
        C("""
        best = None
        residual_variance_rho = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["rho", "single_acc", "forest_acc", "single_var", "mean_var_formula"] and list(table.index) == ["None", "sqrt", "2"], "table with three rows and five columns", "columns: rho, single_acc, forest_acc, single_var, mean_var_formula")
        check(table["rho"].is_monotonic_decreasing and table.loc["None", "rho"] > 0.45 and table.loc["2", "rho"] < 0.42, "the correlation between trees drops with max_features: from over 0.45 (all) to under 0.42 (2)", "mean of the off-diagonal coefficients of np.corrcoef(P)")
        check(table["single_acc"].is_monotonic_decreasing and table.loc["None", "single_acc"] - table.loc["2", "single_acc"] > 0.03, "a single tree with fewer features is worse (by over 3 points with max_features=2)", "mean accuracy of a tree")
        check(table.loc["sqrt", "forest_acc"] > table.loc["None", "forest_acc"] + 0.008 and table.loc["2", "forest_acc"] > table.loc["None", "forest_acc"], "but the forest with sqrt and with 2 features is better than bagging (max_features=all)", "forest_acc")
        check(best in ("sqrt", "2") and close(residual_variance_rho, table.loc[best, "rho"] * table.loc[best, "single_var"]), "the best is sqrt or 2; residual variance = rho * single variance", "the term that does not drop when adding trees")
        check((table["mean_var_formula"] < 0.6 * table["single_var"]).all() and table["mean_var_formula"].is_monotonic_decreasing, "the variance of the average is less than 60% of that of a tree (almost as much as rho), and drops with a smaller max_features", "rho * s2 + (1 - rho) * s2 / B")
    """,
    hint="""
        `rho = (np.corrcoef(P).sum() - len(P)) / (len(P) * (len(P) - 1))`; `single_var = P.var(axis=0).mean()` (variance over the trees, averaged over the examples); `mean_var_formula = rho * single_var + (1 - rho) * single_var / B`. Accuracy: `((P > 0.5) == y_te).mean(axis=1).mean()` for the single tree, `((P.mean(0) > 0.5) == y_te).mean()` for the forest.
    """,
    solution=[
        C("""
        def forest(X, y, B, max_features, seed=0):
            rng = np.random.default_rng(seed)
            trees = []
            for b in range(B):
                i = rng.integers(0, len(X), len(X))
                trees.append(DecisionTreeClassifier(max_features=max_features, random_state=b).fit(X[i], y[i]))
            return trees
        """),
        C("""
        rows = {}
        for mf in options:
            trees = forest(X_tr, y_tr, B, mf)
            P = np.array([t.predict_proba(X_te)[:, 1] for t in trees])
            rho = (np.corrcoef(P).sum() - len(P)) / (len(P) * (len(P) - 1))                     # mean correlation between different pairs of trees
            single_var = P.var(axis=0).mean()                                                    # variance (over the trees) of a prediction, averaged over the examples
            rows[str(mf)] = {"rho": rho, "single_acc": ((P > 0.5) == y_te).mean(axis=1).mean(), "forest_acc": ((P.mean(0) > 0.5) == y_te).mean(),
                             "single_var": single_var, "mean_var_formula": rho * single_var + (1 - rho) * single_var / B}
        table = pd.DataFrame(rows).T
        table.round(4)
        """),
        C("""
        best = table["forest_acc"].idxmax()
        residual_variance_rho = table.loc[best, "rho"] * table.loc[best, "single_var"]
        print(f"best: max_features={best} | variance that does not drop when adding trees (rho * sigma^2): {residual_variance_rho:.4f} | variance of a tree: {table.loc[best, 'single_var']:.4f}")
        """),
    ],
    learned="""
        - **More trees are not enough when they are correlated.** The variance of the average of `B` trees is $\\rho\\sigma^2 + (1-\\rho)\\sigma^2/B$: adding trees eliminates only the second term. With all the features (bagging) $\\rho = 0.53$: the variance will not drop below 53% of that of a tree even with infinite trees.
        - **A small `max_features` reduces $\\rho$** (0.53 → 0.43 with `sqrt` → 0.40 with 2 features): the trees do not all choose the dominant feature at the root, and so they make different errors.
        - **The price is a worse tree:** the accuracy of a single tree drops (0.77 → 0.75 → 0.74), but the average gains more than each tree loses: 0.843 (bagging) against 0.865 (`sqrt`). It is the bias-variance trade-off in a surprising form: **weaker trees, stronger ensemble.**
        - With a `max_features` that is too small the gain ends (the trees become too weak): that is why it is a hyperparameter to choose, on a small grid (`sqrt`, `log2`, fractions, all).
    """,
    pitfall="""
        Choosing `max_features` by looking at the accuracy of **one tree**: you would only see its negative side. You always evaluate the whole forest (OOB or cross-validation).
    """,
    interview=(
        "Why does a random forest work better than bagging?",
        """
        Because at every split it limits the candidate features: the trees end up less correlated (they do not all choose the same dominant feature) and the variance of the average is rho·σ² + (1−rho)·σ²/B, where extra trees alone do not reduce the correlation term. Each tree is a little worse (more bias),
        but the average of less correlated trees gains much more. It is the trade-off that `max_features` controls.
        """),
)

EXERCISES = [A1, A2]
