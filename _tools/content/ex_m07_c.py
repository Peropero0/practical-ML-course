"""Module 7 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="the_tree_does_not_extrapolate_the_trend", kind="C", level=2, minutes=45, module=MODULE,
    title="Breaking the tree with a trend: flat predictions outside the training range",
    goal="""
        Build a series with trend and seasonality, train a linear regression and a regression tree on the first 200 months and predict the next 100: the tree is wrong by almost five times as much as the line, because outside the range of `t` it repeats the last value. Then fix it with a **hybrid** model
        (linear trend + tree on the residuals).
    """,
    dataset="""
        `series()` (given): `y = 2 + 0.05 t + 3 sin(2πt/12) + noise` for `t` = 0..299, with features `t`, `sin`, `cos` of the season. Training: `t < 200`; test (future): `t >= 200`.
    """,
    purpose="""
        Understand a structural limit: a regression tree predicts **leaf means** of values seen in training, so it cannot produce values beyond the observed range nor continue a trend. It is the problem of tree models (forests and boosting included) with nonstationary series.
    """,
    task=[
        "Train `LinearRegression` and `DecisionTreeRegressor(min_samples_leaf=5, random_state=0)` on the training set and save `train_rmse` (dictionary with the training error of both) and `future_rmse` (dictionary with the test error, that is in the future).",
        "Save `future_tree_predictions`: the **distinct** values predicted by the tree on the future considering only the trend part (use only the feature `t`: train a tree on `t` and predict `t` from 200 to 299): how many are there? Save `n_distinct_values` and `last_constant_value`.",
        "**Hybrid:** train the line on the training set, then a tree on the **residuals** (`y − line`) with the same three features and predict the future as `line + tree`. Save `hybrid_rmse`.",
        "Save `error_ratio = future_rmse['tree'] / future_rmse['linear']`.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor


        def series():
            rng = np.random.default_rng(0)
            t = np.arange(300)
            y = 2 + 0.05 * t + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.7, 300)
            return t, np.c_[t, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)], y


        t, F, y = series()
        tr, te = t < 200, t >= 200
        rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    """,
    starter=[
        C("""
        train_rmse = {}
        future_rmse = {}
        """),
        C("""
        n_distinct_values = None
        last_constant_value = None
        """),
        C("""
        hybrid_rmse = None
        error_ratio = None
        """),
    ],
    checks="""
        check(set(train_rmse) == {"linear", "tree"} and abs(train_rmse["tree"] - train_rmse["linear"]) < 0.1, "on the training set the two models are on par (within 0.1): nothing hints at the problem", "compare the training errors")
        check(future_rmse["linear"] < 1.0 and future_rmse["tree"] > 3, "in the future the line stays below 1.0; the tree above 3", "test error (t >= 200)")
        check(n_distinct_values is not None and n_distinct_values <= 3 and abs(last_constant_value - y[tr][-30:].mean()) < 3, "a tree on t predicts very few distinct values in the future (at most 3), close to the level of the last months", "the future falls entirely in the last leaf")
        check(hybrid_rmse is not None and hybrid_rmse < 1.0 and hybrid_rmse < future_rmse["tree"] / 3, "the hybrid in the future has RMSE below 1.0 (over 3 times better than the tree)", "line + tree on the residuals")
        check(error_ratio > 4, "the error of the tree is over 4 times that of the line", "future_rmse['tree'] / future_rmse['linear']")
    """,
    hint="""
        `lin = LinearRegression().fit(F[tr], y[tr])`; `tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(F[tr], y[tr])`. Trend only: `DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(t[tr].reshape(-1, 1), y[tr]).predict(t[te].reshape(-1, 1))`; `np.unique(...)`.
        Hybrid: `res = y[tr] - lin.predict(F[tr]); res_tree = DecisionTreeRegressor(...).fit(F[tr], res)`; prediction `lin.predict(F[te]) + res_tree.predict(F[te])`.
    """,
    solution=[
        C("""
        lin = LinearRegression().fit(F[tr], y[tr])
        tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(F[tr], y[tr])
        train_rmse = {"linear": rmse(lin.predict(F[tr]), y[tr]), "tree": rmse(tree.predict(F[tr]), y[tr])}
        future_rmse = {"linear": rmse(lin.predict(F[te]), y[te]), "tree": rmse(tree.predict(F[te]), y[te])}
        print("RMSE on the training set:", {k: round(v, 2) for k, v in train_rmse.items()}, "| in the future:", {k: round(v, 2) for k, v in future_rmse.items()})
        """),
        C("""
        t_only = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(t[tr].reshape(-1, 1), y[tr])
        future_tree_predictions = t_only.predict(t[te].reshape(-1, 1))
        n_distinct_values = int(np.unique(future_tree_predictions).size)
        last_constant_value = float(future_tree_predictions[-1])
        print(f"distinct values predicted by the tree on t for the 100 future months: {n_distinct_values} (value {last_constant_value:.2f}) | mean of the last 30 training months: {y[tr][-30:].mean():.2f}")
        """),
        C("""
        residuals = y[tr] - lin.predict(F[tr])
        res_tree = DecisionTreeRegressor(min_samples_leaf=5, random_state=0).fit(F[tr], residuals)
        hybrid_rmse = rmse(lin.predict(F[te]) + res_tree.predict(F[te]), y[te])
        error_ratio = future_rmse["tree"] / future_rmse["linear"]
        print(f"RMSE in the future: line {future_rmse['linear']:.2f} | tree {future_rmse['tree']:.2f} ({error_ratio:.1f} times) | hybrid {hybrid_rmse:.2f}")
        """),
    ],
    learned="""
        - **A tree does not extrapolate:** on the training set it is on par with the line (RMSE 0.67 for both: no hint of the problem), but in the future its error is almost five times larger (3.7 against 0.78). With `t` alone as a feature it predicts **a single value** (the mean of the last leaf) for all the 100 future months.
        - The same holds for **random forest and gradient boosting**: they are sums and averages of trees, so they are constant outside the range seen. With data that have a trend (sales, prices, sensors with drift) they are not reliable "as they are".
        - **The remedy is to model the trend separately:** a linear trend (or a differencing) and a tree on the **residuals** for the nonlinear part (seasonality, interactions). The hybrid has error 0.74 in the future: almost that of the line, with the flexibility of the tree on the stationary part.
        - Alternatives: transform the target (differences, ratios with respect to a previous period) so that it is stationary, or use features that carry the trend (`t` as a feature is not enough for a tree: it is the target transformations that help it).
    """,
    pitfall="""
        Validating a tree model with a random K-fold on data with a trend: every validation point has neighbors in training on both sides and the tree interpolates well. The flaw is discovered only in production, in the future (M3, C1: the splitter must mirror the use).
    """,
    interview=(
        "Why is a random forest not suited to predicting a series with a trend?",
        """
        Because trees predict means of values seen in training: outside the range of the data the prediction is flat, so it cannot continue a trend. You remove the trend (linear regression, differencing), model the rest with the tree and put it back together; or you use time series models.
        And you validate with a temporal split, which shows the problem right away.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="tree_instability_and_the_vote", kind="C", level=3, minutes=55, module=MODULE,
    title="Tree instability: 30 resamples, 30 different trees",
    goal="""
        Measure the **variance** of trees: resampling with the bootstrap, trees with the same parameters disagree on a large fraction of the examples. See how depth and minimum leaf size reduce the disagreement, and that the **majority vote** of 30 unlimited trees is much more accurate
        than the single tree (the idea of forests).
    """,
    dataset="""
        `make_classification(1500, 10 features, 4 informative, flip_y=0.1)`, 60/40 split (`X_tr, y_tr, X_te, y_te`). For each configuration 30 trees are trained on bootstrap samples of the training set.
    """,
    purpose="""
        Turn "trees are unstable" into numbers: mean accuracy and variability of the single trees, fraction of examples on which two bootstrap trees disagree, and the variance reduction obtained by averaging. It is the direct motivation for the next module.
    """,
    task=[
        "For `configs = {\"no limits\": {}, \"max_depth=5\": {\"max_depth\": 5}, \"max_depth=3\": {\"max_depth\": 3}, \"min_samples_leaf=20\": {\"min_samples_leaf\": 20}, \"min_samples_leaf=50\": {\"min_samples_leaf\": 50}}` train **30 trees** (`DecisionTreeClassifier(random_state=0, **params)`) on bootstrap samples "
        "(`rng = np.random.default_rng(0)`, indices `rng.integers(0, n, n)`) and collect the predictions on the test set.",
        "Build `table` (rows: configurations; columns `mean_acc` (mean accuracy of the 30 trees), `acc_std` (standard deviation), `disagreement` (mean, over the pairs of the first 15 trees, of the fraction of test examples on which two trees give a different prediction), `mean_leaves`).",
        "For the `no limits` configuration compute the **majority vote** of the 30 trees: `vote_acc` (accuracy of the vote) and `vote_gain = vote_acc − mean_acc` of the same configuration.",
        "Save `disagreement_reduction`: `disagreement` of `no limits` divided by that of `max_depth=3`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        X, y = make_classification(1500, n_features=10, n_informative=4, n_redundant=0, class_sep=0.8, flip_y=0.1, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        configs = {"no limits": {}, "max_depth=5": {"max_depth": 5}, "max_depth=3": {"max_depth": 3},
                   "min_samples_leaf=20": {"min_samples_leaf": 20}, "min_samples_leaf=50": {"min_samples_leaf": 50}}
    """,
    starter=[
        C("""
        rng = np.random.default_rng(0)
        predictions, leaves = {}, {}
        for name, params in configs.items():
            ...
        """),
        C("""
        table = pd.DataFrame(index=list(configs), columns=["mean_acc", "acc_std", "disagreement", "mean_leaves"], dtype=float)
        table
        """),
        C("""
        vote_acc = None
        vote_gain = None
        disagreement_reduction = None
        """),
    ],
    checks="""
        check(set(predictions) == set(configs) and all(np.asarray(p).shape == (30, len(y_te)) for p in predictions.values()), "30 predictions on the test set for each configuration", "one row per tree, one column per test example")
        check(table.notna().all().all() and table.loc["no limits", "mean_acc"] < 0.76 and table.loc["max_depth=3", "mean_acc"] > 0.75, "the single trees without limits are below 0.76; with max_depth=3 above 0.75", "mean accuracy of the 30 trees")
        check(table.loc["no limits", "disagreement"] > 0.2 and table.loc["max_depth=3", "disagreement"] < 0.13 and table.loc["min_samples_leaf=50", "disagreement"] < 0.13, "without limits two trees disagree on over 20% of the examples; with max_depth=3 or leaf 50 on less than 13%", "mean over the pairs of trees")
        check(table.loc["no limits", "mean_leaves"] > 60 and table.loc["max_depth=3", "mean_leaves"] <= 8, "without limits over 60 leaves; max_depth=3 at most 8", "get_n_leaves()")
        check(vote_acc is not None and vote_acc > 0.78 and vote_gain > 0.04, "the vote of 30 trees without limits goes over 0.78: over 4 points above the single tree", "mean of the predictions > 0.5")
        check(disagreement_reduction > 1.8, "the disagreement without limits is over 1.8 times that of max_depth=3", "compare the two rows")
    """,
    hint="""
        Loop: `P = []; F = []; for _ in range(30): i = rng.integers(0, len(X_tr), len(X_tr)); t = DecisionTreeClassifier(random_state=0, **params).fit(X_tr[i], y_tr[i]); P.append(t.predict(X_te)); F.append(t.get_n_leaves())`. Disagreement: `np.mean([(P[i] != P[j]).mean() for i in range(15) for j in range(i + 1, 15)])`.
        Vote: `(np.mean(P, axis=0) > 0.5).astype(int)`.
    """,
    solution=[
        C("""
        rng = np.random.default_rng(0)
        predictions, leaves = {}, {}
        for name, params in configs.items():
            P, F = [], []
            for _ in range(30):
                i = rng.integers(0, len(X_tr), len(X_tr))                                # bootstrap sample of the training set
                tree = DecisionTreeClassifier(random_state=0, **params).fit(X_tr[i], y_tr[i])
                P.append(tree.predict(X_te)); F.append(tree.get_n_leaves())
            predictions[name], leaves[name] = np.array(P), np.array(F)
        print({k: v.shape for k, v in predictions.items()})
        """),
        C("""
        table = pd.DataFrame(index=list(configs), columns=["mean_acc", "acc_std", "disagreement", "mean_leaves"], dtype=float)
        for name in configs:
            P = predictions[name]
            accuracies = (P == y_te).mean(1)
            table.loc[name] = [accuracies.mean(), accuracies.std(),
                               np.mean([(P[i] != P[j]).mean() for i in range(15) for j in range(i + 1, 15)]), leaves[name].mean()]
        table.round(3)
        """),
        C("""
        vote = (predictions["no limits"].mean(0) > 0.5).astype(int)
        vote_acc = float((vote == y_te).mean())
        vote_gain = vote_acc - table.loc["no limits", "mean_acc"]
        disagreement_reduction = table.loc["no limits", "disagreement"] / table.loc["max_depth=3", "disagreement"]
        print(f"accuracy of the single tree without limits: {table.loc['no limits', 'mean_acc']:.3f} | vote of 30 trees: {vote_acc:.3f} (+{vote_gain:.3f})")
        print(f"disagreement between trees without limits / with max_depth=3: {disagreement_reduction:.1f} times")
        """),
    ],
    learned="""
        - **Two trees without limits trained on bootstrap samples of the same training set disagree on about a quarter of the examples** (24%): with the same parameters, on the same problem. The root may stay almost always the same (it is a dominant feature), but the deeper branches follow the noise of the sample.
        - **Regularizing reduces the disagreement by more than two times** (11% with `max_depth=3`, 8% with `min_samples_leaf=50`) and raises the mean accuracy (0.74 → 0.77): the same thing from the point of view of the variance. The price is a less flexible model (more bias), and here it is a price worth paying.
        - **The majority vote of 30 trees without limits beats the accuracy of every single tree** and also of the regularized trees (0.80 against 0.77): averaging removes the "random" errors of each tree and the systematic ones remain. It is exactly the recipe of **bagging** and of random forests (module 8): deep trees (low bias, high variance) with the variance knocked down by the average.
        - Note the cost: 30 trees with over 80 leaves each lose the readability of the single tree. Interpretability and accuracy are in tension here.
    """,
    pitfall="""
        Interpreting the details of a single deep tree (a threshold, the order of the branches) as the structure of the phenomenon: changing a hundred examples changes a quarter of the predictions. If you need stability, regularize or average many trees.
    """,
    interview=(
        "Why does a tree have high variance and how do you reduce it?",
        """
        Because every split depends on few examples and the choices cascade: a small change in the data moves a threshold and changes the whole subtree. You reduce it by regularizing (minimum leaf size, depth, pruning) or by averaging many trees trained on different samples (bagging,
        random forest): the average of estimators that are little biased but very variable has a much smaller variance, without increasing the bias.
        """),
)

EXERCISES = [C1, C2]
