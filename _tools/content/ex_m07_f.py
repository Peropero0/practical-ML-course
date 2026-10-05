"""Module 7 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="gini_entropy_and_random_splitter", kind="F", level=2, minutes=45, module=MODULE,
    title="Gini or entropy? Best split or random split?",
    goal="""
        Systematically compare the impurity criterion (`gini`, `entropy`) and the `splitter` (`best`, `random`) of a tree on three datasets (wines, tumors, digits) with a repeated cross-validation, measuring accuracy, number of leaves and training time, and establish how much these choices matter compared with the measurement noise
        of a single tree.
    """,
    dataset="""
        `load_wine`, `load_breast_cancer`, `load_digits`. Model: `DecisionTreeClassifier(min_samples_leaf=3, random_state=0)` with the four combinations; validation `RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)`, accuracy.
    """,
    purpose="""
        Discover that the criterion (Gini or entropy) is a minor detail compared with regularization, and that the uncertainty of a single tree (the variability between folds) is of the same size as the differences between the options: do not draw conclusions from a single score.
    """,
    task=[
        "For each dataset and each combination (`gini/best`, `entropy/best`, `gini/random`, `entropy/random`) compute with `cross_validate` (with `return_train_score=False`) `acc` (mean), `sd` (standard deviation across the 20 folds) and `t_fit` (mean fit time in seconds). Save `table` with index `(dataset, combination)` and the three columns.",
        "For each combination train a tree on the whole dataset and save `leaves` (`pd.Series` with the same index).",
        "Save `criterion_diff` (dictionary dataset → `acc(entropy/best) − acc(gini/best)`) and `max_criterion_diff` (maximum absolute value).",
        "Save `random_has_more_leaves` (True if for all the datasets `gini/random` has more leaves than `gini/best`) and `typical_sd` (median of the `sd` of `table`). Answer with `criterion_matters = False` if `max_criterion_diff` is smaller than `2 * typical_sd`.",
    ],
    setup="""
        import time
        from sklearn.datasets import load_breast_cancer, load_digits, load_wine
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
        from sklearn.tree import DecisionTreeClassifier

        datasets = {"wine": load_wine(return_X_y=True), "tumors": load_breast_cancer(return_X_y=True), "digits": load_digits(return_X_y=True)}
        combinations = {"gini/best": ("gini", "best"), "entropy/best": ("entropy", "best"), "gini/random": ("gini", "random"), "entropy/random": ("entropy", "random")}
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=0)
    """,
    starter=[
        C("""
        rows = {}
        leaves = {}
        for name, (X, y) in datasets.items():
            for comb, (criterion, splitter) in combinations.items():
                model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_leaf=3, random_state=0)
                ...
        table = pd.DataFrame(rows).T
        leaves = pd.Series(leaves)
        table.round(3)
        """),
        C("""
        criterion_diff = {}
        max_criterion_diff = None
        random_has_more_leaves = None
        typical_sd = None
        criterion_matters = None
        """),
    ],
    checks="""
        check(table.shape == (12, 3) and list(table.columns) == ["acc", "sd", "t_fit"] and len(leaves) == 12, "12 rows (3 datasets x 4 combinations) and three columns; 12 leaf counts", "acc, sd, t_fit")
        check(set(criterion_diff) == {"wine", "tumors", "digits"} and close(max_criterion_diff, max(abs(v) for v in criterion_diff.values())) and close(criterion_diff["wine"], table.loc[("wine", "entropy/best"), "acc"] - table.loc[("wine", "gini/best"), "acc"]), "differences between criteria computed", "acc(entropy/best) - acc(gini/best)")
        check(max_criterion_diff < 0.04, "the difference between the two criteria is always below 4 points", "compare the acc values")
        check(random_has_more_leaves is True and (leaves[[("wine", "gini/random"), ("tumors", "gini/random"), ("digits", "gini/random")]].to_numpy() > leaves[[("wine", "gini/best"), ("tumors", "gini/best"), ("digits", "gini/best")]].to_numpy()).all(), "the random splitter gives more leaves, on all the datasets", "leaves of gini/random against gini/best")
        check(0.015 < typical_sd < 0.06 and criterion_matters is False, "the standard deviation across folds (2-6 points) is of the same order as the differences: the criterion does not matter in a distinguishable way", "median of the sd")
    """,
    hint="""
        `r = cross_validate(model, X, y, cv=cv)`; `acc = r["test_score"].mean()`, `sd = r["test_score"].std()`, `t_fit = r["fit_time"].mean()`. Leaves: `model.fit(X, y).get_n_leaves()`. Key: `leaves[(name, comb)] = ...` and `rows[(name, comb)] = {...}`.
    """,
    solution=[
        C("""
        rows, leaves = {}, {}
        for name, (X, y) in datasets.items():
            for comb, (criterion, splitter) in combinations.items():
                model = DecisionTreeClassifier(criterion=criterion, splitter=splitter, min_samples_leaf=3, random_state=0)
                r = cross_validate(model, X, y, cv=cv)
                rows[(name, comb)] = {"acc": r["test_score"].mean(), "sd": r["test_score"].std(), "t_fit": r["fit_time"].mean()}
                leaves[(name, comb)] = model.fit(X, y).get_n_leaves()
        table = pd.DataFrame(rows).T
        leaves = pd.Series(leaves)
        table.assign(leaves=leaves).round(3)
        """),
        C("""
        criterion_diff = {name: table.loc[(name, "entropy/best"), "acc"] - table.loc[(name, "gini/best"), "acc"] for name in datasets}
        max_criterion_diff = max(abs(v) for v in criterion_diff.values())
        random_has_more_leaves = bool(all(leaves[(name, "gini/random")] > leaves[(name, "gini/best")] for name in datasets))
        typical_sd = float(table["sd"].median())
        criterion_matters = bool(max_criterion_diff >= 2 * typical_sd)
        print("acc(entropy) - acc(gini):", {k: round(v, 3) for k, v in criterion_diff.items()}, f"| maximum {max_criterion_diff:.3f}")
        print(f"typical standard deviation across folds: {typical_sd:.3f} | the criterion matters in a distinguishable way: {criterion_matters} | random has more leaves everywhere: {random_has_more_leaves}")
        """),
    ],
    learned="""
        - **Gini against entropy: differences of 1-3 points, of the same order as the noise.** The standard deviation across the folds of a single tree is 2-5 points: a difference of one point cannot be distinguished. Both criteria are concave functions of the impurity with a maximum at 0.5: they almost always choose the same split. Neither is "better": try both if you have time and do not
          expect miracles.
        - **`splitter="random"` does not worsen the accuracy** but produces **bigger** trees (about 1.3-1.7 times the leaves) because it picks a random threshold instead of the best one for each feature. It costs less per node (no exhaustive search) and it is used (with other tricks) in "extremely randomized trees".
        - **The real point is another:** regularization (minimum leaf size, depth, pruning) moves the accuracy by 10 points (B1), the criterion by 1-3. Optimizing the criterion before the regularization is a way of wasting time. And, with a cross-validation of only 5 folds, differences of that kind would sometimes go one way and sometimes the other.
        - **The time** is almost the same between Gini and entropy (the entropy needs one more logarithm), and with `random` it is lower on datasets with many features (digits).
    """,
    pitfall="""
        Reporting "entropy is better than Gini" from a single cross-validation with a difference of one point. A tree is an unstable model: without repeating the validation and without looking at the variability between folds there is no way to tell a real effect from noise.
    """,
    interview=(
        "Which criterion do you use to split a tree: Gini or entropy?",
        """
        In practice they give almost identical trees: I pick the default (Gini, slightly faster) and focus on what really matters, which is the regularization (minimum leaf size, depth, pruning) chosen in cross-validation. If I want to compare the two criteria I do it with a repeated validation and looking at the variability.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="boundary_shapes_and_invariance", kind="F", level=3, minutes=55, module=MODULE,
    title="Tree, logistic and KNN on four boundary shapes (and after a monotonic transformation)",
    goal="""
        Compare a tree (with `max_depth` and `min_samples_leaf` chosen in cross-validation), a logistic regression and a KNN on four boundary shapes in 2D (axis-parallel steps, XOR, diagonal, circle) and repeat the comparison after applying a **monotonic transformation** (`exp`) to the data: the tree does not change, the other two do.
    """,
    dataset="""
        `shapes()` (given): 1500 uniform points in [−3, 3]² with four labelings (`steps`, `XOR`, `diagonal`, `circle`) and 5% of swapped labels. 60/40 split (`random_state=0`). Models: tree with `GridSearchCV` (depth and minimum leaf size), `LogisticRegression` and `KNeighborsClassifier(15)` with `StandardScaler`.
    """,
    purpose="""
        Learn to choose the model from the **geometry of the boundary**: trees excel when the boundary is made of thresholds on single features (steps, XOR) and struggle with diagonals; the logistic is the opposite. And see the invariance of trees to monotonic transformations, which makes them independent of the scale and of the distribution of the features.
    """,
    task=[
        "For each shape train the three models (the tree with `GridSearchCV(DecisionTreeClassifier(random_state=0), {\"max_depth\": [2, 3, 4, 5, 6, 8, None], \"min_samples_leaf\": [1, 5, 20]}, cv=cv)` with `cv = StratifiedKFold(5, shuffle=True, random_state=0)`) and save the test accuracy in `table` (index: shapes; columns `tree`, `logistic`, `knn`).",
        "Save `depths` (dictionary shape → `max_depth` chosen by the search for the tree).",
        "Repeat after the `exp` transformation applied to **all** the features (training and test): `exp_table` (same rows and columns; the tree with the hyperparameters already chosen).",
        "Save `change` (`pd.DataFrame` with `exp_table − table`) and `max_tree_change` (maximum absolute value of the `tree` column), `max_other_change` (maximum absolute value of the other two columns).",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier


        def shapes():
            rng = np.random.default_rng(0)
            n = 1500
            X = rng.uniform(-3, 3, (n, 2))
            labels = {
                "steps": ((X[:, 0] > 0.5) & (X[:, 1] < 1.0)) | ((X[:, 0] < -1.5) & (X[:, 1] > -0.5)),
                "XOR": (X[:, 0] > 0) ^ (X[:, 1] > 0),
                "diagonal": X[:, 0] + X[:, 1] > 0,
                "circle": (X ** 2).sum(1) < 4,
            }
            result = {}
            for name, y in labels.items():
                y = y.astype(int)
                y = np.where(rng.random(n) < 0.05, 1 - y, y)                    # 5% of wrong labels
                result[name] = (X, y)
            return result


        data = shapes()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        grid = {"max_depth": [2, 3, 4, 5, 6, 8, None], "min_samples_leaf": [1, 5, 20]}
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=list(data), columns=["tree", "logistic", "knn"], dtype=float)
        depths = {}
        best = {}
        for name, (X, y) in data.items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
            ...
        table.round(3)
        """),
        C("""
        exp_table = pd.DataFrame(index=list(data), columns=["tree", "logistic", "knn"], dtype=float)
        for name, (X, y) in data.items():
            X_tr, X_te, y_tr, y_te = train_test_split(np.exp(X), y, test_size=0.4, random_state=0)
            ...
        exp_table.round(3)
        """),
        C("""
        change = None
        max_tree_change = None
        max_other_change = None
        """),
    ],
    checks="""
        check(table.notna().all().all() and set(depths) == set(data), "complete table and depth for each shape", "one row per shape")
        check((table.loc[["steps", "XOR", "circle"], "tree"] > table.loc[["steps", "XOR", "circle"], "logistic"] + 0.15).all() and table.loc["XOR", "logistic"] < 0.6, "the tree beats the logistic by over 15 points on steps, XOR and circle; on the XOR the logistic is at chance", "compare the columns")
        check(table.loc["diagonal", "logistic"] > table.loc["diagonal", "tree"] + 0.01 and table.loc["diagonal", "tree"] > 0.88, "on the diagonal the logistic beats the tree (which still goes over 0.88 with some depth)", "the diagonal is not parallel to the axes")
        check(depths["steps"] <= 5 and depths["diagonal"] is not None and depths["diagonal"] >= 4 and depths["circle"] >= depths["steps"], "chosen depth: the circle needs more depth than the steps; the diagonal needs at least 4", "best_params_['max_depth']")
        check(exp_table.notna().all().all(), "complete table with the exp transformation", "np.exp(X) for training and test")
        check(max_tree_change < 0.02 and max_other_change > 0.04, "after exp the tree changes by less than 2 points; logistic or KNN change by over 4 points", "change = exp_table - table")
        check(change.loc["diagonal", "logistic"] < -0.05, "the logistic on the diagonal gets worse by over 5 points with exp (the boundary is no longer a line)", "change['logistic']")
    """,
    hint="""
        For each shape: `search = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=cv).fit(X_tr, y_tr)`; `table.loc[name, "tree"] = search.score(X_te, y_te)`; `depths[name] = search.best_params_["max_depth"]`; `best[name] = search.best_params_`.
        With `exp`: `DecisionTreeClassifier(random_state=0, **best[name]).fit(...)`. Logistic and KNN: `make_pipeline(StandardScaler(), LogisticRegression())`, `make_pipeline(StandardScaler(), KNeighborsClassifier(15))`.
    """,
    solution=[
        C("""
        def evaluate(X_tr, X_te, y_tr, y_te, tree_params):
            tree = DecisionTreeClassifier(random_state=0, **tree_params).fit(X_tr, y_tr)
            logistic = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
            knn = make_pipeline(StandardScaler(), KNeighborsClassifier(15)).fit(X_tr, y_tr)
            return [tree.score(X_te, y_te), logistic.score(X_te, y_te), knn.score(X_te, y_te)]


        table = pd.DataFrame(index=list(data), columns=["tree", "logistic", "knn"], dtype=float)
        depths, best = {}, {}
        for name, (X, y) in data.items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
            search = GridSearchCV(DecisionTreeClassifier(random_state=0), grid, cv=cv).fit(X_tr, y_tr)
            best[name], depths[name] = search.best_params_, search.best_params_["max_depth"]
            table.loc[name] = evaluate(X_tr, X_te, y_tr, y_te, search.best_params_)
        print("chosen depths:", depths)
        table.round(3)
        """),
        C("""
        exp_table = pd.DataFrame(index=list(data), columns=["tree", "logistic", "knn"], dtype=float)
        for name, (X, y) in data.items():
            X_tr, X_te, y_tr, y_te = train_test_split(np.exp(X), y, test_size=0.4, random_state=0)         # monotonic transformation of all the features
            exp_table.loc[name] = evaluate(X_tr, X_te, y_tr, y_te, best[name])
        exp_table.round(3)
        """),
        C("""
        change = exp_table - table
        max_tree_change = float(change["tree"].abs().max())
        max_other_change = float(change[["logistic", "knn"]].abs().max().max())
        print(change.round(3))
        print(f"maximum change: tree {max_tree_change:.3f} | logistic and KNN {max_other_change:.3f}")
        """),
    ],
    learned="""
        - **The boundary decides the model.** With boundaries made of thresholds parallel to the axes (steps, XOR) the tree reaches 0.94-0.95 while the logistic is at 0.72 (steps) or at chance (XOR: no line separates the four quadrants). With the circle the tree gets 0.91 (it needs a staircase: depth 8 against 4-5 in the other cases).
          With the diagonal the logistic (0.94) beats the tree (0.91): describing an oblique line with right-angle thresholds takes many splits.
        - **KNN is the generalist in 2D:** 0.93-0.94 on all the shapes, without having to guess the geometry (but remember the curse of dimensionality: in many dimensions it is no longer so, M5).
        - **Invariance of trees:** after `exp` the tree changes by less than half a point (0.2 points on the steps shape: the few differences come from thresholds halfway between values that are now spaced differently), while logistic and KNN change by up to 11 points (the diagonal is no longer a line; the distances get distorted). Trees depend only on the **order** of the values:
          no scaling, no logarithms "to help", no worries about scale outliers.
        - **A practical tip:** with tabular data the relationships are often "threshold-like" (age above 18, income above a value) and with interactions: it is the ground of trees (and of tree ensembles, modules 8 and 9). With physical measurements and smooth, diagonal relationships, linear models and networks are more efficient.
    """,
    pitfall="""
        Concluding "trees are better than linear models" (or the opposite) from a single dataset. The answer depends on the geometry of the problem; and the first thing to do with new data is to try both.
    """,
    interview=(
        "When does a tree work better than a logistic regression?",
        """
        When the relationship with the target is made of thresholds and interactions (effects that exist only for certain ranges of several variables) rather than smooth linear combinations: XOR, "if A and not B" rules, step effects. With oblique boundaries or many weakly informative features the regularized logistic is often better
        and more stable. Another advantage of trees is the invariance to monotonic transformations of the features: no scaling. I compare them in cross-validation on the same metric.
        """),
)

EXERCISES = [F1, F2]
