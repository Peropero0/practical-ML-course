"""Module 7 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="classification_tree_from_scratch", kind="A", level=3, minutes=60, module=MODULE,
    title="A classification tree from scratch: Gini, best split, recursion",
    goal="""
        Implement a classification tree with numpy: Gini impurity, search for the **best split** over all the features and thresholds (with cumulative sums, without recounting at every threshold), recursive construction with `max_depth` and `min_samples_leaf`, and prediction.
        Check that the root, the structure and the predictions match `DecisionTreeClassifier`.
    """,
    dataset="""
        `load_breast_cancer` (569 nodules, 30 features, 2 classes), stratified 70/30 split: `X_tr, y_tr, X_te, y_te`.
    """,
    purpose="""
        Understand what `fit` of a tree does: a greedy algorithm that tries every threshold and picks the largest purity gain. You will see that the cost is dominated by the sorting of the features and that cumulative sums make the search for thresholds linear.
    """,
    task=[
        "`gini(y)`: Gini impurity of a vector of 0/1 labels (`1 − Σ p_k²`).",
        "`best_split(X, y, min_leaf)` → `(gain, feature, threshold)`: for each feature sort the examples, compute with `np.cumsum` the counts of the two classes to the left of each position and the gain $G(\\text{parent}) - \\frac{n_L G_L + n_R G_R}{n}$. Consider only the positions where the value changes and where "
        "both children have at least `min_leaf` examples; the threshold is the midpoint between two consecutive values. If no split has a positive gain return `(0, None, None)`. On ties in gain keep the **first** one found (lowest feature index, lowest threshold).",
        "`build(X, y, max_depth, min_leaf)`: recursive tree (dictionaries or tuples). A leaf holds the class counts; stop if the depth is `max_depth`, the node is pure or there is no valid split.",
        "`predict(tree, X)`: class for each row (the most frequent one in the leaf). Then compare with `DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)`: save `train_agreement`, `test_agreement` (fraction of equal predictions), `same_root` (feature and threshold of the root) and `same_leaves` (same number of leaves: implement `count_leaves(tree)`).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        data = load_breast_cancer()
        X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, stratify=data.target, random_state=0)
        sk = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0).fit(X_tr, y_tr)
    """,
    starter=[
        C("""
        def gini(y):
            ...


        def best_split(X, y, min_leaf):
            ...
        """),
        C("""
        def build(X, y, max_depth, min_leaf, depth=0):
            ...


        def predict(tree, X):
            ...


        def count_leaves(tree):
            ...
        """),
        C("""
        tree = build(X_tr, y_tr, 4, 5)
        train_agreement = None
        test_agreement = None
        same_root = None
        same_leaves = None
        """),
    ],
    checks="""
        check(close(gini(np.array([0, 0, 1, 1])), 0.5) and close(gini(np.array([1, 1, 1])), 0.0) and close(gini(np.array([0, 0, 0, 1])), 0.375), "Gini: 0.5 for half and half, 0 for a pure node, 0.375 for 3 against 1", "1 - sum(p ** 2)")
        g, f, s = best_split(X_tr, y_tr, 5)
        check(f == sk.tree_.feature[0] and abs(s - sk.tree_.threshold[0]) < 1e-3 and g > 0.3, "the best split at the root matches the one of scikit-learn", "cumulative sums on the sorted data; threshold = midpoint")
        check(best_split(np.array([[1.0], [1.0], [1.0], [1.0]]), np.array([0, 1, 0, 1]), 1) == (0, None, None), "no split possible with a constant feature", "consider only the points where the value changes")
        check(train_agreement == 1.0 and test_agreement >= 0.98, "identical predictions on the training set and at least 98% on the test", "scikit-learn compares in float32: a few thresholds may differ")
        check(same_root is True and same_leaves is True and count_leaves(tree) == sk.get_n_leaves(), "same root and same number of leaves", "count_leaves(tree) == sk.get_n_leaves()")
        small = build(X_tr, y_tr, 1, 5)
        check(count_leaves(small) == 2 and (predict(small, X_tr) == DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_tr, y_tr).predict(X_tr)).all(), "with max_depth=1 (a stump): two leaves and the same predictions as scikit-learn", "maximum depth 1")
    """,
    hint="""
        For each feature `j`: `o = np.argsort(X[:, j], kind="stable")`; `xs, ys = X[o, j], y[o]`; `cs = np.cumsum(np.eye(2)[ys], axis=0)` (counts of the two classes on the left); for each `i` from `min_leaf - 1` to `n - min_leaf - 1` with `xs[i] != xs[i + 1]`:
        `left = cs[i]`, `right = cs[-1] - left`. Gini from counts: `1 - ((c / c.sum()) ** 2).sum()`. Compare with `gain > best + 1e-12` to keep the first one.
    """,
    solution=[
        C("""
        def gini(y):
            p = np.bincount(y, minlength=2) / len(y)
            return 1 - (p ** 2).sum()


        def best_split(X, y, min_leaf):
            n, p = X.shape
            g0 = gini(y)
            best = (0, None, None)
            for j in range(p):
                o = np.argsort(X[:, j], kind="stable")
                xs, ys = X[o, j], y[o]
                cs = np.cumsum(np.eye(2)[ys], axis=0)                              # cumulative class counts: left of each position
                total = cs[-1]
                for i in range(min_leaf - 1, n - min_leaf):                         # children with at least min_leaf examples
                    if xs[i] == xs[i + 1]:
                        continue                                                    # no threshold between equal values
                    left, right = cs[i], total - cs[i]
                    nl, nr = left.sum(), right.sum()
                    gain = g0 - (nl * (1 - ((left / nl) ** 2).sum()) + nr * (1 - ((right / nr) ** 2).sum())) / n
                    if gain > best[0] + 1e-12:
                        best = (gain, j, (xs[i] + xs[i + 1]) / 2)
            return best
        """),
        C("""
        def build(X, y, max_depth, min_leaf, depth=0):
            counts = np.bincount(y, minlength=2)
            if depth == max_depth or counts.max() == len(y):
                return {"leaf": counts}
            _, j, s = best_split(X, y, min_leaf)
            if j is None:
                return {"leaf": counts}
            m = X[:, j] <= s
            return {"feature": j, "threshold": s,
                    "left": build(X[m], y[m], max_depth, min_leaf, depth + 1),
                    "right": build(X[~m], y[~m], max_depth, min_leaf, depth + 1)}


        def predict(tree, X):
            def one(x):
                node = tree
                while "leaf" not in node:
                    node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
                return node["leaf"].argmax()
            return np.array([one(x) for x in X])


        def count_leaves(tree):
            return 1 if "leaf" in tree else count_leaves(tree["left"]) + count_leaves(tree["right"])
        """),
        C("""
        tree = build(X_tr, y_tr, 4, 5)
        train_agreement = float((predict(tree, X_tr) == sk.predict(X_tr)).mean())
        test_agreement = float((predict(tree, X_te) == sk.predict(X_te)).mean())
        same_root = bool(tree["feature"] == sk.tree_.feature[0] and abs(tree["threshold"] - sk.tree_.threshold[0]) < 1e-3)
        same_leaves = bool(count_leaves(tree) == sk.get_n_leaves())
        print(f"agreement with scikit-learn: training {train_agreement:.3f}, test {test_agreement:.3f} | same root: {same_root} ({data.feature_names[tree['feature']]} <= {tree['threshold']:.2f}) | leaves: {count_leaves(tree)} against {sk.get_n_leaves()}")
        print(f"test accuracy of your tree: {(predict(tree, X_te) == y_te).mean():.3f}")
        """),
    ],
    learned="""
        - **A tree is an exhaustive search over features and thresholds.** With cumulative sums on the sorted data the gain of each threshold is computed in constant time: the cost of a node is that of sorting every feature, $O(p\\,n\\log n)$. This is why trees are fast to train even with many rows.
        - The thresholds are **midpoints** between consecutive values, and there is none between equal values: with discrete features (age, counts) there are few thresholds and the tree is less flexible.
        - The comparison with scikit-learn is exact on the training set and almost exact on the test: scikit-learn works in **`float32`**, so a point very close to a threshold can end up on the other side. In a dataset with continuous values it rarely happens (here it does not), with discrete values more often: a detail to remember when comparing implementations.
        - The recursion is **greedy**: each node maximizes the immediate gain. The resulting tree is not the best possible at that depth, but it is almost always good and costs very little.
    """,
    pitfall="""
        Computing the gain of each threshold by recounting the classes with a loop over the examples: $O(n^2)$ per feature. The version with cumulative sums is linear after the sorting, and it is the idea that makes trees practical.
    """,
    interview=(
        "How does a decision tree choose the split and what is the cost?",
        """
        For each feature it sorts the examples and tries all the thresholds between consecutive different values, computing the impurity (Gini or entropy) of the two children with cumulative counts; it picks the feature-threshold pair with the maximum gain and repeats recursively on the children until the stopping criterion.
        The cost per node is dominated by the sorting, O(p n log n); overall it is O(p n log n) per level of the tree. It is a greedy algorithm: it does not guarantee the optimal tree.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="regression_tree_and_importances_from_scratch", kind="A", level=3, minutes=55, module=MODULE,
    title="Regression tree and feature importance from scratch",
    goal="""
        Implement a regression tree (squared error reduction, prediction = mean of the leaf) and compute the **impurity importance**: the sum, for each feature, of the impurity reductions of its splits weighted by the fraction of examples. Check that the importances match
        `feature_importances_` of scikit-learn.
    """,
    dataset="""
        `load_diabetes` (442 patients, 10 already standardized features, target = disease progression), 70/30 split: `X_tr, y_tr, X_te, y_te`.
    """,
    purpose="""
        Understand what `feature_importances_` really is (an accounting of impurity reductions, not a causal measure) and how regression reuses the same scheme as classification with another impurity (the variance).
    """,
    task=[
        "`best_split_reg(X, y, min_leaf)` → `(gain, feature, threshold)`: as in A1, but the impurity of a node is the **variance** of the target, computed with the cumulative sums of `y` and `y²`. The gain is `var(parent) − (n_L var_L + n_R var_R) / n`.",
        "`build_reg(X, y, max_depth, min_leaf, N, importances, depth=0)`: recursive tree; a leaf holds the **mean** of `y`. Every time you choose a split on feature `j` add to `importances[j]` the value `gain * len(y) / N` (`N` = examples of the whole training set).",
        "`predict_reg(tree, X)`. Build the tree with `max_depth=4, min_leaf=5` and save `norm_importances` (the importances divided by their sum).",
        "Compare with `DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0)`: `same_importances` (`np.allclose`), `train_agreement` and `test_agreement` (fraction of predictions equal within 1e-9). Also save `top_feature` (name of the feature with the largest importance) and `top_two_share` (sum of the two largest importances).",
    ],
    setup="""
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        data = load_diabetes()
        X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target, test_size=0.3, random_state=0)
        sk = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=0).fit(X_tr, y_tr)
    """,
    starter=[
        C("""
        def best_split_reg(X, y, min_leaf):
            ...
        """),
        C("""
        def build_reg(X, y, max_depth, min_leaf, N, importances, depth=0):
            ...


        def predict_reg(tree, X):
            ...
        """),
        C("""
        importances = np.zeros(X_tr.shape[1])
        tree = build_reg(X_tr, y_tr, 4, 5, len(y_tr), importances)
        norm_importances = None
        same_importances = None
        train_agreement = None
        test_agreement = None
        top_feature = None
        top_two_share = None
        """),
    ],
    checks="""
        g, f, s = best_split_reg(X_tr, y_tr, 5)
        check(f == sk.tree_.feature[0] and abs(s - sk.tree_.threshold[0]) < 1e-6 and g > 0, "the split at the root matches the one of scikit-learn", "variance with the cumulative sums of y and y^2")
        check(norm_importances is not None and close(norm_importances.sum(), 1.0) and same_importances is True, "normalized importances matching scikit-learn", "sum(gain * n_node / N) per feature, then divide by the sum")
        check(train_agreement == 1.0 and test_agreement >= 0.98, "equal predictions on the training set (100%) and on at least 98% of the test", "mean of the leaf; scikit-learn uses float32 for the thresholds")
        check(top_feature == data.feature_names[int(np.argmax(sk.feature_importances_))], "the most important feature matches the one of scikit-learn", "argmax of the importances")
        check(0.7 < top_two_share < 0.9 and (norm_importances[[3, 5]] < norm_importances[8]).all(), "the top two features are worth between 70% and 90% of the importance", "sum of the two largest importances")
    """,
    hint="""
        Variance from cumulative sums: `cs = np.cumsum(ys); cs2 = np.cumsum(ys ** 2)`; to the left of position `i` (`nl = i + 1`): `cs2[i] / nl - (cs[i] / nl) ** 2`; to the right: `((cs2[-1] - cs2[i]) / nr) - ((cs[-1] - cs[i]) / nr) ** 2`.
        In the recursion: `importances[j] += gain * len(y) / N`; stop if `np.ptp(y) == 0` or `len(y) < 2 * min_leaf`.
    """,
    solution=[
        C("""
        def best_split_reg(X, y, min_leaf):
            n, p = X.shape
            var0 = (y ** 2).mean() - y.mean() ** 2
            best = (0, None, None)
            for j in range(p):
                o = np.argsort(X[:, j], kind="stable")
                xs, ys = X[o, j], y[o]
                cs, cs2 = np.cumsum(ys), np.cumsum(ys ** 2)
                for i in range(min_leaf - 1, n - min_leaf):
                    if xs[i] == xs[i + 1]:
                        continue
                    nl, nr = i + 1, n - i - 1
                    var_left = cs2[i] / nl - (cs[i] / nl) ** 2
                    var_right = (cs2[-1] - cs2[i]) / nr - ((cs[-1] - cs[i]) / nr) ** 2
                    gain = var0 - (nl * var_left + nr * var_right) / n
                    if gain > best[0] + 1e-12:
                        best = (gain, j, (xs[i] + xs[i + 1]) / 2)
            return best
        """),
        C("""
        def build_reg(X, y, max_depth, min_leaf, N, importances, depth=0):
            if depth == max_depth or len(y) < 2 * min_leaf or np.ptp(y) == 0:
                return {"leaf": y.mean()}
            gain, j, s = best_split_reg(X, y, min_leaf)
            if j is None:
                return {"leaf": y.mean()}
            importances[j] += gain * len(y) / N                                 # impurity reduction weighted by the share of examples of the node
            m = X[:, j] <= s
            return {"feature": j, "threshold": s,
                    "left": build_reg(X[m], y[m], max_depth, min_leaf, N, importances, depth + 1),
                    "right": build_reg(X[~m], y[~m], max_depth, min_leaf, N, importances, depth + 1)}


        def predict_reg(tree, X):
            def one(x):
                node = tree
                while "leaf" not in node:
                    node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
                return node["leaf"]
            return np.array([one(x) for x in X])
        """),
        C("""
        importances = np.zeros(X_tr.shape[1])
        tree = build_reg(X_tr, y_tr, 4, 5, len(y_tr), importances)
        norm_importances = importances / importances.sum()
        same_importances = bool(np.allclose(norm_importances, sk.feature_importances_))
        train_agreement = float((np.abs(predict_reg(tree, X_tr) - sk.predict(X_tr)) < 1e-9).mean())
        test_agreement = float((np.abs(predict_reg(tree, X_te) - sk.predict(X_te)) < 1e-9).mean())
        top_feature = data.feature_names[int(norm_importances.argmax())]
        top_two_share = float(np.sort(norm_importances)[-2:].sum())
        print("importances:", dict(zip(data.feature_names, norm_importances.round(3))))
        print(f"equal to scikit-learn's: {same_importances} | agreement on the predictions: training {train_agreement:.3f}, test {test_agreement:.3f} | top feature: {top_feature} | top two: {top_two_share:.1%}")
        """),
    ],
    learned="""
        - **`feature_importances_` is an accounting**: for every split you compute how much impurity (here variance) it removed, weight it by the share of examples that go through that node and sum per feature. The two implementations match to the decimal: there is nothing else behind it.
        - **The importances are relative to the tree:** they are normalized to sum 1 and depend on the depth and on the available features. The top two features (`s5`, `bmi`) are worth over 80%; with correlated features the importance is split arbitrarily among them.
        - Regression reuses the scheme of classification with a different impurity: the **variance** (which is also the squared error of predicting the mean of the node). The leaf predicts the mean, and the gain is the reduction of the squared error.
        - Remember the limits (D2 and lesson 7.3): the impurity importance is computed **on the training set**, it favors features with many distinct values and it does not say whether the feature helps out of sample.
    """,
    pitfall="""
        Presenting `feature_importances_` as "the influence of the variables on the phenomenon". It is a statistic of the trained tree: it changes with the depth, with correlated features and with the sample.
    """,
    interview=(
        "How is feature importance computed in a tree?",
        """
        For each feature I sum the impurity reduction of the splits that use it, weighted by the fraction of examples that go through the node, and normalize to sum 1 (in forests you average over the trees). It is fast but it has limits: it is computed on the training set, it favors features with many distinct values and it is split among correlated features.
        For a more reliable judgment I use the permutation importance on validation data.
        """),
)

EXERCISES = [A1, A2]
