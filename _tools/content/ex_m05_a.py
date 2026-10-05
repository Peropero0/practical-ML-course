"""Module 5 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="knn_classifier_from_scratch", kind="A", level=2, minutes=45, module=MODULE,
    title="A KNN in numpy: vectorized distances, voting and weights",
    goal="""
        Implement a KNN classifier without loops over the examples: distance matrix (with the identity $\\lVert a-b\\rVert^2 = \\lVert a\\rVert^2+\\lVert b\\rVert^2-2a\\cdot b$),
        selection of the `k` neighbors, uniform voting and voting weighted by the inverse of the distance, and handling of zero distance. Check that the predictions match `KNeighborsClassifier`.
    """,
    dataset="""
        `load_wine` (178 wines, 13 chemical measurements, 3 classes), split 70/30 and standardized with the scaler learned on the training set: `Z_tr, y_tr, Z_te, y_te`.
    """,
    purpose="""
        See that KNN is **all `predict`**: a distance matrix, an `argpartition` and a vote. You will find the two details that make a difference (rounding errors in the distances and distance 0) and how
        scikit-learn resolves ties.
    """,
    task=[
        "`distances(A, B)`: `(len(A), len(B))` matrix of Euclidean distances **without loops**, with the identity above (careful: because of rounding errors the square can come out negative: use `np.maximum(..., 0)` before the root).",
        "`nearest(D, k)`: for every row of `D` the indices of the `k` smallest values, sorted by increasing distance (use `np.argpartition` and then sort only the `k` found).",
        "`predict_knn(A, X, y, k, weights)`: prediction for every row of `A` with training set `(X, y)`. `weights=\"uniform\"`: class with the most votes (on a tie, the class with the **smallest label**); "
        "`weights=\"distance\"`: every neighbor is worth `1/distance`; if a neighbor has distance 0 **only** the neighbors at distance 0 count.",
        "Compare with `KNeighborsClassifier` for `k` in `(1, 5, 15)` and for both weights: save in `agreement` a dictionary `(k, weights) -> fraction of equal predictions` on the test set. Then check the limit case: `predict_knn(Z_tr, Z_tr, y_tr, 3, \"distance\")` must match `y_tr`.",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.metrics import pairwise_distances
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        X, y = load_wine(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        scaler = StandardScaler().fit(X_tr)
        Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)
        print(Z_tr.shape, Z_te.shape, "| classes:", np.bincount(y_tr))
    """,
    starter=[
        C("""
        def distances(A, B):
            ...


        def nearest(D, k):
            ...
        """),
        C("""
        def predict_knn(A, X, y, k, weights="uniform"):
            ...
        """),
        C("""
        agreement = {}
        """),
    ],
    checks="""
        check(nearest(np.array([[3.0, 1.0, 2.0, 5.0]]), 2).tolist() == [[1, 2]], "nearest returns the k smallest indices, in increasing order", "argpartition then sort only the k found")
        D = distances(Z_te, Z_tr)
        check(D.shape == (len(Z_te), len(Z_tr)) and nearest(D, 5).shape == (len(Z_te), 5), "correct shapes", "distances: (len(A), len(B))")
        check(nearest(np.array([[0.0, 0.0, 0.0]]), 1).shape == (1, 1), "nearest also works with equal distances", "mind the ties")
        check(nearest(D, 5).tolist() == np.argsort(D, axis=1)[:, :5].tolist(), "the indices of the neighbors match a full sort", "argpartition(D, k - 1)[:, :k] then sort by distance")
        check(np.allclose(D, pairwise_distances(Z_te, Z_tr), atol=1e-6) and not np.isnan(distances(Z_tr, Z_tr)).any(),
              "distances equal to pairwise_distances, with no NaN even between identical points", "np.maximum(..., 0) before the root")
        for (k, weights), fraction in agreement.items():
            print(f"   k={k:2d} weights={weights:8s} agreement with scikit-learn: {fraction:.3f}")
        check(set(agreement) == {(k, p) for k in (1, 5, 15) for p in ("uniform", "distance")} and all(v == 1.0 for v in agreement.values()), "all the predictions match scikit-learn", "check the ties (smallest label) and the weights 1/d")
        check((predict_knn(Z_tr, Z_tr, y_tr, 3, "distance") == y_tr).all(), "with 1/distance weights every training point predicts itself (distance 0)", "if there is a neighbor at distance 0 only those at distance 0 count")
    """,
    hint="""
        `distances`: `sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T; np.sqrt(np.maximum(sq, 0))`. `nearest`: `idx = np.argpartition(D, k - 1, axis=1)[:, :k]; order = np.argsort(np.take_along_axis(D, idx, 1), axis=1); np.take_along_axis(idx, order, 1)`.
        Vote: for every row `np.bincount(y[idx], weights=w, minlength=n_classes).argmax()` (the `argmax` of a vector with equal maxima takes the first, that is the smallest class).
    """,
    solution=[
        C("""
        def distances(A, B):
            square = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
            return np.sqrt(np.maximum(square, 0))              # rounding can give small negatives


        def nearest(D, k):
            idx = np.argpartition(D, k - 1, axis=1)[:, :k]                        # the k smallest, unsorted (linear cost)
            order = np.argsort(np.take_along_axis(D, idx, 1), axis=1)             # I sort only these k
            return np.take_along_axis(idx, order, 1)
        """),
        C("""
        def predict_knn(A, X, y, k, weights="uniform"):
            D = distances(A, X)
            idx = nearest(D, k)
            d = np.take_along_axis(D, idx, 1)
            n_classes = y.max() + 1
            predictions = np.empty(len(A), dtype=int)
            for i in range(len(A)):
                if weights == "uniform":
                    w = np.ones(k)
                elif (d[i] == 0).any():
                    w = (d[i] == 0).astype(float)                                # only those that coincide with the point count
                else:
                    w = 1 / d[i]
                predictions[i] = np.bincount(y[idx[i]], weights=w, minlength=n_classes).argmax()     # on a tie: smallest label
            return predictions
        """),
        C("""
        agreement = {}
        for k in (1, 5, 15):
            for weights in ("uniform", "distance"):
                sk = KNeighborsClassifier(k, weights=weights).fit(Z_tr, y_tr).predict(Z_te)
                agreement[(k, weights)] = float((predict_knn(Z_te, Z_tr, y_tr, k, weights) == sk).mean())
        print(agreement)
        """),
    ],
    learned="""
        - A KNN's `predict` is three lines of algebra: the distance matrix (with the dot-product identity, which exploits numpy's optimized matrix multiplication), the indices of the `k` minima with
          `argpartition` (linear cost, much better than sorting everything) and a vote. The model is the training set.
        - **Two numeric details**: the square computed with the identity can come out as `-1e-16` and the root gives `NaN`, which is why it is brought to zero with `np.maximum`; and with `1/distance` weights a neighbor at distance 0 would cause a
          division by zero, and scikit-learn resolves it by giving weight only to the neighbors at distance 0.
        - The tie is resolved **deterministically but arbitrarily** (smallest label): two identical models with renamed labels can give different answers. With odd `k` and two classes there are no ties.
        - Cost: the distance matrix is `n_test × n_train`; with hundreds of thousands of rows it no longer fits in memory and you work in blocks: it is what `KNeighborsClassifier` and the approximate search libraries do.
    """,
    pitfall="""
        Computing the distances with two `for` loops "for clarity": it works with 178 wines and becomes unusable with 100,000 rows. The vectorized version is not an optional optimization, it is what makes KNN usable.
    """,
    interview=(
        "What is the complexity of `predict` for a KNN and how is it reduced?",
        """
        For every test point it is O(n·d) with the exhaustive computation (n training examples, d features), plus O(n) to select the k minima with a partition. It is reduced with search structures (kd-tree, ball-tree: good with few dimensions)
        or with approximate indexes (LSH, HNSW) in high dimension, at the cost of some errors in the neighbors found. In production you also precompute the scalers and keep the training set in memory.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="leave_one_out_for_all_k", kind="A", level=3, minutes=50, module=MODULE,
    title="Leave-one-out for all k with a single distance matrix",
    goal="""
        Compute the leave-one-out accuracy of a KNN for **all the `k` from 1 to 60 in a single pass**: an n×n distance matrix with the diagonal excluded, a sort per row and the cumulated votes. Check
        the exact equality with `cross_val_score(..., cv=LeaveOneOut())` and measure how much you save.
    """,
    dataset="""
        `load_breast_cancer` (569 nodules, 30 measurements, 2 classes) standardized once: `Z, y`. (Standardizing before a LOO is a small incorrectness that we accept here to isolate the problem: your goal is the algorithm.)
    """,
    purpose="""
        Exploit the structure of KNN: the predictions for `k` and for `k+1` share almost everything, so with a single sort of the distances you get **the whole validation curve** at the cost of a single evaluation. A standard trick
        to choose `k` on small datasets.
    """,
    task=[
        "`loo_matrix(Z)`: n×n matrix of Euclidean distances with the **diagonal set to `np.inf`** (a point cannot be its own neighbor).",
        "`loo_accuracy(Z, y, k_max)`: array of `k_max` accuracies, for `k = 1..k_max`. Sort every row (`argsort`), take the labels of the `k_max` neighbors, build the cumulated counts per class with `np.cumsum` on a one-hot, and the prediction for `k` is the class with the most votes among the first `k` neighbors "
        "(on a tie, the smallest label: `argmax`).",
        "Check: for `k` in `(1, 4, 7, 25)` the accuracy must match **exactly** `cross_val_score(KNeighborsClassifier(k), Z, y, cv=LeaveOneOut()).mean()`. Measure with `time.time()` `t_slow` (the four scikit-learn LOOs) and `t_fast` (your function with `k_max=60`) and save `speedup = t_slow / t_fast`.",
        "Save `best_k` (the `k` with maximum accuracy; on a tie, the smallest) and `simple_rule_k`: the **maximum** `k` whose accuracy is within **one percentage point** of the maximum.",
    ],
    setup="""
        import time
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import LeaveOneOut, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        X, y = load_breast_cancer(return_X_y=True)
        Z = StandardScaler().fit_transform(X)
        print(Z.shape, "| classes:", np.bincount(y))
    """,
    starter=[
        C("""
        def loo_matrix(Z):
            ...


        def loo_accuracy(Z, y, k_max):
            ...
        """),
        C("""
        t0 = time.time()
        slow = {k: cross_val_score(KNeighborsClassifier(k), Z, y, cv=LeaveOneOut()).mean() for k in (1, 4, 7, 25)}
        t_slow = time.time() - t0
        t0 = time.time()
        curve = loo_accuracy(Z, y, 60)
        t_fast = time.time() - t0
        speedup = t_slow / t_fast
        print(f"scikit-learn (4 values of k): {t_slow:.1f}s | yours (60 values of k): {t_fast:.3f}s | ratio: {speedup:.0f}x")
        """),
        C("""
        best_k = None
        simple_rule_k = None
        """),
    ],
    checks="""
        D = loo_matrix(Z)
        check(D.shape == (569, 569) and np.isinf(np.diag(D)).all() and np.isfinite(D[~np.eye(569, dtype=bool)]).all(), "569 x 569 matrix with the diagonal at infinity", "np.fill_diagonal(D, np.inf)")
        check(len(curve) == 60 and all(abs(curve[k - 1] - slow[k]) < 1e-12 for k in (1, 4, 7, 25)), "the LOO accuracy matches scikit-learn exactly for k = 1, 4, 7, 25", "votes among the first k neighbors, tie -> smallest label")
        check(speedup > 5, "your function is over 5 times faster (60 values of k against 4)", "a single sort of the rows")
        check(best_k == int(np.argmax(curve)) + 1, "best k: the one with maximum accuracy (on a tie, the smallest)", "np.argmax(curve) + 1")
        check(simple_rule_k == max(k for k in range(1, 61) if curve[k - 1] >= curve.max() - 0.01) and simple_rule_k >= best_k, "k of the simple rule: the largest within one point of the maximum", "smoother model at equal performance")
    """,
    hint="""
        `order = np.argsort(D, axis=1)[:, :k_max]`; `labels = y[order]` (n × k_max); `oh = np.eye(n_classes)[labels]` (n × k_max × n_classes); `votes = np.cumsum(oh, axis=1)`; the prediction with `k` neighbors is `votes[:, k - 1, :].argmax(1)`;
        `accuracy = (votes.argmax(2) == y[:, None]).mean(0)`.
    """,
    solution=[
        C("""
        def loo_matrix(Z):
            sq = (Z ** 2).sum(1)
            D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0))
            np.fill_diagonal(D, np.inf)                                # no point is its own neighbor
            return D


        def loo_accuracy(Z, y, k_max):
            D = loo_matrix(Z)
            order = np.argsort(D, axis=1)[:, :k_max]                   # the k_max neighbors of every point, from the nearest
            oh = np.eye(y.max() + 1)[y[order]]                         # (n, k_max, classes): the class of each neighbor
            votes = np.cumsum(oh, axis=1)                              # votes[i, k-1, c] = votes for c among the first k neighbors of i
            return (votes.argmax(2) == y[:, None]).mean(0)             # accuracy for k = 1..k_max
        """),
        C("""
        t0 = time.time()
        slow = {k: cross_val_score(KNeighborsClassifier(k), Z, y, cv=LeaveOneOut()).mean() for k in (1, 4, 7, 25)}
        t_slow = time.time() - t0
        t0 = time.time()
        curve = loo_accuracy(Z, y, 60)
        t_fast = time.time() - t0
        speedup = t_slow / t_fast
        print(f"scikit-learn (4 values of k): {t_slow:.1f}s | yours (60 values of k): {t_fast:.3f}s | ratio: {speedup:.0f}x")
        """),
        C("""
        best_k = int(np.argmax(curve)) + 1
        simple_rule_k = max(k for k in range(1, 61) if curve[k - 1] >= curve.max() - 0.01)
        print(f"LOO accuracy for k = 1, 5, 15, 30, 60: {curve[[0, 4, 14, 29, 59]].round(4)}")
        print(f"best k: {best_k} (accuracy {curve.max():.4f}) | k of the simple rule (within 1 point): {simple_rule_k}")
        """),
    ],
    learned="""
        - With a single sort per row you get **the whole curve** of the accuracy as a function of `k`: the predictions for `k` are the cumulated votes of the first `k` neighbors, and the votes for `k+1` are obtained from those for `k` by adding one neighbor.
          It is a property specific to KNN (parametric models have nothing like it) and explains why LOO is cheap for KNN.
        - The curve is **flat** around the maximum: many `k` are equivalent within the uncertainty of the estimate (about ±1 point with 569 examples). Choosing the argmax is choosing the noise; the rule "the largest `k` within one point
          of the maximum" (smoother model) is more robust.
        - LOO has a single notable cost: the n×n matrix (569² = 324,000 numbers here, but 10⁸ with 10,000 rows). With larger datasets you move to a K-fold or work in blocks.
        - Note about a small sin: the scaler was learned on all the data. For choosing `k` the bias is negligible; for estimating the final performance it is not.
    """,
    pitfall="""
        Choosing `k` by looking at the single maximum of the LOO curve: with 569 examples the difference between `k = 7` and `k = 15` is less than the noise of the estimate. Look at the shape of the curve and choose the simplest model among the equivalent ones.
    """,
    interview=(
        "Why is leave-one-out cross-validation cheap for KNN?",
        """
        Because KNN has no training: the prediction for a point uses its neighbors in the training set excluding the point itself, so a single distance matrix with the diagonal excluded is enough. With a single sort per row you get the
        predictions for every `k` at almost no cost. What remains is the O(n²) cost of the matrix, and the variance of the LOO estimate (the n estimates are highly correlated), which is why on large datasets I prefer a repeated K-fold.
        """),
)

EXERCISES = [A1, A2]
