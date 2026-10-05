"""Module 5 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="decision_boundary_and_k", kind="D", level=2, minutes=45, module=MODULE,
    title="Reading the decision boundary: roughness, regions and k",
    goal="""
        Turn the KNN decision boundary from an image into **numbers**: for nine values of `k` compute on a grid the roughness of the boundary (fraction of adjacent cells with different classes) and the number of connected regions, and read from the data
        where there is overfitting, where the model is too rigid and how large `k` must be to "simplify" the boundary.
    """,
    dataset="""
        `make_moons(600, noise=0.3)` split 50/50 (300 training examples, 300 test ones). `grid_map(k)` (given) returns the prediction of the model on a 100×100 grid covering the plane (matrix of 0s and 1s).
    """,
    purpose="""
        Learn to read a decision boundary without trusting the eye: the number of connected regions and the length of the boundary are measures of the complexity of the model, and `k` is its only regulator.
    """,
    task=[
        "For `k` in `[1, 3, 5, 9, 15, 31, 61, 121, 201]` build `table` (index `k`, columns `train_acc`, `test_acc`, `roughness`, `n_regions`). `roughness`: fraction of pairs of adjacent cells (horizontal and vertical) of the map with different classes. "
        "`n_regions`: number of connected components of the map (count the regions of 0 and of 1 with `scipy.ndimage.label`, 4-connectivity).",
        "Save `overfit_k` (the `k` with the largest difference `train_acc − test_acc`) and `best_k` (the `k` with maximum `test_acc`; on a tie the smallest).",
        "Save `k1_regions` (number of regions with `k = 1`) and `two_region_k` (the smallest `k` whose map has exactly **two** regions).",
        "Answer with booleans: `roughness_decreases` (the roughness **never** increases as `k` grows: true or false?), `train_below_test_at_large_k` (is there a `k` where `train_acc < test_acc`?), `k1_has_train_one` (the training accuracy of `k = 1` is 1).",
    ],
    setup="""
        from scipy.ndimage import label
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        X, y = make_moons(600, noise=0.3, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        g1, g2 = np.meshgrid(np.linspace(-2.5, 3.5, 100), np.linspace(-1.8, 2.3, 100))
        points = np.c_[g1.ravel(), g2.ravel()]
        ks = [1, 3, 5, 9, 15, 31, 61, 121, 201]


        def grid_map(k):
            \"\"\"KNN prediction (k neighbors) on a 100 x 100 grid: matrix of 0s and 1s.\"\"\"
            return KNeighborsClassifier(k).fit(X_tr, y_tr).predict(points).reshape(100, 100)
    """,
    starter=[
        C("""
        rows = {}
        for k in ks:
            G = grid_map(k)
            rows[k] = ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        overfit_k = None
        best_k = None
        k1_regions = None
        two_region_k = None
        """),
        C("""
        roughness_decreases = None
        train_below_test_at_large_k = None
        k1_has_train_one = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["train_acc", "test_acc", "roughness", "n_regions"] and list(table.index) == ks, "table with the four columns for nine values of k", "index k, columns train_acc, test_acc, roughness, n_regions")
        G1 = grid_map(1)
        rough_ref = ((G1[:, 1:] != G1[:, :-1]).sum() + (G1[1:, :] != G1[:-1, :]).sum()) / (G1[:, 1:].size + G1[1:, :].size)
        check(close(table.loc[1, "roughness"], rough_ref), "roughness computed as the fraction of different adjacent pairs", "compare G[:, 1:] with G[:, :-1] and G[1:, :] with G[:-1, :]")
        check(table.loc[1, "n_regions"] == label(G1 == 0)[1] + label(G1 == 1)[1], "number of regions: connected components of 0 and of 1", "label(G == 0)[1] + label(G == 1)[1]")
        check(overfit_k == (table["train_acc"] - table["test_acc"]).idxmax() and overfit_k == 1, "the maximum train-test gap is at k = 1", "idxmax of the difference")
        check(best_k == table["test_acc"].idxmax() and 9 <= best_k <= 61, "the best k on the test set is intermediate (between 9 and 61)", "idxmax of test_acc")
        check(k1_regions >= 8 and two_region_k <= 15 and table.loc[201, "n_regions"] == 2, "with k = 1 at least 8 regions; from k = 15 at most two; with k = 201 exactly two", "n_regions in table")
        check(roughness_decreases is False and table.loc[1, "roughness"] > 1.5 * table.loc[15, "roughness"] and abs(table.loc[15, "roughness"] - table.loc[61, "roughness"]) < 0.002, "the roughness does NOT always fall: it drops until k = 15 (at k = 1 it is over 1.5 times) and then stays on a plateau", "look at the roughness column: is_monotonic_decreasing")
        check(k1_has_train_one is True and train_below_test_at_large_k is False and abs(table.loc[201, "train_acc"] - table.loc[201, "test_acc"]) < 0.02 and table.loc[201, "test_acc"] < table["test_acc"].max() - 0.08, "k = 1: perfect training; the training is never below the test; at k = 201 train and test coincide but are both low (underfitting)", "compare train_acc and test_acc row by row")
    """,
    hint="""
        Roughness: `((G[:, 1:] != G[:, :-1]).sum() + (G[1:, :] != G[:-1, :]).sum()) / (G[:, 1:].size + G[1:, :].size)`. Regions: `label(G == 0)[1] + label(G == 1)[1]`. Each row: `{"train_acc": m.score(X_tr, y_tr), ...}` with `m = KNeighborsClassifier(k).fit(X_tr, y_tr)`.
    """,
    solution=[
        C("""
        rows = {}
        for k in ks:
            m = KNeighborsClassifier(k).fit(X_tr, y_tr)
            G = m.predict(points).reshape(100, 100)
            differences = (G[:, 1:] != G[:, :-1]).sum() + (G[1:, :] != G[:-1, :]).sum()                 # adjacent pairs with different classes
            rows[k] = {"train_acc": m.score(X_tr, y_tr), "test_acc": m.score(X_te, y_te),
                       "roughness": differences / (G[:, 1:].size + G[1:, :].size),
                       "n_regions": label(G == 0)[1] + label(G == 1)[1]}
        table = pd.DataFrame(rows).T
        table["n_regions"] = table["n_regions"].astype(int)
        table.round(3)
        """),
        C("""
        overfit_k = int((table["train_acc"] - table["test_acc"]).idxmax())
        best_k = int(table["test_acc"].idxmax())
        k1_regions = int(table.loc[1, "n_regions"])
        two_region_k = int(table.index[(table["n_regions"] == 2).to_numpy().argmax()])
        print(f"k with the maximum train-test gap: {overfit_k} | best k on the test set: {best_k} | regions at k=1: {k1_regions} | first k with two regions: {two_region_k}")
        """),
        C("""
        roughness_decreases = bool(table["roughness"].is_monotonic_decreasing)
        train_below_test_at_large_k = bool((table["train_acc"] < table["test_acc"]).any())
        k1_has_train_one = bool(table.loc[1, "train_acc"] == 1.0)
        print("decreasing roughness:", roughness_decreases, "| there is a k with train < test:", train_below_test_at_large_k, "| perfect training with k = 1:", k1_has_train_one)
        """),
    ],
    learned="""
        - **The number of regions is a complexity gauge**: with `k = 1` the map has 11 regions (the two moons plus nine islands created by the noisy points), with `k = 3` five, with `k = 5` four and from `k = 9` two remain, the two moons. The roughness falls by more than a third
          between `k = 1` and `k = 15` (0.018 against 0.009) and then stays on a **plateau**: when the boundary is the true one, increasing `k` no longer shortens it, it only moves it.
        - The **train-test gap** is maximal at `k = 1` (training 1.00, test 0.88): it is the classic symptom of overfitting and is read without plots.
        - The best `k` on the test set is intermediate (31 out of 300 examples, but between 9 and 61 the accuracy is practically equal: 0.88-0.89): a boundary neither jagged nor too smooth. With `k = 201` (2/3 of the training set) the model becomes almost a line and collapses to 0.77.
        - At `k = 201` training and test coincide (0.77 and 0.77) but are **both low**: it is underfitting, not good generalization. A small gap between training and test is good news only if the two accuracies are high.
    """,
    pitfall="""
        Choosing `k` "by eye" from the boundary plot: a smooth boundary is pretty, but the only judge is validation. The numeric measures (regions, roughness) help you understand, not choose.
    """,
    interview=(
        "How do you explain to a colleague that `k` regulates the complexity?",
        """
        With `k = 1` the prediction at a point depends only on the nearest neighbor: the boundary follows every example and every error, it has dozens of regions and the training is perfect. As `k` grows the prediction is an average over more points, the boundary simplifies and in the end
        it becomes the most frequent class. It moves the model along the bias-variance tradeoff; I choose it with cross-validation.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="knn_probabilities", kind="D", level=3, minutes=50, module=MODULE,
    title="What `predict_proba` means in a KNN",
    goal="""
        Interpret the `predict_proba` of a KNN: why it takes few values, what effect it has on the log-loss and on the ROC curve, whether it is calibrated, and how a single prediction is "explained" by looking at the neighbors that voted.
    """,
    dataset="""
        `make_classification` (3000 examples, 6 features of which 3 informative), standardized, 50/50 split: `Z_tr, y_tr, Z_te, y_te`. Models (given): KNN with `k` = 1, 5, 25 (uniform), 25 with 1/distance weights, and a logistic as a reference (the problem is linear enough to work well for all).
    """,
    purpose="""
        Dismantle an illusion: the `predict_proba` of a KNN **is not a calibrated probability**, it is a fraction of neighbors. It has practical consequences on metrics (huge log-loss with small `k`), curves (stepped ROC) and calibration.
    """,
    task=[
        "Save `n_values` (dictionary `model name → number of distinct values` in `predict_proba(Z_te)[:, 1]`) for the five models (`knn1`, `knn5`, `knn25`, `knn25_d`, `lr`). How is it linked to `k` for the uniform ones?",
        "Save `table` (index: the five models; columns `auc`, `logloss`, `roc_points` (length of `roc_curve(...)[0]`), `ece`) with the `ece` already defined (`ece(y, p)`, 10 groups of equal size).",
        "Save `why_logloss_is_high = 'b'`: a) the KNN ranks the cases badly; b) with small `k` many probabilities are exactly 0 or 1 and errors with probability 0/1 cost `-log(1e-15)` ≈ 34.5 each; c) the log-loss cannot be used with KNNs.",
        "**Explaining a prediction.** For the first test example that `knn25` gets wrong (`error_idx`, already computed) save `neighbors_table` (DataFrame with one row per neighbor: `index`, `distance`, `class`, sorted by distance) and `votes` (dictionary class → number of votes) using `knn25.kneighbors`. "
        "Save `margin` = votes of the predicted class minus votes of the other one.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score, roc_curve
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import StandardScaler

        X, y = make_classification(3000, n_features=6, n_informative=3, n_redundant=0, n_clusters_per_class=1, class_sep=0.8, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        scaler = StandardScaler().fit(X_tr)
        Z_tr, Z_te = scaler.transform(X_tr), scaler.transform(X_te)

        models = {
            "knn1": KNeighborsClassifier(1), "knn5": KNeighborsClassifier(5), "knn25": KNeighborsClassifier(25),
            "knn25_d": KNeighborsClassifier(25, weights="distance"), "lr": LogisticRegression(max_iter=1000),
        }
        probabilities = {name: m.fit(Z_tr, y_tr).predict_proba(Z_te)[:, 1] for name, m in models.items()}
        knn25 = models["knn25"]
        error_idx = int(np.flatnonzero(knn25.predict(Z_te) != y_te)[0])           # first test example that knn25 gets wrong


        def ece(y, p, k=10):
            g = pd.DataFrame({"y": y, "p": p, "g": pd.qcut(p, k, duplicates="drop")}).groupby("g", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
            return float((g["n"] / g["n"].sum() * (g["p"] - g["y"]).abs()).sum())
    """,
    starter=[
        C("""
        n_values = None
        """),
        C("""
        table = pd.DataFrame(index=list(models), columns=["auc", "logloss", "roc_points", "ece"], dtype=float)
        table
        """),
        C("""
        why_logloss_is_high = None
        """),
        C("""
        neighbors_table = None
        votes = None
        margin = None
        """),
    ],
    checks="""
        check(n_values["knn1"] == 2 and n_values["knn5"] == 6 and n_values["knn25"] == 26 and n_values["knn25_d"] > 100 and n_values["lr"] > 1000, "k+1 values for the uniform KNNs (2, 6, 26); over a hundred with 1/distance weights; thousands with the logistic", "np.unique(p).size")
        check(table.notna().all().all() and close(table.loc["knn5", "roc_points"], len(roc_curve(y_te, probabilities["knn5"])[0])), "complete table", "five models, four columns")
        check(table.loc["knn1", "logloss"] > 5 and table.loc["knn1", "logloss"] > 5 * table.loc["knn5", "logloss"] and table.loc["knn25", "logloss"] < table.loc["knn5", "logloss"] and table.loc["lr", "logloss"] < table.loc["knn25", "logloss"], "log-loss: over 5 with k=1 (more than 5 times that of k=5), then it drops with k=25 and the logistic is the best", "use sklearn.metrics.log_loss on the raw probabilities")
        check(table.loc["knn1", "roc_points"] == 3 and 10 < table.loc["knn25", "roc_points"] <= 27 and table.loc["lr", "roc_points"] > 50, "the ROC curve has 3 points for k=1, at most 27 for k=25 (scikit-learn drops collinear points), over 50 for the logistic", "len(roc_curve(y, p)[0])")
        check(table.loc["knn25", "auc"] > table.loc["knn1", "auc"] + 0.1 and table.loc["lr", "ece"] < table.loc["knn25", "ece"] / 1.5, "k = 25 has an AUC better than k = 1 by over 0.1; the logistic has an ECE at least 1.5 times better than k = 25", "compare auc and ece")
        check(table.loc["knn1", "ece"] < 0.05, "with k = 1 the ECE is small (below 0.05) but it is an illusion: the groups with equal probabilities collapse into one", "pd.qcut(..., duplicates='drop') with only two distinct values")
        check(why_logloss_is_high == "b", "correct answer: wrong 0/1 probabilities cost 34.5", "b")
        check(list(neighbors_table.columns) == ["index", "distance", "class"] and len(neighbors_table) == 25 and neighbors_table["distance"].is_monotonic_increasing and sum(votes.values()) == 25, "25 neighbors, increasing distances, 25 votes", "knn25.kneighbors(Z_te[[error_idx]])")
        check(margin == max(votes.values()) - min(votes.values()) and int(y_te[error_idx]) != max(votes, key=votes.get), "the margin is the difference in votes; the true class lost the vote", "votes of the predicted class minus votes of the other")
    """,
    hint="""
        Neighbors: `d, idx = knn25.kneighbors(Z_te[[error_idx]])`; `classes = y_tr[idx[0]]`; `votes = {0: int((classes == 0).sum()), 1: int((classes == 1).sum())}`.
        Log-loss: `log_loss(y_te, p)` (with 0/1 probabilities `scikit-learn` clips them to `1e-15`).
    """,
    solution=[
        C("""
        n_values = {name: int(np.unique(p).size) for name, p in probabilities.items()}
        print(n_values)
        """),
        C("""
        table = pd.DataFrame(index=list(models), columns=["auc", "logloss", "roc_points", "ece"], dtype=float)
        for name, p in probabilities.items():
            table.loc[name] = [roc_auc_score(y_te, p), log_loss(y_te, p), len(roc_curve(y_te, p)[0]), ece(y_te, p)]
        table.round(3)
        """),
        C("""
        why_logloss_is_high = "b"
        """),
        M("""
        With `k = 1` every probability is 0 or 1: when the neighbor is wrong (about one case in five) the probability assigned to the true class is zero and the log-loss clips it to `1e-15`, that is a loss of 34.5 per example. The mean
        is dominated by those cases. It is a property of the metric applied to uncalibrated probabilities, not a defect of the ordering (the AUC of `knn1` is poor for another reason: it has only two values, that is a single threshold).
        """),
        C("""
        distances, indices = knn25.kneighbors(Z_te[[error_idx]])
        neighbors_table = pd.DataFrame({"index": indices[0], "distance": distances[0], "class": y_tr[indices[0]]})
        votes = {int(c): int((neighbors_table["class"] == c).sum()) for c in (0, 1)}
        predicted = max(votes, key=votes.get)
        margin = votes[predicted] - votes[1 - predicted]
        print(f"test example {error_idx}: true class {y_te[error_idx]}, predicted {predicted} | votes {votes} | margin {margin}")
        neighbors_table.round(3).head(5)
        """),
    ],
    learned="""
        - **The `predict_proba` of a uniform KNN takes only `k + 1` values** (0, 1/k, ..., 1): with `k = 5` that is 6. The ROC curve therefore has at most `k + 2` points and the AUC is a coarse estimate; with 1/distance weights the values become thousands (almost one per example), but the probabilities remain
          fractions of "weighted" neighbors, not risk estimates.
        - **The log-loss of a KNN with small `k` is enormous**: with `k = 1` it is 5.3 (the logistic: 0.18), with `k = 5` 0.62, with `k = 25` 0.34. With `k = 1` the probabilities are exactly 0 or 1 and every error costs 34.5; with `k = 5` five agreeing neighbors are enough for a
          "sure" error. Don't compare the log-loss of a KNN with small `k` with that of other models without thinking about this effect.
        - Calibration: the logistic has an ECE of 0.06, the KNN with `k = 25` of 0.12 (double: fractions of neighbors are not true probabilities) and the 1/distance weights don't change much (0.11). **Beware of `k = 1`, which looks the best (0.025):** with only two distinct values the ten groups
          collapse into one and the ECE compares only the overall mean. A calibration metric with quantile groups is unreliable with discrete scores. If you need true probabilities, recalibrate (`CalibratedClassifierCV`).
        - KNN is **explainable by construction**: for every prediction you can show the `k` training examples that voted, with distances and classes. It is the only model of this course for which the explanation is "the most similar case to this one is that one".
    """,
    pitfall="""
        Using the `predict_proba` of a KNN with small `k` as a "probability" in a risk or expected cost computation. They are fractions of a few neighbors: noisy, discrete and uncalibrated.
    """,
    interview=(
        "Can you give a probability with a KNN?",
        """
        Yes, but with caution: the fraction of neighbors is a coarse, discrete estimate, poorly calibrated with small `k`. I use a larger `k`, check the calibration with a curve or an indicator like the ECE and, if needed, recalibrate it (sigmoid or isotonic)
        on data not used for training. If the probabilities are central to the use case, I consider a model that estimates them directly, like the logistic.
        """),
)

EXERCISES = [D1, D2]
