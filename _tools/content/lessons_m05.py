"""Module 5 lessons: k-nearest neighbors."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_05_knn"
TITLE = "K-nearest neighbors"
LESSONS = [
    ("01_the_neighbors_model.ipynb", "5.1", "The neighbors model",
     "Predicting with the k nearest points: what `k` means, how the boundary looks and why the model is the dataset."),
    ("02_scale_distance_and_dimensions.ipynb", "5.2", "Scale, distance and the curse of dimensionality",
     "Why scaling decides everything, how metric and weights change and what happens when the useless features increase."),
    ("03_knn_in_practice.ipynb", "5.3", "KNN in practice",
     "Prediction cost, choice of `k`, regression with neighbors, KNNImputer and when it makes sense to use it."),
]
PATH = [
    ("Lesson 5.1", ["A1", "C2", "D1", "E1"]),
    ("Lesson 5.2", ["E3", "F1", "C1", "E2", "D2"]),
    ("Lesson 5.3", ["A2", "B1", "B2", "B3", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_5_1() -> List[Cell]:
    """Cells of lesson 5.1."""
    return [
        M("""
        # Lesson 5.1 · The neighbors model

        KNN is the simplest model there is and for that very reason it is an excellent tool to understand what a model is: **it learns nothing**. It memorizes the training set and,
        for every new point, looks at the `k` nearest examples and reports their most frequent class (or the mean of the target, in regression). "Similar to what I already know" is the only idea.

        **What you will learn**

        - What `fit` and `predict` do in a model that has no parameters to estimate.
        - Why `k` is the complexity regulator (bias and variance) and how it shows on the decision boundary.
        - How to read `predict_proba` and `kneighbors`.
        """),
        _setup("""
        from sklearn.datasets import make_moons
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        print(X_tr.shape, "| classes:", np.bincount(y_tr))
        """),
        M("""
        ## What "training" a KNN means

        `fit` does not compute coefficients: it stores `X_tr` and `y_tr` (and builds, if needed, a structure to search quickly). All the work happens in **`predict`**:
        for every test point it computes the distances from all the training examples, takes the `k` smallest and votes.

        ```
        predict(x):  d = distances(x, X_tr) → the k indices with minimum d → most frequent class among their y
        ```
        """),
        C("""
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
        x = X_te[:1]
        distances, indices = knn.kneighbors(x)                      # the 5 neighbors of a point, with the distances
        print("indices of the neighbors:", indices[0], "| distances:", distances[0].round(3))
        print("classes of the neighbors:", y_tr[indices[0]], "-> predict:", knn.predict(x), "| predict_proba:", knn.predict_proba(x))
        """),
        M("""
        `predict_proba` is not a calibrated probability: it is the **fraction of neighbors** of each class (with `k = 5` it has only the values 0, 0.2, ..., 1). You come back to it in exercise D2.

        ## `k` regulates the complexity

        - `k = 1`: every point takes the label of the nearest neighbor. The boundary follows every single example, noise included: **overfitting** (low bias, high variance), and on the training set the accuracy is 1.
        - `k = n`: everyone predicts the most frequent class: **underfitting**.
        - In between the boundary smooths out. Look at it.
        """),
        C("""
        grid1, grid2 = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
        points = np.c_[grid1.ravel(), grid2.ravel()]
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
        for a, k in zip(ax, [1, 15, 100]):
            m = KNeighborsClassifier(k).fit(X_tr, y_tr)
            a.contourf(grid1, grid2, m.predict(points).reshape(grid1.shape), alpha=0.25, cmap="coolwarm")
            a.scatter(*X_tr.T, c=y_tr, s=12, cmap="coolwarm", edgecolor="k", linewidth=0.3)
            a.set_title(f"k = {k}: train {m.score(X_tr, y_tr):.2f} | test {m.score(X_te, y_te):.2f}")
        plt.tight_layout(); plt.show()
        """),
        M("""
        With `k = 1` the boundary is jagged and the training accuracy is 1.00, but the test one is lower. With `k = 100` the boundary is almost a line and the model underestimates the
        curvature. The compromise is found with cross-validation, like for every hyperparameter (lesson 5.3).

        **Watch out for a detail.** With `k = 1` the prediction on a **training** point is the label of the point itself (the nearest neighbor of a training point is itself, distance 0): that is why
        the training accuracy is 1 and says nothing. Whoever evaluates a KNN on the data it was trained on gets a meaningless number (exercise E1).

        ## Weights and voting

        By default every neighbor weighs the same (`weights="uniform"`). With `weights="distance"` every neighbor weighs `1/distance`: the really close neighbors count more and ties are almost always broken. For the same reason as before, with the weights the training point has distance 0 from itself and weighs infinitely: the **training** accuracy with `weights="distance"` is always 1, whatever `k` is (look at the table below).
        """),
        C("""
        for w in ["uniform", "distance"]:
            m = KNeighborsClassifier(15, weights=w).fit(X_tr, y_tr)
            print(f"{w:9s} | train {m.score(X_tr, y_tr):.3f} | test {m.score(X_te, y_te):.3f}")
        """),
        M("""
        > **Common mistake.** Thinking that a KNN "without training" cannot overfit. KNN with small `k` is one of the models most prone to overfitting: all the complexity sits in `k`.

        ## Check your understanding

        """ + details(
            "Why is the accuracy on the training set always 1 with `k=1` (without duplicates with different labels)?",
            "Because the nearest neighbor of every training point is the point itself, at distance 0, and its label is the true one. To evaluate a KNN you need a test set or cross-validation, never the training set.") + "\n\n" + details(
            "If the two sides of a tie have the same number of neighbors, what does scikit-learn do?",
            "With `weights=\"uniform\"` it chooses the class with the smallest label (the tie is resolved deterministically but arbitrarily). That is why with two classes odd `k` are preferred, and with more classes `weights=\"distance\"` is better.")),
    ]


def lesson_5_2() -> List[Cell]:
    """Cells of lesson 5.2."""
    return [
        M("""
        # Lesson 5.2 · Scale, distance and the curse of dimensionality

        All of KNN depends on **one** thing: what you mean by "near". The distance sums the contributions of all the features, and so the features with a large scale or no information dominate
        the result. This lesson shows two very concrete effects.

        **What you will learn**

        - Why scaling is mandatory and not a detail, and what different metrics do.
        - Why adding useless features **ruins** KNN, and what it means that in many dimensions "everyone is far".
        """),
        _setup("""
        from sklearn.datasets import load_wine, make_moons
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        wine = load_wine(as_frame=True)
        X, y = wine.data, wine.target
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        print(X.shape, "| std dev of the features: min", X.std().min().round(2), "max", X.std().max().round(1))
        X.std().round(2).sort_values(ascending=False).head(4)
        """),
        M("""
        ## The scale decides who counts

        The Euclidean distance sums the squares of the differences. In the wine dataset `proline` has a standard deviation of about 315, the second (`magnesium`) 14 and all the others about 3 or less: in the distance
        `proline` does almost **everything** and the other 12 columns weigh very little.
        """),
        C("""
        V = X.to_numpy()
        squares = (V[:, None, :] - V[None, :, :]) ** 2                   # contribution of every feature to the squared distance, for every pair of wines
        print("mean share of proline in the squared distance:", round(squares[:, :, list(X.columns).index("proline")].sum() / squares.sum(), 4))

        without = KNeighborsClassifier(7)
        with_scaler = make_pipeline(StandardScaler(), KNeighborsClassifier(7))
        print("accuracy without scaling:", cross_val_score(without, X, y, cv=cv).mean().round(3), "| with StandardScaler:", cross_val_score(with_scaler, X, y, cv=cv).mean().round(3))
        """),
        M("""
        With scaling the accuracy goes up by almost thirty points (from 0.66 to 0.95). It is not a novelty of KNN: the same holds for all distance-based models (and for PCA, module 6). The scaler goes inside the `Pipeline`, so it is learned
        on the training part of every fold alone.

        ## Metrics

        The default metric is Minkowski with `p=2` (Euclidean). Other useful choices: `manhattan` (sum of absolute values: less sensitive to large gaps on a single feature), `chebyshev` (the maximum gap), `cosine`
        (angle between the vectors: ignores the length, useful for text and counts). Scaling changes much more than the metric, as you will see in exercise F1.

        ## More features does not mean more information

        Let's add columns of pure noise to a 2-dimensional problem (the moons) and look at the accuracy.
        """),
        C("""
        rng = np.random.default_rng(0)
        Xm, ym = make_moons(1500, noise=0.25, random_state=0)
        rows = []
        for n_noise in [0, 2, 5, 10, 20, 50, 100]:
            Xr = np.c_[Xm, rng.normal(size=(len(Xm), n_noise))]
            rows.append({"noise columns": n_noise, "accuracy (k=5)": cross_val_score(KNeighborsClassifier(5), Xr, ym, cv=cv).mean()})
        pd.DataFrame(rows).round(3).set_index("noise columns").T
        """),
        M("""
        The two useful dimensions do not change, but every noise column adds a random term to the squared distance: the "true" neighbors are covered by the accumulated random distance. With 100 useless columns the accuracy falls from 0.93 to 0.57, little better than chance (0.5).
        It is the **curse of dimensionality**: in many dimensions the distances between points concentrate (the nearest neighbor is almost as far as the farthest) and "near" loses meaning.

        > **Common mistake.** Adding all the available features "because maybe they help". With distance-based models every useless feature has a cost; feature selection (or a PCA, module 6) is part of the model.

        ## Check your understanding

        """ + details(
            "Do I need to scale the dummy (0/1) variables of a one-hot before the KNN?",
            "It depends on how much you want them to weigh: a 0/1 dummy has a smaller scale than a standardized feature, and a category with many levels adds many dummies to the distance. A common choice is to scale the numerics, leave the dummies at 0/1 and check with cross-validation whether a different weight helps.") + "\n\n" + details(
            "Why does the cosine work with text?",
            "Because a long document and a short one on the same topic have count vectors with the same direction but different lengths: the Euclidean distance considers them far, the cosine (which looks only at the angle) near.")),
    ]


def lesson_5_3() -> List[Cell]:
    """Cells of lesson 5.3."""
    return [
        M("""
        # Lesson 5.3 · KNN in practice

        With scaling in place and few useful features KNN is a surprisingly strong baseline. It has, however, three characteristics to know: prediction cost, sensitivity to the choice of `k` and to the size of the data, and
        a use that goes beyond classification (regression, imputation of missing values).

        **What you will learn**

        - How much `predict` costs and how the search algorithms (`brute`, `kd_tree`, `ball_tree`) change things.
        - How `k` is chosen and how regression with neighbors works.
        - `KNNImputer` as an alternative to imputation with the mean.
        """),
        _setup("""
        import time
        from sklearn.impute import KNNImputer
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, validation_curve
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_classification
        """),
        M("""
        ## The cost is all in `predict`

        `fit` is almost free, `predict` computes the distance from every training point. With `brute` the cost is $O(n\\cdot d)$ for every prediction. With `kd_tree` and `ball_tree` scikit-learn organizes the data in a
        tree to skip most of the comparisons: very effective with **few dimensions** (up to about 10-15), of little use with many.
        """),
        C("""
        rng = np.random.default_rng(0)
        for d in [3, 30]:
            Xb = rng.normal(size=(30000, d)); Xq = rng.normal(size=(1000, d))
            row = []
            for alg in ["brute", "kd_tree", "ball_tree"]:
                m = KNeighborsClassifier(5, algorithm=alg).fit(Xb, rng.integers(0, 2, len(Xb)))
                t = time.time(); m.predict(Xq); row.append(f"{alg} {time.time() - t:.2f}s")
            print(f"{d:2d} dimensions, 30000 training examples, 1000 predictions: " + " | ".join(row))
        """),
        M("""
        With 3 dimensions the tree is much faster than the exhaustive computation (about 5 times); with 30 it becomes **a disadvantage** (over twenty times slower: building and visiting the tree costs more than the direct comparison). The default (`algorithm="auto"`) chooses for you.
        It is also the reason why a KNN in production with millions of examples becomes a problem of **approximate search** (ANN indexes), not of scikit-learn.

        ## Choosing `k`

        It is chosen with cross-validation like any hyperparameter, looking at the whole curve and not only at the maximum (D1 of module 3). As a starting point the rule of thumb $k \\approx \\sqrt{n}$ holds, then you adjust.
        """),
        C("""
        Xc, yc = make_classification(1500, n_features=8, n_informative=4, n_redundant=0, class_sep=0.8, random_state=0)
        ks = [1, 3, 5, 9, 15, 25, 41, 61, 91]
        train, val = validation_curve(make_pipeline(StandardScaler(), KNeighborsClassifier()), Xc, yc, param_name="kneighborsclassifier__n_neighbors", param_range=ks,
                                      cv=StratifiedKFold(5, shuffle=True, random_state=0))
        pd.DataFrame({"train": train.mean(1), "validation": val.mean(1)}, index=ks).round(3).T
        """),
        M("""
        The training score collapses as `k` grows while the validation one goes up, touches a maximum and then goes down: it is the U-shaped curve of complexity. Note that the maximum is **flat**: many values of `k` are equivalent
        within the uncertainty of the estimate; when in doubt you choose the simplest (larger `k`, smoother model).

        ## Regression with neighbors

        `KNeighborsRegressor` predicts the **mean** of the target of the `k` neighbors. It does not extrapolate (outside the training range the prediction stays flat) but it captures nonlinear relationships without writing any function.
        """),
        C("""
        x = np.sort(rng.uniform(0, 6, 120)); yr = np.sin(x) + rng.normal(0, 0.25, 120)
        grid = np.linspace(-1, 7, 300).reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ax.scatter(x, yr, s=12, color="gray")
        for k in [1, 10, 40]:
            ax.plot(grid, KNeighborsRegressor(k).fit(x.reshape(-1, 1), yr).predict(grid), label=f"k = {k}")
        ax.legend(); plt.show()
        """),
        M("""
        With `k = 1` the curve "jumps" from one point to the next (it follows the noise); with `k = 40` it is too smooth; outside [0, 6] the prediction is constant: **KNN does not extrapolate**.

        ## KNNImputer: neighbors to fill the missing values

        The same idea for missing values: `KNNImputer` fills a missing value with the mean of the `k` neighbors (computed on the available columns). It requires **scaled** features and works well when the columns are correlated with each other:
        you will see it in exercise B3.

        > **Common mistake.** Using KNN because "it is simple" on a dataset with hundreds of features and millions of rows: it is not simple to put in production, and in high dimension it is almost always beaten by a linear model or by trees.

        ## Check your understanding

        """ + details(
            "When does KNN make sense?",
            "As a quick baseline to try, with few scaled and meaningful features, data that is not huge and irregular boundaries; in recommendation and retrieval systems (with approximate indexes). It does not make sense with many irrelevant features, "
            "huge data to serve with low latency, or when you need to extrapolate.") + "\n\n" + details(
            "How do you choose between `kd_tree` and `brute`?",
            "Usually you don't: `algorithm=\"auto\"` decides. In general trees help with few dimensions and many examples; with many dimensions (over 15-20) or few examples the exhaustive computation is just as fast.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_the_neighbors_model.ipynb", lesson_5_1),
                    ("02_scale_distance_and_dimensions.ipynb", lesson_5_2),
                    ("03_knn_in_practice.ipynb", lesson_5_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
