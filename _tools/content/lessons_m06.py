"""Module 6 lessons: PCA."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_06_pca"
TITLE = "PCA: principal component analysis"
LESSONS = [
    ("01_principal_components.ipynb", "6.1", "The principal components",
     "Directions of maximum variance, SVD, scores and reconstruction: what `PCA` really computes."),
    ("02_how_many_components.ipynb", "6.2", "How many components to keep",
     "Scree plot, cumulative explained variance, choice with cross-validation and reconstruction error."),
    ("03_pca_in_practice.ipynb", "6.3", "PCA in practice",
     "Inside a pipeline, to visualize, to remove noise, and its limits (nonlinearity, signal in low-variance directions)."),
]
PATH = [
    ("Lesson 6.1", ["A1", "D1", "E3", "C2"]),
    ("Lesson 6.2", ["A2", "B1", "D2", "E1"]),
    ("Lesson 6.3", ["B2", "B3", "C1", "E2", "F1", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_6_1() -> List[Cell]:
    """Cells of lesson 6.1."""
    return [
        M("""
        # Lesson 6.1 · The principal components

        PCA answers one question: **along which directions do the data vary the most?** It finds a new system of axes (the principal components), ordered by the variance they capture, and lets you
        keep only the first ones. It is not a prediction model: it is a **transformation of the data** used to reduce dimensions, visualize, remove noise or decorrelate the features.

        **What you will learn**

        - What PCA computes (centering, SVD) and how to read `components_`, `explained_variance_ratio_` and the scores.
        - Why the scale of the columns decides the result.
        - How to reconstruct the data from the components.
        """),
        _setup("""
        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        # two correlated variables: a cloud stretched along a diagonal
        x1 = rng.normal(0, 2, 300)
        X2d = np.c_[x1, 0.8 * x1 + rng.normal(0, 0.7, 300)]
        pca2 = PCA().fit(X2d)
        print("explained variance per component:", pca2.explained_variance_ratio_.round(3))
        print("first component (direction):", pca2.components_[0].round(3))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(*X2d.T, s=8, alpha=0.4)
        for comp, var in zip(pca2.components_, pca2.explained_variance_):
            ax.arrow(*pca2.mean_, *(comp * 2 * np.sqrt(var)), color="red", width=0.04, head_width=0.25)
        ax.set_aspect("equal"); plt.show()
        """),
        M("""
        ## How it works

        1. **Center** every column (subtract the mean): PCA studies the variance around the center of the cloud.
        2. Find the direction along which the data have **maximum variance**: it is the first component. The second is the direction of maximum variance **orthogonal to the first**, and so on.
        3. The result is a rotation: the rotated points (the **scores**) have columns that are uncorrelated with each other, ordered by decreasing variance.

        Numerically it is obtained with the **SVD** of the centered matrix $X_c = U\\,S\\,V^\\top$: the rows of $V^\\top$ are the components (`components_`), the variances are $S^2/(n-1)$.
        """),
        C("""
        X = load_wine(as_frame=True).data
        Xc = (X - X.mean()).to_numpy()
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        variances = S ** 2 / (len(X) - 1)
        pca = PCA().fit(X)
        print("SVD by hand   :", (variances / variances.sum())[:3].round(4))
        print("sklearn PCA   :", pca.explained_variance_ratio_[:3].round(4))
        print("components equal up to the sign:", np.allclose(np.abs(Vt), np.abs(pca.components_)))
        """),
        M("""
        The result is identical: `PCA` is an SVD of the centered data. The **sign** of the components is arbitrary (`v` and `-v` are the same direction): scikit-learn fixes one with a convention, other libraries another.
        Never compare signs across different models.

        ## The scale decides the result

        In the wine dataset the variance of `proline` is enormously larger than all the others. See what PCA does on the raw data and on the standardized ones.
        """),
        C("""
        raw = PCA().fit(X)
        scaled = PCA().fit(StandardScaler().fit_transform(X))
        print("explained variance (raw data)         :", raw.explained_variance_ratio_[:4].round(3))
        print("explained variance (standardized data):", scaled.explained_variance_ratio_[:4].round(3))
        print("column with the largest loading in PC1 (raw):", X.columns[np.abs(raw.components_[0]).argmax()])
        """),
        M("""
        On the raw data the first component **is `proline`**: it explains 99.8% of the variance simply because that column has the largest variance. It is not a discovery about wines: it is an effect of the units of measure. After
        standardization the variance spreads over more components and PCA discovers the true correlations between the measurements.

        **Rule:** before a PCA you standardize, unless the columns already have the same unit and the same scale and you really want to preserve the differences in variance (pixels, homogeneous measurements).

        > **Common mistake.** Interpreting the `explained_variance_ratio_` of a PCA on raw data as if it said "how important" a direction is. It says how much variance there is, which depends on the units.

        ## Projecting and reconstructing

        `transform` projects the data onto the components (scores); `inverse_transform` brings them back to the original space. With all the components the reconstruction is exact; with `k` components it is the best possible **rank-`k` approximation**
        (the closest in squared error).
        """),
        C("""
        Z = StandardScaler().fit_transform(X)
        for k in [1, 2, 5, 13]:
            p = PCA(k).fit(Z)
            error = ((Z - p.inverse_transform(p.transform(Z))) ** 2).mean()
            print(f"{k:2d} components | mean squared reconstruction error: {error:.4f} | unexplained variance: {1 - p.explained_variance_ratio_.sum():.4f}")
        """),
        M("""
        The mean reconstruction error coincides with the unexplained variance (for standardized data, where the total variance per column is 1): **the explained variance is a measure of reconstruction fidelity**, not of usefulness for predicting anything (you will see it in exercise C1).

        ## Check your understanding

        """ + details(
            "Why does PCA center the data?",
            "Because variance is measured around the mean: without centering, the first direction would simply be the one going from the origin to the mean of the cloud. scikit-learn's `PCA` centers by itself (and saves the mean in `mean_`); `TruncatedSVD`, which does not center, "
            "is not the same thing (exercise E2).") + "\n\n" + details(
            "Are the components the most important features of the dataset?",
            "No. They are **linear combinations** of all the features (the weights are in `components_`, called loadings), ordered by variance and not by relationship with a target. PCA does not look at `y`: the most informative direction for predicting could be one of the last.")),
    ]


def lesson_6_2() -> List[Cell]:
    """Cells of lesson 6.2."""
    return [
        M("""
        # Lesson 6.2 · How many components to keep

        Reducing the dimensions means choosing `k`. There is no absolutely right `k`: it depends on what you want to do (reconstruct, visualize, feed a model). This lesson shows the three criteria in use and their limits.

        **What you will learn**

        - Scree plot, cumulative variance and the practical rules (threshold on the variance, elbow).
        - Why the best choice for a supervised model is cross-validation.
        - What `PCA(n_components=0.95)` does.
        """),
        _setup("""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_digits(return_X_y=True)                        # 1797 8x8 images = 64 pixels
        pca = PCA().fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        ax[0].plot(range(1, 65), pca.explained_variance_ratio_, "o-", ms=3); ax[0].set(title="scree plot", xlabel="component", ylabel="share of variance")
        ax[1].plot(range(1, 65), cumulative, "o-", ms=3); ax[1].axhline(0.95, color="r", ls=":"); ax[1].set(title="cumulative variance", xlabel="number of components")
        plt.tight_layout(); plt.show()
        for threshold in [0.8, 0.9, 0.95, 0.99]:
            print(f"components to explain {threshold:.0%} of the variance: {int(np.searchsorted(cumulative, threshold)) + 1}")
        """),
        M("""
        ## Three rules, three answers

        - **Threshold on the cumulative variance** (90%, 95%, 99%): simple, but arbitrary. With `PCA(n_components=0.95)` scikit-learn chooses by itself the minimum `k` that reaches the threshold.
        - **Elbow of the scree plot**: you look for the point after which every component adds little. Subjective, and on real data there is often no sharp elbow.
        - **Cross-validation on the final task**: if PCA feeds a classifier, `k` is a hyperparameter like `C`. It is the criterion to prefer because it measures what you care about.
        """),
        C("""
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        rows = {}
        for k in [2, 5, 10, 20, 30, 40, 64]:
            model = make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=3000))
            rows[k] = cross_val_score(model, X, y, cv=cv).mean()
        print("cross-validated accuracy as a function of the components kept:")
        print(pd.Series(rows).round(3).to_string())
        """),
        M("""
        The curve climbs fast and then flattens: with 20-30 components (a third to a half of the 64 pixels) the accuracy is 0.94-0.95, and from about 40 components on it is within a few thousandths of the 0.969 of the full set. The full set is the best of all, so on these clean data **the reduction does not improve the accuracy: it compresses, and the price is between one and three points of accuracy at 20-30 components**. (Removing noise can pay off when the data are noisy, as in lesson 6.3.)
        The point where the curve flattens is the one you care about, and it does not necessarily coincide with 95% of the variance: 95% is reached with 29 components (accuracy around 0.95), while the curve only flattens around 40, which is about 99% of the variance (41 components).

        ## How much variance is enough?

        The explained variance measures the fidelity of the **reconstruction**, not the ability to discriminate the classes. Components with little variance can contain the information the target needs (C1). That is why thresholds on the variance are a starting point to
        explore, not a selection criterion.

        > **Common mistake.** Choosing `k` by looking at the explained variance on the whole dataset and then evaluating the model in cross-validation with that `k`: PCA was learned on the validation folds too. Put PCA in the `Pipeline` and search `k` with `GridSearchCV`.

        ## Check your understanding

        """ + details(
            "What happens with `PCA(n_components=0.95)`?",
            "With a number between 0 and 1, scikit-learn keeps the minimum number of components that explain at least that fraction of variance. It is convenient, but the number of components becomes dependent on the training data and changes from one fold to another: "
            "in cross-validation it is part of the model.") + "\n\n" + details(
            "Do I have to keep the components with explained variance greater than 1 (Kaiser rule)?",
            "The Kaiser rule makes sense only on standardized data (a component with variance below 1 explains less than a single original variable). It is a shortcut: on data with many redundant columns it tends to keep too many or too few components. Prefer cross-validation.")),
    ]


def lesson_6_3() -> List[Cell]:
    """Cells of lesson 6.3."""
    return [
        M("""
        # Lesson 6.3 · PCA in practice

        Three typical uses, and the situations in which PCA misleads.

        **What you will learn**

        - PCA inside a `Pipeline` (for a KNN, for a regression).
        - PCA to **visualize** and to **remove noise**.
        - The limits: nonlinear structure and informative low-variance components (the units of measure were in lesson 6.1).
        """),
        _setup("""
        from sklearn.datasets import load_digits, make_circles
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_digits(return_X_y=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        """),
        M("""
        ## 1. Visualizing

        The first two components give a 2D map of the data: it is the fastest way to "look at" a dataset with many columns.
        """),
        C("""
        P = PCA(2).fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        sc = ax.scatter(*P.T, c=y, cmap="tab10", s=8)
        ax.set(xlabel="PC1", ylabel="PC2"); plt.colorbar(sc, label="digit"); plt.show()
        print("variance explained by the first two components:", PCA(2).fit(X).explained_variance_ratio_.sum().round(3))
        """),
        M("""
        Only 29% of the variance sits in the first two components: some digits (the 0, for example) form a well separated group, others stay mixed. A 2D map is a **projection**, and the distances in the map are not the true ones. It must be read with caution (exercise D2).

        ## 2. Removing noise

        If the signal sits in the first components and the noise is spread over all of them, reconstructing with few components **filters** the noise.
        """),
        C("""
        rng = np.random.default_rng(0)
        noisy = X + rng.normal(0, 4, X.shape)                                    # strong noise (the digits have pixels from 0 to 16)
        for k in [64, 30, 15, 8]:
            p = PCA(k).fit(noisy)
            reconstructed = p.inverse_transform(p.transform(noisy))
            print(f"{k:2d} components | error with respect to the clean image: {((reconstructed - X) ** 2).mean():.2f}")
        """),
        M("""
        With all 64 components the error is that of the noise (16); with 30 components it drops to 9.1 and with 15 to 7.1, less than half: the reconstruction with few components **is closer to the clean original than the noisy image is**. With too few
        (8) the error goes up again (8.3) because part of the signal is lost too. It is the same bias-variance tradeoff, applied to the reconstruction.

        ## 3. Inside a pipeline

        PCA always goes inside the `Pipeline`, so it is learned on the training part of every fold alone. With a KNN on these **clean** digits it does not improve the accuracy (0.986 on the 64 pixels, 0.983 with 20 components): it compresses to a third of the dimensions for a cost of a few tenths of a point. The advantage on noise you measure in exercise B2.
        """),
        C("""
        for name, m in {"KNN on 64 pixels": KNeighborsClassifier(5),
                        "PCA(20) + KNN": make_pipeline(PCA(20), KNeighborsClassifier(5)),
                        "PCA(10) + KNN": make_pipeline(PCA(10), KNeighborsClassifier(5))}.items():
            print(f"{name:16s} accuracy {cross_val_score(m, X, y, cv=cv).mean():.3f}")
        """),
        M("""
        ## 4. Warning: PCA is linear and does not look at `y`

        Two concentric circles are not separable with a line, and no rotation (which is what PCA does) makes them separable: PCA cannot help. And if the signal sits in a low-variance direction, PCA discards it.
        """),
        C("""
        Xc, yc = make_circles(1000, noise=0.05, factor=0.4, random_state=0)
        for k in [1, 2]:
            m = make_pipeline(PCA(k), LogisticRegression())
            print(f"concentric circles, PCA({k}) + logistic: accuracy {cross_val_score(m, Xc, yc, cv=cv).mean():.2f}")
        """),
        M("""
        With one or two components (in 2D, `PCA(2)` is just a rotation) the accuracy stays at 0.47 and 0.46, that is at chance: the problem is not linear and linear PCA does not change that (see F2 for the nonlinear alternatives, KernelPCA).

        > **Common mistake.** Using PCA as "selection of the best features": it orders the directions by variance, not by usefulness toward a target.

        ## Check your understanding

        """ + details(
            "When is it worth applying PCA before a supervised model?",
            "When there are many correlated or noisy columns (KNN, unregularized linear regression, models sensitive to collinearity), when the data are too large, or for a first visualization. With trees and boosting almost never: they are already "
            "robust to redundant features and the rotation can even make things worse (trees split along the original axes).") + "\n\n" + details(
            "Is PCA interpretable?",
            "Less than the original columns: every component is a combination of all of them. You interpret it by looking at the loadings (`components_`) and asking which variables "
            "move together, but a model on the components loses the direct reading \"this column does this\".")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_principal_components.ipynb", lesson_6_1),
                    ("02_how_many_components.ipynb", lesson_6_2),
                    ("03_pca_in_practice.ipynb", lesson_6_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
