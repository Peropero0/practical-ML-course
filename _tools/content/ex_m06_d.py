"""Module 6 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_loadings_and_variance_on_tumors", kind="D", level=2, minutes=45, module=MODULE,
    title="Reading a PCA: loadings, explained variance and relationship with the diagnosis",
    goal="""
        Interpret a PCA on 30 measurements of breast nodules: how much variance each component captures, which variables define it (loadings), why the first component is almost a "general factor", how the explained variance links to the redundancy between variables
        (three nearly identical size measurements) and whether the first component separates benign from malignant tumors.
    """,
    dataset="""
        `load_breast_cancer` (569 nodules, 30 features: mean, standard error and worst value of 10 measurements, target 1 = benign), standardized: `Z` (569 × 30), with `names` (the column names) and `y`.
    """,
    purpose="""
        Read the output of a PCA the way you would read a table in a paper: not "how many components" but **what** they represent, which columns move together and what it means to have 44% of the variance on the first component.
    """,
    task=[
        "Train `PCA().fit(Z)` and save `ratio` (`explained_variance_ratio_`) and `k_for_threshold`: dictionary `{0.8: k, 0.9: k, 0.95: k}` with the minimum number of components to reach each cumulative variance threshold.",
        "For the **first component**: `top_pc1` (list of the names of the 5 variables with the largest |loading|) and `pc1_sign_share` (fraction of loadings with the same sign as the sum of the loadings: if it is close to 1 all the variables contribute in the same direction).",
        "**Redundancy:** `size_corr` (the minimum correlation between `mean radius`, `mean perimeter` and `mean area`) and `three_sizes_ratio` (variance explained by the first component of a PCA done only on those three standardized columns).",
        "**Relationship with the diagnosis:** `diagnosis_corr` (array of 6 values: |correlation| of each of the first 6 components with `y`) and `most_correlated_pc` (order number, from 1). Then `acc_k` (dictionary `k → cross-validated accuracy` of `LogisticRegression` on the first `k` components, with `k` in `[1, 2, 5, 10, 30]`, `StratifiedKFold(5, shuffle=True, random_state=0)`).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        data = load_breast_cancer()
        names, y = list(data.feature_names), data.target
        Z = StandardScaler().fit_transform(data.data)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        pca = PCA().fit(Z)
        ratio = None
        k_for_threshold = None
        """),
        C("""
        top_pc1 = None
        pc1_sign_share = None
        """),
        C("""
        size_corr = None
        three_sizes_ratio = None
        """),
        C("""
        diagnosis_corr = None
        most_correlated_pc = None
        acc_k = None
        """),
    ],
    checks="""
        check(close(ratio, pca.explained_variance_ratio_) and 0.42 < ratio[0] < 0.46 and 0.18 < ratio[1] < 0.2, "the first component explains 44% of the variance, the second 19%", "explained_variance_ratio_")
        check(k_for_threshold == {0.8: 5, 0.9: 7, 0.95: 10}, "5 components for 80%, 7 for 90%, 10 for 95% (out of 30 variables)", "np.searchsorted(np.cumsum(ratio), threshold) + 1")
        check(len(top_pc1) == 5 and "mean concave points" in top_pc1 and pc1_sign_share > 0.85, "the variables with the largest weight describe shape and concavity; almost all variables contribute with the same sign", "np.abs(pca.components_[0]) sorted, and the sign of each loading")
        check(size_corr > 0.98 and three_sizes_ratio > 0.99, "radius, perimeter and area are almost the same variable: a single component explains over 99%", "minimum correlation between the three columns and PCA on those three")
        check(len(diagnosis_corr) == 6 and most_correlated_pc == 1 and diagnosis_corr[0] > 0.75 and (diagnosis_corr[1:] < 0.25).all(), "only the first component is linked to the diagnosis (correlation over 0.75); the others below 0.25", "np.corrcoef(scores[:, i], y)")
        check(set(acc_k) == {1, 2, 5, 10, 30} and acc_k[1] > 0.85 and acc_k[10] > acc_k[1] and acc_k[30] > 0.95, "a single component already gives over 0.85; with 10 components it improves; with 30 over 0.95", "LogisticRegression on PCA(k) in a pipeline")
    """,
    hint="""
        Loadings: `pd.Series(pca.components_[0], index=names).abs().nlargest(5).index`. Sign: `np.sign(l.sum())` against `np.sign(l)`. Three sizes: `columns = [names.index(c) for c in ["mean radius", "mean perimeter", "mean area"]]`; `np.corrcoef(Z[:, columns].T).min()`;
        `PCA().fit(Z[:, columns]).explained_variance_ratio_[0]`. Components in the pipeline: `make_pipeline(PCA(k), LogisticRegression(max_iter=2000))` (the data are already standardized).
    """,
    solution=[
        C("""
        pca = PCA().fit(Z)
        ratio = pca.explained_variance_ratio_
        cumulative = np.cumsum(ratio)
        k_for_threshold = {s: int(np.searchsorted(cumulative, s)) + 1 for s in (0.8, 0.9, 0.95)}
        print("variance share of the first 5 components:", ratio[:5].round(3), "| components per threshold:", k_for_threshold)
        """),
        C("""
        l1 = pd.Series(pca.components_[0], index=names)
        top_pc1 = list(l1.abs().nlargest(5).index)
        pc1_sign_share = float((np.sign(l1) == np.sign(l1.sum())).mean())
        print("variables with the largest weight in PC1:", top_pc1)
        print(f"PC1 loadings with the same sign as the sum: {pc1_sign_share:.0%}")
        """),
        C("""
        columns = [names.index(c) for c in ["mean radius", "mean perimeter", "mean area"]]
        size_corr = np.corrcoef(Z[:, columns].T).min()
        three_sizes_ratio = PCA().fit(Z[:, columns]).explained_variance_ratio_[0]
        print(f"minimum correlation between radius, perimeter and area: {size_corr:.3f} | a PCA on those three columns: the first component explains {three_sizes_ratio:.1%}")
        """),
        C("""
        scores = pca.transform(Z)
        diagnosis_corr = np.abs([np.corrcoef(scores[:, i], y)[0, 1] for i in range(6)])
        most_correlated_pc = int(diagnosis_corr.argmax()) + 1
        acc_k = {k: cross_val_score(make_pipeline(PCA(k), LogisticRegression(max_iter=2000)), Z, y, cv=cv).mean() for k in [1, 2, 5, 10, 30]}
        print("correlation with the diagnosis of the first 6 components:", diagnosis_corr.round(2))
        print("accuracy with k components:", {k: round(v, 3) for k, v in acc_k.items()})
        """),
    ],
    learned="""
        - **The first component is a "general factor":** it explains 44% of the variance and **all 30** variables have the same sign in the loadings (the strongest are the concavity and shape measurements). "Larger and more irregular" tumors have higher values in almost all the measurements:
          it is exactly the axis that separates benign from malignant (correlation with the diagnosis 0.79, against less than 0.2 for all the others).
        - **Redundancy lowers the effective dimensionality:** radius, perimeter and area (means) are correlated over 0.98 and are in practice a single variable (a PCA on them gives 99% with one component). With 10 measurements repeated in three versions (mean, error, worst),
          in 30 columns there are many near-copies: hence 5 components for 80% and 10 for 95%.
        - **One component is enough for 91%:** in a simple problem like this one a single component already gives a good classifier (0.91), with 2 you reach 0.95 and with 10 0.98, equal to the one with all 30 variables (0.979). It is a case where the explained variance and the usefulness for the target coincide (contrast with C1).
        - Loadings are read in groups: variables with the same sign and similar weight "move together". The **sign** of a component is arbitrary: saying "PC1 is high in malignant tumors" or "low" depends on the convention of that run; what matters is the structure of the weights, not the absolute sign.
    """,
    pitfall="""
        Giving a name to a component ("the severity component") as if it were a measured variable. It is a linear combination that summarizes the variance, and with different data (or another scaler) the same component can change composition.
    """,
    interview=(
        "How do you interpret the principal components?",
        """
        I look at the explained variance (scree plot) and at the loadings: which variables have a high weight and with which sign, so I see which ones move together. I verify the relationship with external variables (for example the correlation of the scores with the target), check the stability (bootstrap) and
        don't read the sign in absolute terms. I remember that a component is a direction of maximum variance and not a causal factor.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="what_pca_sees_in_the_digits", kind="D", level=3, minutes=50, module=MODULE,
    title="What PCA sees in the digits: pixels, centroids and lost information",
    goal="""
        Read a PCA on images: which pixels never vary, where the variance sits (center versus edge), how many components are needed for a given share of variance, how the classes arrange themselves in the reduced space (which digits get confused) and how much classification information
        is lost by reducing to 2, 10 or 30 dimensions (with a centroid classifier).
    """,
    dataset="""
        `load_digits` (1797 8×8 images = 64 pixels, 10 classes): `X` (1797 × 64), `y`, and `images = X.reshape(-1, 8, 8)`.
    """,
    purpose="""
        Link the PCA numbers to the images: the pixels are also spatial coordinates, and the components are "images" (weights on the pixels). You will learn to ask the data "where is the variance" and to measure what a projection to 2 dimensions costs.
    """,
    task=[
        "Save `constant_pixels` (number of pixels with zero variance) and `edge_variance_share`: share of the total variance of the pixels of the **outer ring** of the image (rows 0 and 7 and columns 0 and 7: 28 pixels) out of the 64.",
        "Train `PCA().fit(X)`; save `k90` and `k99` (number of components for 90% and 99% of the cumulative variance) and `pc1_image` (the first component as an 8×8 matrix) with `pc1_center` = mean of |weight| of the 16 central pixels (rows 2-5, columns 2-5) minus the mean of the 28 edge pixels.",
        "With the first two components compute the **centroids** of the 10 classes (`centroids`, 10 × 2 matrix) and the matrix of the distances between centroids; save `closest_pair` (the pair `(a, b)`, with `a < b`, of digits with the closest centroids) and `isolated_class` (the digit whose closest centroid is the farthest of all).",
        "**Lost information:** with `NearestCentroid` in cross-validation (`StratifiedKFold(5, shuffle=True, random_state=0)`), on the components alone, compute `centroid_acc` (dictionary `k → accuracy`) for `k` in `[2, 10, 30, 64]`, with PCA inside the pipeline.",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import NearestCentroid
        from sklearn.pipeline import make_pipeline

        X, y = load_digits(return_X_y=True)
        images = X.reshape(-1, 8, 8)
        edge = np.zeros((8, 8), dtype=bool)
        edge[[0, 7], :] = True
        edge[:, [0, 7]] = True
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        constant_pixels = None
        edge_variance_share = None
        """),
        C("""
        pca = PCA().fit(X)
        k90 = None
        k99 = None
        pc1_image = None
        pc1_center = None
        """),
        C("""
        P2 = PCA(2).fit_transform(X)
        centroids = None
        closest_pair = None
        isolated_class = None
        """),
        C("""
        centroid_acc = None
        """),
    ],
    checks="""
        check(constant_pixels == (X.var(0) == 0).sum() and constant_pixels >= 3, "some pixels (at the corners) never vary", "(X.var(0) == 0).sum()")
        check(close(edge_variance_share, X.reshape(-1, 64).var(0)[edge.ravel()].sum() / X.var(0).sum()) and edge_variance_share < 0.25, "the outer ring (28 pixels out of 64) contains less than a quarter of the total variance", "sum of the variances of the 28 edge pixels / total")
        check(k90 == int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.9)) + 1 and 18 <= k90 <= 24 and 38 <= k99 <= 48, "about 21 components for 90% and about 41 for 99% (out of 64 pixels)", "np.searchsorted(np.cumsum(explained_variance_ratio_), threshold) + 1")
        check(pc1_image.shape == (8, 8) and pc1_center > 0.05, "the first component weighs the central pixels more than the edge ones", "components_[0].reshape(8, 8)")
        check(centroids.shape == (10, 2) and close(centroids[3], P2[y == 3].mean(0)) and closest_pair[0] < closest_pair[1] and isolated_class == 0, "centroids computed; the digit 0 is the most isolated", "mean of the scores per class and distance matrix")
        check(set(centroid_acc) == {2, 10, 30, 64} and centroid_acc[2] < 0.65 and centroid_acc[10] > 0.85 and centroid_acc[64] > 0.88, "with 2 components a centroid classifier has less than 0.65; with 10 over 0.85; with 64 over 0.88", "NearestCentroid inside the pipeline with PCA(k)")
        check(centroid_acc[10] > centroid_acc[2] + 0.25, "going from 2 to 10 components gains over 25 points", "the 2D map loses a lot of class information")
    """,
    hint="""
        Edge: `X.var(0).reshape(8, 8)[edge].sum() / X.var(0).sum()`. Centroids: `np.array([P2[y == c].mean(0) for c in range(10)])`; distances: `np.linalg.norm(centroids[:, None] - centroids[None], axis=2)` with the diagonal at infinity; pair: `np.unravel_index(D.argmin(), D.shape)`.
        Isolated: `D.min(axis=1).argmax()`. Center: `np.abs(pca.components_[0].reshape(8, 8))[2:6, 2:6].mean() - np.abs(...)[edge].mean()`.
    """,
    solution=[
        C("""
        pixel_var = X.var(0)
        constant_pixels = int((pixel_var == 0).sum())
        edge_variance_share = float(pixel_var.reshape(8, 8)[edge].sum() / pixel_var.sum())
        print(f"pixels with zero variance: {constant_pixels} | share of the variance in the 28 edge pixels: {edge_variance_share:.1%}")
        """),
        C("""
        pca = PCA().fit(X)
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        k90, k99 = int(np.searchsorted(cumulative, 0.9)) + 1, int(np.searchsorted(cumulative, 0.99)) + 1
        pc1_image = pca.components_[0].reshape(8, 8)
        absolute = np.abs(pc1_image)
        pc1_center = float(absolute[2:6, 2:6].mean() - absolute[edge].mean())
        print(f"components for 90% and 99% of the variance: {k90} and {k99} | mean weight center - edge in PC1: {pc1_center:.3f}")
        print(np.round(pc1_image, 2))
        """),
        C("""
        P2 = PCA(2).fit_transform(X)
        centroids = np.array([P2[y == c].mean(0) for c in range(10)])
        D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        a, b = np.unravel_index(D.argmin(), D.shape)
        closest_pair = (int(min(a, b)), int(max(a, b)))
        isolated_class = int(D.min(axis=1).argmax())
        print("pair of digits with the closest centroids:", closest_pair, "| most isolated digit:", isolated_class)
        print("distance to the closest centroid, per digit:", D.min(axis=1).round(2))
        """),
        C("""
        centroid_acc = {k: cross_val_score(make_pipeline(PCA(k), NearestCentroid()), X, y, cv=cv).mean() for k in [2, 10, 30, 64]}
        print({k: round(v, 3) for k, v in centroid_acc.items()})
        """),
    ],
    learned="""
        - **The pixels are coordinates:** 3 pixels (at the corners) never vary and carry no information, and the outer ring of 28 pixels (44% of the pixels) contains only 19% of the variance. The variance sits in the center of the image, where the strokes of the digits pass: the first component weighs mostly the center
          compared to the edge.
        - **How much variance, how many components:** 21 components for 90% and 41 for 99% out of 64 pixels. In images the redundancy is very high (neighboring pixels are correlated) and the effective dimension is much smaller than the nominal one.
        - **A 2-dimensional map loses a lot of class information:** with a centroid classifier you go from 0.61 with 2 components (the two dimensions capture 29% of the variance) to 0.88 with 10. Some digits stay far from the others (the 0 is the most isolated) and others overlap:
          reading a 2D map with too much confidence leads to concluding that "the classes separate" or "don't separate" because of an artifact of the projection.
        - **The number of components for the variance is not the one for classification**: with 30 components (96% of the variance) the centroid accuracy (0.90) is equal to that with all 64 pixels. For visualization 2 are enough, for classification 20-30.
    """,
    pitfall="""
        Judging the separability of the classes from a plot of the first two components. Classes that overlap in 2D can be well separated in 10 dimensions (and vice versa, less often): the map is a projection, not a measurement.
    """,
    interview=(
        "A 2D PCA plot shows overlapping classes. What do you conclude?",
        """
        Little: two components capture a fraction of the variance (here 29%) and the overlap can be an artifact of the projection. I measure the separability with a classifier in more dimensions (cross-validation) and use a nonlinear technique (t-SNE, UMAP) only
        to explore, remembering that the distances in those maps are not quantitative.
        """),
)

EXERCISES = [D1, D2]
