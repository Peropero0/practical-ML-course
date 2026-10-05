"""Module 6 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="the_signal_in_the_discarded_component", kind="C", level=3, minutes=50, module=MODULE,
    title="The signal sits in the ninth component: PCR fails",
    goal="""
        Build a dataset in which the target depends on a direction with **tiny variance** and verify that a regression on the first principal components has R² zero until you include that component (the ninth out of ten, with 0.2% of the variance). Then compare with the regression on the original features (which sees all the directions) and with PLS, and understand what "explained variance" really measures.
    """,
    dataset="""
        `generate()` (given): 600 examples, 10 features obtained by rotating 10 independent variables with standard deviations `[5, 4, 3, 2.5, 2, 1.5, 1, 0.6, 0.4, 0.3]`. The target depends **only** on the ninth variable (standard deviation 0.4), with noise.
    """,
    purpose="""
        Dismantle the equation "high variance = useful information". PCA orders the directions by variance; usefulness for predicting `y` is another matter, and without looking at `y` PCA can throw away exactly the direction that is needed.
    """,
    task=[
        "Compute `r2_per_k` (`pd.Series`, index `k = 1..10`): mean R² in `KFold(5, shuffle=True, random_state=0)` of the pipeline `PCA(k)` → `LinearRegression`.",
        "Save `variance_ratio` (`explained_variance_ratio_` of the complete PCA) and `corr_with_y` (absolute value of the correlation of each component with `y`, on the scores of the complete PCA). Save `useful_component` (order number, from 1, of the component most correlated with `y`) and `useful_component_variance` (its share of variance).",
        "Save `full_r2` (cross-validated R² of the linear regression on all the original features) and `min_useful_k`: the smallest `k` for which `r2_per_k` exceeds 0.5.",
        "**Alternatives:** `ridge_r2` = mean cross-validated R² of `RidgeCV(alphas=np.logspace(-4, 0, 9))` on the original features, and `pls1_r2` = mean cross-validated R² of `PLSRegression(n_components=1)`. Which of the two sees the useful direction?",
    ],
    setup="""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, RidgeCV
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import make_pipeline


        def generate():
            rng = np.random.default_rng(0)
            n, p = 600, 10
            sd = np.array([5, 4, 3, 2.5, 2, 1.5, 1, 0.6, 0.4, 0.3])
            Z = rng.normal(size=(n, p)) * sd
            Q, _ = np.linalg.qr(rng.normal(size=(p, p)))                    # rotation: the original variables are mixtures
            y = Z[:, 8] * 3 + rng.normal(0, 0.5, n)                          # only the ninth direction (sd 0.4) matters
            X = Z @ Q.T
            return X - X.mean(0), y


        X, y = generate()
        cv = KFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        r2_per_k = pd.Series(dtype=float)
        r2_per_k
        """),
        C("""
        variance_ratio = None
        corr_with_y = None
        useful_component = None
        useful_component_variance = None
        """),
        C("""
        full_r2 = None
        min_useful_k = None
        ridge_r2 = None
        pls1_r2 = None
        """),
    ],
    checks="""
        check(list(r2_per_k.index) == list(range(1, 11)) and r2_per_k.loc[1:8].max() < 0.05, "with 1-8 components the R2 is practically zero (below 0.05)", "PCA(k) + LinearRegression, cross-validated R2")
        check(r2_per_k[9] > 0.8 and r2_per_k[10] > 0.8, "with 9 components the R2 goes above 0.8", "include the ninth component")
        check(useful_component == 9 and useful_component_variance < 0.005 and corr_with_y[8] > 0.85 and (np.delete(corr_with_y, 8) < 0.2).all(), "the useful component is the ninth: 0.2% of the variance, correlation over 0.85 with y; the others below 0.2", "PCA().fit_transform(X) and np.corrcoef with y")
        check(min_useful_k == 9 and full_r2 > 0.8 and abs(full_r2 - r2_per_k[10]) < 1e-9, "minimum k 9; the regression on the 10 features (or on 10 components) is the same", "PCA(10) is just a rotation")
        check(ridge_r2 is not None and ridge_r2 > 0.8 and pls1_r2 is not None and pls1_r2 < 0.1, "Ridge on the original features sees the useful direction (R2 above 0.8); PLS with one component does not (below 0.1)", "RidgeCV with small alphas; PLSRegression(n_components=1)")
    """,
    hint="""
        `scores = PCA().fit_transform(X)`; `corr_with_y = np.abs([np.corrcoef(scores[:, i], y)[0, 1] for i in range(10)])`; `useful_component = int(np.argmax(corr_with_y)) + 1`.
    """,
    solution=[
        C("""
        r2_per_k = pd.Series({k: cross_val_score(make_pipeline(PCA(k), LinearRegression()), X, y, cv=cv, scoring="r2").mean() for k in range(1, 11)})
        r2_per_k.round(3)
        """),
        C("""
        pca = PCA().fit(X)
        variance_ratio = pca.explained_variance_ratio_
        scores = pca.transform(X)
        corr_with_y = np.abs([np.corrcoef(scores[:, i], y)[0, 1] for i in range(10)])
        useful_component = int(np.argmax(corr_with_y)) + 1
        useful_component_variance = variance_ratio[useful_component - 1]
        print("share of variance per component:", variance_ratio.round(3))
        print("correlation with y per component:", corr_with_y.round(2))
        print(f"the useful component is number {useful_component}, with {useful_component_variance:.2%} of the variance")
        """),
        C("""
        full_r2 = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="r2").mean()
        min_useful_k = int(r2_per_k.index[(r2_per_k > 0.5).to_numpy().argmax()])
        ridge_r2 = cross_val_score(RidgeCV(alphas=np.logspace(-4, 0, 9)), X, y, cv=cv, scoring="r2").mean()
        pls1_r2 = cross_val_score(PLSRegression(n_components=1), X, y, cv=cv, scoring="r2").mean()
        print(f"R2 with all the features: {full_r2:.3f} | first k with R2 > 0.5: {min_useful_k} | Ridge: {ridge_r2:.3f} | PLS with 1 component: {pls1_r2:.3f}")
        """),
    ],
    learned="""
        - **PCA does not look at `y`.** With the first 8 components (99.6% of the variance) the R² is zero, with 9 it goes up to 0.85: all the useful information is in a direction that contains 0.2% of the variance. Keeping "95% of the variance" would have thrown away 100% of the signal.
        - **"Explained variance" measures reconstruction fidelity, not usefulness for a target.** They are two different questions: the first concerns the columns (X), the second the link with `y`. They often coincide in real data, where the directions with the most variance are also the most informative, but there is no guarantee.
        - **Ridge on the original features works (0.85)**: it discards no direction. **PLS with one component does not (0.02)**: it chooses the direction of maximum *covariance* with `y`, that is variance × correlation, and the high-variance directions with a small random correlation (0.05) beat the low-variance
          one with high correlation (0.91). No method that summarizes the features into a few directions is immune: the rule is not to throw away directions without checking that the downstream task does not get worse, and to choose the number of components with cross-validation.
        - In real data this scenario shows up when the useful variable is a small difference between two large, correlated quantities (for example the difference between two nearly equal measurements): its variance is tiny and PCA eliminates it first.
    """,
    pitfall="""
        Reducing the dimensions with PCA "keeping 95% of the variance" before a supervised model, without checking that the downstream performance does not get worse. The threshold on the variance says nothing about the target.
    """,
    interview=(
        "Why does PCA not guarantee that the chosen components are useful for predicting?",
        """
        Because it is unsupervised: it orders the directions by variance of X and not by relationship with y. If the signal sits in a low-variance direction (a difference between two large, correlated variables, for example) it gets discarded first. That is why `k` must be chosen with
        cross-validation on the final task, or you use supervised methods (PLS, LDA) or models that see the original features (Ridge, trees).
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="three_outliers_hijack_pc1", kind="C", level=2, minutes=40, module=MODULE,
    title="Three outliers out of 303 points hijack the first component",
    goal="""
        Add three anomalous points to a cloud of 300 points and measure how much the first principal component rotates (by 90 degrees: it moves from one column to another). Fix with a clipping (winsorizing) of the extreme values and verify that
        the direction goes back to the true one.
    """,
    dataset="""
        `A`: 300 points in 4 dimensions with standard deviations `[3, 2, 1, 0.7]` (the true first component is column 0). `B` = `A` plus three outliers that differ only in column 2 (values around ±58).
    """,
    purpose="""
        Understand that PCA is based on **variance**, hence on squares: a single very distant point has a disproportionate weight. The defect does not show in the accuracy of a model: it shows in the direction of the components and in the explained variance.
    """,
    task=[
        "Compute `pc1_A` and `pc1_B` (first component of `PCA` on `A` and on `B`) and `A_column`, `B_column` (index of the column with the largest absolute loading in each). Save `angle_degrees` = angle between the two directions (`arccos(|pc1_A · pc1_B|)` in degrees).",
        "Save `ratio_A` and `ratio_B` (share of variance of the first component before and after) and `outlier_share` (share of the 3 outliers over the 303 points, as a percentage).",
        "**Fix:** clip every column of `B` between the 1st and the 99th percentile (`np.clip`, percentiles computed on the column) → `B_clip`. Save `clip_column` (column with the largest loading in PC1) and `clip_angle_degrees` (angle between the PC1 of `B_clip` and the PC1 of `A`).",
        "Also check with the **Mahalanobis distances** that the outliers are identifiable: save `outlier_indices` (the three points of `B` with the largest Mahalanobis distance, computed with the covariance of `A`).",
    ],
    setup="""
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(1)
        A = rng.multivariate_normal([0, 0, 0, 0], np.diag([9, 4, 1, 0.5]), 300)
        outliers = np.array([[0, 0, 60, 0], [0, 0, -55, 5], [0, 0, 58, -3]])
        B = np.r_[A, outliers]
        print(A.shape, B.shape)
    """,
    starter=[
        C("""
        pc1_A = None
        pc1_B = None
        A_column = None
        B_column = None
        angle_degrees = None
        ratio_A = None
        ratio_B = None
        outlier_share = None
        """),
        C("""
        B_clip = None
        clip_column = None
        clip_angle_degrees = None
        """),
        C("""
        outlier_indices = None
        """),
    ],
    checks="""
        check(A_column == 0 and B_column == 2, "without outliers PC1 is column 0; with the three outliers it is column 2", "np.abs(pc1).argmax()")
        check(angle_degrees > 85 and close(outlier_share, 100 * 3 / 303), "PC1 rotates by about 90 degrees because of three points (1% of the data)", "arccos(|pc1_A . pc1_B|)")
        check(ratio_B > ratio_A and ratio_A < 0.65 and ratio_B > 0.7, "the variance explained by PC1 goes from 0.59 to 0.73: misleading", "explained_variance_ratio_[0]")
        check(B_clip.shape == B.shape and clip_column == 0 and clip_angle_degrees < 15, "after the clipping PC1 is column 0 again (angle with the true one below 15 degrees)", "np.clip(column, percentile 1, percentile 99) for every column")
        check(sorted(outlier_indices) == [300, 301, 302], "the Mahalanobis distance identifies the three outliers (indices 300, 301, 302)", "use the covariance of A and the mean of A")
    """,
    hint="""
        Angle: `np.degrees(np.arccos(min(1, abs(pc1_A @ pc1_B))))`. Clipping: `lo, hi = np.percentile(B, [1, 99], axis=0); B_clip = np.clip(B, lo, hi)`. Mahalanobis: `d = B - A.mean(0); m2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(A.T)), d)`; `np.argsort(m2)[-3:]`.
    """,
    solution=[
        C("""
        pca_A, pca_B = PCA().fit(A), PCA().fit(B)
        pc1_A, pc1_B = pca_A.components_[0], pca_B.components_[0]
        A_column, B_column = int(np.abs(pc1_A).argmax()), int(np.abs(pc1_B).argmax())
        angle_degrees = float(np.degrees(np.arccos(min(1, abs(pc1_A @ pc1_B)))))
        ratio_A, ratio_B = pca_A.explained_variance_ratio_[0], pca_B.explained_variance_ratio_[0]
        outlier_share = 100 * 3 / len(B)
        print(f"column with the largest loading in PC1: {A_column} -> {B_column} | angle between the two PC1: {angle_degrees:.0f} degrees")
        print(f"variance explained by PC1: {ratio_A:.3f} -> {ratio_B:.3f} | outliers: {outlier_share:.1f}% of the points")
        """),
        C("""
        lo, hi = np.percentile(B, [1, 99], axis=0)
        B_clip = np.clip(B, lo, hi)
        pc1_clip = PCA().fit(B_clip).components_[0]
        clip_column = int(np.abs(pc1_clip).argmax())
        clip_angle_degrees = float(np.degrees(np.arccos(min(1, abs(pc1_clip @ pc1_A)))))
        print(f"after the clipping: PC1 on column {clip_column}, angle with the true PC1 {clip_angle_degrees:.1f} degrees")
        """),
        C("""
        d = B - A.mean(0)
        mahalanobis2 = np.einsum("ij,jk,ik->i", d, np.linalg.inv(np.cov(A.T)), d)
        outlier_indices = list(np.argsort(mahalanobis2)[-3:])
        print("points with the largest Mahalanobis distance:", sorted(outlier_indices), "| distances^2:", np.sort(mahalanobis2)[-3:].round(0))
        """),
    ],
    learned="""
        - **Three points out of 303 (1%) rotate PC1 by almost 90 degrees**: PCA maximizes the variance, and a point at distance 60 contributes $60^2 = 3600$ to the variance of a column whose true variance is 1. The "principal" component simply becomes the direction of the outliers.
        - **The symptom is misleading:** the share of variance explained by PC1 goes from 0.59 to 0.73: it looks like the data are simpler than before, when they are just dirtier. A PCA that "works better" after the addition of suspicious data is a warning.
        - Remedies: **clip** the extreme values (winsorizing, as here), remove the outliers identified with the Mahalanobis distance or an `EllipticEnvelope`, use a **robust PCA** (based on robust covariance estimators like `MinCovDet`). The order matters: the outliers must be handled **before** PCA and learned on the training set alone.
        - The Mahalanobis distance identifies isolated outliers well; with many outliers the covariance itself estimated with the dirty data gets contaminated (masking): that is why here it is computed with `A`, that is with data that we assume clean.
    """,
    pitfall="""
        Applying PCA to uninspected data and interpreting the first component as "the main factor". Look at the outliers first (box plot, Mahalanobis distances) and compare PC1 with and without the extreme points.
    """,
    interview=(
        "Is PCA sensitive to outliers?",
        """
        Very: it is based on variance and covariance, that is on squares, so a few distant points can hijack the first component. I inspect the outliers first (Mahalanobis, isolation forest), clip or remove them on the training set alone, or use a robust version (MCD covariance, robust PCA).
        I always check that the first components are stable with a bootstrap or by removing a few points.
        """),
)

EXERCISES = [C1, C2]
