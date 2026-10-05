"""Module 6 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="pca_fitted_on_the_test_set", kind="E", level=2, minutes=40, module=MODULE, start="must_fail",
    title="PCA redone on the test set: accuracy below chance",
    goal="""
        Find out why a classifier trained on the components of the training set gets 0.13 accuracy (worse than chance, with 3 classes) on the test set: the colleague **redid the PCA on the test set**, so the components of the test set are different axes (and with different signs) from those of the training set.
        Fix it with `transform` and with a `Pipeline`.
    """,
    dataset="""
        `load_wine`, stratified 70/30 split, standardized with the scaler learned on the training set: `Z_tr, Z_te`, `y_tr, y_te`. PCA with 5 components + `LogisticRegression`.
    """,
    purpose="""
        Recognize the most typical bug of "unsupervised" transformers: `fit_transform` on the test set. It produces no errors (the shapes are right) and the output numbers look reasonable, but they are coordinates in a **different frame of reference** from the one the model was trained on.
    """,
    task=[
        "Run the colleague's cell: `bug_acc` is very low.",
        "Compare the components: `cosines` = list of the dot products `train_pca.components_[i] @ test_pca.components_[i]` for the 5 components (where `test_pca` is the PCA redone on the test set) and `flipped_signs` = how many have a **negative** dot product. A value close to 1 means the same direction; negative, the opposite direction.",
        "Fix it: apply to the test data the PCA **trained on the training set** (`transform`, not `fit_transform`). Save `ok_acc`.",
        "Put everything in a `Pipeline` (`StandardScaler` → `PCA(5)` → `LogisticRegression`) on the raw data `X_tr_raw, X_te_raw` and save `pipeline_acc`. It must match `ok_acc`.",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_wine(return_X_y=True)
        X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        scaler = StandardScaler().fit(X_tr_raw)
        Z_tr, Z_te = scaler.transform(X_tr_raw), scaler.transform(X_te_raw)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        train_pca = PCA(5).fit(Z_tr)
        P_tr = train_pca.transform(Z_tr)
        test_pca = PCA(5).fit(Z_te)                     # "to get the components of the test set"
        P_te = test_pca.transform(Z_te)
        model = LogisticRegression(max_iter=1000).fit(P_tr, y_tr)
        bug_acc = model.score(P_te, y_te)
        print(f"test accuracy: {bug_acc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        cosines = None
        flipped_signs = None
        """),
        C("""
        ok_acc = None
        pipeline_acc = None
        """),
    ],
    checks="""
        check(bug_acc < 0.5, "the colleague gets less than 0.5 with three classes", "don't modify the colleague's code")
        check(cosines is not None and len(cosines) == 5 and flipped_signs >= 2 and cosines[0] > 0.9 and min(cosines) < 0, "the first component is almost the same; at least two components have the opposite direction", "dot product between the rows of components_")
        check(ok_acc is not None and ok_acc > 0.9, "with transform (the same components as the training set) the accuracy goes back above 0.9", "train_pca.transform(Z_te)")
        check(pipeline_acc is not None and close(pipeline_acc, ok_acc), "the Pipeline gives exactly the same result", "make_pipeline(StandardScaler(), PCA(5), LogisticRegression())")
    """,
    hint="""
        `cosines = [float(train_pca.components_[i] @ test_pca.components_[i]) for i in range(5)]`. Fix: `model.score(train_pca.transform(Z_te), y_te)`. Pipeline: `make_pipeline(StandardScaler(), PCA(5), LogisticRegression(max_iter=1000)).fit(X_tr_raw, y_tr).score(X_te_raw, y_te)`.
    """,
    solution=[
        C("""
        train_pca = PCA(5).fit(Z_tr)
        P_tr = train_pca.transform(Z_tr)
        test_pca = PCA(5).fit(Z_te)                     # "to get the components of the test set"
        P_te = test_pca.transform(Z_te)
        model = LogisticRegression(max_iter=1000).fit(P_tr, y_tr)
        bug_acc = model.score(P_te, y_te)
        print(f"test accuracy: {bug_acc:.3f}")
        """),
        M("""
        **Where the bug is.** `PCA(5).fit(Z_te)` computes the components of the **test data**: they are other directions, and with signs that may be flipped. The logistic learned "a high coordinate 2 means class 0" in the frame of reference of the training set, but in the test set the same coordinate
        can measure another direction (or the same one reversed): the predictions are systematically wrong, even worse than chance.
        """),
        C("""
        cosines = [float(train_pca.components_[i] @ test_pca.components_[i]) for i in range(5)]
        flipped_signs = int(sum(c < 0 for c in cosines))
        print("dot products between the training and test components:", np.round(cosines, 2), "| components with the opposite direction:", flipped_signs)
        """),
        C("""
        ok_acc = model.score(train_pca.transform(Z_te), y_te)                                   # the TRAINING components applied to the test set
        pipeline = make_pipeline(StandardScaler(), PCA(5), LogisticRegression(max_iter=1000)).fit(X_tr_raw, y_tr)
        pipeline_acc = pipeline.score(X_te_raw, y_te)
        print(f"accuracy with transform: {ok_acc:.3f} | with the Pipeline: {pipeline_acc:.3f}")
        """),
    ],
    learned="""
        - **A transformer is trained on the training set and applied (`transform`) to the test set.** Redoing `fit` or `fit_transform` on the test set creates a new frame of reference: for PCA it means different axes and arbitrary **signs** (the second and third components have dot product −0.87 and −0.84 with the training components),
          hence incompatible coordinates. Accuracy 0.13: worse than chance (0.4) because it is systematically wrong, not random.
        - There is no error and no warning: the shapes are right, the values plausible. The warning sign is a result **worse than chance** from a model that worked on the training set.
        - The rule holds for everything that learns from the data: `StandardScaler`, imputer, encoder, PCA. The `Pipeline` makes it impossible to violate, because `predict` calls only `transform`.
        - A milder variant is to `fit` the PCA on **all** the data (training and test) before the split: the signs stay coherent and the effect is small, but the test set influenced the components (mild leakage). The pipeline avoids this too.
    """,
    pitfall="""
        Thinking that "fitting the PCA on the test set" is fine because it doesn't use the labels. The components depend on the distribution of the data: applying a model trained in one frame of reference to points expressed in another makes no sense.
    """,
    interview=(
        "Why can't you do fit_transform on the test set too?",
        """
        Because fit learns parameters from the data (means, deviations, components) and the downstream model was trained with the parameters of the training set. Redoing the fit on the test set would change the frame of reference (with PCA: different axes and signs), so the scores would not be comparable with the ones the model saw.
        You use transform with the objects already trained, and a Pipeline guarantees it by construction.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="truncatedsvd_instead_of_pca", kind="E", level=2, minutes=40, module=MODULE, start="must_fail",
    title="TruncatedSVD does not center: the first component is the mean",
    goal="""
        Find out why a reduction to 2 dimensions with `TruncatedSVD` on the digits gives worse results than `PCA` (accuracy 0.41 against 0.60) and an "explained variance" that looks small: `TruncatedSVD` **does not center the data**, so its first component is almost the direction of the mean. Fix it by centering first.
    """,
    dataset="""
        `load_digits`: 1797 images, 64 pixels (0-16, all nonnegative: the mean is far from zero). Validation `StratifiedKFold(5, shuffle=True, random_state=0)`.
    """,
    purpose="""
        Recognize when two algorithms with similar names are not equivalent. `TruncatedSVD` (SVD without centering) is designed for **sparse** matrices (text, counts), where centering would destroy the sparsity. On dense data with a nonzero mean it is not a PCA.
    """,
    task=[
        "Run the colleague's cell: `svd_acc` (`TruncatedSVD(2)` + logistic) and `pca_acc` (`PCA(2)` + logistic): the colleague chose the wrong one.",
        "Check what the first component contains: `svd_cosine` (|cosine| between `TruncatedSVD.components_[0]` and the normalized mean vector of the data) and `pca_cosine` (the same for `PCA.components_[0]`).",
        "Compare the explained variance: `svd_evr` and `pca_evr` (sum of `explained_variance_ratio_` with 2 components). Which one is physically comparable?",
        "Fix it: `centered_svd_acc` = accuracy with a pipeline `StandardScaler(with_std=False)` (centering only) → `TruncatedSVD(2, algorithm=\"arpack\")` → logistic. It must match (within 0.01) `pca_acc`.",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA, TruncatedSVD
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_digits(return_X_y=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        mean_direction = X.mean(0) / np.linalg.norm(X.mean(0))
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        svd_acc = cross_val_score(make_pipeline(TruncatedSVD(2, random_state=0), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        pca_acc = cross_val_score(make_pipeline(PCA(2), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        print(f"accuracy with 2 components: TruncatedSVD {svd_acc:.3f} | PCA {pca_acc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        svd_cosine = None
        pca_cosine = None
        svd_evr = None
        pca_evr = None
        """),
        C("""
        centered_svd_acc = None
        """),
    ],
    checks="""
        check(svd_acc < 0.5 and pca_acc > 0.55, "TruncatedSVD below 0.5, PCA above 0.55", "don't modify the colleague's code")
        check(svd_cosine is not None and svd_cosine > 0.99 and pca_cosine < 0.05, "the first component of TruncatedSVD coincides with the direction of the mean (cosine > 0.99); the PCA one is almost orthogonal (< 0.05)", "np.abs(components_[0] @ mean_direction)")
        check(svd_evr is not None and svd_evr < 0.2 and pca_evr > 0.27, "TruncatedSVD 'explains' less than 20%, PCA over 27%: they are not comparable", "sum(explained_variance_ratio_)")
        check(centered_svd_acc is not None and abs(centered_svd_acc - pca_acc) < 0.01, "after centering the data TruncatedSVD and PCA give the same result (within 0.01)", "StandardScaler(with_std=False) before TruncatedSVD")
    """,
    hint="""
        `svd = TruncatedSVD(2, random_state=0).fit(X); abs(svd.components_[0] @ mean_direction)`. Centered pipeline: `make_pipeline(StandardScaler(with_std=False), TruncatedSVD(2, algorithm="arpack"), LogisticRegression(max_iter=3000))`.
    """,
    solution=[
        C("""
        svd_acc = cross_val_score(make_pipeline(TruncatedSVD(2, random_state=0), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        pca_acc = cross_val_score(make_pipeline(PCA(2), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        print(f"accuracy with 2 components: TruncatedSVD {svd_acc:.3f} | PCA {pca_acc:.3f}")
        """),
        M("""
        **Where the bug is.** `PCA` centers the data (subtracts the mean of every column) and finds the directions of maximum variance **around the mean**. `TruncatedSVD` does the SVD of the data **as they are**: if all the pixels are positive the direction that explains the most "energy" is simply that
        of the mean (the mean image), which does not distinguish the digits. The second component is the true first component of the PCA: a dimension has been lost.
        """),
        C("""
        svd = TruncatedSVD(2, random_state=0).fit(X)
        pca = PCA(2).fit(X)
        svd_cosine = float(abs(svd.components_[0] @ mean_direction))
        pca_cosine = float(abs(pca.components_[0] @ mean_direction))
        svd_evr, pca_evr = svd.explained_variance_ratio_.sum(), pca.explained_variance_ratio_.sum()
        print(f"cosine between the first component and the direction of the mean: TruncatedSVD {svd_cosine:.4f} | PCA {pca_cosine:.4f}")
        print(f"'explained variance' with 2 components: TruncatedSVD {svd_evr:.3f} | PCA {pca_evr:.3f}")
        """),
        C("""
        centered = make_pipeline(StandardScaler(with_std=False), TruncatedSVD(2, algorithm="arpack"), LogisticRegression(max_iter=3000))
        centered_svd_acc = cross_val_score(centered, X, y, cv=cv).mean()
        print(f"TruncatedSVD after centering: {centered_svd_acc:.3f} (PCA: {pca_acc:.3f})")
        """),
    ],
    learned="""
        - **PCA = SVD of the centered data; `TruncatedSVD` = SVD without centering.** They are the same thing only if the data already have mean zero. With nonnegative pixels the first component of `TruncatedSVD` is the mean image (cosine 0.99998) and the two useful axes reduce to one: 0.41 accuracy
          against 0.60.
        - The **"explained variance"** of `TruncatedSVD` (0.18 for two components) is not comparable with that of PCA (0.29): it is computed on the variance of the transformed columns relative to the variance of the original data, but without having removed the mean. Don't use that number to choose `k`.
        - `TruncatedSVD` exists for a valid reason: on **sparse matrices** (text with TF-IDF, counts) centering would make the matrix dense and unmanageable. In that case you use it (the technique is called LSA) knowing that the first component will contain the "mean" of the corpus.
        - Rule: dense data → `PCA`; sparse data → `TruncatedSVD`; if you want PCA on sparse data, center implicitly with other solvers or reduce the dimension first.
    """,
    pitfall="""
        Choosing `TruncatedSVD` "because it is faster" on dense data. It is not the same transformation: it changes the result, and no error warns you.
    """,
    interview=(
        "What is the difference between PCA and TruncatedSVD?",
        """
        PCA centers the data and then does the SVD, so it finds the directions of maximum variance around the mean. TruncatedSVD does the SVD directly on the matrix, without centering: it is suited to sparse matrices (where centering would destroy the sparsity, as in LSA for text), but on dense data with
        a nonzero mean the first component aligns with the mean. If the data are already centered they coincide.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="pca_on_the_raw_wine_data", kind="E", level=2, minutes=35, module=MODULE, start="must_fail",
    title="A colleague discovers that wines are a one-dimensional problem",
    goal="""
        Find out why a PCA on the wines gives 99.8% of the variance to the first component ("the dataset is almost one-dimensional") and a classifier on that single component reaches 0.70: the PCA was done **on the raw data**, where `proline` (scale of hundreds) decides alone. Fix it by
        standardizing and verify how false the conclusion was.
    """,
    dataset="""
        `load_wine`: 178 wines, 13 measurements with very different scales (`proline` up to 1680, `hue` around 1), 3 classes. Validation `StratifiedKFold(5, shuffle=True, random_state=0)`.
    """,
    purpose="""
        Recognize the most common mistake in the use of PCA (not standardizing) and its signature: a first component with 99+% of the variance and a loading almost entirely on a single column. It is not a discovery about the phenomenon: it is an effect of the units of measure.
    """,
    task=[
        "Run the colleague's cell: `raw_pc1_ratio` and `raw_1_acc` (logistic on the first component alone, in cross-validation).",
        "Save `pc1_column` (name of the column with the largest loading in PC1) and `proline_variance_share` (variance of `proline` divided by the sum of the variances of all the columns).",
        "Fix it: pipeline `StandardScaler` → `PCA(k)` → logistic. Save `scaled_pc1_ratio` (variance explained by PC1 after scaling), `scaled_1_acc` (k = 1) and `scaled_2_acc` (k = 2).",
        "Save `scaled_k_for_90`: components needed for 90% of the variance on the standardized data (against 1 in the raw data).",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_wine(return_X_y=True, as_frame=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        pca = PCA().fit(X)
        raw_pc1_ratio = pca.explained_variance_ratio_[0]
        raw_1_acc = cross_val_score(make_pipeline(PCA(1), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        print(f"variance explained by PC1: {raw_pc1_ratio:.3f} -> 'the problem is one-dimensional'. Accuracy with 1 component: {raw_1_acc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        pc1_column = None
        proline_variance_share = None
        """),
        C("""
        scaled_pc1_ratio = None
        scaled_1_acc = None
        scaled_2_acc = None
        scaled_k_for_90 = None
        """),
    ],
    checks="""
        check(raw_pc1_ratio > 0.99 and raw_1_acc < 0.75, "the colleague: PC1 with over 99% of the variance and accuracy below 0.75", "don't modify the colleague's code")
        check(pc1_column == "proline" and proline_variance_share is not None and proline_variance_share > 0.99, "PC1 is the column `proline`, which has over 99% of the total variance", "X.var() and components_[0]")
        check(scaled_pc1_ratio is not None and 0.3 < scaled_pc1_ratio < 0.4 and scaled_k_for_90 == 8, "after scaling PC1 explains 36%, and 8 components are needed for 90%", "StandardScaler before PCA")
        check(scaled_2_acc is not None and scaled_2_acc > 0.95 and scaled_2_acc > raw_1_acc + 0.2, "with two components of the standardized data the accuracy is over 0.95 (against 0.70)", "make_pipeline(StandardScaler(), PCA(2), LogisticRegression())")
        check(scaled_1_acc > raw_1_acc + 0.1, "even with a single component scaling improves by over 0.1", "compare scaled_1_acc and raw_1_acc")
    """,
    hint="""
        `X.var().idxmax()` and `X.var()["proline"] / X.var().sum()`. Column of PC1: `X.columns[np.abs(pca.components_[0]).argmax()]`. Components for 90%: `np.searchsorted(np.cumsum(PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_), 0.9) + 1`.
    """,
    solution=[
        C("""
        pca = PCA().fit(X)
        raw_pc1_ratio = pca.explained_variance_ratio_[0]
        raw_1_acc = cross_val_score(make_pipeline(PCA(1), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        print(f"variance explained by PC1: {raw_pc1_ratio:.3f} -> 'the problem is one-dimensional'. Accuracy with 1 component: {raw_1_acc:.3f}")
        """),
        M("""
        **Where the bug is.** PCA maximizes the variance, and the variance depends on the **units**: `proline` has a standard deviation of 315 against 0.1-15 for the others. The first component is `proline` and nothing else. The 99.8% does not say that the wines are one-dimensional: it says that a column has
        an enormous scale.
        """),
        C("""
        pc1_column = X.columns[np.abs(pca.components_[0]).argmax()]
        proline_variance_share = X["proline"].var() / X.var().sum()
        print(f"column with the largest loading in PC1: {pc1_column} | share of the total variance due to proline: {proline_variance_share:.4f}")
        """),
        C("""
        scaled = PCA().fit(StandardScaler().fit_transform(X))
        scaled_pc1_ratio = scaled.explained_variance_ratio_[0]
        scaled_k_for_90 = int(np.searchsorted(np.cumsum(scaled.explained_variance_ratio_), 0.9)) + 1
        scaled_1_acc = cross_val_score(make_pipeline(StandardScaler(), PCA(1), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        scaled_2_acc = cross_val_score(make_pipeline(StandardScaler(), PCA(2), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        print(f"after scaling: PC1 explains {scaled_pc1_ratio:.1%}, {scaled_k_for_90} components are needed for 90%")
        print(f"accuracy with 1 component: {scaled_1_acc:.3f} | with 2: {scaled_2_acc:.3f} (raw, 1 component: {raw_1_acc:.3f})")
        """),
    ],
    learned="""
        - **On unstandardized data PCA measures the units, not the data.** The PC1 of the raw wines is `proline` (99.8% of the total variance due to a single column). A "99.8%" is a warning sign, not a result: when a component explains almost everything, look at its loadings right away.
        - **After scaling the true structure emerges:** PC1 explains 36%, 8 components are needed for 90%, and with **two** components the accuracy is 0.96 (against 0.70 with the "99.8% component"). The colleague's conclusion ("the wines are one-dimensional") was false, and an analysis built on top of it would have gone in the wrong direction.
        - The rule: **standardize before a PCA** when the columns have different units. The exception is when the columns are already homogeneous (pixels, measurements with the same unit) and you want to preserve the differences in variance (see B1).
        - This bug is a twin of E3 of module 5 (the `Normalizer` instead of the `StandardScaler`): in both cases there is no error but a number that "looks" like information. The check is always the same: **look at the loadings and compare with a correct baseline.**
    """,
    pitfall="""
        Reporting the explained variance of a PCA without saying whether the data were standardized. It is not interpretable: it holds for those units, not for the phenomenon.
    """,
    interview=(
        "Why do you standardize before a PCA?",
        """
        Because PCA looks for the directions of maximum variance, and the variance depends on the units of measure: a column expressed in small units (hence with large numbers) dominates the components. By standardizing every column has variance 1 and PCA works on the correlation matrix, that is on the relationships
        between the variables. If the columns already have the same unit and the differences in variance are informative (pixels), I don't standardize.
        """),
)

EXERCISES = [E1, E2, E3]
