"""Module 6 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="pca_vs_feature_selection", kind="F", level=3, minutes=55, module=MODULE,
    title="PCA or feature selection? Three data shapes, five strategies",
    goal="""
        Compare five dimensionality reduction strategies in front of a logistic regression (no reduction, `PCA(5)`, `PCA(15)`, selection of 5 features with `SelectKBest`, `PCA(5)` without a scaler) on three kinds of data: **redundant** features (5 latent factors in 40 columns), signal in
        **low-variance directions** and **sparse** signal (5 informative columns out of 60). Average over 3 datasets per kind.
    """,
    dataset="""
        `generate(kind, seed)` (given): 400 examples per dataset; `kind` in `"redundant"`, `"low variance"`, `"sparse"`. Final model: `LogisticRegression(max_iter=2000)`; validation `StratifiedKFold(5, shuffle=True, random_state=0)`, accuracy.
    """,
    purpose="""
        Replace "use PCA to reduce the dimensions" with a rule based on the shape of the data: PCA helps (or at least does no harm) when variance and information coincide; it hurts when the signal is in a few columns among many or in low-variance directions. Supervised feature
        selection is often a better fit.
    """,
    task=[
        "Build `models` (dictionary name → function that creates the `Pipeline`): `none` (`StandardScaler` → logistic), `PCA(5)` and `PCA(15)` (`StandardScaler` → `PCA(k)` → logistic), `Select(5)` (`StandardScaler` → `SelectKBest(f_classif, k=5)` → logistic) and `PCA(5) raw` (`PCA(5)` → logistic, no scaler).",
        "Build `table` (index: the three kinds of data; columns: the five models) with the cross-validation accuracy **averaged over `seed` = 0, 1, 2**.",
        "Save `best` (dictionary kind → best model), `pca_loses` (sorted list of the kinds where `PCA(5)` is worse than `none` by more than 0.05) and `pca_ties` (the kinds where the difference between `PCA(5)` and `none` is less than 0.03).",
        "Save `scaler_matters` (dictionary kind → `PCA(5)` with scaler minus `PCA(5) raw`): where does scaling the PCA change the result the most?",
    ],
    setup="""
        from sklearn.decomposition import PCA
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def generate(kind, seed=0, n=400):
            rng = np.random.default_rng(seed)
            if kind == "redundant":                        # 5 latent factors explain 40 correlated columns; y depends on 2 factors
                L = rng.normal(size=(n, 5))
                X = L @ rng.normal(size=(5, 40)) + rng.normal(0, 1, (n, 40))
                y = (L[:, 0] + L[:, 1] + rng.normal(0, 1, n) > 0).astype(int)
            elif kind == "low variance":                   # 20 high-variance noise columns and 3 low-variance informative ones
                z = rng.normal(size=(n, 3))
                y = (z.sum(1) + rng.normal(0, 0.5, n) > 0).astype(int)
                X = np.c_[rng.normal(0, 5, (n, 20)), 0.5 * z]
            else:                                          # 5 informative columns out of 60
                X = rng.normal(size=(n, 60))
                y = (X[:, :5].sum(1) + rng.normal(0, 1, n) > 0).astype(int)
            return X, y


        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        kinds = ["redundant", "low variance", "sparse"]
    """,
    starter=[
        C("""
        models = {
            "none": lambda: ...,
        }
        """),
        C("""
        table = pd.DataFrame(index=kinds, columns=list(models), dtype=float)
        for kind in kinds:
            for name, create in models.items():
                ...
        table.round(3)
        """),
        C("""
        best = {}
        pca_loses = None
        pca_ties = None
        scaler_matters = {}
        """),
    ],
    checks="""
        check(list(models) == ["none", "PCA(5)", "PCA(15)", "Select(5)", "PCA(5) raw"], "five models with the given names", "none, PCA(5), PCA(15), Select(5), PCA(5) raw")
        check(table.shape == (3, 5) and table.notna().all().all(), "complete 3 x 5 table", "one accuracy for each kind and model")
        check(table.loc["low variance", "PCA(5)"] < table.loc["low variance", "none"] - 0.1 and table.loc["sparse", "PCA(5)"] < 0.6, "PCA(5) fails with low-variance signal (over 0.1 below 'none') and with sparse signal (below 0.6, almost chance)", "compare the rows of the table")
        check(pca_loses == ["low variance", "sparse"] and pca_ties == ["redundant"], "PCA(5) loses on 'low variance' and 'sparse'; it ties on 'redundant'", "difference between PCA(5) and none")
        check(best["low variance"] == "Select(5)" and best["sparse"] == "Select(5)", "supervised selection is the best when the signal is in a few columns", "idxmax per row")
        check(scaler_matters["low variance"] > 0.1 and abs(scaler_matters["redundant"]) < 0.05, "the scaler matters a lot for 'low variance' (over 0.1) and almost nothing for 'redundant'", "PCA(5) with scaler minus PCA(5) raw")
    """,
    hint="""
        Example: `"PCA(5)": lambda: make_pipeline(StandardScaler(), PCA(5), LogisticRegression(max_iter=2000))`. Average over the seeds: `np.mean([cross_val_score(create(), *generate(kind, seed), cv=cv).mean() for seed in range(3)])`.
        Kinds where PCA(5) loses: `sorted(t for t in kinds if table.loc[t, "PCA(5)"] < table.loc[t, "none"] - 0.05)`.
    """,
    solution=[
        C("""
        def logistic():
            return LogisticRegression(max_iter=2000)


        models = {
            "none": lambda: make_pipeline(StandardScaler(), logistic()),
            "PCA(5)": lambda: make_pipeline(StandardScaler(), PCA(5), logistic()),
            "PCA(15)": lambda: make_pipeline(StandardScaler(), PCA(15), logistic()),
            "Select(5)": lambda: make_pipeline(StandardScaler(), SelectKBest(f_classif, k=5), logistic()),
            "PCA(5) raw": lambda: make_pipeline(PCA(5), logistic()),
        }
        """),
        C("""
        table = pd.DataFrame(index=kinds, columns=list(models), dtype=float)
        for kind in kinds:
            for name, create in models.items():
                table.loc[kind, name] = np.mean([cross_val_score(create(), *generate(kind, seed), cv=cv).mean() for seed in range(3)])
        table.round(3)
        """),
        C("""
        best = table.idxmax(axis=1).to_dict()
        pca_loses = sorted(t for t in kinds if table.loc[t, "PCA(5)"] < table.loc[t, "none"] - 0.05)
        pca_ties = sorted(t for t in kinds if abs(table.loc[t, "PCA(5)"] - table.loc[t, "none"]) < 0.03)
        scaler_matters = (table["PCA(5)"] - table["PCA(5) raw"]).to_dict()
        print("best:", best, "\\nPCA(5) loses on:", pca_loses, "| ties on:", pca_ties, "\\neffect of the scaler on PCA(5):", {k: round(v, 3) for k, v in scaler_matters.items()})
        """),
    ],
    learned="""
        - **Redundant** (5 factors in 40 columns): all the strategies sit between 0.79 and 0.80, `PCA(5)` (0.788) included: the 5 components hold the signal and the remaining 35 directions are noise. PCA compresses without losing anything; in accuracy it gains nothing, because a regularized logistic regression
          already handles redundancy well (differences of one point are noise).
        - **Low variance:** `PCA(5)` collapses (0.64 against 0.88): the 5 highest-variance directions are the noise columns. With 15 components it recovers part of the signal (0.76), and without a scaler it is at chance (0.51). It is the scenario of C1 on a classification problem.
        - **Sparse** (5 columns out of 60): `PCA(5)` is almost at chance (0.55): all the columns have the same variance and PCA has no reason to prefer the informative ones. **Supervised selection** (`SelectKBest`) wins when the signal is in a few columns, and on this data it even beats no reduction
          (0.87 against 0.83): it removes noise by using `y`.
        - **Rule of choice:** PCA when you want to compress redundant columns or visualize, and you know that variance and information coincide; selection (or L1 regularization) when you think that a few columns matter; no reduction with small data and regularized models. Always compare in cross-validation with the "no reduction" baseline.
    """,
    pitfall="""
        Putting PCA in every pipeline "to reduce noise". PCA reduces noise only if the noise has less variance than the signal; with low-variance or sparse signal it amplifies the noise relatively. It has to earn its place against the baseline, like every step of a pipeline.
    """,
    interview=(
        "PCA or feature selection: how do you choose?",
        """
        It depends on where I think the signal is. If the columns are many and redundant (correlated sensors, pixels) PCA compresses well without losing anything. If a few predictors matter and the rest is noise, PCA mixes them with the noise and supervised selection (filters, L1, trees) works better. If the
        signal is in low-variance directions, PCA throws it away. In practice I try both in a pipeline with cross-validation and compare them with the no-reduction baseline.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="linear_pca_vs_kernel_pca", kind="F", level=3, minutes=55, module=MODULE,
    title="Linear PCA against KernelPCA on nonlinear data (and what it costs)",
    goal="""
        Show that on nonlinear data (two concentric circles, two moons) linear PCA does not help a linear classifier, while `KernelPCA` with an RBF kernel makes the problem separable with a few components; choose components and `gamma` on a grid and measure how the computation time grows with the number of examples.
    """,
    dataset="""
        `make_circles(1000, noise=0.05, factor=0.4)` and `make_moons(1000, noise=0.15)`. Model: `LogisticRegression`; validation `StratifiedKFold(5, shuffle=True, random_state=0)`.
    """,
    purpose="""
        Understand what a **nonlinear** PCA does (the kernel trick: PCA in the feature space induced by the kernel) and what it costs: the kernel matrix is `n × n`, so the cost grows with the square (or more) of the number of examples. This is why KernelPCA is only used with small or medium datasets.
    """,
    task=[
        "For each dataset compute `base` (accuracy of the logistic alone on the raw data) and `pca2` (`PCA(2)` → logistic).",
        "For `KernelPCA(k, kernel=\"rbf\", gamma=g)` → logistic, with `k` in `[2, 5, 10]` and `g` in `[1, 3, 10]`, compute the accuracy and save it in `grid` (dictionary `dataset → DataFrame` with `k` rows and `g` columns). Save `best` (dictionary dataset → tuple `(k, g)` with the highest accuracy; on ties, the smaller `k` and then the smaller `g`) and `min_k` (dictionary dataset → the smallest `k` for which **at least one** `g` goes over 0.95).",
        "**Cost:** measure in seconds `KernelPCA(5, kernel=\"rbf\", gamma=3).fit(X)` with `n` in `[500, 1000, 2000, 4000]` points of `make_moons(n, noise=0.15)`: save `times` (`pd.Series` indexed by `n`) and `exponent` = slope of the line `log(time) ~ log(n)` (with `np.polyfit`).",
    ],
    setup="""
        import time
        from sklearn.datasets import make_circles, make_moons
        from sklearn.decomposition import PCA, KernelPCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline

        datasets = {"circles": make_circles(1000, noise=0.05, factor=0.4, random_state=0), "moons": make_moons(1000, noise=0.15, random_state=0)}
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        base, pca2 = {}, {}
        for name, (X, y) in datasets.items():
            ...
        pd.DataFrame({"logistic": base, "PCA(2)": pca2}).round(3)
        """),
        C("""
        grid = {}
        best = {}
        min_k = {}
        """),
        C("""
        times = pd.Series(dtype=float)
        exponent = None
        """),
    ],
    checks="""
        check(set(base) == {"circles", "moons"} and base["circles"] < 0.55 and pca2["circles"] < 0.55 and 0.8 < base["moons"] < 0.9, "circles: logistic and PCA(2) at chance (below 0.55); moons: the logistic reaches 0.85", "PCA(2) is a rotation: it changes nothing")
        check(all(grid[n].shape == (3, 3) for n in ("circles", "moons")) and grid["circles"].loc[5].max() > 0.99 and grid["moons"].loc[5].max() > 0.96, "KernelPCA with 5 components: circles over 0.99, moons over 0.96", "RBF kernel, k=5")
        check(best["circles"][0] >= 5 and min_k["circles"] == 5 and min_k["moons"] == 5, "5 components are needed (2 is not enough with any gamma)", "the smallest k with accuracy above 0.95")
        check(grid["circles"].loc[2].max() < 0.8 and grid["moons"].loc[2].max() < 0.95, "with 2 components KernelPCA does not solve the problem (circles below 0.8, moons below 0.95)", "two components are not enough")
        check(list(times.index) == [500, 1000, 2000, 4000] and times.is_monotonic_increasing and exponent > 1.3 and times[4000] > 5 * times[1000], "the time grows more than linearly with n (exponent over 1.3; 4000 points over 5 times slower than 1000)", "the kernel matrix is n x n")
    """,
    hint="""
        `cross_val_score(make_pipeline(KernelPCA(k, kernel="rbf", gamma=g), LogisticRegression(max_iter=1000)), X, y, cv=cv).mean()`. Times: `t0 = time.time(); KernelPCA(5, kernel="rbf", gamma=3).fit(X); time.time() - t0`. Exponent: `np.polyfit(np.log(times.index), np.log(times.values), 1)[0]`.
    """,
    solution=[
        C("""
        base, pca2 = {}, {}
        for name, (X, y) in datasets.items():
            base[name] = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=cv).mean()
            pca2[name] = cross_val_score(make_pipeline(PCA(2), LogisticRegression(max_iter=1000)), X, y, cv=cv).mean()
        pd.DataFrame({"logistic": base, "PCA(2)": pca2}).round(3)
        """),
        C("""
        ks, gammas = [2, 5, 10], [1, 3, 10]
        grid, best, min_k = {}, {}, {}
        for name, (X, y) in datasets.items():
            g = pd.DataFrame(index=ks, columns=gammas, dtype=float)
            for k in ks:
                for gamma in gammas:
                    g.loc[k, gamma] = cross_val_score(make_pipeline(KernelPCA(k, kernel="rbf", gamma=gamma), LogisticRegression(max_iter=1000)), X, y, cv=cv).mean()
            grid[name] = g
            best[name] = min((k, gamma) for k in ks for gamma in gammas if g.loc[k, gamma] >= g.to_numpy().max() - 1e-12)
            min_k[name] = min(k for k in ks if g.loc[k].max() > 0.95)
            print(name); print(g.round(3))
        print("best:", best, "| minimum k to go over 0.95:", min_k)
        """),
        C("""
        def kpca_time(n):
            X, _ = make_moons(n, noise=0.15, random_state=0)
            t0 = time.time()
            KernelPCA(5, kernel="rbf", gamma=3).fit(X)
            return time.time() - t0


        times = pd.Series({n: kpca_time(n) for n in [500, 1000, 2000, 4000]})
        exponent = np.polyfit(np.log(times.index.to_numpy(dtype=float)), np.log(times.to_numpy()), 1)[0]
        print(times.round(2).to_string())
        print(f"the time grows like n^{exponent:.2f}")
        """),
    ],
    learned="""
        - **Linear PCA does not solve nonlinear problems:** on the circles the logistic and `PCA(2)` are at chance (0.46) because two concentric circles are not separable by any line, and rotating the axes does not change the geometry. On the moons 0.865.
        - **`KernelPCA` with an RBF kernel does PCA in an implicit space of much higher dimension:** with 5 components the logistic reaches 1.00 on the circles and 0.98 on the moons. With 2 components it is not enough (0.46-0.70 on the circles): the dimension matters. `gamma` (the width of the kernel) is a hyperparameter like
          `k`: but here its influence is small once `k` is sufficient.
        - **The cost:** the kernel matrix is `n × n` and the solvers decompose it, so the time grows roughly like the square of `n` (here the measured exponent is 1.8). With 4000 points it is already over ten times slower than with 1000 (0.9 s against 0.07), and with 100 000 examples it is unusable without approximations (Nystroem, random Fourier features).
        - The alternatives: **Nystroem** (kernel approximation with a subset), **Isomap** and **t-SNE/UMAP** (for visualization), or a nonlinear model directly (trees, kernel SVM, networks). KernelPCA is a niche choice: interesting for understanding, rarely the most practical.
    """,
    pitfall="""
        Expecting KernelPCA to be "PCA but better". It has more hyperparameters (kernel, `gamma`), it is not simply invertible (no exact `inverse_transform`), it costs `O(n^2)` in memory and it gives no interpretable loadings.
    """,
    interview=(
        "When is linear PCA not enough?",
        """
        When the structure of the data is nonlinear (curved manifolds, nested clusters): two concentric circles, images with deformations. The options are KernelPCA (RBF kernel), Isomap or embedding methods (t-SNE, UMAP) for visualization, autoencoders for compression. Kernel methods
        cost O(n^2) in memory and time, so they are used with small datasets or with approximations (Nystroem). And before complicating things, I check whether a nonlinear model directly on the original data already solves the problem.
        """),
)

EXERCISES = [F1, F2]
