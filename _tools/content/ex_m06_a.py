"""Module 6 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="pca_from_scratch_with_svd", kind="A", level=2, minutes=45, module=MODULE,
    title="PCA in numpy with the SVD (and with the eigenvalues)",
    goal="""
        Implement `fit`, `transform` and `inverse_transform` of a PCA with the SVD of the centered data, check that they match `sklearn.decomposition.PCA` (up to the sign), get the same variances with the eigenvalues of the covariance matrix and
        show that the mean squared reconstruction error with `k` components is the sum of the **discarded** variances.
    """,
    dataset="""
        `load_wine` (178 wines, 13 measurements) standardized: `Z` (178 × 13).
    """,
    purpose="""
        Remove all mystery from PCA: it is a centering, an SVD and a matrix multiplication. Understand the role of the sign (arbitrary), the relationship between SVD and covariance and why "explained variance" and "reconstruction error" are the same thing.
    """,
    task=[
        "`pca_fit(X, k)` → `(mean, components, variances)`: mean per column, the first `k` components (`k × p` matrix, one per row) and their variances $S^2/(n-1)$ (**only the first `k`**) from the SVD of `X - mean` (`np.linalg.svd(..., full_matrices=False)`).",
        "`pca_transform(X, mean, components)` (the scores) and `pca_inverse(P, mean, components)` (the reconstruction in the original space).",
        "Compare with `PCA(k)`: `equal_components` (the absolute values coincide), `equal_variances` and `equal_reconstruction` (the reconstruction does **not** depend on the sign). Also save `total_variance` (sum of the variances of all the columns of `Z`, with `ddof=1`) and check that it equals the sum of **all** the variances of the components (compute them with `k = 13`).",
        "**Eigenvalues:** compute `eigenvalues` = decreasing eigenvalues of `np.cov(Z.T)` with `np.linalg.eigh` and check that they coincide with the variances from the SVD.",
        "**Reconstruction error:** for `k = 3` compute `mse` (mean squared error over all the cells between `Z` and the reconstruction) and `theoretical_mse` (= sum of the discarded variances × `(n-1)/n` / `p`). They must coincide.",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        Z = StandardScaler().fit_transform(load_wine().data)
        n, p = Z.shape
        print(Z.shape)
    """,
    starter=[
        C("""
        def pca_fit(X, k):
            ...


        def pca_transform(X, mean, components):
            ...


        def pca_inverse(P, mean, components):
            ...
        """),
        C("""
        mean, components, variances = pca_fit(Z, 3)
        sk = PCA(3).fit(Z)
        equal_components = None
        equal_variances = None
        equal_reconstruction = None
        total_variance = None
        sum_of_all_variances = None
        """),
        C("""
        eigenvalues = None
        """),
        C("""
        mse = None
        theoretical_mse = None
        """),
    ],
    checks="""
        check(components.shape == (3, 13) and variances.shape == (3,), "3 components of 13 columns and 3 variances", "components: one per row")
        check(equal_components is True and equal_variances is True, "components (up to the sign) and variances match scikit-learn's PCA", "compare np.abs(components) with np.abs(sk.components_)")
        check(equal_reconstruction is True, "the reconstruction matches exactly: it does not depend on the sign", "pca_inverse(pca_transform(Z)) against sk.inverse_transform(sk.transform(Z))")
        check(close(total_variance, Z.var(0, ddof=1).sum()) and close(sum_of_all_variances, total_variance), "the sum of all the variances of the components is the total variance (13 for standardized data)", "the rotation preserves the total variance")
        check(close(eigenvalues, pca_fit(Z, 13)[2]) and (np.diff(eigenvalues) <= 0).all(), "the eigenvalues of the covariance are the variances from the SVD, in decreasing order", "np.linalg.eigh returns increasing eigenvalues: reverse them")
        check(close(mse, theoretical_mse) and close(mse, ((Z - PCA(3).fit(Z).inverse_transform(PCA(3).fit(Z).transform(Z))) ** 2).mean()), "reconstruction error = sum of the discarded variances (x (n-1)/n / p)", "total_variance - sum(variances) is the unexplained variance")
    """,
    hint="""
        `Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False); components = Vt[:k]; variances = S[:k] ** 2 / (len(X) - 1)`. Scores: `(X - mean) @ components.T`. Reconstruction: `P @ components + mean`.
        `np.linalg.eigh(np.cov(Z.T))[0][::-1]`. Error: `(total_variance - variances.sum()) * (n - 1) / n / p`.
    """,
    solution=[
        C("""
        def pca_fit(X, k):
            mean = X.mean(0)
            U, S, Vt = np.linalg.svd(X - mean, full_matrices=False)                # SVD of the centered data
            return mean, Vt[:k], S[:k] ** 2 / (len(X) - 1)                           # the rows of Vt are the components


        def pca_transform(X, mean, components):
            return (X - mean) @ components.T                                         # projection onto the components


        def pca_inverse(P, mean, components):
            return P @ components + mean
        """),
        C("""
        mean, components, variances = pca_fit(Z, 3)
        sk = PCA(3).fit(Z)
        equal_components = bool(np.allclose(np.abs(components), np.abs(sk.components_)))
        equal_variances = bool(np.allclose(variances, sk.explained_variance_))
        equal_reconstruction = bool(np.allclose(pca_inverse(pca_transform(Z, mean, components), mean, components), sk.inverse_transform(sk.transform(Z))))
        total_variance = Z.var(0, ddof=1).sum()
        sum_of_all_variances = pca_fit(Z, 13)[2].sum()
        print("equal components (up to the sign):", equal_components, "| equal variances:", equal_variances, "| equal reconstruction:", equal_reconstruction)
        print(f"total variance {total_variance:.4f} = sum of the variances of the components {sum_of_all_variances:.4f}")
        print("signs: does my first component have the same sign as scikit-learn's?", bool(np.sign(components[0] @ sk.components_[0]) > 0))
        """),
        C("""
        eigenvalues = np.linalg.eigh(np.cov(Z.T))[0][::-1]                            # eigh returns the eigenvalues in increasing order
        print("first three eigenvalues:", eigenvalues[:3].round(4), "| variances from the SVD:", variances.round(4))
        """),
        C("""
        reconstruction = pca_inverse(pca_transform(Z, mean, components), mean, components)
        mse = ((Z - reconstruction) ** 2).mean()
        theoretical_mse = (total_variance - variances.sum()) * (n - 1) / n / p
        print(f"mean squared reconstruction error with 3 components: {mse:.5f} | sum of the discarded variances (scaled): {theoretical_mse:.5f}")
        """),
    ],
    learned="""
        - **PCA = centering + SVD.** The rows of `Vt` are the directions (the components), the squares of the singular values divided by `n - 1` are the variances: the same thing you get from the **eigenvectors and eigenvalues of the covariance matrix**. With the SVD it is numerically
          more stable (it never forms the covariance matrix).
        - **The sign is arbitrary:** `v` and `-v` describe the same direction. The components of this implementation and of scikit-learn coincide up to the sign; the reconstruction, which does not depend on it, coincides exactly. Never compare signs.
        - The rotation **preserves the total variance** (13 for standardized data) and distributes it in decreasing order among the components. That is why "explained variance" and "reconstruction error" are the same information: the mean squared error with `k` components is the variance you discarded.
        - With `k` components the reconstruction is the **best rank-`k` approximation** (Eckart-Young theorem): no other projection onto `k` dimensions does better in squared error.
    """,
    pitfall="""
        Forgetting to center (or to add back the mean in `inverse_transform`): the SVD of uncentered data puts in the first component the direction of the mean, not that of maximum variance (see E2).
    """,
    interview=(
        "How is PCA computed and what is the relationship with the eigenvalues?",
        """
        You center the data and compute the SVD of the centered matrix X = U S V^T: the rows of V^T are the components and S^2/(n-1) are the variances. It is equivalent to the spectral decomposition of the covariance (the eigenvectors are the components, the eigenvalues the variances), but the SVD is
        numerically more stable and for large matrices truncated or randomized variants are used. The projection onto the first k components is the best rank-k approximation in squared error.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="power_iteration_and_spectral_gap", kind="A", level=3, minutes=55, module=MODULE,
    title="The components with the power method: convergence and the gap between eigenvalues",
    goal="""
        Compute the first principal components with the **power method** (repeatedly multiplying a vector by the covariance matrix) and **deflation** (removing the component found), check the equality with `np.linalg.eigh`, and measure that the speed of
        convergence depends on the ratio $\\lambda_2/\\lambda_1$ (the spectral gap).
    """,
    dataset="""
        `wine_C` (13 × 13): correlation matrix of the wines. `narrow_C` (13 × 13): a symmetric matrix built with the same spectrum except the first two eigenvalues, made almost equal (ratio 0.98).
    """,
    purpose="""
        Understand how the components are computed when the matrix is huge (the power method is at the base of the randomized algorithms and of those for sparse matrices), and understand what the speed depends on: if two components have almost equal variances, telling them apart is hard
        (and numerically unstable).
    """,
    task=[
        "`power(M, tol, max_iter, seed)` → `(v, lam, n_iter, errors)`: start from a normalized random vector (`np.random.default_rng(seed).normal`), repeat `v ← M v / ‖M v‖` until the angle with the previous iteration is below `tol` (`1 − |v_new · v_old| < tol`), "
        "and return the eigenvector `v`, the eigenvalue `lam = v @ M @ v`, the number of iterations and the list of errors `errors[t] = 1 − |v_t · true_v|` (where `true_v` is the true component, passed as an optional argument `true_v=None`; if it is `None`, don't record errors).",
        "`top_k(M, k)` with **deflation**: find the first component, subtract `lam · v v^T` from `M` and repeat. Return eigenvalues (array) and eigenvectors (`k × p` matrix, one per row). Compare with `np.linalg.eigh` (eigenvectors up to the sign).",
        "On the wine matrix save `wine_iters` (iterations to converge with `tol = 1e-12`) and on `narrow_C` `narrow_iters`. Save `iteration_ratio = narrow_iters / wine_iters`.",
        "**Convergence law:** with `true_v` known, estimate `empirical_rate` as the median of `errors[t+1] / errors[t]` for the iterations with `errors[t] > 1e-10` (the error in `1 − cos` drops like the **square** of the ratio $\\lambda_2/\\lambda_1$ per iteration: compare `empirical_rate` with `(lam2/lam1) ** 2`).",
    ],
    setup="""
        from sklearn.datasets import load_wine

        Xw = load_wine().data
        wine_C = np.corrcoef(Xw.T)
        eig, vec = np.linalg.eigh(wine_C)
        eig, vec = eig[::-1], vec[:, ::-1]
        narrow_eig = eig.copy(); narrow_eig[1] = 0.98 * eig[0]                    # second eigenvalue almost equal to the first
        narrow_C = (vec * narrow_eig) @ vec.T
        print("lambda2/lambda1 ratio: wine", round(eig[1] / eig[0], 3), "| narrow", round(narrow_eig[1] / narrow_eig[0], 3))
    """,
    starter=[
        C("""
        def power(M, tol=1e-12, max_iter=100000, seed=0, true_v=None):
            ...


        def top_k(M, k):
            ...
        """),
        C("""
        wine_iters = None
        narrow_iters = None
        iteration_ratio = None
        """),
        C("""
        empirical_rate = None
        theoretical_rate = None
        """),
    ],
    checks="""
        v, lam, n_it, err = power(wine_C, true_v=vec[:, 0])
        check(abs(v @ vec[:, 0]) > 1 - 1e-10 and close(lam, eig[0]) and len(err) == n_it, "first eigenvector and eigenvalue correct; one error per iteration", "v <- M v / ||M v||")
        lams, vs = top_k(wine_C, 3)
        check(close(lams, eig[:3], rtol=1e-6) and all(abs(vs[i] @ vec[:, i]) > 1 - 1e-8 for i in range(3)), "first three eigenvalues and eigenvectors (up to the sign) match eigh", "deflation: M - lam * outer(v, v)")
        check(wine_iters is not None and wine_iters < 100 and narrow_iters > 20 * wine_iters and close(iteration_ratio, narrow_iters / wine_iters), "wine: fewer than 100 iterations; with the narrow gap over 20 times more", "the convergence rate is (lambda2/lambda1)")
        check(empirical_rate is not None and abs(empirical_rate - theoretical_rate) / theoretical_rate < 0.1 and close(theoretical_rate, (eig[1] / eig[0]) ** 2), "the empirical error rate is (lambda2/lambda1)^2 within 10%", "median of errors[t+1]/errors[t]")
    """,
    hint="""
        `v = rng.normal(size=len(M)); v /= np.linalg.norm(v); while ...: w = M @ v; w /= np.linalg.norm(w); converged if 1 - abs(w @ v) < tol; v = w`. Deflation: `M = M - lam * np.outer(v, v)`.
        Rate: `e = np.array(errors); e = e[e > 1e-10]; np.median(e[1:] / e[:-1])`.
    """,
    solution=[
        C("""
        def power(M, tol=1e-12, max_iter=100000, seed=0, true_v=None):
            v = np.random.default_rng(seed).normal(size=len(M))
            v /= np.linalg.norm(v)
            errors = []
            for t in range(1, max_iter + 1):
                w = M @ v
                w /= np.linalg.norm(w)
                if true_v is not None:
                    errors.append(1 - abs(w @ true_v))
                converged = 1 - abs(w @ v) < tol                       # the angle with the previous iteration is negligible
                v = w
                if converged:
                    break
            return v, v @ M @ v, t, errors


        def top_k(M, k):
            M = M.copy()
            eigenvalues, eigenvectors = [], []
            for _ in range(k):
                v, lam, _, _ = power(M)
                eigenvalues.append(lam); eigenvectors.append(v)
                M = M - lam * np.outer(v, v)                          # deflation: I remove the component just found
            return np.array(eigenvalues), np.array(eigenvectors)
        """),
        C("""
        lams, vs = top_k(wine_C, 3)
        print("eigenvalues (power):", lams.round(5), "| eigh:", eig[:3].round(5))
        _, _, wine_iters, wine_errors = power(wine_C, true_v=vec[:, 0])
        _, _, narrow_iters, _ = power(narrow_C, seed=0)
        iteration_ratio = narrow_iters / wine_iters
        print(f"iterations: wine {wine_iters}, narrow matrix {narrow_iters} ({iteration_ratio:.0f} times more)")
        """),
        C("""
        e = np.array(wine_errors)
        e = e[e > 1e-10]
        empirical_rate = float(np.median(e[1:] / e[:-1]))
        theoretical_rate = (eig[1] / eig[0]) ** 2
        print(f"empirical error reduction rate: {empirical_rate:.4f} | (lambda2/lambda1)^2 = {theoretical_rate:.4f}")
        """),
    ],
    learned="""
        - The power method converges to the principal component by multiplying several times by the matrix: every step amplifies the component along the dominant eigenvector by a factor $\\lambda_1/\\lambda_2$ with respect to the second. The error in `1 − cos` drops like
          $(\\lambda_2/\\lambda_1)^{2t}$: with the wines (ratio 0.53) 23 iterations are enough, with a gap of 0.98 you need 513, that is **over twenty times as many**.
        - **The gap between eigenvalues controls everything**: if two components have almost equal variances, their direction is ill-defined (a small noise rotates one into the other) and iterative algorithms slow down. It is the reason why components with close eigenvalues must not be
          interpreted one by one.
        - **Deflation** lets you compute only the first `k` components without the complete decomposition: it is the basis of the `arpack` and `randomized` solvers that scikit-learn uses when `k` is much smaller than `p` (with `svd_solver="auto"`).
        - Numerical error accumulates in deflation (every component is computed from the reduced matrix): for many components block methods are preferred (subspace iteration, Lanczos).
    """,
    pitfall="""
        Interpreting the second and the third component as "distinct factors" when their eigenvalues are almost equal: PCA cannot tell them apart, and an arbitrary rotation of the two is just as valid.
    """,
    interview=(
        "How do you compute PCA on a matrix with millions of rows and thousands of columns?",
        """
        You don't need the complete decomposition: the first k components are enough. You use iterative or randomized methods (power/subspace iteration, Lanczos, randomized SVD) that work on the matrix with matrix-vector products, or `IncrementalPCA` in blocks if the data don't fit in memory.
        The speed depends on the gap between eigenvalues: with small gaps you need more iterations or oversampling.
        """),
)

EXERCISES = [A1, A2]
