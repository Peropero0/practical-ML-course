"""Module 1 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_01_linear_regression"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="ols_normal_equation_and_svd", kind="A", level=2, minutes=30, module=MODULE,
    title="OLS with the normal equation and with the SVD",
    goal="""
        Implement OLS regression in two ways, with the normal equation and with the SVD decomposition, and find out at
        what point the two methods stop giving the same result.
    """,
    dataset="""
        Two synthetic datasets, built in the setup cell:

        - `X`, `y`: 100 rows, 3 independent features and known true coefficients (`y = 3 + 2·x₁ − x₂ + 0.5·x₃ + noise`);
        - `x_poly`, `y_poly`: 60 points in [0, 1] of a sinusoid with noise, for a polynomial regression. The columns
          1, x, x², x³... of a Vandermonde matrix are nearly collinear: the problem is **ill-conditioned**.
    """,
    purpose="""
        Understand why the textbook formula, β = (AᵀA)⁻¹Aᵀy, is not the algorithm used in practice, and what it
        concretely means for a matrix to be ill-conditioned.
    """,
    task=[
        "Write `ols_normal(X, y)`: it adds the column of ones to X, solves the normal equation with `np.linalg.solve` "
        "(not with the explicit inverse) and returns the pair `(intercept, coef)`.",
        "Write `ols_svd(X, y)`: it computes `U, S, Vt = np.linalg.svd(A, full_matrices=False)` and rebuilds "
        "β = V · diag(1/S) · Uᵀ · y, zeroing the singular values smaller than `rcond * S.max()`.",
        "On the first dataset check that both functions match `LinearRegression`.",
        "On the second dataset, for degrees 4, 6, 8 and 10, build the Vandermonde matrix `A` and fill the DataFrame "
        "`table` with the columns `degree`, `cond_A`, `cond_AtA`, `err_normal`, `err_svd`. The error is the maximum "
        "absolute difference of the coefficients from the reference `np.linalg.lstsq(A, y_poly, rcond=None)[0]`.",
        "Look at the table and assign to `critical_degree` the first degree at which `err_normal` exceeds 1e-3.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 3))
        y = 3 + X @ np.array([2.0, -1.0, 0.5]) + rng.normal(scale=0.5, size=100)

        x_poly = np.linspace(0, 1, 60)
        y_poly = np.sin(2 * np.pi * x_poly) + rng.normal(scale=0.1, size=60)
    """,
    starter=[
        C("""
        def ols_normal(X, y):
            \"\"\"Returns (intercept, coef) by solving the normal equation.\"\"\"
            A = ...            # X with a column of ones in first position
            beta = ...         # np.linalg.solve(A.T @ A, A.T @ y)
            return beta[0], beta[1:]


        def ols_svd(X, y, rcond=1e-12):
            \"\"\"Returns (intercept, coef) using the SVD of A.\"\"\"
            ...
        """),
        C("""
        rows = []
        for degree in [4, 6, 8, 10]:
            A = np.vander(x_poly, degree + 1, increasing=True)        # columns: 1, x, x^2, ..., x^degree
            reference = np.linalg.lstsq(A, y_poly, rcond=None)[0]
            ...
            rows.append({"degree": degree, "cond_A": ..., "cond_AtA": ..., "err_normal": ..., "err_svd": ...})

        table = pd.DataFrame(rows)
        table
        """),
        C("""
        critical_degree = ...
        """),
    ],
    checks="""
        ref = LinearRegression().fit(X, y)
        i_n, c_n = ols_normal(X, y)
        i_s, c_s = ols_svd(X, y)
        check(close(i_n, ref.intercept_) and close(c_n, ref.coef_),
              "ols_normal matches LinearRegression",
              "ols_normal does not match LinearRegression: check the column of ones and the normal equation")
        check(close(i_s, ref.intercept_) and close(c_s, ref.coef_),
              "ols_svd matches LinearRegression",
              "ols_svd does not match LinearRegression: check 1/S and the reconstruction of beta")

        check(list(table.columns) == ["degree", "cond_A", "cond_AtA", "err_normal", "err_svd"],
              "the table has the required columns", "the columns of `table` are not the required ones")
        check((table.err_svd < 1e-6).all(),
              "the SVD stays accurate at every degree", "err_svd should stay below 1e-6 at all degrees")
        check(table.err_normal.is_monotonic_increasing,
              "the error of the normal equation grows with the degree", "err_normal should grow with the degree")
        ratio = np.log10(table.cond_AtA) / np.log10(table.cond_A)
        check(np.allclose(ratio, 2, atol=0.25),
              "cond(AtA) is about cond(A) squared",
              "cond_AtA should be about the square of cond_A: did you compute np.linalg.cond(A.T @ A)?")
        check(critical_degree == 8, "correct critical degree",
              "look at which row of `table` has err_normal exceeding 1e-3 for the first time")
    """,
    hint="""
        For the SVD: `S_inv = np.where(S > rcond * S.max(), 1 / S, 0.0)` and then `beta = Vt.T @ (S_inv * (U.T @ y))`.
        For the table: the condition number is `np.linalg.cond(matrix)`; to use your functions on the Vandermonde matrix
        pass `A[:, 1:]` (without the column of ones: they take care of adding it) and recompose
        `np.r_[intercept, coef]`.
    """,
    solution=[
        M("""
        The normal equation is solved with `np.linalg.solve(AtA, Aty)`: it is more stable and faster than computing the inverse. The
        SVD, instead, works directly on A: if A = U·Σ·Vᵀ, then the least squares solution is V·Σ⁺·Uᵀ·y, where
        Σ⁺ holds the inverses of the singular values, and zero in place of the negligible ones.
        """),
        C("""
        def _with_ones(X):
            return np.column_stack([np.ones(len(X)), X])


        def ols_normal(X, y):
            A = _with_ones(X)
            beta = np.linalg.solve(A.T @ A, A.T @ y)
            return beta[0], beta[1:]


        def ols_svd(X, y, rcond=1e-12):
            A = _with_ones(X)
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            S_inv = np.where(S > rcond * S.max(), 1 / S, 0.0)        # tiny singular values get zeroed
            beta = Vt.T @ (S_inv * (U.T @ y))
            return beta[0], beta[1:]
        """),
        M("""
        On the well-conditioned problem the two functions agree up to machine precision. The interesting comparison
        is on the Vandermonde matrix, where the condition number grows fast with the degree.
        """),
        C("""
        rows = []
        for degree in [4, 6, 8, 10]:
            A = np.vander(x_poly, degree + 1, increasing=True)
            reference = np.linalg.lstsq(A, y_poly, rcond=None)[0]

            beta_n = np.r_[ols_normal(A[:, 1:], y_poly)[0], ols_normal(A[:, 1:], y_poly)[1]]
            beta_s = np.r_[ols_svd(A[:, 1:], y_poly)[0], ols_svd(A[:, 1:], y_poly)[1]]

            rows.append({
                "degree": degree,
                "cond_A": np.linalg.cond(A),
                "cond_AtA": np.linalg.cond(A.T @ A),
                "err_normal": np.abs(beta_n - reference).max(),
                "err_svd": np.abs(beta_s - reference).max(),
            })

        table = pd.DataFrame(rows)
        pd.options.display.float_format = "{:.3g}".format
        table
        """),
        C("""
        critical_degree = int(table.loc[table.err_normal > 1e-3, "degree"].min())
        critical_degree
        """),
    ],
    learned="""
        - `cond(AᵀA) = cond(A)²`: squaring the matrix halves the available significant digits. At degree 10 the
          condition number of A is of the order of 10⁷, that of AᵀA of 10¹⁴, and with 16 digits in double precision
          barely two are left: the error of the normal equation is of the order of 10¹.
        - The SVD does not go through AᵀA and keeps an error close to machine precision even where the normal equation is
          useless. That is why `LinearRegression` uses `scipy.linalg.lstsq`.
        - "Correct formula" and "correct algorithm" are not the same thing: in finite arithmetic how you compute matters.
    """,
    pitfall="""
        The conditioning problem does not concern only polynomials: it shows up whenever two or more features are nearly
        collinear or have very different scales. With the normal equation it happens silently: no error, just wrong
        coefficients. Scaling the features (and not using `np.linalg.inv`) reduces the risk but does not eliminate it.
    """,
    interview=(
        "A colleague computes the regression with (XᵀX)⁻¹Xᵀy and sometimes gets absurd coefficients. Where do you start?",
        """
        I check `np.linalg.cond(X.T @ X)`: if it is close to 10¹⁶ the matrix is numerically singular, typically because of collinear
        features or very different scales. Then I switch to a solver based on QR or SVD (`np.linalg.lstsq`,
        `LinearRegression`), which does not square the conditioning, I scale the features and, if the collinearity is
        real, I consider a Ridge.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="gradient_descent", kind="A", level=2, minutes=35, module=MODULE,
    title="Gradient descent for linear regression",
    goal="""
        Implement gradient descent for OLS, show that on unscaled features it diverges, and compute the maximum stable
        learning rate starting from the Hessian.
    """,
    dataset="""
        Synthetic, 300 rows: `area` (square meters, order of magnitude 10²) and `income` (euros, order of magnitude
        10⁴), with `y = 50 + 2·area + 0.003·income + noise`. The two features have units of measure
        deliberately far apart.
    """,
    purpose="""
        Connect three things that textbooks usually present separately: the OLS cost function, its gradient and
        convexity. You will see that the scale of the features decides **how large the learning rate can be**, not just the speed of
        convergence.
    """,
    task=[
        "Write `gradient_descent(A, y, lr, n_iter)`: A already contains the column of ones; start from `w = 0`; at every iteration "
        "record the MSE in `history` and update `w ← w − lr · (2/n) · Aᵀ(Aw − y)`. If the MSE is not finite or exceeds 1e30 "
        "break the loop. Return `(w, history)` with `history` as a numpy array.",
        "On the **raw** data (`A_raw`) run `lr=0.01` for 200 iterations. Assign to `diverges_raw` `True` if "
        "the last recorded MSE is greater than the first.",
        "Standardize the features (subtract the mean and divide by the standard deviation) to build `A_std`, run "
        "`lr=0.1` for 500 iterations and compare `w_gd` with the exact solution `w_ols` (`np.linalg.lstsq`).",
        "Compute the Hessian of the MSE `H = (2/n) AᵀA` on the standardized data and the maximum stable learning rate "
        "`lr_max = 2 / λ_max(H)`. Check that `0.9 * lr_max` converges (`conv_09`) and that `1.1 * lr_max` diverges (`div_11`).",
        "Repeat the computation of `lr_max` on the raw data (`lr_max_raw`) and notice by how many orders of magnitude it is smaller.",
    ],
    setup="""
        rng = np.random.default_rng(0)
        n = 300
        area = rng.normal(90, 30, n)                     # square meters
        income = rng.normal(30_000, 10_000, n)           # euros
        X = np.column_stack([area, income])
        y = 50 + 2.0 * area + 0.003 * income + rng.normal(0, 5, n)

        A_raw = np.column_stack([np.ones(n), X])
    """,
    starter=[
        C("""
        def gradient_descent(A, y, lr, n_iter):
            n = len(y)
            w = np.zeros(A.shape[1])
            history = []
            for _ in range(n_iter):
                ...
            return w, np.array(history)
        """),
        C("""
        w_raw, history_raw = gradient_descent(A_raw, y, lr=0.01, n_iter=200)
        diverges_raw = ...
        """),
        C("""
        X_std = ...
        A_std = np.column_stack([np.ones(n), X_std])
        w_gd, history_std = gradient_descent(A_std, y, lr=0.1, n_iter=500)
        w_ols = np.linalg.lstsq(A_std, y, rcond=None)[0]
        """),
        C("""
        H = ...
        lr_max = ...
        w_a, history_a = gradient_descent(A_std, y, lr=0.9 * lr_max, n_iter=500)
        w_b, history_b = gradient_descent(A_std, y, lr=1.1 * lr_max, n_iter=500)
        conv_09 = ...
        div_11 = ...
        """),
        C("""
        lr_max_raw = ...
        """),
    ],
    checks="""
        check(isinstance(history_std, np.ndarray) and len(history_std) == 500,
              "the function returns the MSE history as an array", "`history` must be an array with one MSE per iteration")
        check(bool(diverges_raw), "on the raw data gradient descent diverges",
              "with lr=0.01 on unscaled data the MSE should explode")
        check(close(w_gd, w_ols, atol=1e-3), "on the standardized data you reach the OLS solution",
              "w_gd does not match the exact solution: check the gradient (2/n) A^T (A w - y)")
        check(np.all(np.diff(history_std) <= 1e-12), "the MSE never grows (convex function, small lr)",
              "with a small learning rate the MSE should go down at every step")
        lr_expected = 2 / np.linalg.eigvalsh((2 / n) * A_std.T @ A_std).max()
        check(close(lr_max, lr_expected, rtol=1e-6), "correct lr_max",
              "lr_max = 2 / (largest eigenvalue of H), with H = (2/n) A^T A")
        check(bool(conv_09) and bool(div_11), "0.9*lr_max converges and 1.1*lr_max diverges",
              "conv_09 must be True and div_11 True: compare the last MSE with the first")
        check(lr_max_raw < 1e-6 and lr_max / lr_max_raw > 1e6,
              "on the raw data the stable learning rate is millions of times smaller",
              "compute lr_max on the raw matrix: it should come out of the order of 1e-9")
    """,
    hint="""
        The gradient is `(2 / n) * A.T @ (A @ w - y)`. The eigenvalues of a symmetric matrix are computed with
        `np.linalg.eigvalsh(H)`. To declare divergence compare `history[-1]` with `history[0]`, remembering that
        the loop stops when the MSE is no longer finite.
    """,
    solution=[
        M("""
        The MSE is `(1/n)·‖Aw − y‖²`: its gradient is `(2/n)·Aᵀ(Aw − y)` and its Hessian `H = (2/n)·AᵀA`, constant.
        A quadratic function with a positive definite Hessian is convex, so gradient descent converges to the OLS
        solution if the learning rate is small enough. "Small enough" has a precise answer: `lr < 2/λ_max(H)`.
        """),
        C("""
        def gradient_descent(A, y, lr, n_iter):
            n = len(y)
            w = np.zeros(A.shape[1])
            history = []
            for _ in range(n_iter):
                residual = A @ w - y
                with np.errstate(all="ignore"):
                    mse = np.mean(residual ** 2)
                history.append(mse)
                if not np.isfinite(mse) or mse > 1e30:
                    break
                w = w - lr * (2 / n) * A.T @ residual
            return w, np.array(history)
        """),
        C("""
        w_raw, history_raw = gradient_descent(A_raw, y, lr=0.01, n_iter=200)
        diverges_raw = history_raw[-1] > history_raw[0]
        print("initial MSE:", f"{history_raw[0]:.3g}", "| last recorded:", f"{history_raw[-1]:.3g}",
              "| iterations done:", len(history_raw))
        """),
        C("""
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        A_std = np.column_stack([np.ones(n), X_std])
        w_gd, history_std = gradient_descent(A_std, y, lr=0.1, n_iter=500)
        w_ols = np.linalg.lstsq(A_std, y, rcond=None)[0]
        print("w_gd :", w_gd.round(4))
        print("w_ols:", w_ols.round(4))
        """),
        C("""
        H = (2 / n) * A_std.T @ A_std
        lr_max = 2 / np.linalg.eigvalsh(H).max()
        print("lr_max (standardized data):", round(lr_max, 4))

        w_a, history_a = gradient_descent(A_std, y, lr=0.9 * lr_max, n_iter=500)
        w_b, history_b = gradient_descent(A_std, y, lr=1.1 * lr_max, n_iter=500)
        conv_09 = history_a[-1] < history_a[0] and close(w_a, w_ols, atol=1e-3)
        div_11 = history_b[-1] > history_b[0]
        print("0.9 * lr_max converges:", conv_09, "| 1.1 * lr_max diverges:", div_11)
        """),
        C("""
        H_raw = (2 / n) * A_raw.T @ A_raw
        lr_max_raw = 2 / np.linalg.eigvalsh(H_raw).max()
        print("lr_max (raw data):", f"{lr_max_raw:.3g}", "| ratio:", f"{lr_max / lr_max_raw:.3g}")
        """),
    ],
    learned="""
        - The stability of gradient descent on a quadratic is governed by λ_max(H): `lr < 2/λ_max`. If you exceed it
          the error is multiplied by a factor greater than 1 at every step.
        - Income (order 10⁴) makes λ_max grow by almost 9 orders of magnitude: to avoid diverging you need a learning rate
          of the order of 10⁻⁹, with which the other direction (the area) practically never moves. Standardizing
          brings all the eigenvalues to the same order of magnitude and lets you use large steps.
        - Convexity means there is a single minimum; it does **not** mean that any learning rate reaches it.
    """,
    pitfall="""
        At the first `nan` in the loss you tend to blame the data or the code. With gradient descent the most frequent cause is a
        learning rate too large for the scale of the features. Before looking for bugs elsewhere, reduce lr by an order of
        magnitude and scale the features.
    """,
    interview=(
        "Why is gradient descent slower (or does it diverge) when the features have very different scales, even though OLS has "
        "an exact solution?",
        """
        Because the surface of the MSE becomes a very long and narrow valley: the Hessian has very different eigenvalues. The
        learning rate is limited by the largest eigenvalue (2/λ_max) but the speed along the direction with the smallest eigenvalue
        is proportional to lr·λ_min, so the ratio λ_max/λ_min (the conditioning) decides how many iterations are needed.
        Standardizing brings the eigenvalues closer to each other.
        """),
)

# ---------------------------------------------------------------------------------------- A3

A3 = dict(
    id="A3", slug="standard_errors_formula_and_bootstrap", kind="A", level=3, minutes=40, module=MODULE,
    title="Standard errors of the coefficients: closed formula versus bootstrap",
    goal="""
        Compute the uncertainty of the OLS coefficients in two ways, with the analytic formula and with the pairs bootstrap, and
        show that the two results coincide only if the errors are homoscedastic.
    """,
    dataset="""
        Synthetic, generated by the function `generate(heteroscedastic)`: 150 rows, 3 Gaussian features, true coefficients
        `[1, 2, -1, 0.5]` (intercept included). With `heteroscedastic=False` the noise has constant standard deviation (2.0);
        with `True` it grows with the absolute value of the first feature.
    """,
    purpose="""
        You already know the theory: `Var(β̂) = σ² (AᵀA)⁻¹`. Here you implement it, compare it with an estimate that makes no assumptions about the
        variance of the errors, and see in a concrete case what happens to the t tests and the confidence intervals when
        homoscedasticity does not hold.
    """,
    task=[
        "Write `standard_errors_formula(X, y)`: it returns an array with the standard errors of `[intercept, coef...]` computed "
        "as the square root of the diagonal of `σ̂² (AᵀA)⁻¹`, with `σ̂² = SSR / (n − p − 1)`.",
        "Write `standard_errors_bootstrap(X, y, B=2000, seed=0)`: it resamples B times the **rows** (x, y pairs) with "
        "replacement, re-estimates the OLS with `np.linalg.lstsq` and returns `(se, coef_boot)`, where `se` is the standard deviation "
        "(`ddof=1`) of each coefficient and `coef_boot` the B×4 matrix of the estimates.",
        "On the homoscedastic dataset compute `ratio_homo = se_boot / se_formula` (one value per coefficient).",
        "On the heteroscedastic dataset compute `ratio_het` the same way. Assign to `underestimates_x1` `True` if for the "
        "coefficient of the **first feature** (index 1) the ratio exceeds 1.3, that is if the formula underestimates the uncertainty.",
        "With the homoscedastic bootstrap build the 95% confidence interval (2.5 and 97.5 percentiles) of each "
        "coefficient and save in `ci_width` the width of the intervals.",
    ],
    setup="""
        TRUE_BETA = np.array([1.0, 2.0, -1.0, 0.5])


        def generate(heteroscedastic, seed=1, n=150):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(n, 3))
            sigma = 2.0 * (1.5 * np.abs(X[:, 0]) + 0.1) if heteroscedastic else 2.0
            y = TRUE_BETA[0] + X @ TRUE_BETA[1:] + rng.normal(size=n) * sigma
            return X, y


        X_homo, y_homo = generate(False)
        X_het, y_het = generate(True)
    """,
    starter=[
        C("""
        def standard_errors_formula(X, y):
            A = np.column_stack([np.ones(len(X)), X])
            n, k = A.shape                       # k = number of coefficients, intercept included
            ...


        def standard_errors_bootstrap(X, y, B=2000, seed=0):
            rng = np.random.default_rng(seed)
            A = np.column_stack([np.ones(len(X)), X])
            coef_boot = np.empty((B, A.shape[1]))
            for b in range(B):
                ...
            return coef_boot.std(axis=0, ddof=1), coef_boot
        """),
        C("""
        se_formula_homo = standard_errors_formula(X_homo, y_homo)
        se_boot_homo, coef_boot_homo = standard_errors_bootstrap(X_homo, y_homo)
        ratio_homo = ...
        """),
        C("""
        se_formula_het = standard_errors_formula(X_het, y_het)
        se_boot_het, _ = standard_errors_bootstrap(X_het, y_het)
        ratio_het = ...
        underestimates_x1 = ...
        """),
        C("""
        ci = np.percentile(coef_boot_homo, [2.5, 97.5], axis=0)
        ci_width = ...
        """),
    ],
    checks="""
        A = np.column_stack([np.ones(len(X_homo)), X_homo])
        e = y_homo - A @ np.linalg.lstsq(A, y_homo, rcond=None)[0]
        expected = np.sqrt(np.diag(e @ e / (len(y_homo) - 4) * np.linalg.inv(A.T @ A)))
        check(close(se_formula_homo, expected), "the analytic standard errors are correct",
              "standard_errors_formula does not match sqrt(diag(sigma2 * inv(A^T A))): check the degrees of freedom n-p-1")
        check(coef_boot_homo.shape == (2000, 4), "the bootstrap returns B x 4 estimates", "coef_boot must have shape (B, 4)")
        check(np.all((ratio_homo > 0.85) & (ratio_homo < 1.2)),
              "with homoscedastic errors formula and bootstrap agree (ratios between 0.85 and 1.2)",
              "on the homoscedastic dataset the two methods should give similar standard errors")
        check(bool(underestimates_x1) and ratio_het[1] > 1.3,
              "with heteroscedastic errors the formula underestimates the uncertainty of the first feature",
              "on the heteroscedastic dataset the ratio for the first feature should exceed 1.3")
        check(close(ci_width, 3.92 * se_boot_homo, rtol=0.15),
              "the 95% intervals are about 3.92 standard errors wide",
              "ci_width should be ci[1] - ci[0], that is about 2 * 1.96 * se")
    """,
    hint="""
        Bootstrap: `idx = rng.integers(0, n, n)` draws the indices with replacement; then
        `coef_boot[b] = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0]`. Formula: `sigma2 = ssr / (n - k)` and
        `np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))`.
    """,
    solution=[
        C("""
        def standard_errors_formula(X, y):
            A = np.column_stack([np.ones(len(X)), X])
            n, k = A.shape
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            residuals = y - A @ beta
            sigma2 = residuals @ residuals / (n - k)          # unbiased estimate of the error variance
            return np.sqrt(np.diag(sigma2 * np.linalg.inv(A.T @ A)))


        def standard_errors_bootstrap(X, y, B=2000, seed=0):
            rng = np.random.default_rng(seed)
            A = np.column_stack([np.ones(len(X)), X])
            coef_boot = np.empty((B, A.shape[1]))
            for b in range(B):
                idx = rng.integers(0, len(y), len(y))         # resample the ROWS, (x, y) pairs together
                coef_boot[b] = np.linalg.lstsq(A[idx], y[idx], rcond=None)[0]
            return coef_boot.std(axis=0, ddof=1), coef_boot
        """),
        C("""
        se_formula_homo = standard_errors_formula(X_homo, y_homo)
        se_boot_homo, coef_boot_homo = standard_errors_bootstrap(X_homo, y_homo)
        ratio_homo = se_boot_homo / se_formula_homo
        print("homoscedastic   - bootstrap/formula ratio:", ratio_homo.round(2))
        """),
        C("""
        se_formula_het = standard_errors_formula(X_het, y_het)
        se_boot_het, _ = standard_errors_bootstrap(X_het, y_het)
        ratio_het = se_boot_het / se_formula_het
        underestimates_x1 = bool(ratio_het[1] > 1.3)
        print("heteroscedastic - bootstrap/formula ratio:", ratio_het.round(2))
        """),
        C("""
        ci = np.percentile(coef_boot_homo, [2.5, 97.5], axis=0)
        ci_width = ci[1] - ci[0]
        print("95% intervals:")
        print(pd.DataFrame(ci.T, columns=["2.5%", "97.5%"], index=["intercept", "x1", "x2", "x3"]).assign(true=TRUE_BETA).round(2))
        """),
        M("""
        Cross-check with the reference statistics library in Python: the analytic standard errors must
        coincide with those of `statsmodels` (which uses the same formula).
        """),
        C("""
        import statsmodels.api as sm
        ols = sm.OLS(y_homo, sm.add_constant(X_homo)).fit()
        print("statsmodels:", ols.bse.round(4))
        print("yours      :", se_formula_homo.round(4))
        """),
    ],
    learned="""
        - The formula `σ̂² (AᵀA)⁻¹` holds **only** if the errors have constant variance and are independent. When the
          variance grows with the feature, the extreme points (high leverage) are also the ones with the largest error, and the formula
          treats them as if they were normal noise: the uncertainty on the slope is underestimated (here by about 40%).
        - The pairs bootstrap assumes nothing about the shape of the variance: it is an honest estimate even when the
          classical assumptions fail. It costs B fits instead of one, but on small data it is irrelevant.
        - A 95% confidence interval is about 4 standard errors wide: if the standard error is wrong, so are the
          p-values and the intervals.
    """,
    pitfall="""
        The bootstrap must be done on the **rows** (x, y pairs). Resampling only the residuals is a different procedure (residual
        bootstrap) that presupposes homoscedasticity itself, the very assumption you want to test here.
    """,
    interview=(
        "In a regression the p-values of the coefficients look great but the residual plot shows a funnel. What do you do?",
        """
        The funnel indicates heteroscedasticity: the coefficient estimates remain unbiased, but the classical standard errors are
        wrong, and so are the p-values and intervals. I would recompute the uncertainty with robust standard errors (HC) or with the
        pairs bootstrap, and I would check whether a transformation of the target (for example the logarithm) stabilizes the variance.
        """),
)

EXERCISES = [A1, A2, A3]
