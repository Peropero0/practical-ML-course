"""Module 6 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_06_pca"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="choosing_k_in_a_pipeline_on_digits", kind="B", level=2, minutes=50, module=MODULE,
    title="How many components? PCA in a pipeline on the digits",
    goal="""
        Choose the number of components of a PCA in a `StandardScaler` → `PCA` → `LogisticRegression` `Pipeline` with cross-validation, compare with the model without reduction, and find out that **standardizing the pixels changes how many components are needed**: `PCA(0.95)` chooses
        a different number depending on the scaler (and from one fold to another).
    """,
    dataset="""
        `load_digits`: 1797 8×8 images, 64 pixels (0-16), 10 classes. Validation: `StratifiedKFold(5, shuffle=True, random_state=0)`, accuracy.
    """,
    purpose="""
        Treat `k` as a hyperparameter (like `C`), read the accuracy curve as a function of `k` with an explicit choice rule and understand that `n_components=0.95` is not a number: it is a rule, and the number it gives depends on the data and on the preprocessing.
    """,
    task=[
        "Compute `acc_k` (`pd.Series` indexed by `k`) with the cross-validated accuracy of `make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=3000))` for `k` in `[2, 5, 10, 15, 20, 30, 40, 64]`.",
        "Save `acc_without_pca` (same pipeline without PCA) and `rule_k`: the **smallest** `k` whose accuracy is within **0.01** of the maximum of `acc_k`. Save `rule_k_loss = acc_without_pca - acc_k[rule_k]`.",
        "With `cross_validate(..., return_estimator=True)` on the pipeline `StandardScaler` → `PCA(0.95)` → logistic, save `scaled_k_per_fold` (list of `n_components_` per fold). Repeat **without** `StandardScaler` (`PCA(0.95)` on the raw pixels) and save `raw_k_per_fold` and `raw_acc` (mean accuracy).",
        "Save `k_difference`: mean of `scaled_k_per_fold` minus mean of `raw_k_per_fold`.",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_digits(return_X_y=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        ks = [2, 5, 10, 15, 20, 30, 40, 64]
    """,
    starter=[
        C("""
        acc_k = pd.Series(dtype=float)
        acc_k
        """),
        C("""
        acc_without_pca = None
        rule_k = None
        rule_k_loss = None
        """),
        C("""
        scaled_k_per_fold = None
        raw_k_per_fold = None
        raw_acc = None
        k_difference = None
        """),
    ],
    checks="""
        check(list(acc_k.index) == ks and acc_k[2] < 0.6 and acc_k[64] > 0.96, "curve over eight values of k: with 2 components below 0.6, with 64 above 0.96", "pipeline with StandardScaler, PCA(k), logistic")
        check(acc_k.is_monotonic_increasing, "the accuracy grows with k", "more components, more information")
        check(acc_without_pca > 0.96 and abs(acc_without_pca - acc_k[64]) < 1e-9, "with 64 components (no reduction) the accuracy is the one without PCA: a rotation does not change the logistic", "PCA(64) is just a rotation of the axes")
        check(rule_k == min(k for k in ks if acc_k[k] >= acc_k.max() - 0.01) and 30 <= rule_k <= 40 and 0 < rule_k_loss < 0.012, "k of the rule: 30-40 components out of 64, with a loss below 1.2 points", "the smallest k within 0.01 of the maximum")
        check(len(scaled_k_per_fold) == 5 and all(35 <= k <= 45 for k in scaled_k_per_fold) and all(24 <= k <= 32 for k in raw_k_per_fold), "PCA(0.95): 35-45 components with the scaler, 24-32 without", "cross_validate(..., return_estimator=True) and pipeline[k].n_components_")
        check(k_difference > 8 and raw_acc > 0.94, "the scaler requires over 8 more components for the same 95% of variance; without the scaler the accuracy stays above 0.94", "means of k_per_fold")
    """,
    hint="""
        To get the components per fold: `r = cross_validate(pipe, X, y, cv=cv, return_estimator=True); [e["pca"].n_components_ for e in r["estimator"]]` (use a name for the step, for example `Pipeline([("pca", PCA(0.95)), ...])`, or `e[1].n_components_` with `make_pipeline`).
    """,
    solution=[
        C("""
        def pipeline(k, scale=True):
            steps = ([StandardScaler()] if scale else []) + [PCA(k), LogisticRegression(max_iter=3000)]
            return make_pipeline(*steps)


        acc_k = pd.Series({k: cross_val_score(pipeline(k), X, y, cv=cv).mean() for k in ks})
        acc_k.round(4)
        """),
        C("""
        acc_without_pca = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), X, y, cv=cv).mean()
        rule_k = min(k for k in ks if acc_k[k] >= acc_k.max() - 0.01)
        rule_k_loss = acc_without_pca - acc_k[rule_k]
        print(f"without PCA {acc_without_pca:.4f} | k of the rule (within 0.01 of the maximum): {rule_k} components out of 64, accuracy {acc_k[rule_k]:.4f} (loss {rule_k_loss:.4f})")
        """),
        C("""
        r_scaled = cross_validate(pipeline(0.95), X, y, cv=cv, return_estimator=True)
        r_raw = cross_validate(pipeline(0.95, scale=False), X, y, cv=cv, return_estimator=True)
        scaled_k_per_fold = [e[-2].n_components_ for e in r_scaled["estimator"]]           # the second-to-last step is the PCA
        raw_k_per_fold = [e[-2].n_components_ for e in r_raw["estimator"]]
        raw_acc = r_raw["test_score"].mean()
        k_difference = np.mean(scaled_k_per_fold) - np.mean(raw_k_per_fold)
        print("components for 95% of the variance, per fold | with scaler:", scaled_k_per_fold, "| without scaler:", raw_k_per_fold)
        print(f"accuracy without scaler: {raw_acc:.4f} | extra components with the scaler: {k_difference:.1f}")
        """),
    ],
    learned="""
        - **`k` is a hyperparameter:** the accuracy grows with `k` and flattens (0.54 with 2 components, 0.94 with 20, 0.963 with 40, 0.969 with 64). With the rule "the smallest `k` within one point of the maximum" you keep 40 components out of 64 while losing less than a point.
        - **PCA did not improve the accuracy** (0.969 without reduction): with a logistic there is no noise to remove and a rotation adds no information. The gain is **compression** (fewer dimensions, less memory and downstream time), at almost no cost.
        - **`PCA(0.95)` is a rule, not a number:** with the scaler you need to keep about 40 components (39-40 depending on the fold), without the scaler 28 (28-29). Standardizing the pixels gives the same weight to the almost always dark ones (edges), which are largely noise, and the variance
          spreads over more components. Scaling is not "always right" (M5 F1 showed the same for distances): on images with homogeneous pixels it is often not needed.
        - PCA inside the `Pipeline` is redone at every fold: the number of components can vary (here by one). With a fixed `k` the behavior is more predictable; with a threshold the final model has a number of components that depends on the training data.
    """,
    pitfall="""
        Computing PCA on the whole dataset "to decide `k`" and then putting `PCA(k)` in cross-validation. It is almost harmless with PCA (it does not look at `y`) but the chosen number already depends on the validation folds: when in doubt, `GridSearchCV` on the parameter `pca__n_components` inside the pipeline.
    """,
    interview=(
        "How do you choose the number of principal components?",
        """
        It depends on the use. For visualization 2-3. For reconstruction a threshold on the cumulative variance or the required reconstruction error. For a downstream model I treat `k` as a hyperparameter and search it in cross-validation inside the pipeline, with a parsimony rule
        (the smallest k within the standard error of the best). I remember that the explained variance measures reconstruction fidelity, not usefulness for the target.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="pca_to_remove_noise", kind="B", level=3, minutes=55, module=MODULE,
    title="Removing noise from images with PCA",
    goal="""
        Add Gaussian noise to the digits, reconstruct with `k` components and measure the error with respect to the **clean** image; choose `k` with a criterion that uses only the noisy data (the Marchenko-Pastur edge: the components with an eigenvalue above the expected noise) and
        compare with the absolute best `k`; finally verify that a classifier trained on the components is more robust than one on the pixels.
    """,
    dataset="""
        `load_digits`, stratified 60/40 split, with Gaussian noise of standard deviation 4 added to training and test (`X_tr_n`, `X_te_n`; the clean versions `X_tr`, `X_te` serve only to **measure**). The pixels go from 0 to 16.
    """,
    purpose="""
        See PCA as a **filter**: the signal (the shapes of the digits) sits in the first components, the noise is spread over all of them, so reconstructing with few components reduces it. And learn a practical criterion to choose `k` without having the clean image.
    """,
    task=[
        "For `k` in `[2, 5, 10, 15, 20, 25, 30, 40, 64]` train `PCA(k)` on the **noisy training set**, reconstruct the noisy test set and compute the mean squared error with respect to the **clean** test set: save `mse_per_k` (`pd.Series`), `oracle_k` (the best) and `noisy_mse` (error of the noisy image without reconstruction).",
        "**Criterion without oracle:** with `sigma = 4`, `n, p = X_tr_n.shape` and `threshold = sigma**2 * (1 + np.sqrt(p / n))**2` (the maximum edge of the eigenvalues of the noise alone), save `mp_k` = number of components with `explained_variance_ > threshold` of the complete PCA on the noisy training set. "
        "Save `mp_mse` (error with `mp_k` components) and `reduction` = `1 - mse_per_k[oracle_k] / noisy_mse`.",
        "**Classification:** train `LogisticRegression(max_iter=3000)` on the noisy pixels of the training set and evaluate on the noisy test set (`pixel_acc`); then `PCA(oracle_k)` + logistic (`pca_acc`). Also save `clean_acc` (logistic trained and evaluated on the clean data).",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline

        X, y = load_digits(return_X_y=True)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0)
        sigma = 4
        rng = np.random.default_rng(0)
        X_tr_n = X_tr + rng.normal(0, sigma, X_tr.shape)
        X_te_n = X_te + rng.normal(0, sigma, X_te.shape)
        ks = [2, 5, 10, 15, 20, 25, 30, 40, 64]
    """,
    starter=[
        C("""
        mse_per_k = pd.Series(dtype=float)
        noisy_mse = None
        oracle_k = None
        """),
        C("""
        mp_k = None
        mp_mse = None
        reduction = None
        """),
        C("""
        pixel_acc = None
        pca_acc = None
        clean_acc = None
        """),
    ],
    checks="""
        check(close(noisy_mse, ((X_te_n - X_te) ** 2).mean()) and close(noisy_mse, 16, rtol=0.1), "the noisy image has an error of about 16 (= sigma^2)", "np.mean((X_te_n - X_te) ** 2)")
        check(list(mse_per_k.index) == ks and mse_per_k[64] > 0.97 * noisy_mse, "with 64 components (no filter) the error stays that of the noise", "reconstruction with all the components = noisy image")
        check(oracle_k == mse_per_k.idxmin() and 10 <= oracle_k <= 20 and mse_per_k[oracle_k] < 0.5 * noisy_mse, "the best k is between 10 and 20 and halves the error (over 50%)", "idxmin of mse_per_k")
        check(20 <= mp_k <= 35 and mp_mse < 0.6 * noisy_mse and mp_mse > mse_per_k[oracle_k], "the noise-edge criterion gives k between 20 and 35: almost optimal, but more conservative than the oracle", "components with an eigenvalue above the threshold")
        check(close(reduction, 1 - mse_per_k[oracle_k] / noisy_mse) and reduction > 0.5, "the error is reduced by over 50%", "1 - oracle_mse / noisy_mse")
        check(pca_acc > pixel_acc + 0.02 and clean_acc > 0.95 and pixel_acc < 0.9, "on the noisy images the logistic on the components beats the one on the pixels by over 2 points (clean: above 0.95)", "compare pca_acc, pixel_acc, clean_acc")
    """,
    hint="""
        `p = PCA(k).fit(X_tr_n); R = p.inverse_transform(p.transform(X_te_n)); ((R - X_te) ** 2).mean()`. Criterion: `(PCA().fit(X_tr_n).explained_variance_ > threshold).sum()`. Classification: `make_pipeline(PCA(oracle_k), LogisticRegression(max_iter=3000)).fit(X_tr_n, y_tr).score(X_te_n, y_te)`.
    """,
    solution=[
        C("""
        noisy_mse = ((X_te_n - X_te) ** 2).mean()
        mse_per_k = {}
        for k in ks:
            p = PCA(k).fit(X_tr_n)                                            # PCA sees only noisy data
            reconstructed = p.inverse_transform(p.transform(X_te_n))
            mse_per_k[k] = ((reconstructed - X_te) ** 2).mean()                # but the error is measured against the clean image
        mse_per_k = pd.Series(mse_per_k)
        oracle_k = int(mse_per_k.idxmin())
        print(f"error of the noisy image: {noisy_mse:.2f}")
        print(mse_per_k.round(2).to_string())
        """),
        C("""
        n, p_ = X_tr_n.shape
        threshold = sigma ** 2 * (1 + np.sqrt(p_ / n)) ** 2                       # maximum eigenvalue expected if there were only noise
        full = PCA().fit(X_tr_n)
        mp_k = int((full.explained_variance_ > threshold).sum())
        pk = PCA(mp_k).fit(X_tr_n)
        mp_mse = ((pk.inverse_transform(pk.transform(X_te_n)) - X_te) ** 2).mean()
        reduction = 1 - mse_per_k[oracle_k] / noisy_mse
        print(f"noise threshold: {threshold:.1f} | components above the threshold: {mp_k} | error with those: {mp_mse:.2f} (oracle k={oracle_k}: {mse_per_k[oracle_k]:.2f}) | error reduction with the oracle: {reduction:.0%}")
        """),
        C("""
        pixel_acc = LogisticRegression(max_iter=3000).fit(X_tr_n, y_tr).score(X_te_n, y_te)
        pca_acc = make_pipeline(PCA(oracle_k), LogisticRegression(max_iter=3000)).fit(X_tr_n, y_tr).score(X_te_n, y_te)
        clean_acc = LogisticRegression(max_iter=3000).fit(X_tr, y_tr).score(X_te, y_te)
        print(f"logistic on noisy pixels {pixel_acc:.3f} | on PCA({oracle_k}) of noisy data {pca_acc:.3f} | on clean data {clean_acc:.3f}")
        """),
    ],
    learned="""
        - **Reconstruction with few components filters the noise:** with 15 components the error with respect to the clean image drops from 16 to 7.2 (−55%), better than the noisy image and much better than the 64 pixels. With too many components (64) the error is that of the noise; with too few
          (2) the signal is lost (14).
        - The tradeoff is bias-variance applied to the reconstruction: every added component brings a bit of signal and a quantity of noise. The minimum is where the marginal signal equals the noise.
        - **Without the clean image** you can choose `k` with the Marchenko-Pastur edge: the components with an eigenvalue above $\\sigma^2(1+\\sqrt{p/n})^2$ are not compatible with noise alone. Here it indicates 25 (error 8.0), more conservative than the optimum (15, error 7.2) but almost as good and
          chosen **only with the noisy data** and knowledge of `sigma`.
        - The filter also helps a classifier trained on the noisy data (0.86 against 0.82 on the pixels): the advantage is modest compared to the clean case (0.97), because the noise is strong, but it is free. It is the reason why PCA appears in so many pipelines on sensor data and images.
    """,
    pitfall="""
        Choosing `k` by minimizing the reconstruction error **on the noisy training set**: it always decreases with `k` and would choose 64, that is the whole noise. The right error is the one against a clean reference, or a criterion that accounts for the expected noise.
    """,
    interview=(
        "How can you use PCA for denoising and how do you choose how many components to keep?",
        """
        I reconstruct the data with the first k components, which contain the correlated signal while the noise, spread over all the directions, is mostly discarded. I choose k with a clean reference if I have one, or by comparing the eigenvalues with the expected noise level
        (Marchenko-Pastur edge), or with cross-validation on a downstream task. It works if the signal is low-dimensional and the noise is uncorrelated; with correlated noise or a nonlinear signal you need something else (autoencoder, kernel PCA).
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="regression_on_components_and_pls", kind="B", level=3, minutes=55, module=MODULE,
    title="Many correlated features: OLS, Ridge, regression on the components and PLS",
    goal="""
        Compare four ways of handling 40 correlated features (generated from 5 latent factors) with only 100 training examples: linear regression, Ridge, **principal component regression** (PCR: PCA + linear regression) with `k` chosen in cross-validation, and **PLS**
        (components chosen by looking at `y` too).
    """,
    dataset="""
        `generate()` (given): 200 examples, 40 features built as combinations of 5 latent factors plus noise; `y` depends on 3 of the 5 factors (plus noise). 50/50 split: `X_tr, X_te, y_tr, y_te`.
    """,
    purpose="""
        Recognize the classic case for which PCR exists (highly correlated features, few useful directions) and see its limit: the components are chosen without looking at `y`. PLS is the supervised alternative.
    """,
    task=[
        "Compute `test_r2` (dictionary) with the test R² of `OLS` (`LinearRegression`), `Ridge` (`RidgeCV(alphas=np.logspace(-2, 4, 20))`), `PCR` and `PLS` (see below).",
        "**PCR:** for `k` in `range(1, 21)` compute `cv_pcr` (mean R² in `KFold(5, shuffle=True, random_state=0)` on the training set) of the pipeline `PCA(k)` → `LinearRegression`. Save `pcr_k`: the **smallest** `k` with `cv_pcr` within 0.01 of the maximum.",
        "**PLS:** same procedure with `PLSRegression(n_components=k)` for `k` in `range(1, 11)` (`cv_pls`) and `pls_n` (smallest `k` within 0.01 of the maximum).",
        "Save `pls_has_fewer_components` (True if `pls_n < pcr_k`) and fill `test_r2` with the final models trained with `pcr_k` and `pls_n`.",
    ],
    setup="""
        from sklearn.cross_decomposition import PLSRegression
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LinearRegression, RidgeCV
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline


        def generate():
            rng = np.random.default_rng(0)
            n = 200
            L = rng.normal(size=(n, 5))                                       # 5 latent factors
            X = L @ rng.normal(size=(5, 40)) + rng.normal(0, 1.0, (n, 40))    # 40 correlated features
            y = L @ np.array([3, -2, 1, 0.5, 0]) + rng.normal(0, 1.5, n)
            return train_test_split(X, y, test_size=0.5, random_state=0)


        X_tr, X_te, y_tr, y_te = generate()
        cv = KFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        cv_pcr = pd.Series(dtype=float)
        pcr_k = None
        """),
        C("""
        cv_pls = pd.Series(dtype=float)
        pls_n = None
        pls_has_fewer_components = None
        """),
        C("""
        test_r2 = {}
        test_r2
        """),
    ],
    checks="""
        check(list(cv_pcr.index) == list(range(1, 21)) and cv_pcr.idxmax() >= 4 and cv_pcr.max() > 0.7, "PCR curve over components 1-20: maximum above 0.7 with at least 4 components", "PCA(k) + LinearRegression")
        check(cv_pcr[1] < 0.3 and 4 <= pcr_k <= 10, "with a single component the R2 is low (below 0.3); the chosen k is between 4 and 10", "smallest k within 0.01 of the maximum")
        check(list(cv_pls.index) == list(range(1, 11)) and pls_n <= 4 and pls_has_fewer_components is True, "PLS: at most 4 components, fewer than PCR", "PLS chooses the directions by looking at y")
        check(set(test_r2) == {"OLS", "Ridge", "PCR", "PLS"} and test_r2["OLS"] < 0.7 and test_r2["Ridge"] > 0.78 and test_r2["PCR"] > 0.78 and test_r2["PLS"] > 0.75, "test R2: OLS below 0.7; Ridge and PCR above 0.78; PLS above 0.75", "final models trained on the training set")
        check(test_r2["PCR"] > test_r2["OLS"] + 0.1, "PCR beats OLS by over 0.1 of R2", "many correlated features and few examples: OLS overfits")
    """,
    hint="""
        `cross_val_score(make_pipeline(PCA(k), LinearRegression()), X_tr, y_tr, cv=cv, scoring="r2").mean()`. PLS: `cross_val_score(PLSRegression(k), X_tr, y_tr, cv=cv, scoring="r2")`. Final models: `.fit(X_tr, y_tr).score(X_te, y_te)`. Rule: `min(k for k in series.index if series[k] >= series.max() - 0.01)`.
    """,
    solution=[
        C("""
        cv_pcr = pd.Series({k: cross_val_score(make_pipeline(PCA(k), LinearRegression()), X_tr, y_tr, cv=cv, scoring="r2").mean() for k in range(1, 21)})
        pcr_k = min(k for k in cv_pcr.index if cv_pcr[k] >= cv_pcr.max() - 0.01)
        print(cv_pcr.round(3).to_string())
        print("k of the PCR (smallest within 0.01 of the maximum):", pcr_k)
        """),
        C("""
        cv_pls = pd.Series({k: cross_val_score(PLSRegression(k), X_tr, y_tr, cv=cv, scoring="r2").mean() for k in range(1, 11)})
        pls_n = min(k for k in cv_pls.index if cv_pls[k] >= cv_pls.max() - 0.01)
        pls_has_fewer_components = bool(pls_n < pcr_k)
        print(cv_pls.round(3).to_string())
        print("PLS components:", pls_n, "| fewer than PCR:", pls_has_fewer_components)
        """),
        C("""
        test_r2 = {
            "OLS": LinearRegression().fit(X_tr, y_tr).score(X_te, y_te),
            "Ridge": RidgeCV(alphas=np.logspace(-2, 4, 20)).fit(X_tr, y_tr).score(X_te, y_te),
            "PCR": make_pipeline(PCA(pcr_k), LinearRegression()).fit(X_tr, y_tr).score(X_te, y_te),
            "PLS": PLSRegression(pls_n).fit(X_tr, y_tr).score(X_te, y_te),
        }
        pd.Series(test_r2).round(3)
        """),
    ],
    learned="""
        - **OLS suffers from collinearity with few examples:** with 40 features and 100 training examples the test R² is 0.64. The directions with small variance (in the noise) have enormous coefficients that overfit.
        - **PCR removes those directions:** keeping only 5 components (the 5 latent factors) the R² goes up to 0.80, on par with Ridge (0.81). With a single component the R² is almost null: the first factor has the largest variance but is not the most useful for `y`.
        - **PLS finds the directions by looking at `y`:** with only 2 components it gets 0.78, very close to PCR with 5. In PCR the components are chosen by variance (unsupervised), in PLS by covariance with `y`: more efficient when the signal sits in a few non-dominant directions.
        - Ridge, PCR and PLS solve the same problem (collinearity) with different logics: Ridge **shrinks** all the directions in inverse proportion to their variance, PCR **truncates** the low-variance directions, PLS **chooses** the ones useful for the target. In practice Ridge is the first to try: a single hyperparameter, no discrete choices.
    """,
    pitfall="""
        Assuming that PCR is "the solution to collinearity" without checking that the signal does not sit in the discarded components: C1 shows a case where PCR with few components fails completely.
    """,
    interview=(
        "You have many correlated variables in a regression. What options do you have?",
        """
        Ridge (shrinks the coefficients, a single hyperparameter), Lasso or elastic net if I want to select, PCR (regression on the first components) if I think the signal sits in the high-variance directions, PLS if I want target-oriented components.
        I compare them in cross-validation with the same split; Ridge is usually the starting point because it is stable and requires no discrete choices.
        """),
)

EXERCISES = [B1, B2, B3]
