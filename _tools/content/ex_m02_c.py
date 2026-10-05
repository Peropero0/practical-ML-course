"""Module 2 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="scaling_and_knn_distances", kind="C", level=2, minutes=35, module=MODULE,
    title="Features in euros sink the KNN: where the distance goes",
    goal="""
        Build a dataset in which twenty irrelevant features, expressed in large units, alone decide the distances between the points, and
        measure by how much the KNN accuracy collapses and how much of the distance each group of features explains, with and without scaling.
    """,
    dataset="""
        Synthetic: `make_moons` (600 points, two moon-shaped classes, noise 0.25) provides two **informative** features on a scale
        of about 1. We add twenty more, `junk`, of pure Gaussian noise multiplied by 1000 (like amounts in euros). `X` thus has 22
        columns: the first two are informative, the other twenty are not. The target is the class of the moons.
    """,
    purpose="""
        See where the Euclidean distance goes: it is a sum of squares per feature, and the term with the largest unit
        dominates it. And understand that scaling **fixes the scale, not the relevance**: a useless feature stays useless.
    """,
    task=[
        "With `KNeighborsClassifier(n_neighbors=5)` and the cross-validation `cv` compute `acc_raw` (mean accuracy on the "
        "unscaled data, 22 columns).",
        "Compute `acc_scaled`: the same thing inside `make_pipeline(StandardScaler(), KNeighborsClassifier(5))`.",
        "Compute `acc_without_junk`: scaled pipeline on the two informative columns only (`X[:, :2]`).",
        "Write `distance_share(X, n_pairs=3000, seed=0)`: draw `n_pairs` random pairs of rows (`rng.integers`) and return, for each "
        "column, the fraction of the **total squared distance** due to that column (sum over the pairs of the square of the differences "
        "of that column, divided by the sum over all the columns).",
        "Compute `raw_share = distance_share(X)` and `scaled_share = distance_share(StandardScaler().fit_transform(X))`, then the total share of the two informative columns in `info_raw` and `info_scaled`.",
    ],
    setup="""
        from sklearn.datasets import make_moons
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rng = np.random.default_rng(0)
        X_info, y = make_moons(n_samples=600, noise=0.25, random_state=0)      # two informative features, scale about 1
        junk = rng.normal(size=(600, 20)) * 1000                               # twenty irrelevant features, in "euros"
        X = np.column_stack([X_info, junk])
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        acc_raw = ...
        acc_scaled = ...
        acc_without_junk = ...
        print(f"raw {acc_raw:.3f} | scaled {acc_scaled:.3f} | without the useless features {acc_without_junk:.3f}")
        """),
        C("""
        def distance_share(X, n_pairs=3000, seed=0):
            ...

        raw_share = ...
        scaled_share = ...
        info_raw = ...
        info_scaled = ...
        """),
    ],
    checks="""
        check(acc_raw < 0.62, "without scaling the KNN is close to chance (accuracy below 0.62)", "with the feature in 'euros' the KNN should collapse")
        check(acc_scaled > acc_raw + 0.05, "scaling recovers part of the loss", "acc_scaled must exceed acc_raw")
        check(acc_without_junk > acc_scaled + 0.15,
              "but removing the useless features helps much more: scaling does not fix relevance",
              "without the junk columns the accuracy must clearly exceed the scaled one")
        check(len(raw_share) == 22 and abs(raw_share.sum() - 1) < 1e-9, "the 22 shares sum to 1", "distance_share returns one fraction per column, and the sum is 1")
        check(info_raw < 1e-5, "without scaling the two informative columns explain less than 0.001% of the distance", "info_raw = raw_share[:2].sum()")
        check(0.05 < info_scaled < 0.2, "after scaling the informative columns weigh about 2/22, that is less than 20%",
              "after scaling every column weighs about 1/22: the two informative ones about 9%")
    """,
    hint="""
        For the pairs: `i, j = rng.integers(0, len(X), n_pairs), rng.integers(0, len(X), n_pairs)`; then `d2 = (X[i] - X[j]) ** 2`
        (n_pairs x columns matrix), `d2.sum(axis=0) / d2.sum()`. For the cross-validation: `cross_val_score(model, X, y, cv=cv).mean()`.
    """,
    solution=[
        C("""
        acc_raw = cross_val_score(KNeighborsClassifier(5), X, y, cv=cv).mean()
        acc_scaled = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X, y, cv=cv).mean()
        acc_without_junk = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), X[:, :2], y, cv=cv).mean()
        print(f"raw {acc_raw:.3f} | scaled {acc_scaled:.3f} | without the useless features {acc_without_junk:.3f}")
        """),
        C("""
        def distance_share(X, n_pairs=3000, seed=0):
            r = np.random.default_rng(seed)
            i, j = r.integers(0, len(X), n_pairs), r.integers(0, len(X), n_pairs)
            d2 = (X[i] - X[j]) ** 2                      # contribution of every column to the squared distance
            return d2.sum(axis=0) / d2.sum()


        raw_share = distance_share(X)
        scaled_share = distance_share(StandardScaler().fit_transform(X))
        info_raw, info_scaled = raw_share[:2].sum(), scaled_share[:2].sum()
        print(f"share of the distance of the 2 informative columns: raw {info_raw:.2e} | scaled {info_scaled:.3f}")
        """),
    ],
    learned="""
        - Without scaling the twenty `junk` columns explain practically 100% of the distance: the "neighbors" of a point are the points with similar
          `junk`, that is random ones. The KNN drops to about 0.5, that is chance level.
        - With scaling every column weighs about one twenty-second and the accuracy goes back up, but it does **not** reach the one obtained by removing
          `junk`: the twenty useless columns keep pushing the neighbors at random for about 90% of the distance.
        - Scaling solves a problem of **units**, not of **relevance**. For the KNN you also need feature selection or per-feature
          weights: it is one of the reasons why the KNN suffers in high dimensions.
    """,
    pitfall="""
        Scaling and considering the job done. Irrelevant features have the same weight as relevant ones in a Euclidean
        distance; in a real dataset with dozens of columns, most of them useless, the scaled KNN can still
        do badly.
    """,
    interview=(
        "I standardized the features but the KNN still does badly. What do I check?",
        """
        That the irrelevant features are not diluting the distance: with many barely informative columns the neighbors become almost
        equidistant (curse of dimensionality). I try feature selection, a dimensionality reduction (PCA),
        the metric (Manhattan is less sensitive) or weighting the features. And I check that k is chosen in cross-validation.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="ordinal_on_nominal_variable", kind="C", level=2, minutes=40, module=MODULE,
    title="Encoding a nominal with integers: the invented order",
    goal="""
        Show that a linear model with a nominal variable encoded as integers depends on the **arbitrary order** of the
        categories, while the one-hot does not, and that a tree copes because it isolates the categories with its splits.
    """,
    dataset="""
        Synthetic: 1200 rows with a nominal variable `zone` (8 levels, A-H) that shifts the target by `+6, −6, +5, −5, +7, −7, +4, −4`
        (a **non-monotonic** effect in alphabetical order: high and low zones alternate), a numeric feature `x` (`+2·x`) and noise. The function `evaluate(encoder,
        model)` builds the pipeline and returns the mean cross-validated R².
    """,
    purpose="""
        Get a hands-on feel for why "encode with 0, 1, 2, ..." is not neutral: it hands a linear model a monotonic relationship
        between code and target that does not exist. You will see that the R² changes a lot **just by changing the order** of the categories.
    """,
    task=[
        "Compute `r2_ord_lin` (`OrdinalEncoder()` with alphabetical order + `LinearRegression`), `r2_ohe_lin` (`OneHotEncoder(sparse=False)` "
        "+ `LinearRegression`), `r2_ord_tree` and `r2_ohe_tree` (`DecisionTreeRegressor(max_depth=8, random_state=0)` in place of the regression).",
        "For 200 random permutations of the order of the categories (`np.random.default_rng(s).permutation(categories)` for `s = 0..199`) "
        "compute the R² of the linear regression with `OrdinalEncoder(categories=[order])` and save it in `r2_perm`.",
        "Compute `spread = r2_perm.max() − r2_perm.min()` and `r2_perm_max`.",
        "Choose in `answer` (\"a\", \"b\" or \"c\") the correct explanation:\n"
        "   - **a**: the OrdinalEncoder assigns codes at random and the result changes at every run;\n"
        "   - **b**: a linear model sees the code as a quantity: it imposes that the effect grows (or falls) linearly along the chosen "
        "order, and with a non-monotonic effect no order works well;\n"
        "   - **c**: linear regression does not accept categorical variables.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from sklearn.tree import DecisionTreeRegressor

        rng = np.random.default_rng(0)
        n = 1200
        effect = {"A": 6, "B": -6, "C": 5, "D": -5, "E": 7, "F": -7, "G": 4, "H": -4}      # high and low alternate: non-monotonic
        df = pd.DataFrame({"zone": rng.choice(list(effect), n), "x": rng.normal(size=n)})
        y = df.zone.map(effect).to_numpy() + 2 * df.x.to_numpy() + rng.normal(size=n)
        categories = np.array(sorted(effect))


        def evaluate(encoder, model):
            pipe = make_pipeline(ColumnTransformer([("zone", encoder, ["zone"])], remainder="passthrough"), model)
            return cross_val_score(pipe, df[["zone", "x"]], y, cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    """,
    starter=[
        C("""
        r2_ord_lin = ...
        r2_ohe_lin = ...
        r2_ord_tree = ...
        r2_ohe_tree = ...
        print(f"linear:  ordinal {r2_ord_lin:.3f} | one-hot {r2_ohe_lin:.3f}")
        print(f"tree  :  ordinal {r2_ord_tree:.3f} | one-hot {r2_ohe_tree:.3f}")
        """),
        C("""
        r2_perm = []
        for s in range(200):
            ...
        r2_perm = np.array(r2_perm)
        spread = ...
        r2_perm_max = ...
        """),
        C("""
        answer = ...
        """),
    ],
    checks="""
        check(r2_ohe_lin > 0.85, "the one-hot with the linear model recovers the effect of the zones (R2 > 0.85)", "the linear one-hot should have a high R2")
        check(r2_ord_lin < 0.3, "with integer codes the linear model fails (R2 < 0.3)", "with alphabetical order the effect is not monotonic")
        check(r2_ohe_lin > r2_ord_lin + 0.5, "huge difference between the two encodings, on the same data", "r2_ohe_lin must exceed r2_ord_lin by more than 0.5")
        check(r2_ord_tree > 0.8, "the tree copes even with integer codes (it isolates the categories with its splits)", "a depth 8 tree should have R2 > 0.8")
        check(len(r2_perm) == 200 and spread > 0.15, "changing ONLY the order of the categories the R2 varies a lot", "the spread between the permutations should exceed 0.15")
        check(r2_ohe_lin > r2_perm_max + 0.05 and np.median(r2_perm) < 0.3,
              "the one-hot beats even the best random order, and a typical order is terrible (median below 0.3)",
              "the one-hot must exceed the maximum of the permutations, whose median is low")
        check(answer == "b", "correct explanation", "reread: what does a linear model assume about the numeric code?")
    """,
    hint="""
        `evaluate(OrdinalEncoder(), LinearRegression())`. For a permutation: `order = np.random.default_rng(s).permutation(categories)` and
        `evaluate(OrdinalEncoder(categories=[order]), LinearRegression())`.
    """,
    solution=[
        C("""
        r2_ord_lin = evaluate(OrdinalEncoder(), LinearRegression())
        r2_ohe_lin = evaluate(OneHotEncoder(sparse=False), LinearRegression())
        tree = lambda: DecisionTreeRegressor(max_depth=8, random_state=0)
        r2_ord_tree = evaluate(OrdinalEncoder(), tree())
        r2_ohe_tree = evaluate(OneHotEncoder(sparse=False), tree())
        print(f"linear:  ordinal {r2_ord_lin:.3f} | one-hot {r2_ohe_lin:.3f}")
        print(f"tree  :  ordinal {r2_ord_tree:.3f} | one-hot {r2_ohe_tree:.3f}")
        """),
        C("""
        r2_perm = []
        for s in range(200):
            order = np.random.default_rng(s).permutation(categories)
            r2_perm.append(evaluate(OrdinalEncoder(categories=[order]), LinearRegression()))
        r2_perm = np.array(r2_perm)
        spread = r2_perm.max() - r2_perm.min()
        r2_perm_max = r2_perm.max()
        print(f"R2 with random order: minimum {r2_perm.min():.3f} | median {np.median(r2_perm):.3f} | maximum {r2_perm_max:.3f}")
        """),
        C("""
        answer = "b"
        """),
    ],
    learned="""
        - With integer codes the linear model estimates **a single slope** per zone: the effect is forced to grow (or fall)
          linearly along the chosen order. With a non-monotonic effect the R² is close to zero, with the one-hot it is about 0.9.
        - The result depends on the **order** of the categories, that is on an arbitrary choice (here the alphabetical order). Among the 200
          permutations the R² goes from about 0.1 to about 0.88 (median 0.17): the best order puts the "high" zones together and the "low" ones, but you
          know it only by looking at the target, and anyway it does not reach the one-hot.
        - A tree uses the splits to isolate the categories: with enough depth it does not see the difference between the two
          encodings. It is the reason why trees and boosting are tolerant of ordinal encoding (it is not a rule: with
          many categories you need many splits).
    """,
    pitfall="""
        Using `LabelEncoder` or `OrdinalEncoder` "because it is more convenient" on a nominal and then passing the data to a linear model, a
        KNN or a network. No error, no warning: just a worse model, and nobody knows why.
    """,
    interview=(
        "When is the OrdinalEncoder acceptable on a nominal variable?",
        """
        With tree models (trees, forests, boosting): the model works with thresholds and can isolate a category with two splits,
        so it is not bound to the order. With few categories and enough data it works; with many categories you need many splits and
        a better encoding is preferable (target encoding, native handling of categoricals). With linear models, KNN, SVM or networks
        it is wrong: you need the one-hot or another encoding.
        """),
)

# ---------------------------------------------------------------------------------------- C3

C3 = dict(
    id="C3", slug="mean_imputation_distorts", kind="C", level=3, minutes=45, module=MODULE,
    title="Imputation with the mean erases variance and correlation",
    goal="""
        Measure, as the share of missing values grows, how imputation with the mean, deterministic regression and
        regression with noise alter the variance, correlation and regression coefficient of a variable.
    """,
    dataset="""
        Synthetic: 4000 rows with `x1 ~ N(0,1)` and `x2 = 0.8·x1 + 0.6·e` (true correlation 0.8, variance 1), and `y = x1 + x2 + noise`.
        The true coefficient of `x2` in the regression of `y` on `x1` and `x2` is **1**. We make a share `p` of `x2` missing, at random (MCAR).
    """,
    purpose="""
        Understand that imputation is not neutral with respect to the statistics you care about: the mean **shortens the variance**
        by (1 − p) and **weakens the correlation**; deterministic regression **inflates** it; only imputation with noise preserves
        both.
    """,
    task=[
        "Write `imputations(p, seed=0)`: it makes a fraction `p` of `x2` missing (mask `rng.random(n) < p`) and returns a dictionary with "
        "three imputed versions of `x2`: `\"mean\"` (`SimpleImputer`), `\"regression\"` (`IterativeImputer(random_state=0)` on `[x1, x2]`) and "
        "`\"with_noise\"` (`IterativeImputer(sample_posterior=True, random_state=0)`).",
        "Write `metrics(x2_imp)`: it returns a dictionary with `var_ratio` (variance of `x2_imp` divided by the variance of the true `x2`), `corr` "
        "(correlation between `x1` and `x2_imp`) and `coef_x2` (coefficient of `x2_imp` in `LinearRegression` of `y` on `[x1, x2_imp]`).",
        "For `p` in `[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]` and for each method, fill the DataFrame `results` (index `(p, method)`, columns "
        "`var_ratio`, `corr`, `coef_x2`).",
        "Compute `theoretical_var = 1 − p` and `theoretical_corr = sqrt(1 − p)·0.8` for the mean imputation only: they should coincide with "
        "what you measured.",
    ],
    setup="""
        from sklearn.experimental import enable_iterative_imputer          # noqa: F401
        from sklearn.impute import IterativeImputer, SimpleImputer
        from sklearn.linear_model import LinearRegression

        RHO, n = 0.8, 4000
        rng = np.random.default_rng(0)
        x1 = rng.normal(size=n)
        x2 = RHO * x1 + np.sqrt(1 - RHO ** 2) * rng.normal(size=n)
        y = x1 + x2 + rng.normal(size=n)
        PS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    """,
    starter=[
        C("""
        def imputations(p, seed=0):
            ...


        def metrics(x2_imp):
            ...
        """),
        C("""
        rows = {}
        for p in PS:
            for method, x2_imp in imputations(p).items():
                rows[(p, method)] = ...
        results = pd.DataFrame(rows).T
        results.index.names = ["p", "method"]
        results.round(3)
        """),
        C("""
        theoretical_var = ...
        theoretical_corr = ...
        """),
    ],
    checks="""
        check(set(imputations(0.3)) == {"mean", "regression", "with_noise"}, "three imputation methods", "keys: mean, regression, with_noise")
        check(len(results) == 18 and set(results.columns) == {"var_ratio", "corr", "coef_x2"}, "18 rows and the three metrics", "6 values of p x 3 methods")
        mean_rows = results.xs("mean", level="method")
        check(np.all(np.abs(mean_rows.var_ratio.to_numpy() - (1 - np.array(PS))) < 0.06),
              "with the mean the variance drops by about a factor (1 - p)", "var_ratio of the mean must follow 1 - p")
        check(np.all(np.abs(mean_rows["corr"].to_numpy() - np.sqrt(1 - np.array(PS)) * RHO) < 0.06),
              "with the mean the correlation drops by about sqrt(1 - p)", "corr of the mean must follow sqrt(1 - p) * 0.8")
        check(close(theoretical_var, 1 - np.array(PS)) and close(theoretical_corr, np.sqrt(1 - np.array(PS)) * RHO),
              "correct theoretical formulas", "theoretical_var = 1 - p, theoretical_corr = sqrt(1 - p) * 0.8")
        check(mean_rows.coef_x2.loc[0.6] < mean_rows.coef_x2.loc[0.1] - 0.3, "the coefficient of x2 is attenuated by the mean, more and more as p grows",
              "coef_x2 with the mean must drop sharply from p=0.1 to p=0.6")
        reg = results.xs("regression", level="method")
        check(reg["corr"].loc[0.5] > RHO + 0.05, "deterministic regression INFLATES the correlation (the imputed values are an exact function of x1)",
              "with p=0.5 the correlation after deterministic regression should exceed 0.85")
        noisy = results.xs("with_noise", level="method")
        check(abs(noisy["corr"].loc[0.5] - RHO) < 0.06 and abs(noisy.var_ratio.loc[0.5] - 1) < 0.1,
              "with noise correlation and variance stay the true ones", "with p=0.5 with_noise should have corr about 0.8 and var_ratio about 1")
    """,
    hint="""
        To mask: `mask = np.random.default_rng(seed).random(n) < p; x2_miss = np.where(mask, np.nan, x2)`. To impute on two columns:
        `IterativeImputer(...).fit_transform(np.column_stack([x1, x2_miss]))[:, 1]`. For the mean `SimpleImputer(strategy="mean")`.
        Coefficient: `LinearRegression().fit(np.column_stack([x1, x2_imp]), y).coef_[1]`.
    """,
    solution=[
        C("""
        def imputations(p, seed=0):
            mask = np.random.default_rng(seed).random(n) < p
            x2_miss = np.where(mask, np.nan, x2)
            X_miss = np.column_stack([x1, x2_miss])
            return {
                "mean": SimpleImputer(strategy="mean").fit_transform(X_miss)[:, 1],
                "regression": IterativeImputer(random_state=0).fit_transform(X_miss)[:, 1],
                "with_noise": IterativeImputer(sample_posterior=True, random_state=0).fit_transform(X_miss)[:, 1],
            }


        def metrics(x2_imp):
            return {"var_ratio": x2_imp.var() / x2.var(),
                    "corr": np.corrcoef(x1, x2_imp)[0, 1],
                    "coef_x2": LinearRegression().fit(np.column_stack([x1, x2_imp]), y).coef_[1]}
        """),
        C("""
        rows = {}
        for p in PS:
            for method, x2_imp in imputations(p).items():
                rows[(p, method)] = metrics(x2_imp)
        results = pd.DataFrame(rows).T
        results.index.names = ["p", "method"]
        results.round(3)
        """),
        C("""
        theoretical_var = 1 - np.array(PS)
        theoretical_corr = np.sqrt(1 - np.array(PS)) * RHO
        print("p                :", PS)
        print("expected var     :", theoretical_var.round(2), "| measured:", results.xs("mean", level="method").var_ratio.round(2).to_numpy())
        print("expected corr    :", theoretical_corr.round(2), "| measured:", results.xs("mean", level="method")["corr"].round(2).to_numpy())
        """),
    ],
    learned="""
        - **Mean**: the imputed values are all equal, so they do not vary together with anything. The variance drops by a factor
          (1 − p) and the correlation with the other variables by sqrt(1 − p): with p = 0.5, from 0.8 to about 0.57. The regression coefficient
          of that variable gets **attenuated** toward zero.
        - **Deterministic regression**: every imputed value is an exact function of the other columns, so the relationship looks
          stronger than the true one: the correlation goes above 0.8 and the variance stays too low.
        - **Regression with noise** (`sample_posterior=True`): it adds the residual noise to the predictions and preserves both variance and
          correlation. It is the idea behind multiple imputation. Careful: the coefficient of `x2` toward `y` is still
          attenuated (in the table it drops below 0.5 with p = 0.5), because the imputation here does not know `y`: the added values are
          noise with respect to `y`. To not lose that relationship the imputation model must also include the target
          (computed only on the training set).
        - For predicting (not for estimating relationships) the difference is often small; for **inference** (coefficients, correlations) the
          mean is a method to avoid.
    """,
    pitfall="""
        Choosing the imputer by looking only at the error on the imputed values (RMSE, as in B2). An imputer with the lowest RMSE, like
        deterministic regression, produces a distribution with the wrong variability: the right criterion depends on the use
        you will make of the data.
    """,
    interview=(
        "Why is imputation with the mean criticized in statistical analyses, but often used in machine learning?",
        """
        Because in statistical analysis what matters are variances, correlations and coefficients, and the mean distorts them systematically (it shortens the
        variance by a factor 1 − p and weakens the correlations). In machine learning the goal is predictive performance: with
        few missing values and a flexible model the distortion costs little, and the mean is simple, fast and robust. In both cases
        I add the missingness indicator and check the effect with cross-validation.
        """),
)

EXERCISES = [C1, C2, C3]
