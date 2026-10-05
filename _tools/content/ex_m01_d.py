"""Module 1 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_01_linear_regression"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_the_coefficients", kind="D", level=2, minutes=35, module=MODULE,
    title="Reading the coefficients: units, standardization and the logarithm",
    goal="""
        Answer questions about the effect of each variable in a salary model, showing that the largest coefficient
        is not the most important variable, and translating the coefficients of a log-linear model into
        percentages.
    """,
    dataset="""
        `ds.salaries()`: 600 employees with `years_experience`, `weekly_hours`, `big_city` (0/1), `degree` (0/1) and the
        gross annual `salary` in euros. The generating process is hidden: the logarithm of the salary is linear in the
        features.
    """,
    purpose="""
        Learn to **read** a model, not just to train it: what a coefficient means when the features have
        different units, and what it means when the target is a logarithm. It is the kind of question you get in interviews and
        in model reviews.
    """,
    task=[
        "Train `raw = LinearRegression()` on `salary` with the features as they are. Save `raw_coefs` (a `Series` with "
        "the names as index) and `raw_most_influential`: the name of the variable with the largest coefficient **in absolute value**.",
        "Standardize the features (mean 0, standard deviation 1), retrain and save `std_coefs` (euros per one standard deviation "
        "of change in the feature) and `std_most_influential`.",
        "Train `log_model = LinearRegression()` on `np.log(salary)` with the original features. Save `log_coefs` and "
        "the exact percentage effect `effect_pct = (exp(coef) − 1) · 100` as a `Series`. Extract `degree_effect_pct` and "
        "`year_effect_pct`.",
        "Compare two identical people (10 years of experience, 40 hours) except that `profile_a` lives in a big city and has "
        "a degree and `profile_b` does not. Compute `salary_ratio` as the ratio between the predicted salaries (with `np.exp` of the "
        "predictions of the log model) and `ratio_from_coefs` as `exp(coef_city + coef_degree)`.",
        "Assign to `true_false` a dictionary `{\"a\": ..., \"b\": ..., \"c\": ...}` with `True` or `False` for the three statements:\n"
        "   - **a**: the largest coefficient in absolute value of the model on the raw target always indicates the most important variable;\n"
        "   - **b**: in the model on the logarithm, every extra year of experience raises the salary by about 3%;\n"
        "   - **c**: in the model on the logarithm, the gain in euros due to the degree is the same for someone who earns 20,000 euros and for "
        "someone who earns 60,000.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        df = ds.salaries()
        X = df.drop(columns="salary")
        y = df["salary"]
        profile_a = pd.DataFrame({"years_experience": [10.0], "weekly_hours": [40], "big_city": [1], "degree": [1]})
        profile_b = pd.DataFrame({"years_experience": [10.0], "weekly_hours": [40], "big_city": [0], "degree": [0]})
        df.describe().round(1)
    """,
    starter=[
        C("""
        raw = ...
        raw_coefs = ...
        raw_most_influential = ...
        raw_coefs.round(1)
        """),
        C("""
        X_std = ...
        std_coefs = ...
        std_most_influential = ...
        std_coefs.round(0)
        """),
        C("""
        log_model = ...
        log_coefs = ...
        effect_pct = ...
        degree_effect_pct = ...
        year_effect_pct = ...
        effect_pct.round(2)
        """),
        C("""
        salary_ratio = ...
        ratio_from_coefs = ...
        """),
        C("""
        true_false = {"a": ..., "b": ..., "c": ...}
        """),
    ],
    checks="""
        check(raw_most_influential == raw_coefs.abs().idxmax(), "variable with the largest raw coefficient identified",
              "use raw_coefs.abs().idxmax()")
        check(std_most_influential == std_coefs.abs().idxmax() and std_most_influential == "years_experience",
              "after standardization the most influential variable is a different one",
              "retrain on standardized features (StandardScaler) and look for the largest coefficient in absolute value")
        check(raw_most_influential != std_most_influential,
              "the largest coefficient depends on the unit of measure: the two answers are different",
              "the two 'most influential' variables should be different")
        check(7 < degree_effect_pct < 12, "plausible percentage effect of the degree",
              "degree_effect_pct should be about 9-10%: did you use np.exp(coef) - 1?")
        check(2.5 < year_effect_pct < 3.6, "plausible percentage effect of one year of experience",
              "year_effect_pct should be about 3%")
        check(close(salary_ratio, ratio_from_coefs), "the effect in the log model is multiplicative",
              "the ratio between the predicted salaries must match exp(coef_city + coef_degree)")
        check(true_false == {"a": False, "b": True, "c": False}, "statements judged correctly",
              "reread the meaning of the coefficients: they depend on the unit of measure, and in the log model the effects are multiplicative")
    """,
    hint="""
        `pd.Series(model.coef_, index=X.columns)`. To standardize while keeping the names: `pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)`.
        In the log model, the prediction in euros is `np.exp(log_model.predict(profile_a))[0]`.
    """,
    solution=[
        C("""
        raw = LinearRegression().fit(X, y)
        raw_coefs = pd.Series(raw.coef_, index=X.columns)
        raw_most_influential = raw_coefs.abs().idxmax()
        print("euros per unit of each feature:")
        print(raw_coefs.round(1).to_string())
        print("-> the largest coefficient:", raw_most_influential)
        """),
        M("""
        `big_city` wins because its "step" is a jump from 0 to 1: it is an effect of a few thousand euros for a whole
        unit. `years_experience` is worth little per year, but the years vary over a range of 30 units. Comparing the
        coefficients means comparing the effect of **one unit of each variable**: units that are not comparable.
        """),
        C("""
        X_std = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        std_coefs = pd.Series(LinearRegression().fit(X_std, y).coef_, index=X.columns)
        std_most_influential = std_coefs.abs().idxmax()
        print("euros per standard deviation of each feature:")
        print(std_coefs.round(0).to_string())
        print("-> the largest coefficient:", std_most_influential)
        """),
        M("""
        The model on the logarithm of the salary has coefficients that are **changes of the logarithm**: small numbers that
        must be read as relative changes. To avoid mistakes you exponentiate: `exp(β) − 1` is the exact percentage
        change, and for small β it is almost equal to β.
        """),
        C("""
        log_model = LinearRegression().fit(X, np.log(y))
        log_coefs = pd.Series(log_model.coef_, index=X.columns)
        effect_pct = (np.exp(log_coefs) - 1) * 100
        degree_effect_pct = effect_pct["degree"]
        year_effect_pct = effect_pct["years_experience"]
        print(pd.DataFrame({"coef (log)": log_coefs.round(4), "effect %": effect_pct.round(2)}).to_string())
        """),
        C("""
        salary_a = np.exp(log_model.predict(profile_a))[0]
        salary_b = np.exp(log_model.predict(profile_b))[0]
        salary_ratio = salary_a / salary_b
        ratio_from_coefs = np.exp(log_coefs["big_city"] + log_coefs["degree"])
        print(f"salary A {salary_a:,.0f} | salary B {salary_b:,.0f} | ratio {salary_ratio:.3f} = exp(sum of the coefs) {ratio_from_coefs:.3f}")
        print(f"difference in euros: {salary_a - salary_b:,.0f}")
        """),
        C("""
        true_false = {"a": False, "b": True, "c": False}
        """),
    ],
    learned="""
        - In the model on the raw target the coefficient of `big_city` (about 5,300 euros) is the largest in absolute value,
          but after standardization the most influential variable is `years_experience` (about 8,900 euros per standard
          deviation). The coefficients measure "per unit of each feature": to compare importance you need comparable
          units (standardized coefficients, or better still the effect of a realistic change).
        - In the log model the effects are **multiplicative**: degree and big city combine with a product (1.09 · 1.16),
          not with a sum. And the effect in euros depends on the starting level: +9% on 20,000 is 1,800 euros, +9% on 60,000
          is 5,400.
        - A coefficient of 0.0298 in the log model is ≈ 3%: the difference between 0.0298 and `exp(0.0298) - 1 = 3.03%` becomes
          relevant only for large coefficients (β = 0.5 is +65%, not +50%).
    """,
    pitfall="""
        Saying "the variable with the largest coefficient is the most important". And its twin: standardizing a binary variable
        and reading the result as the "effect of one standard deviation" makes little sense (for a 0/1 the standard
        deviation depends on the proportion of ones). For binary variables it is preferable to read directly the effect of the step
        from 0 to 1.
    """,
    interview=(
        "The coefficient of 'years of experience' in your model is 1,051 euros, the one of 'big city' is 5,348. The boss says "
        "'the city matters five times more than experience'. What do you answer?",
        """
        That the two numbers are not comparable: 1,051 euros is the effect of **one year** of experience, 5,348 that of the jump
        from "not a big city" to "big city". To compare importance I would use standardized coefficients or a concrete question:
        how much does the predicted salary change between an employee at the 10th and one at the 90th percentile of experience,
        versus changing city? In this dataset experience, over a realistic range, weighs more than the city.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="residual_diagnostics_four_mysteries", kind="D", level=3, minutes=45, module=MODULE,
    title="Residual diagnostics: four mysteries",
    goal="""
        Implement three diagnostic tests on the residuals (Durbin-Watson, curvature, Breusch-Pagan) and use them to
        diagnose four datasets: one respects the assumptions, the other three violate one each.
    """,
    dataset="""
        `ds.residual_mysteries()` returns a dictionary with four DataFrames `"A"`, `"B"`, `"C"`, `"D"`, each with a feature `x`
        and a target `y` (200 rows, in order of acquisition). How they were generated is hidden on purpose. The possible
        violations are: `"nonlinearity"`, `"heteroscedasticity"`, `"autocorrelation"`. One of the four is healthy (`"ok"`).
    """,
    purpose="""
        Move from "I look at the plot" to "I measure". Every pathology has a dedicated statistic and a different remedy, and in
        an automatic flow (a model-checking pipeline) plots are not enough.
    """,
    task=[
        "Write `residuals_of(df)`: OLS fit of `y` on `x` with `np.linalg.lstsq` and return the pair `(x, e)` with the residuals `e`.",
        "Write `durbin_watson(e)` = `Σ (e_t − e_{t−1})² / Σ e_t²`. It is about 2 without autocorrelation, and drops toward 0 with "
        "positive autocorrelation.",
        "Write `curvature_r2(x, e)`: R² of the regression of the residuals on `[1, x, x²]`. A high value means that the residuals "
        "have a curved structure left out of the model.",
        "Write `breusch_pagan(x, e)`: regress `e²` on `[1, x]`, take `stat = n · R²` and return `(stat, p_value)` with "
        "`p_value = scipy.stats.chi2.sf(stat, df=1)`. A small p-value rejects homoscedasticity.",
        "Build the DataFrame `table` with index `A`, `B`, `C`, `D` and columns `dw`, `curv_r2`, `p_bp`.",
        "Assign to `diagnosis` a dictionary with the violation of each dataset. Suggested rules: `dw < 1.2` → "
        "`\"autocorrelation\"`; `curv_r2 > 0.10` → `\"nonlinearity\"`; `p_bp < 0.01` → `\"heteroscedasticity\"`; otherwise `\"ok\"`.",
    ],
    setup="""
        from scipy import stats
        from common.plots import residual_panel

        mysteries = ds.residual_mysteries()
        {name: d.shape for name, d in mysteries.items()}
    """,
    starter=[
        C("""
        def residuals_of(df):
            ...

        def durbin_watson(e):
            ...

        def curvature_r2(x, e):
            ...

        def breusch_pagan(x, e):
            ...
        """),
        C("""
        rows = {}
        for name, d in mysteries.items():
            x, e = residuals_of(d)
            rows[name] = {"dw": ..., "curv_r2": ..., "p_bp": ...}
        table = pd.DataFrame(rows).T
        table
        """),
        C("""
        diagnosis = {"A": ..., "B": ..., "C": ..., "D": ...}
        """),
    ],
    checks="""
        check(close(durbin_watson(np.array([1.0, -1.0, 1.0, -1.0])), 3.0), "durbin_watson is correct on a known case",
              "durbin_watson([1,-1,1,-1]) must equal 12/4 = 3")
        rng_t = np.random.default_rng(0)
        x_t = rng_t.uniform(-3, 3, 300)
        check(curvature_r2(x_t, x_t ** 2 - (x_t ** 2).mean()) > 0.99, "curvature_r2 recognizes a perfect parabola",
              "regressing on [1, x, x^2] an exact parabola must give R2 = 1")
        x_h = rng_t.uniform(1, 10, 500)
        check(breusch_pagan(x_h, rng_t.normal(size=500) * x_h)[1] < 1e-3, "breusch_pagan recognizes growing variance",
              "with errors whose standard deviation grows with x the p-value must be tiny")
        check(breusch_pagan(x_h, rng_t.normal(size=500))[1] > 0.01, "breusch_pagan does not fire on homoscedastic errors",
              "with errors independent of x the p-value should not be small")
        check(list(table.index) == ["A", "B", "C", "D"] and list(table.columns) == ["dw", "curv_r2", "p_bp"],
              "the table has the required index and columns", "index A-D and columns dw, curv_r2, p_bp")
        check(diagnosis == {"A": "heteroscedasticity", "B": "ok", "C": "nonlinearity", "D": "autocorrelation"},
              "correct diagnosis for all four datasets",
              "look at which statistic is out of range in each row of `table`: at most one should fire")
    """,
    hint="""
        A generic linear regression with numpy: `beta = np.linalg.lstsq(B, z, rcond=None)[0]`, `R² = 1 - SSR/SST` with
        `SST = ((z - z.mean())**2).sum()`. For the curvature `B = np.column_stack([np.ones_like(x), x, x**2])` and `z = e`;
        for Breusch-Pagan `B = np.column_stack([np.ones_like(x), x])` and `z = e**2`.
    """,
    solution=[
        C("""
        def residuals_of(df):
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            A = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(A, y, rcond=None)[0]
            return x, y - A @ beta


        def _regression_r2(B, z):
            beta = np.linalg.lstsq(B, z, rcond=None)[0]
            return 1 - ((z - B @ beta) ** 2).sum() / ((z - z.mean()) ** 2).sum()


        def durbin_watson(e):
            return np.sum(np.diff(e) ** 2) / np.sum(e ** 2)


        def curvature_r2(x, e):
            return _regression_r2(np.column_stack([np.ones_like(x), x, x ** 2]), e)


        def breusch_pagan(x, e):
            r2 = _regression_r2(np.column_stack([np.ones_like(x), x]), e ** 2)
            stat = len(x) * r2
            return stat, stats.chi2.sf(stat, df=1)
        """),
        C("""
        rows = {}
        for name, d in mysteries.items():
            x, e = residuals_of(d)
            rows[name] = {"dw": durbin_watson(e), "curv_r2": curvature_r2(x, e), "p_bp": breusch_pagan(x, e)[1]}
        table = pd.DataFrame(rows).T
        pd.options.display.float_format = "{:.4g}".format
        table
        """),
        M("""
        Every row has exactly **one** statistic out of range: A has a tiny Breusch-Pagan p-value (the variance depends on
        x), C has a curvature R² close to 1 (the residuals are shaped like a parabola), D has a Durbin-Watson close to 0 (consecutive
        residuals look alike). B triggers nothing.
        """),
        C("""
        def classify(row):
            if row.dw < 1.2:
                return "autocorrelation"
            if row.curv_r2 > 0.10:
                return "nonlinearity"
            if row.p_bp < 0.01:
                return "heteroscedasticity"
            return "ok"

        diagnosis = {name: classify(row) for name, row in table.iterrows()}
        diagnosis
        """),
        C("""
        for name in ["A", "C", "D"]:
            x, e = residuals_of(mysteries[name])
            d = mysteries[name]
            residual_panel(d["y"], d["y"] - e, f"Dataset {name}: {diagnosis[name]}")
        """),
    ],
    learned="""
        - **Nonlinearity**: the residuals against x have a shape (here a parabola). Remedies: add polynomial terms or
          transform the features. The predictions are systematically wrong: it is the most serious case.
        - **Heteroscedasticity**: the spread of the residuals changes with x (funnel). The estimates remain unbiased, but the classical
          standard errors are wrong: you need robust errors, weighted least squares or a transformation of the target.
        - **Autocorrelation**: consecutive errors are not independent (typical of time data). Here too the coefficient
          estimates are reasonable but the uncertainty is underestimated, and the residuals have a "wave" pattern.
        - The three statistics are almost orthogonal: the right one fires, the others stay in range. In real data they often
          fire together and everything gets worse.
    """,
    pitfall="""
        Fixating on the Q-Q plot and on the normality of the residuals: in dataset A the normality test (Shapiro) rejects, in B it does not, but
        the problem of A is not the shape of the distribution, it is the non-constant variance. Normality is the least important
        of the assumptions for predicting. A test that fires does not tell you **which** assumption is violated: you need the right statistic.
    """,
    interview=(
        "You have a regression on daily sales data and the plot of the residuals against time shows regular waves. "
        "Which assumption is violated and what are the consequences?",
        """
        The independence of the errors: there is autocorrelation. The coefficient estimates remain reasonable but the standard errors
        are underestimated, so p-values and intervals are too optimistic. I would confirm with Durbin-Watson, and move to a
        model that handles the temporal dependence (AR errors, lagged features, seasonal components). For validation I would
        use TimeSeriesSplit, not a random K-fold.
        """),
)

# ---------------------------------------------------------------------------------------- D3

D3 = dict(
    id="D3", slug="anscombe_leverage_and_influence", kind="D", level=2, minutes=40, module=MODULE,
    title="Anscombe: four datasets, one line, three wrong models",
    goal="""
        Verify that the four datasets of Anscombe's quartet have the same summary statistics, and then use leverage and
        Cook's distance to understand which point moves the line in each one.
    """,
    dataset="""
        `ds.anscombe()`: four datasets of 11 points (`"I"`, `"II"`, `"III"`, `"IV"`), with columns `dataset`, `x`, `y`. Built in 1973
        to have almost identical slope, intercept, R² and residual variance, even though they are very different.
    """,
    purpose="""
        Understand **why** summary statistics are not enough and learn the two measures that expose the points that
        command the model. Dataset IV contains an instructive limit case: a point with leverage exactly equal to 1.
    """,
    task=[
        "For each of the four datasets do the OLS fit and build `summary` (index `I`..`IV`, columns `slope`, "
        "`intercept`, `r2`, `resid_std`, the latter with `ddof=2`).",
        "Write `leverages(x)` (diagonal of the hat matrix with intercept) and `cook(x, y)` (Cook's distance of each point, with the formula "
        "of lesson 1.3). If the leverage of a point exceeds `1 - 1e-9`, the formula gives 0/0: assign `np.inf` to that point.",
        "Assign to `influential_point` a dictionary `{name: index_of_the_point}` with the point at maximum Cook's distance in every dataset.",
        "In dataset III remove the influential point and recompute `slope_III_without` and `r2_III_without`. In IV remove the influential point and "
        "assign to `iv_x_constant` `True` if all the remaining x are equal.",
        "Assign to `regression_suitable` a dictionary `{name: True/False}`: for which datasets is an OLS line a good model "
        "**without interventions**?",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression

        anscombe = ds.anscombe()
        sets = {name: g[["x", "y"]].reset_index(drop=True) for name, g in anscombe.groupby("dataset")}
        sets["I"].head(3)
    """,
    starter=[
        C("""
        rows = {}
        for name, d in sets.items():
            ...
        summary = pd.DataFrame(rows).T
        summary.round(3)
        """),
        C("""
        def leverages(x):
            ...

        def cook(x, y):
            ...
        """),
        C("""
        influential_point = {}
        for name, d in sets.items():
            ...
        influential_point
        """),
        C("""
        slope_III_without = ...
        r2_III_without = ...
        iv_x_constant = ...
        """),
        C("""
        regression_suitable = {"I": ..., "II": ..., "III": ..., "IV": ...}
        """),
        C("""
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
        # one panel per dataset: points, OLS line and the influential point highlighted
        """),
    ],
    checks="""
        check(list(summary.columns) == ["slope", "intercept", "r2", "resid_std"],
              "the summary has the required columns", "columns: slope, intercept, r2, resid_std")
        check(np.allclose(summary.slope, 0.5, atol=0.01) and np.allclose(summary.intercept, 3.0, atol=0.01),
              "slope and intercept are practically identical in the four datasets (0.5 and 3.0)",
              "the four lines should have slope 0.5 and intercept 3.0")
        check(np.allclose(summary.r2, 0.667, atol=0.01), "the R2 is identical (0.667) in the four datasets",
              "the R2 should be about 0.667 in all four")
        check(abs(leverages(sets["IV"].x.to_numpy(float)).max() - 1) < 1e-9, "in dataset IV one point has leverage 1",
              "the point with x = 19 is the only one with that value: its leverage is 1")
        check(np.isinf(cook(sets["IV"].x.to_numpy(float), sets["IV"].y.to_numpy(float))).sum() == 1,
              "the Cook's distance of that point is infinite", "assign np.inf to the points with leverage > 1 - 1e-9")
        check(influential_point["III"] == 2 and influential_point["IV"] == 7,
              "influential points: index 2 in III (the vertical outlier), index 7 in IV (the isolated point)",
              "in III the outlier has x = 13, y = 12.74 (index 2); in IV the point with x = 19 has index 7")
        check(abs(slope_III_without - 0.345) < 0.01 and r2_III_without > 0.99,
              "without the outlier dataset III is almost perfectly linear (slope 0.345, R2 ~ 1)",
              "remove the point with index 2 from dataset III and refit")
        check(bool(iv_x_constant), "in dataset IV, without that point, x is constant and the slope is not identifiable",
              "after removing the point x = 19 all the x equal 8")
        check(regression_suitable == {"I": True, "II": False, "III": False, "IV": False}, "correct judgment on the four datasets",
              "only dataset I is a normal linear cloud: II is curved, III has an outlier, IV depends on one point")
    """,
    hint="""
        Cook: `e = y - A @ beta`, `s2 = e @ e / (n - p)`, `D = e**2 / (p * s2) * h / (1 - h)**2`; then
        `D = np.where(h > 1 - 1e-9, np.inf, D)`. To avoid division-by-zero warnings, use `with np.errstate(all="ignore")`.
    """,
    solution=[
        C("""
        rows = {}
        for name, d in sets.items():
            m = LinearRegression().fit(d[["x"]], d["y"])
            residuals = d["y"] - m.predict(d[["x"]])
            rows[name] = {"slope": m.coef_[0], "intercept": m.intercept_, "r2": m.score(d[["x"]], d["y"]),
                          "resid_std": residuals.std(ddof=2)}
        summary = pd.DataFrame(rows).T
        summary.round(3)
        """),
        M("""
        The four rows are practically identical: anyone who limits themselves to these statistics would conclude that the four
        datasets are the same problem. Only Cook's distance and the plots show the difference.
        """),
        C("""
        def leverages(x):
            A = np.column_stack([np.ones(len(x)), x])
            return np.diag(A @ np.linalg.inv(A.T @ A) @ A.T)


        def cook(x, y):
            A = np.column_stack([np.ones(len(x)), x])
            n, p = A.shape
            e = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
            s2 = e @ e / (n - p)
            h = leverages(x)
            with np.errstate(all="ignore"):
                d = e ** 2 / (p * s2) * h / (1 - h) ** 2
            return np.where(h > 1 - 1e-9, np.inf, d)
        """),
        C("""
        influential_point = {}
        for name, d in sets.items():
            x, y = d["x"].to_numpy(float), d["y"].to_numpy(float)
            influential_point[name] = int(np.argmax(cook(x, y)))
        print(influential_point)
        print("maximum leverage in IV:", leverages(sets["IV"].x.to_numpy(float)).max())
        """),
        C("""
        d3 = sets["III"].drop(index=influential_point["III"])
        m3 = LinearRegression().fit(d3[["x"]], d3["y"])
        slope_III_without = m3.coef_[0]
        r2_III_without = m3.score(d3[["x"]], d3["y"])

        d4 = sets["IV"].drop(index=influential_point["IV"])
        iv_x_constant = d4["x"].nunique() == 1
        print(f"III without the outlier: slope {slope_III_without:.3f}, R2 {r2_III_without:.4f}")
        print("IV without the isolated point: are all x equal?", iv_x_constant)
        """),
        C("""
        regression_suitable = {"I": True, "II": False, "III": False, "IV": False}
        """),
        C("""
        fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
        for a, (name, d) in zip(axes.ravel(), sets.items()):
            m = LinearRegression().fit(d[["x"]], d["y"])
            g = np.linspace(3, 20, 50)
            a.scatter(d.x, d.y, s=25)
            a.plot(g, m.predict(pd.DataFrame({"x": g})), color="#1F6E5C")
            i = influential_point[name]
            a.scatter(d.x[i], d.y[i], s=90, facecolors="none", edgecolors="#9C4530", lw=2)
            a.set_title(f"Dataset {name}")
        fig.tight_layout()
        """),
    ],
    learned="""
        - The summary statistics (slope, intercept, R²) do not identify the model: four very different structures give
          the same numbers. Looking at the data is not optional.
        - **II** is a parabola: the line is the wrong model (violation of linearity, visible in the residuals). **III** is a
          perfect line with a single outlier, which by itself explains the wrong slope: remove it and R² goes up to 1.
        - **IV** is the limit case: a single point with an x different from the others determines the whole slope. It has leverage exactly 1: the
          line passes *necessarily* through the point (residual 0), and Cook's distance is 0/0. Without that point x is constant and
          the slope is not even defined: the data contain no information about the x-y relationship.
    """,
    pitfall="""
        Saying "I remove the outlier" without understanding whether it is a measurement error or a piece of reality. In III the point is probably an
        error (it must be corrected or investigated); in IV the isolated point **is the only information** we have about the relationship, and
        removing it means no longer having a model.
    """,
    interview=(
        "In your model a single point has a Cook's distance 40 times the threshold. Do you remove it?",
        """
        First I understand why: I look at its leverage and its residual, and check whether it is a recording error. If it is an error I
        correct it or remove it documenting the choice. If it is a real but extreme observation (for example a customer much
        bigger than the others), I would in effect be removing the part of the range that I want to predict: I prefer a robust model (Huber),
        a transformation that reduces the leverage, or I state that the model does not cover that area. I always report the results
        with and without the point.
        """),
)

EXERCISES = [D1, D2, D3]
