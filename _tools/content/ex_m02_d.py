"""Module 2 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_a_one_hot_and_the_dummy_trap", kind="D", level=2, minutes=35, module=MODULE,
    title="Reading a one-hot: names, memory and the dummy variable trap",
    goal="""
        Interpret the output of a `OneHotEncoder` (categories, names, memory usage), show with the rank of the matrix
        that with the intercept the columns are redundant, and read the coefficients of a regression with `drop="first"`.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the two nominals `city` (12 levels) and `contract_type` (4), 1500 rows, and the target
        `defaulted` (0/1) as the variable to model with a linear regression (linear probability model: the coefficients
        are differences in default rates).
    """,
    purpose="""
        Understand that the full one-hot is **linearly dependent** with the intercept (the sum of the columns of a feature is 1) and
        that `drop="first"` does not change the predictions but **changes the meaning of the coefficients**: they are differences with respect to a reference
        category.
    """,
    task=[
        "With `enc = OneHotEncoder(sparse=False)` on the two columns save `n_columns` (number of columns produced), `names` (list of the names) and "
        "`baseline = enc.categories_[1][0]`, the first category of `contract_type` (the one that `drop=\"first\"` would remove).",
        "With `OneHotEncoder(sparse=True)` (default) save `nnz` (nonzero elements) and `density = nnz / (rows · columns)`. Why "
        "is it exactly 2 / n_columns?",
        "With the full matrix add the column of ones: `A = np.column_stack([np.ones(n), X_ohe])`. Save `rank_without_drop` "
        "(`np.linalg.matrix_rank(A)`) and `columns_of_A`. Then redo it with `drop=\"first\"`: `rank_with_drop` and `columns_with_drop`.",
        "Train `LinearRegression()` on `y = defaulted` with the full one-hot and with `drop=\"first\"`. Save `max_coef_full` and "
        "`max_coef_drop` (maximum absolute value of the coefficients in the two models) and `pred_diff` (maximum absolute difference between "
        "the predictions of the two models on the training set).",
        "With **only** `contract_type` and `drop=\"first\"`, train the regression and save `unemployed_coef` (the coefficient of "
        "`contract_type_unemployed`) and `mean_diff`: the difference between the mean default rate of the unemployed and that of the "
        "`baseline` category, computed with `groupby`.",
        "Assign to `true_false` `{\"a\", \"b\", \"c\"}`:\n"
        "   - **a**: with `drop=\"first\"` the coefficient of a category is the effect with respect to the dropped category, not with respect to the mean;\n"
        "   - **b**: changing the dropped category changes the predictions of the model;\n"
        "   - **c**: with the full one-hot and the intercept, an unregularized OLS can produce enormous, meaningless coefficients, "
        "even though the predictions stay almost correct.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder

        df = ds.loans(with_missing=False)
        X_cat = df[["city", "contract_type"]]
        y = df["defaulted"]
        n = len(df)
    """,
    starter=[
        C("""
        enc = OneHotEncoder(sparse=False).fit(X_cat)
        n_columns = ...
        names = ...
        baseline = ...
        """),
        C("""
        X_sparse = OneHotEncoder().fit_transform(X_cat)
        nnz = ...
        density = ...
        """),
        C("""
        X_ohe = enc.transform(X_cat)
        A = np.column_stack([np.ones(n), X_ohe])
        rank_without_drop, columns_of_A = ..., ...

        enc_d = OneHotEncoder(sparse=False, drop="first").fit(X_cat)
        A_d = np.column_stack([np.ones(n), enc_d.transform(X_cat)])
        rank_with_drop, columns_with_drop = ..., ...
        """),
        C("""
        max_coef_full = ...
        max_coef_drop = ...
        pred_diff = ...
        """),
        C("""
        enc_c = OneHotEncoder(sparse=False, drop="first").fit(df[["contract_type"]])
        model = LinearRegression().fit(enc_c.transform(df[["contract_type"]]), y)
        unemployed_coef = ...
        mean_diff = ...
        """),
        C("""
        true_false = {"a": ..., "b": ..., "c": ...}
        """),
    ],
    checks="""
        check(n_columns == 16 and len(names) == 16 and baseline == "fixed_term", "16 columns, and the alphabetical baseline is 'fixed_term'",
              "12 cities + 4 contracts = 16; categories_[1][0] is the first sorted category of contract_type")
        check(nnz == 2 * n and close(density, 2 / 16), "the one-hot is very sparse: 2 nonzero values per row, density 2/16",
              "every row has exactly one 1 per feature, so 2 nonzero out of 16 columns")
        check(columns_of_A == 17 and rank_without_drop == 15, "without drop the rank is 15 out of 17 columns: two linear dependencies",
              "one dependency per feature (the sum of its columns is the column of ones): 17 - 2 = 15")
        check(columns_with_drop == 15 and rank_with_drop == 15, "with drop='first' the 15 columns are independent (full rank)",
              "with drop='first' 1 + 11 + 3 = 15 columns remain, all independent")
        check(max_coef_full > 100 * max_coef_drop, "with the full one-hot the coefficients explode (over 100 times those with drop)",
              "without drop the OLS is ill-conditioned: the maximum of the coefficients must be huge compared to the one with drop")
        check(pred_diff < 0.1, "the predictions are almost equal (the same model, apart from numerical instability)",
              "the column space is the same: the predictions must differ little (less than 0.1)")
        check(abs(unemployed_coef - mean_diff) < 1e-9,
              "the coefficient matches the difference between the means of the two groups", "with a single dummy feature the coefficient is the difference of means")
        check(true_false == {"a": True, "b": False, "c": True}, "statements judged correctly",
              "changing the reference category changes the coefficients, not the predictions")
    """,
    hint="""
        `enc.get_feature_names_out()` for the names. `X_sparse.nnz` for the nonzero elements. The predictions:
        `LinearRegression().fit(A[:, 1:], y).predict(A[:, 1:])` against the same with `A_d`. The difference between means:
        `df.groupby("contract_type")["defaulted"].mean()` and then `m["unemployed"] - m[baseline]`.
    """,
    solution=[
        C("""
        enc = OneHotEncoder(sparse=False).fit(X_cat)
        n_columns = len(enc.get_feature_names_out())
        names = list(enc.get_feature_names_out())
        baseline = enc.categories_[1][0]
        print(n_columns, "columns | baseline of contract_type:", baseline)
        """),
        C("""
        X_sparse = OneHotEncoder().fit_transform(X_cat)
        nnz = X_sparse.nnz
        density = nnz / (X_sparse.shape[0] * X_sparse.shape[1])
        print(f"nonzero: {nnz} out of {X_sparse.shape[0] * X_sparse.shape[1]} | density {density:.4f} = 2/{X_sparse.shape[1]}")
        """),
        C("""
        X_ohe = enc.transform(X_cat)
        A = np.column_stack([np.ones(n), X_ohe])
        rank_without_drop, columns_of_A = np.linalg.matrix_rank(A), A.shape[1]

        enc_d = OneHotEncoder(sparse=False, drop="first").fit(X_cat)
        A_d = np.column_stack([np.ones(n), enc_d.transform(X_cat)])
        rank_with_drop, columns_with_drop = np.linalg.matrix_rank(A_d), A_d.shape[1]
        print(f"without drop: rank {rank_without_drop} out of {columns_of_A} columns | with drop: rank {rank_with_drop} out of {columns_with_drop}")
        """),
        C("""
        m_full = LinearRegression().fit(X_ohe, y)
        m_drop = LinearRegression().fit(enc_d.transform(X_cat), y)
        max_coef_full = np.abs(np.r_[m_full.intercept_, m_full.coef_]).max()
        max_coef_drop = np.abs(np.r_[m_drop.intercept_, m_drop.coef_]).max()
        pred_diff = np.abs(m_full.predict(X_ohe) - m_drop.predict(enc_d.transform(X_cat))).max()
        print(f"maximum coefficient: full {max_coef_full:.3g} | with drop {max_coef_drop:.3g}")
        print(f"maximum difference between the predictions: {pred_diff:.4f}")
        """),
        C("""
        enc_c = OneHotEncoder(sparse=False, drop="first").fit(df[["contract_type"]])
        model = LinearRegression().fit(enc_c.transform(df[["contract_type"]]), y)
        unemployed_coef = model.coef_[list(enc_c.get_feature_names_out()).index("contract_type_unemployed")]
        means = df.groupby("contract_type")["defaulted"].mean()
        mean_diff = means["unemployed"] - means[baseline]
        print(f"intercept {model.intercept_:.3f} = mean rate of '{baseline}' {means[baseline]:.3f}")
        print(f"coefficient 'unemployed' {unemployed_coef:.3f} = difference of rates {mean_diff:.3f}")
        """),
        C("""
        true_false = {"a": True, "b": False, "c": True}
        """),
    ],
    learned="""
        - For a nominal with k levels, the k one-hot columns sum to 1 everywhere, that is they **coincide with the column of ones**
          of the intercept: with two nominal features the dependencies are two (rank 15 out of 17 columns). `drop="first"` removes them.
        - `drop` does not change the model (the columns span the same space) but it changes the **parametrization**. With a single dummy
          feature the intercept is the mean of the reference category and every coefficient is the difference of means from it.
        - With the full one-hot `LinearRegression` has no unique solution: it solves a nearly singular system and the coefficients
          reach absurd orders of magnitude (here of the order of 10¹²), with predictions that differ by a few percentage points
          from those of the correct model purely through numerical instability.
        - The one-hot is extremely sparse: 2 nonzero values out of 16 per row. For high-cardinality features the sparse form (the
          default) saves a lot of memory; many linear models and scikit-learn estimators accept it directly.
        - scikit-learn's `LogisticRegression` and `Ridge` are regularized and do not suffer from this singularity, but their
          coefficients are still hard to interpret if the columns are redundant (modules 1 and 4).
    """,
    pitfall="""
        Reading the coefficient of a dummy as "the effect of the category" without saying with respect to which reference category. If the
        baseline is chosen alphabetically (`fixed_term`) the choice is arbitrary: changing it changes all the coefficients, not the model.
        In reports always state the baseline.
    """,
    interview=(
        "What is the dummy variable trap and when is it a problem in scikit-learn?",
        """
        It is the linear dependence between the one-hot columns of a feature and the intercept: their sum is the column of ones. With
        an unregularized OLS the matrix is not invertible and the coefficients are not identifiable; you avoid it by removing
        one column (`drop="first"`). In scikit-learn the regularized models (Ridge, `LogisticRegression` with L2, the default) work
        anyway, so often nothing is dropped. If I want to interpret the coefficients as differences from a baseline
        chosen deliberately, I drop that category myself.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="missingness_mechanisms", kind="D", level=3, minutes=45, module=MODULE,
    title="Diagnosing missingness mechanisms: what you can and cannot know",
    goal="""
        For each column with missing values in the loans dataset, establish with statistical tests whether the missingness is compatible
        with MCAR or depends on other variables (MAR), and say what the data **cannot** prove.
    """,
    dataset="""
        `ds.loans()` (with missing values): columns with missing values `age`, `income`, `job_tenure`, `education`, `occupation`. The missingness
        mechanisms were inserted differently for each one and are hidden on purpose. The other columns, including the
        target `defaulted`, are observed (with a few missing values themselves).
    """,
    purpose="""
        Know **how far** the data let you investigate the missingness mechanism: you can detect dependencies on the observed
        variables (MAR), but you cannot prove MCAR, and you cannot even detect MNAR by looking only at the observed ones.
    """,
    task=[
        "Write `dependence_test(df, col)`: for the column `col` build the indicator `m = df[col].isna()` and, for **every other column** "
        "`c`, compare `c` between the rows with `col` missing and those with `col` present, using only the rows where `c` is known. For "
        "numeric columns (including `defaulted`) use `scipy.stats.mannwhitneyu`; for categorical ones `scipy.stats.chi2_contingency` on "
        "`pd.crosstab(m, df[c])`. Return `(corrected_min_p, column)`: the smallest p-value multiplied by the number of tests "
        "(Bonferroni correction, at most 1) and the name of the column that produced it.",
        "For every column with missing values compute `corrected_min_p` and fill the DataFrame `table` (index: columns with missing values; "
        "columns: `missing_share`, `corrected_p`, `depends_on`).",
        "Assign to `mechanism` a dictionary `{column: \"MCAR\" | \"MAR\"}`: `\"MAR\"` if `corrected_p < 0.001`, otherwise `\"MCAR\"`.",
        "Compute `default_if_income_missing` and `default_if_income_present`: the default rate with the income missing and with the "
        "income known.",
        "Assign to `true_false` a dictionary:\n"
        "   - **a**: a non-significant test proves that the missingness of `age` is MCAR;\n"
        "   - **b**: if the missingness depended on the value of income itself (MNAR), these tests would always detect it;\n"
        "   - **c**: the fact that `income` is missing is useful information for predicting `defaulted`.",
    ],
    setup="""
        from scipy import stats

        df = ds.loans()
        with_missing = [c for c in df.columns if df[c].isna().any()]
        df[with_missing].isna().mean().round(3)
    """,
    starter=[
        C("""
        def dependence_test(df, col):
            m = df[col].isna()
            p_values = {}
            for c in df.columns:
                if c == col:
                    continue
                ...
            ...
            return corrected_min_p, column
        """),
        C("""
        rows = {}
        for col in with_missing:
            p, by = dependence_test(df, col)
            rows[col] = {"missing_share": df[col].isna().mean(), "corrected_p": p, "depends_on": by}
        table = pd.DataFrame(rows).T
        mechanism = ...
        table
        """),
        C("""
        default_if_income_missing = ...
        default_if_income_present = ...
        true_false = {"a": ..., "b": ..., "c": ...}
        """),
    ],
    checks="""
        p, by = dependence_test(df, "income")
        check(0 <= p <= 1 and by == "defaulted", "for 'income' the strongest dependence is with the target defaulted",
              "income is missing much more often among defaulters: the test on the target must have the minimum p-value")
        p2, by2 = dependence_test(df, "job_tenure")
        check(by2 == "contract_type" and p2 < 1e-6, "'job_tenure' depends on the contract type", "the missingness depends on contract_type")
        check(list(table.columns) == ["missing_share", "corrected_p", "depends_on"] and set(table.index) == set(with_missing),
              "table with one row per column and the three required columns", "columns: missing_share, corrected_p, depends_on")
        check(mechanism == {"age": "MCAR", "income": "MAR", "job_tenure": "MAR", "education": "MCAR", "occupation": "MCAR"},
              "mechanisms diagnosed correctly", "MAR where corrected_p < 0.001, MCAR elsewhere")
        check(default_if_income_missing > 3 * default_if_income_present, "whoever does not declare income defaults much more often (over 3 times)",
              "compare the default rate between rows with income missing and present")
        check(true_false == {"a": False, "b": False, "c": True}, "statements judged correctly",
              "a non-significant test proves nothing; MNAR is not detectable from the observed ones; the missingness is informative")
    """,
    hint="""
        For every column `c`: `ok = df[c].notna()`; if numeric `stats.mannwhitneyu(df.loc[ok & m, c], df.loc[ok & ~m, c]).pvalue`; if categorical
        `stats.chi2_contingency(pd.crosstab(m[ok], df.loc[ok, c]))[1]`. Numeric: `pd.api.types.is_numeric_dtype(df[c])`. Use the minimum of the p-values
        for the corrected p-value: `min(1, p_min * len(p_values))`.
    """,
    solution=[
        C("""
        def dependence_test(df, col):
            m = df[col].isna()
            p_values = {}
            for c in df.columns:
                if c == col:
                    continue
                ok = df[c].notna()                                   # compare only where c is observed
                if pd.api.types.is_numeric_dtype(df[c]):
                    p_values[c] = stats.mannwhitneyu(df.loc[ok & m, c], df.loc[ok & ~m, c]).pvalue
                else:
                    p_values[c] = stats.chi2_contingency(pd.crosstab(m[ok], df.loc[ok, c]))[1]
            column = min(p_values, key=p_values.get)
            corrected_min_p = min(1.0, p_values[column] * len(p_values))    # Bonferroni
            return corrected_min_p, column
        """),
        C("""
        rows = {}
        for col in with_missing:
            p, by = dependence_test(df, col)
            rows[col] = {"missing_share": df[col].isna().mean(), "corrected_p": p, "depends_on": by}
        table = pd.DataFrame(rows).T
        mechanism = {col: ("MAR" if row.corrected_p < 0.001 else "MCAR") for col, row in table.iterrows()}
        pd.options.display.float_format = "{:.3g}".format
        table
        """),
        C("""
        missing = df["income"].isna()
        default_if_income_missing = df.loc[missing, "defaulted"].mean()
        default_if_income_present = df.loc[~missing, "defaulted"].mean()
        print(f"default rate: income missing {default_if_income_missing:.3f} | income present {default_if_income_present:.3f}")
        true_false = {"a": False, "b": False, "c": True}
        """),
    ],
    learned="""
        - The tests correctly identify the dependencies: `income` is missing based on the target, `job_tenure` based on the contract
          type (whoever is unemployed has no tenure to report), and the other three columns are compatible with MCAR.
        - **A non-significant test does not prove MCAR**: it only says that that test did not find a dependence. With few missing values or
          weak dependencies the tests have no power.
        - **MNAR cannot be seen from the observed ones**: if the missingness depends on the missing value (whoever has a very high income does not
          declare it), the observed variables may show no dependence at all. It is a hypothesis about the world, not a result of the data:
          it is supported with knowledge of the collection process.
        - The very fact that income is missing is information: 56% default among those who do not declare it against 15% among those who do.
          An imputer with `add_indicator=True` keeps it.
    """,
    pitfall="""
        Concluding "MCAR, so I can drop the rows" after running an underpowered test. And the opposite: correcting the
        p-values with Bonferroni and rejecting everything, when the real dependence is weak. In practice you combine the tests with
        domain knowledge and check **empirically** whether the imputation choice changes the performance.
    """,
    interview=(
        "A colleague says: 'I checked, the missing data are MCAR, so I drop the rows'. How do you reply?",
        """
        That MCAR cannot be verified: the tests can only detect dependencies on the observed variables, and a non-significant
        result is not proof (it depends on the power). Besides, MNAR is indistinguishable from the observed ones. Dropping the rows
        with many missing values also loses data and can introduce bias. I would compare dropping, simple imputation and imputation with
        indicator in cross-validation on the real problem, and use knowledge of how the data are collected.
        """),
)

EXERCISES = [D1, D2]
