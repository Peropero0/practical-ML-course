"""Module 2 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="get_dummies_misaligned_columns", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="pd.get_dummies on training and test: same size, different columns",
    goal="""
        Find out why a model with an excellent AUC during development produces predictions worse than chance on new data, even though there is
        no error or warning: the one-hot columns of the test set do not match those of the training set. Fix it in two ways.
    """,
    dataset="""
        Synthetic, from `generate(n, regions, seed)`: a categorical `region` and a numeric feature `x`, with different true effects for
        each region (`Center +2`, `North −2`, `South 0`, `Sardinia +1`). The **training set** contains Center, North and South. The **test set** (which
        simulates the production data) contains North, South and **Sardinia**: a new region, and a known one (Center) that does not show up.
    """,
    purpose="""
        Recognize one of the most common silent bugs: `pd.get_dummies` decides the columns by looking at **the data it receives**, not at those of the
        training set. With the same number of columns and a different order the model applies every coefficient to the wrong column.
    """,
    task=[
        "Run the colleague's cell: look at the shapes of `Xd_tr` and `Xd_te`. They are equal: no error. Save in `different_columns` `True` if "
        "the lists of column names of the two DataFrames are different.",
        "Fix it with a `OneHotEncoder(handle_unknown=\"ignore\", sparse=False)` fitted on the training set alone, on the `region` column, "
        "concatenated with `x`. Save `auc_ok`.",
        "Also fix it in a second way, staying in pandas: `pd.get_dummies` on the test set and then `reindex(columns=Xd_tr.columns, fill_value=0)`. "
        "Save `auc_reindex`.",
        "Explain in `answer` (\"a\", \"b\" or \"c\") why the bug raises no errors:\n"
        "   - **a**: the number of columns matches (4), so the input has the right shape, but the names (and the order) of the columns do not, and the model (in scikit-learn 1.1) "
        "uses only the positions;\n"
        "   - **b**: `LogisticRegression` ignores categorical variables;\n"
        "   - **c**: `pd.get_dummies` fixes the columns on first use.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import OneHotEncoder


        def generate(n, regions, seed):
            r = np.random.default_rng(seed)
            reg = r.choice(regions, n)
            eff = {"Center": 2.0, "North": -2.0, "South": 0.0, "Sardinia": 1.0}
            x = r.normal(size=n)
            logit = np.array([eff[k] for k in reg]) + 1.0 * x
            return pd.DataFrame({"region": reg, "x": x}), r.binomial(1, 1 / (1 + np.exp(-logit)))


        X_tr, y_tr = generate(800, ["Center", "North", "South"], 1)
        X_te, y_te = generate(300, ["North", "Sardinia", "South"], 2)      # in production: a new region, and a known one that does not show up
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        Xd_tr = pd.get_dummies(X_tr)
        Xd_te = pd.get_dummies(X_te)
        print(Xd_tr.shape, Xd_te.shape)                       # same size: no error

        model = LogisticRegression(max_iter=1000).fit(Xd_tr.to_numpy(dtype=float), y_tr)
        auc_bug = roc_auc_score(y_te, model.predict_proba(Xd_te.to_numpy(dtype=float))[:, 1])
        print(f"test AUC: {auc_bug:.3f}")
        """),
        M("### Your diagnosis and the fixes"),
        C("""
        different_columns = None
        auc_ok = auc_bug
        auc_reindex = auc_bug
        answer = None
        """),
    ],
    checks="""
        check(different_columns is True and list(Xd_tr.columns) != list(Xd_te.columns), "the columns of the test set are not those of the training set",
              "compare list(Xd_tr.columns) with list(Xd_te.columns): different_columns must be True")
        check(auc_ok > auc_bug + 0.05, "the OneHotEncoder fitted on the training set clearly improves the test AUC",
              "auc_ok must come from an encoder fitted ONLY on X_tr and applied with transform to X_te")
        check(abs(auc_ok - auc_reindex) < 1e-9, "the reindex on the training columns is equivalent to the OneHotEncoder",
              "get_dummies(X_te).reindex(columns=Xd_tr.columns, fill_value=0) must give the same result")
        check(answer == "a", "correct explanation", "reread: what changes between the two matrices, besides the number of columns?")
    """,
    hint="""
        `ohe = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X_tr[["region"]])` and then
        `np.column_stack([X[["x"]].to_numpy(), ohe.transform(X[["region"]])])`. For the reindex the column order must be that of
        `Xd_tr.columns`; the columns that do not show up in the test set are filled with zeros and the new ones (`region_Sardinia`) discarded.
    """,
    solution=[
        C("""
        # the colleague's code: it runs, but it is wrong
        Xd_tr = pd.get_dummies(X_tr)
        Xd_te = pd.get_dummies(X_te)
        print(Xd_tr.shape, Xd_te.shape)

        model = LogisticRegression(max_iter=1000).fit(Xd_tr.to_numpy(dtype=float), y_tr)
        auc_bug = roc_auc_score(y_te, model.predict_proba(Xd_te.to_numpy(dtype=float))[:, 1])
        print(f"test AUC: {auc_bug:.3f}")
        """),
        M("""
        **Where the bug is.** `pd.get_dummies` creates a column for every category **present in the data it receives**. The training set has
        `region_Center`, `region_North`, `region_South`; the test set has `region_North`, `region_Sardinia`, `region_South`. The matrices both
        have 4 columns (with `x`), but the second column of the training set is Center and that of the test set is North, the third is North versus
        Sardinia: the model applies the coefficient of Center (+2) to the rows of North (which are worth −2), and that of North to Sardinia.
        """),
        C("""
        different_columns = list(Xd_tr.columns) != list(Xd_te.columns)
        print("training:", list(Xd_tr.columns))
        print("test    :", list(Xd_te.columns))
        """),
        C("""
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False).fit(X_tr[["region"]])          # columns decided by the TRAINING set


        def matrix(X):
            return np.column_stack([X[["x"]].to_numpy(), ohe.transform(X[["region"]])])


        model_ok = LogisticRegression(max_iter=1000).fit(matrix(X_tr), y_tr)
        auc_ok = roc_auc_score(y_te, model_ok.predict_proba(matrix(X_te))[:, 1])

        Xd_te_aligned = Xd_te.reindex(columns=Xd_tr.columns, fill_value=0)
        auc_reindex = roc_auc_score(y_te, LogisticRegression(max_iter=1000).fit(Xd_tr.to_numpy(dtype=float), y_tr)
                                    .predict_proba(Xd_te_aligned.to_numpy(dtype=float))[:, 1])
        print(f"AUC: with the bug {auc_bug:.3f} | OneHotEncoder {auc_ok:.3f} | reindex {auc_reindex:.3f}")
        answer = "a"
        """),
    ],
    learned="""
        - The columns of an encoder must be **decided by the training set and rebuilt identically** at every other moment: with an
          encoder with state (`OneHotEncoder`) it is automatic, with `pd.get_dummies` it is not.
        - The bug produces no errors: the shapes coincide by chance (here 4 columns in both matrices). In scikit-learn 1.2 or later, with
          DataFrame input, the check on the feature names would reveal it with an error; in 1.1 it stays silent. Don't count on it: it is a case where
          a test on the columns (`assert list(...) == list(...)`) saves hours.
        - With `handle_unknown="ignore"` the new region (`Sardinia`) produces zeros: the model does not know it, it is the prudent choice. The cost
          is visible in the correct result too: the AUC stays lower than what is achievable with all the information, because there is no
          information about Sardinia.
    """,
    pitfall="""
        Calling `pd.get_dummies` "for convenience" during development and forgetting that in production the rows arrive one at a time
        (one region at a time, so a single column!). The code that works on a whole test set breaks silently,
        or obviously, on a single request.
    """,
    interview=(
        "Why do you use `OneHotEncoder` and not `pd.get_dummies` in a production pipeline?",
        """
        Because the former is a transformer with state: the categories are learned from the training set and saved together with the model, so
        the columns are always the same and new categories have a defined behavior (zeros, or an error if I want one).
        `pd.get_dummies` is a function with no memory: the result depends on the data I pass it, and with a single request
        it would produce a single column. Inside a Pipeline, moreover, it guarantees that training and production use the same code.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="wrong_coefficient_names", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="Coefficients with the wrong names: the column order after the ColumnTransformer",
    goal="""
        Discover that a variable importance analysis gives a wrong answer because the names of the original columns were
        matched to the coefficients in a different order from the one in which the `ColumnTransformer` produced them, and rebuild the right names.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: four numeric columns (`age`, `income`, `job_tenure`, `amount`) and the target `defaulted`. A
        `ColumnTransformer` standardizes `income` and `amount` first and then `age` and `job_tenure`: the order of the output columns is
        **different** from the input one, but the number of columns is the same.
    """,
    purpose="""
        Understand that a transformer in scikit-learn returns an array **without names**, in an order that depends on how it is defined. And
        learn `get_feature_names_out()` to recover them.
    """,
    task=[
        "Run the colleague's cell and look at the variable it declares to be the most influential (`bug_variable`).",
        "Rebuild the correct names with `ct.get_feature_names_out()` (removing the step prefix: `n.split(\"__\")[-1]`) and save them in "
        "`correct_names`.",
        "Build `coef_ok` (a `Series` with `correct_names` as index) and save `right_variable`, the variable with the largest coefficient "
        "**in absolute value**.",
        "Save `income_sign` (`\"positive\"` or `\"negative\"`) and `amount_sign` according to the correct coefficients, and comment on them: do the signs make sense?",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]]
        y = df["defaulted"]

        ct = ColumnTransformer([("a", StandardScaler(), ["income", "amount"]),
                                ("b", StandardScaler(), ["age", "job_tenure"])])
        Z = ct.fit_transform(X)
        model = LogisticRegression(max_iter=2000).fit(Z, y)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        coef_bug = pd.Series(model.coef_[0], index=X.columns)          # X.columns = age, income, job_tenure, amount
        bug_variable = coef_bug.abs().idxmax()
        print(coef_bug.round(2).to_string())
        print("most influential variable according to the colleague:", bug_variable)
        """),
        M("### Your fix"),
        C("""
        correct_names = None
        coef_ok = coef_bug
        right_variable = bug_variable
        income_sign = None
        amount_sign = None
        """),
    ],
    checks="""
        check(correct_names == ["income", "amount", "age", "job_tenure"], "names rebuilt in the output order of the ColumnTransformer",
              "the order is: income, amount, age, job_tenure (get_feature_names_out, without prefix)")
        check(len(coef_ok) == 4 and list(coef_ok.index) == correct_names, "coef_ok has the correct names as index", "coef_ok = pd.Series(model.coef_[0], index=correct_names)")
        check(close(coef_ok.to_numpy(), model.coef_[0]), "the coefficients are the same: only the names change", "don't change the values, only the matching to the names")
        check(right_variable == coef_ok.abs().idxmax() and right_variable != bug_variable,
              "the most influential variable is a different one from the one declared by the colleague", "right_variable = coef_ok.abs().idxmax()")
        check(income_sign == "negative" and amount_sign == "positive",
              "a higher income lowers the risk, a higher amount raises it: sensible signs",
              "look at the sign of the coefficients of income and amount in coef_ok")
    """,
    hint="""
        `correct_names = [n.split("__")[-1] for n in ct.get_feature_names_out()]`. The sign: `"positive" if coef_ok["income"] > 0 else "negative"`.
    """,
    solution=[
        C("""
        # the colleague's code: it runs, but it is wrong
        coef_bug = pd.Series(model.coef_[0], index=X.columns)          # X.columns = age, income, job_tenure, amount
        bug_variable = coef_bug.abs().idxmax()
        print(coef_bug.round(2).to_string())
        print("most influential variable according to the colleague:", bug_variable)
        """),
        M("""
        **Where the bug is.** `Z` has the columns in the order of the transformers (`income, amount, age, job_tenure`), but the
        colleague used `X.columns` (`age, income, job_tenure, amount`) as the index: the coefficient of income was labeled
        `age`, that of amount `income`, and so on. No error: it is four numbers and four names.
        """),
        C("""
        correct_names = [n.split("__")[-1] for n in ct.get_feature_names_out()]
        coef_ok = pd.Series(model.coef_[0], index=correct_names)
        right_variable = coef_ok.abs().idxmax()
        income_sign = "positive" if coef_ok["income"] > 0 else "negative"
        amount_sign = "positive" if coef_ok["amount"] > 0 else "negative"
        print(pd.DataFrame({"wrong": coef_bug.values, "correct": coef_ok.reindex(X.columns).values}, index=X.columns).round(2))
        print("most influential: colleague", bug_variable, "| correct", right_variable)
        print("income sign:", income_sign, "| amount sign:", amount_sign)
        """),
    ],
    learned="""
        - A transformer returns an array **without names** and in the order of its blocks, not of the original columns. Matching the
          original names to the coefficients "by position" is a silent mistake: four numbers, four names, no warning.
        - `get_feature_names_out()` is the right way to rebuild the names (with the step prefixes: `a__income`, `b__age`...). In a
          `Pipeline` you get it with `pipe[:-1].get_feature_names_out()`.
        - A plausibility test helps: a higher income must lower the risk (negative sign), a higher amount must raise it.
          If the signs are counterintuitive, the first hypothesis is a labeling error, before a discovery.
    """,
    pitfall="""
        Trusting the order of the columns "because it is the same as before anyway". With `remainder="passthrough"`, with transformers that
        change the number of columns (one-hot) or with blocks in a different order, that assumption is false. The rule: names are
        rebuilt from the transformer, never by hand.
    """,
    interview=(
        "How do you get the feature names after a Pipeline with ColumnTransformer and OneHotEncoder?",
        """
        With `get_feature_names_out()` on the preprocessor (`pipe[:-1].get_feature_names_out()` or `pipe.named_steps['prep']...`), which returns the names
        in the order of the output columns, with the step prefix (`num__income`, `cat__city_Rome`). I use them to label
        coefficients or importances; I never rebuild them by hand, because the order depends on the blocks and on the number of categories.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="target_encoding_before_cv", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="Target encoding before the cross-validation: a transformer inside the Pipeline",
    goal="""
        Fix an evaluation inflated by target encoding run on the whole dataset before the cross-validation, by writing a
        scikit-learn compatible transformer (`fit`, `transform`, `fit_transform`) and using it inside a `Pipeline`.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the `occupation` column (40 levels, with a true effect) and `noise`, a category with 300 random levels
        with no link to `defaulted`. The functions `te_estimate`, `te_apply`, `te_oof` (those of exercise A3) are already
        defined in the setup cell.
    """,
    purpose="""
        Close the circle of A3: know **where** to put an encoder that uses the target. A transformer inside a `Pipeline` receives
        `fit_transform(X, y)` on the training folds and only `transform(X)` on the validation folds: it is the only place where target
        encoding does not leak the target.
    """,
    task=[
        "Run the colleague's cell: it encodes all the rows with `te_estimate` + `te_apply` **before** the cross-validation, then evaluates "
        "a `LogisticRegression`. Save `auc_bug` (it is very high).",
        "Write `MyTargetEncoder(BaseEstimator, TransformerMixin)` with `__init__(self, m=5)`, `fit(X, y)` (save the mappings for every column "
        "of `X`, a DataFrame), `transform(X)` (apply the mappings; unknown categories → global mean) and `fit_transform(X, y)` which "
        "returns the **out-of-fold** encoding of the training set (use `te_oof`) after calling `fit`.",
        "Build `make_pipeline(MyTargetEncoder(m=5), LogisticRegression())` and evaluate it with the same `cv`: save `auc_ok`.",
        "Repeat the same pipeline on the `noise` column alone: `noise_only_auc` (it should stay close to 0.5).",
    ],
    setup="""
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline

        df = ds.loans(with_missing=False)
        X = df[["occupation"]].copy()
        X["noise"] = np.array([f"r{i}" for i in np.random.default_rng(1).integers(0, 300, len(df))])
        y = df["defaulted"].to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)


        def te_estimate(cat, y, m):
            d = pd.DataFrame({"c": np.asarray(cat), "y": np.asarray(y, dtype=float)})
            g = d.groupby("c")["y"].agg(["sum", "count"])
            overall = d["y"].mean()
            return ((g["sum"] + m * overall) / (g["count"] + m)).to_dict(), overall


        def te_apply(cat, mapping, overall):
            return pd.Series(np.asarray(cat)).map(mapping).fillna(overall).to_numpy(dtype=float)


        def te_oof(cat, y, m, k=5, seed=0):
            cat, y = np.asarray(cat), np.asarray(y)
            out = np.zeros(len(cat))
            for tr, va in KFold(k, shuffle=True, random_state=seed).split(cat):
                mapping, overall = te_estimate(cat[tr], y[tr], m)
                out[va] = te_apply(cat[va], mapping, overall)
            return out
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        # encoding of the WHOLE dataset with the mapping computed on the WHOLE dataset, before the cross-validation
        Z = np.column_stack([te_apply(X[c], *te_estimate(X[c], y, 5)) for c in X.columns])
        auc_bug = cross_val_score(LogisticRegression(), Z, y, cv=cv, scoring="roc_auc").mean()
        print(f"colleague's cross-validated AUC: {auc_bug:.3f}")
        """),
        M("### Your transformer"),
        C("""
        class MyTargetEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, m=5):
                self.m = m

            def fit(self, X, y):
                ...
                return self

            def transform(self, X):
                ...

            def fit_transform(self, X, y):
                ...
        """),
        C("""
        auc_ok = auc_bug
        noise_only_auc = auc_bug
        """),
    ],
    checks="""
        check(auc_bug > 0.75, "the colleague's number is very high: it is the target leaking", "auc_bug should exceed 0.75: don't modify the colleague's code")
        check(0.55 < auc_ok < 0.72, "with the pipeline the AUC drops to a realistic value (between 0.55 and 0.72)",
              "auc_ok must come from cross_val_score on make_pipeline(MyTargetEncoder(m=5), LogisticRegression())")
        check(noise_only_auc < 0.6, "on the noise alone the cross-validated AUC is close to 0.5", "on the noise alone with the pipeline the AUC must stay below 0.6")
        check(isinstance(MyTargetEncoder(), TransformerMixin), "the transformer is scikit-learn compatible", "inherit from BaseEstimator and TransformerMixin")
        enc = MyTargetEncoder(m=5).fit(X, y)
        new = enc.transform(pd.DataFrame({"occupation": ["occ_zz"], "noise": ["r_new"]}))
        check(np.isfinite(new).all() and close(new, [[y.mean(), y.mean()]]), "unknown categories get the global mean",
              "a never-seen category must be encoded with the global mean of the target")
        check(not np.allclose(MyTargetEncoder(m=5).fit_transform(X, y), enc.transform(X)),
              "fit_transform (out-of-fold) is different from fit + transform on the same training set",
              "fit_transform must use te_oof; transform uses the mapping of the whole training set")
    """,
    hint="""
        In `fit`: `self.columns_ = list(X.columns)` and `self.maps_ = {c: te_estimate(X[c], y, self.m) for c in self.columns_}`. In `transform`:
        `np.column_stack([te_apply(X[c], *self.maps_[c]) for c in self.columns_])`. The `fit_transform`: call `self.fit(X, y)` and then
        `np.column_stack([te_oof(X[c], y, self.m) for c in self.columns_])`.
    """,
    solution=[
        C("""
        # the colleague's code: it runs, but it is wrong
        Z = np.column_stack([te_apply(X[c], *te_estimate(X[c], y, 5)) for c in X.columns])
        auc_bug = cross_val_score(LogisticRegression(), Z, y, cv=cv, scoring="roc_auc").mean()
        print(f"colleague's cross-validated AUC: {auc_bug:.3f}")
        """),
        M("""
        **Where the bug is.** `te_estimate(X[c], y, 5)` uses the target of **all** the rows, including those that then end up in the
        validation fold of the cross-validation: every row is encoded with a mean that contains its own target. For the noise
        column (300 levels, about 5 rows each) the encoding is almost the target itself, and the AUC goes up as if the noise were a
        signal. The fix: the encoder must sit **inside** the pipeline, so that at every fold the mapping is computed only on the
        training part of the fold.
        """),
        C("""
        class MyTargetEncoder(BaseEstimator, TransformerMixin):
            def __init__(self, m=5):
                self.m = m

            def fit(self, X, y):
                X = pd.DataFrame(X)
                self.columns_ = list(X.columns)
                self.maps_ = {c: te_estimate(X[c], y, self.m) for c in self.columns_}
                return self

            def transform(self, X):
                X = pd.DataFrame(X)
                return np.column_stack([te_apply(X[c], *self.maps_[c]) for c in self.columns_])

            def fit_transform(self, X, y):
                self.fit(X, y)                                    # the complete mappings are needed for transform on the validation set
                X = pd.DataFrame(X)
                return np.column_stack([te_oof(X[c], y, self.m) for c in self.columns_])     # but the training set is encoded out-of-fold
        """),
        C("""
        pipe = make_pipeline(MyTargetEncoder(m=5), LogisticRegression())
        auc_ok = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
        noise_only_auc = cross_val_score(pipe, X[["noise"]], y, cv=cv, scoring="roc_auc").mean()
        print(f"cross-validated AUC: colleague {auc_bug:.3f} | correct pipeline {auc_ok:.3f} | noise only {noise_only_auc:.3f}")
        """),
    ],
    learned="""
        - A `Pipeline` calls `fit_transform(X_train, y_train)` on the training folds and `transform(X_val)` on the validation folds:
          overriding `fit_transform` with the out-of-fold encoding gives the right behavior. The bug is repeating the
          same mapping on everything.
        - The colleague's AUC (about 0.85) is a mirage: with the pipeline it drops to about 0.6, the real value of the occupation. On the noise
          alone it falls from high values to about 0.5.
        - Writing a compatible transformer is little code: `BaseEstimator` for `get_params`/`clone`, `TransformerMixin` for the default
          `fit_transform`, `fit` that returns `self`. Module 3 uses this pattern for custom preprocessors.
    """,
    pitfall="""
        Relying on the default `fit_transform` (`TransformerMixin` calls `fit(X, y).transform(X)`): in target encoding it means
        encoding the training set with the in-sample mapping, that is the very bug you were trying to avoid. For transformers that use `y` the
        `fit_transform` must be overridden.
    """,
    interview=(
        "Why must a transformer that uses the target be written with a `fit_transform` different from `fit` + `transform`?",
        """
        Because in training the encoding of every row must not contain its target: `fit_transform` computes it out-of-fold. In
        validation or in production, instead, the rows are new and the mapping of the whole training set is used (`transform`). If `fit_transform`
        were `fit` + `transform` on the same training set, the model would see a feature that has already seen the target and would learn to trust it
        too much: in production it will not be like that.
        """),
)

EXERCISES = [E1, E2, E3]
