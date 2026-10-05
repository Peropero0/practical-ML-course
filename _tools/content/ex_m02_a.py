"""Module 2 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="scalers_from_scratch", kind="A", level=2, minutes=35, module=MODULE,
    title="Three scalers from scratch: fit, transform and the constant column case",
    goal="""
        Implement `StandardScaler`, `MinMaxScaler` and `RobustScaler` with the same interface as scikit-learn (`fit`,
        `transform`, `fit_transform`, `inverse_transform`), check them against the original versions and handle the edge case
        of a constant column.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns (`age`, `income`, `job_tenure`, `amount`) plus
        a fifth **constant** column (7.0) added by hand. The split is 75/25, `X_tr` and `X_te`. Income has a long
        tail with a few very high values.
    """,
    purpose="""
        Discover that a scaler is an object with **state**: `fit` computes some numbers, `transform` applies them without recomputing them.
        Understand how a constant value (zero variance) would break the division, and why `fit_transform` on the test set is a bug.
    """,
    task=[
        "Write `MyStandardScaler`: `fit` computes `mean_` and `scale_` (standard deviation with `ddof=0`); `transform` computes "
        "`(X − mean_) / scale_`. Where `scale_` is 0, replace it with 1 (the constant column becomes all zeros, not NaN).",
        "Write `MyMinMaxScaler` (`min_` and `scale_ = max − min`, with 0 replaced by 1; `transform = (X − min_) / scale_`) and "
        "`MyRobustScaler` (`center_ = median`, `scale_ = q75 − q25` with `np.percentile`; 0 replaced by 1).",
        "Add to each class `fit_transform` and `inverse_transform` (the exact inverse operation).",
        "Check that on the training and test data the results coincide with those of `sklearn.preprocessing`.",
        "Compute `refit_difference`: the maximum absolute difference between `MyStandardScaler().fit_transform(X_te)` and "
        "`scaler_tr.transform(X_te)`, where `scaler_tr` is fitted on `X_tr`. It is the measure of how much the wrong code "
        "changes the data.",
    ],
    setup="""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        df = ds.loans(with_missing=False)
        X_num = df[["age", "income", "job_tenure", "amount"]].to_numpy()
        X_num = np.column_stack([X_num, np.full(len(X_num), 7.0)])         # a constant column: the edge case
        X_tr, X_te = train_test_split(X_num, test_size=0.25, random_state=0)
    """,
    starter=[
        C("""
        class MyStandardScaler:
            def fit(self, X):
                ...
                return self

            def transform(self, X):
                ...

            def fit_transform(self, X):
                ...

            def inverse_transform(self, Z):
                ...
        """),
        C("""
        class MyMinMaxScaler:
            ...


        class MyRobustScaler:
            ...
        """),
        C("""
        scaler_tr = MyStandardScaler().fit(X_tr)
        refit_difference = ...
        refit_difference
        """),
    ],
    checks="""
        pairs = [(MyStandardScaler, StandardScaler), (MyMinMaxScaler, MinMaxScaler), (MyRobustScaler, RobustScaler)]
        for Mine, Ref in pairs:
            mine, ref = Mine(), Ref()
            check(close(mine.fit_transform(X_tr), ref.fit_transform(X_tr), atol=1e-9),
                  f"{Mine.__name__}: fit_transform matches scikit-learn on the training set",
                  f"{Mine.__name__}.fit_transform(X_tr) does not match {Ref.__name__}")
            check(close(mine.transform(X_te), ref.transform(X_te), atol=1e-9),
                  f"{Mine.__name__}: transform uses the training statistics",
                  f"{Mine.__name__}.transform(X_te) does not match: did you recompute the statistics on the test set?")
            check(close(mine.inverse_transform(mine.transform(X_te)), X_te, atol=1e-6),
                  f"{Mine.__name__}: inverse_transform rebuilds the data", "transform followed by inverse_transform must give back X")
            check(not np.isnan(mine.transform(X_te)).any() and np.all(mine.transform(X_te)[:, 4] == 0),
                  f"{Mine.__name__}: the constant column becomes zero, with no NaN", "where the scale is 0 use 1")
        s = MyStandardScaler().fit(X_tr)
        expected = np.abs(MyStandardScaler().fit_transform(X_te) - s.transform(X_te)).max()
        check(close(refit_difference, expected) and refit_difference > 0.1,
              "refitting the scaler on the test set changes the data substantially",
              "refit_difference = max |fit_transform(X_te) - scaler_tr.transform(X_te)|")
    """,
    hint="""
        To neutralize the zero scale: `scale = np.where(scale == 0, 1.0, scale)`. For the Robust:
        `q25, q75 = np.percentile(X, [25, 75], axis=0)`. The inverse of the standard one is `Z * scale_ + mean_`.
    """,
    solution=[
        C("""
        def _safe_scale(s):
            return np.where(s == 0, 1.0, s)                 # constant column: dividing by 1 leaves zeros, not NaN


        class MyStandardScaler:
            def fit(self, X):
                self.mean_ = X.mean(axis=0)
                self.scale_ = _safe_scale(X.std(axis=0))
                return self

            def transform(self, X):
                return (X - self.mean_) / self.scale_

            def fit_transform(self, X):
                return self.fit(X).transform(X)

            def inverse_transform(self, Z):
                return Z * self.scale_ + self.mean_


        class MyMinMaxScaler:
            def fit(self, X):
                self.min_ = X.min(axis=0)
                self.scale_ = _safe_scale(X.max(axis=0) - self.min_)
                return self

            def transform(self, X):
                return (X - self.min_) / self.scale_

            def fit_transform(self, X):
                return self.fit(X).transform(X)

            def inverse_transform(self, Z):
                return Z * self.scale_ + self.min_


        class MyRobustScaler:
            def fit(self, X):
                self.center_ = np.median(X, axis=0)
                q25, q75 = np.percentile(X, [25, 75], axis=0)
                self.scale_ = _safe_scale(q75 - q25)
                return self

            def transform(self, X):
                return (X - self.center_) / self.scale_

            def fit_transform(self, X):
                return self.fit(X).transform(X)

            def inverse_transform(self, Z):
                return Z * self.scale_ + self.center_
        """),
        C("""
        scaler_tr = MyStandardScaler().fit(X_tr)
        refit_difference = np.abs(MyStandardScaler().fit_transform(X_te) - scaler_tr.transform(X_te)).max()
        print("maximum difference between wrong and right code on the test set:", round(refit_difference, 3))
        print("means of the correctly scaled test set:", scaler_tr.transform(X_te).mean(axis=0).round(3))
        """),
    ],
    learned="""
        - A scaler is an object with state: `fit` computes some numbers on the training set, `transform` uses them. Recomputing them on the test set
          (`fit_transform(X_te)`) shifts and rescales the test differently from the training: the model sees data that do not
          resemble those it was trained on.
        - A constant column has standard deviation zero: without the `scale == 0 → 1` protection the division produces NaN and
          the NaN spreads through the whole model. scikit-learn has the same protection (`_handle_zeros_in_scale`).
        - The three scalers differ only in the center (mean, minimum, median) and the scale (standard deviation, total
          range, interquartile range): everything else is identical.
    """,
    pitfall="""
        Computing the statistics on the complete dataset "for convenience" before the split. The model never sees the test set, but
        the scaler does: the statistic of the test set enters the training data. The effect on the metrics is small with scaling and
        large with encoders that use the target (exercise A3).
    """,
    interview=(
        "In production your service receives one request at a time. How do you apply the scaler to a single row?",
        """
        With `transform`, using the mean and standard deviation saved from training: that is why the scaler must be serialized together with the
        model (better the whole Pipeline). With `fit_transform` on one row the standard deviation would be zero and the result
        would be meaningless.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="one_hot_encoder_from_scratch", kind="A", level=2, minutes=35, module=MODULE,
    title="OneHotEncoder from scratch: unknown categories and `drop`",
    goal="""
        Implement a one-hot encoder that learns the categories on the training set, always produces the same columns, handles
        never-seen categories and supports `drop="first"`, checking it against `sklearn.preprocessing.OneHotEncoder`.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: two nominal columns, `city` (12 levels) and `contract_type` (4), as an array of
        strings. In the test set the first five rows have the city replaced by `"city_XX"`, **never seen** in training.
    """,
    purpose="""
        See what `fit` hides: the sorted list of the categories of the **training set** decides the columns forever. And understand why
        an unknown category and `drop` do not combine: the row of zeros would be ambiguous with the dropped category.
    """,
    task=[
        "Write `MyOneHot(drop=None)`. `fit(X)`: for each column save the sorted categories (`np.unique`) in `categories_`.",
        "`transform(X)` returns a `float` array with one column for every category of every feature, in the order of the features. An "
        "unknown category produces zeros in all the columns of that feature, **except** if `drop=\"first\"`: in that case "
        "it raises `ValueError`.",
        "With `drop=\"first\"` drop the first column of each feature.",
        "Write `names(feature_names)`: the list of the column names, `featurename_category`.",
        "Check the result against `OneHotEncoder(sparse=False, handle_unknown=\"ignore\")` (on the test with the unknown "
        "category) and against `OneHotEncoder(sparse=False, drop=\"first\")` (on the clean test, without unknowns).",
    ],
    setup="""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder

        df = ds.loans(with_missing=False)
        cat = df[["city", "contract_type"]].to_numpy(dtype=object)
        cat_tr, cat_te = train_test_split(cat, test_size=0.25, random_state=0)
        cat_te_clean = cat_te.copy()
        cat_te = cat_te.copy()
        cat_te[:5, 0] = "city_XX"                          # a city never seen in training
    """,
    starter=[
        C("""
        class MyOneHot:
            def __init__(self, drop=None):
                self.drop = drop

            def fit(self, X):
                ...
                return self

            def transform(self, X):
                ...

            def names(self, feature_names):
                ...
        """),
        C("""
        mine = MyOneHot().fit(cat_tr)
        ref = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(cat_tr)
        mine_d = MyOneHot(drop="first").fit(cat_tr)
        ref_d = OneHotEncoder(sparse=False, drop="first").fit(cat_tr)
        mine.transform(cat_te).shape
        """),
    ],
    checks="""
        check(close(mine.transform(cat_tr), ref.transform(cat_tr)), "matches scikit-learn on the training set",
              "the columns must follow the order (feature, then sorted categories)")
        check(close(mine.transform(cat_te), ref.transform(cat_te)), "matches scikit-learn on the test set with the unknown city",
              "a never-seen category must give zeros in all the columns of that feature")
        check(mine.transform(cat_te)[:5, :12].sum() == 0, "the rows with the unknown city have 12 zeros in the city columns",
              "check the first five rows of the test set")
        check(list(mine.names(["city", "contract_type"])) == list(ref.get_feature_names_out(["city", "contract_type"])),
              "the column names match", "name = feature + '_' + category")
        check(close(mine_d.transform(cat_tr), ref_d.transform(cat_tr)) and mine_d.transform(cat_tr).shape[1] == 14,
              "drop='first' removes one column per feature (16 - 2 = 14)", "with drop='first' 14 columns remain")
        check(close(mine_d.transform(cat_te_clean), ref_d.transform(cat_te_clean)), "drop='first' matches on the clean test set", "review drop")
        try:
            mine_d.transform(cat_te)
            raised = False
        except ValueError:
            raised = True
        check(raised, "drop='first' with an unknown category raises ValueError",
              "with drop='first' the row of zeros would be ambiguous: raise ValueError")
    """,
    hint="""
        For each feature j: `out = (X[:, j].reshape(-1, 1) == self.categories_[j].reshape(1, -1)).astype(float)` (comparison by broadcasting);
        if `drop == "first"` discard `out[:, 1:]`. Concatenate with `np.hstack`. To recognize the unknowns: `~np.isin(X[:, j], self.categories_[j])`.
    """,
    solution=[
        C("""
        class MyOneHot:
            def __init__(self, drop=None):
                self.drop = drop

            def fit(self, X):
                self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
                return self

            def transform(self, X):
                blocks = []
                for j, cat_j in enumerate(self.categories_):
                    if self.drop == "first" and (~np.isin(X[:, j], cat_j)).any():
                        raise ValueError(f"unknown category in feature {j}: not allowed with drop='first'")
                    block = (X[:, j].reshape(-1, 1) == cat_j.reshape(1, -1)).astype(float)     # broadcasting
                    blocks.append(block[:, 1:] if self.drop == "first" else block)
                return np.hstack(blocks)

            def names(self, feature_names):
                return [f"{name}_{c}" for name, cats in zip(feature_names, self.categories_)
                        for c in (cats[1:] if self.drop == "first" else cats)]
        """),
        C("""
        mine = MyOneHot().fit(cat_tr)
        ref = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(cat_tr)
        mine_d = MyOneHot(drop="first").fit(cat_tr)
        ref_d = OneHotEncoder(sparse=False, drop="first").fit(cat_tr)
        print("shape on the test set:", mine.transform(cat_te).shape, "| with drop='first':", mine_d.transform(cat_te_clean).shape)
        print("first row of the test set (unknown city):", mine.transform(cat_te)[0, :12])
        """),
    ],
    learned="""
        - The columns are decided once and for all by `fit` on the training set. That is why `transform` always produces the same
          number of columns in the same order, even on a test set with different categories: it is the advantage of an encoder with
          state over `pd.get_dummies` (you will see it break in exercise E1).
        - An unknown category has no column: the only sensible choice is the row of zeros ("none of the known
          categories") or an error. With `drop="first"` the row of zeros would mean "first category": ambiguous, and that is
          why scikit-learn (1.1) forbids the combination.
        - `drop="first"` does not change the predictions of a linear model with intercept, but it changes the **meaning** of the
          coefficients: they are differences with respect to the dropped category (you will see it in D1).
    """,
    pitfall="""
        Building the columns by looking at the test set too (for example `np.unique` on the union). With real data the test set often contains
        categories absent from the training set: the columns must come only from the training set, otherwise in production you will have
        a different number of columns from the one the model was trained with.
    """,
    interview=(
        "A one-hot in production gives a dimension error after a new city shows up. What happened and how do you prevent it?",
        """
        Probably the columns were built with `pd.get_dummies` on the incoming data, so the number of columns
        depends on the categories present: a new city adds a column and the model, trained on a fixed number,
        refuses. I use an encoder with state (`OneHotEncoder(handle_unknown="ignore")`) inside the Pipeline: the columns are
        fixed by the training set and the new categories produce zeros. In addition I monitor their frequency as a drift signal.
        """),
)

# ---------------------------------------------------------------------------------------- A3

A3 = dict(
    id="A3", slug="target_encoding_from_scratch", kind="A", level=3, minutes=45, module=MODULE,
    title="Target encoding from scratch: smoothing and out-of-fold encoding",
    goal="""
        Implement target encoding with smoothing, in a naive version and in an out-of-fold version, and measure how much the naive version
        leaks the target: on a column of **pure noise** it should be worth 0.5 of AUC.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: `occupation` (40 levels, with a real effect on the risk) and `noise`, a
        categorical column with 300 levels assigned **at random**, with no link to the target `defaulted`. 75/25 split.
    """,
    purpose="""
        Understand why target encoding is the most powerful and the most dangerous method for high-cardinality variables: the mean of the
        target of a category computed including the row itself **contains the target of that row**.
    """,
    task=[
        "Write `te_estimate(cat, y, m)`: for each category compute `(sum_y + m · global_mean) / (count + m)` (the *smoothing* "
        "pulls the rare categories toward the global mean). Return `(mapping, overall)`, with `mapping` a dictionary category → value.",
        "Write `te_apply(cat, mapping, overall)`: it encodes an array of categories; categories not present in the mapping get "
        "`overall`.",
        "Write `te_oof(cat, y, m, k=5, seed=0)`: out-of-fold encoding. With `KFold(k, shuffle=True, random_state=seed)`, every row "
        "of a fold is encoded with the mapping computed on the **other** folds.",
        "With `m = 5` build for `occupation` and for `noise`: the naive encoding on the training set (`te_estimate` and `te_apply` "
        "on the same training set), the out-of-fold one and the one of the test set (mapping of the whole training set). Save in `table` "
        "(rows `occupation`, `noise`; columns `naive_train_auc`, `oof_train_auc`, `test_auc`) the AUC with which **each encoded "
        "column, alone**, predicts the target (`roc_auc_score(y, encoding)`).",
    ],
    setup="""
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold, train_test_split

        df = ds.loans(with_missing=False)
        y = df["defaulted"].to_numpy()
        occ = df["occupation"].to_numpy()
        noise = np.array([f"r{i}" for i in np.random.default_rng(1).integers(0, 300, len(df))])   # no link to y
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.25, random_state=0)
        y_tr, y_te = y[idx_tr], y[idx_te]
    """,
    starter=[
        C("""
        def te_estimate(cat, y, m):
            ...
            return mapping, overall


        def te_apply(cat, mapping, overall):
            ...


        def te_oof(cat, y, m, k=5, seed=0):
            ...
        """),
        C("""
        m = 5
        rows = {}
        for name, col in [("occupation", occ), ("noise", noise)]:
            c_tr, c_te = col[idx_tr], col[idx_te]
            ...
            rows[name] = {"naive_train_auc": ..., "oof_train_auc": ..., "test_auc": ...}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
    ],
    checks="""
        mapping, glob = te_estimate(np.array(["a", "a", "b"]), np.array([1, 0, 1]), m=1)
        check(close(glob, 2 / 3) and close(mapping["a"], (1 + 2 / 3) / 3) and close(mapping["b"], (1 + 2 / 3) / 2),
              "te_estimate is correct on a hand example", "with m=1: a -> (1 + 2/3)/3, b -> (1 + 2/3)/2")
        check(close(te_apply(np.array(["a", "z"]), mapping, glob), [mapping["a"], glob]),
              "te_apply uses the global mean for unknown categories", "a category not in the mapping gets `overall`")
        c_toy = np.array(list("aabbabab") * 5)
        y_toy = np.tile([1, 0, 0, 1, 1, 1, 0, 0], 5)
        enc = te_oof(c_toy, y_toy, m=2, k=5, seed=0)
        check(enc.shape == (40,) and not np.isnan(enc).any(), "te_oof returns one value per row", "one value for every row")
        check(not np.allclose(enc, te_apply(c_toy, *te_estimate(c_toy, y_toy, 2))),
              "the out-of-fold encoding does NOT match the naive one", "every row must be encoded without itself")
        check(table.loc["noise", "naive_train_auc"] > 0.75,
              "on pure noise the naive encoding seems to predict well (high training AUC): it is leakage",
              "naive_train_auc of the noise should be high: are you encoding the training set with the mapping computed on the training set?")
        check(abs(table.loc["noise", "oof_train_auc"] - 0.5) < 0.08,
              "on pure noise the out-of-fold encoding gives a training AUC close to 0.5",
              "with the out-of-fold encoding the noise should have an AUC of about 0.5")
        check(table.loc["noise", "test_auc"] < 0.65, "on the test set the noise predicts nothing",
              "the test AUC of the noise must stay close to 0.5")
        check(table.loc["occupation", "naive_train_auc"] - table.loc["occupation", "oof_train_auc"] > 0.06,
              "even on a real variable the naive encoding is inflated", "the naive training AUC must exceed the out-of-fold one by at least 0.06")
    """,
    hint="""
        With pandas: `g = pd.DataFrame({"c": cat, "y": y}).groupby("c")["y"].agg(["sum", "count"])` and then
        `((g["sum"] + m * overall) / (g["count"] + m)).to_dict()`. To apply: `pd.Series(cat).map(mapping).fillna(overall).to_numpy()`.
        For the out-of-fold: `for tr, va in KFold(...).split(cat): mapping, g = te_estimate(cat[tr], y[tr], m); out[va] = te_apply(cat[va], mapping, g)`.
    """,
    solution=[
        C("""
        def te_estimate(cat, y, m):
            d = pd.DataFrame({"c": cat, "y": y.astype(float)})
            g = d.groupby("c")["y"].agg(["sum", "count"])
            overall = d["y"].mean()
            mapping = ((g["sum"] + m * overall) / (g["count"] + m)).to_dict()
            return mapping, overall


        def te_apply(cat, mapping, overall):
            return pd.Series(cat).map(mapping).fillna(overall).to_numpy(dtype=float)


        def te_oof(cat, y, m, k=5, seed=0):
            out = np.zeros(len(cat))
            for tr, va in KFold(k, shuffle=True, random_state=seed).split(cat):
                mapping, overall = te_estimate(cat[tr], y[tr], m)             # statistics computed WITHOUT the fold `va`
                out[va] = te_apply(cat[va], mapping, overall)
            return out
        """),
        C("""
        m = 5
        rows = {}
        for name, col in [("occupation", occ), ("noise", noise)]:
            c_tr, c_te = col[idx_tr], col[idx_te]
            mapping, overall = te_estimate(c_tr, y_tr, m)
            naive_tr = te_apply(c_tr, mapping, overall)                  # the training set encoded with its own labels
            oof_tr = te_oof(c_tr, y_tr, m)
            te = te_apply(c_te, mapping, overall)
            rows[name] = {"naive_train_auc": roc_auc_score(y_tr, naive_tr), "oof_train_auc": roc_auc_score(y_tr, oof_tr),
                          "test_auc": roc_auc_score(y_te, te)}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
    ],
    learned="""
        - On pure noise, which by construction has no link to the target, the naive encoding gets a training AUC of
          about 0.85: the downstream model would "learn" to trust a column that contains nothing. The out-of-fold encoding
          brings the value back to about 0.5, and on the test set the noise is worth about 0.5-0.6.
        - Even on the real variable (`occupation`) the naive version is inflated (about 0.67 against 0.57 out-of-fold on the
          training set): the downstream model would trust this column too much and generalize worse.
        - *Smoothing* (parameter `m`) is the second defense: a category with few examples has an unreliable mean, and the
          formula pulls it toward the global mean. It does not replace the out-of-fold: it goes alongside it.
    """,
    pitfall="""
        Applying target encoding with `fit_transform` on the whole dataset before the split or the cross-validation. Even with
        smoothing, every row carries with it a part of its own target. It always goes inside the pipeline, with a transformer that
        uses the out-of-fold encoding during the fit (you will write it in exercise E3).
    """,
    interview=(
        "What is target leakage in target encoding, and how do you avoid it?",
        """
        If the encoding of a row uses the mean of the target of its category computed including the row itself, the encoded
        value contains its target: the smaller the category, the stronger the effect (with a single row, the encoding is the
        target). You avoid it with out-of-fold encoding (every row encoded with statistics computed without it), with
        smoothing toward the global mean for rare categories, and by keeping everything inside the cross-validation.
        """),
)

EXERCISES = [A1, A2, A3]
