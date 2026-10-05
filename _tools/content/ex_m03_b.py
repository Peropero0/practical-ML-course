"""Module 3 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

_SETUP_TITANIC = """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = ds.titanic()
    X, y = df.drop(columns="survived"), df["survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    NUM = ["age", "siblings_spouses", "parents_children", "fare"]
    CAT = ["sex", "port", "deck", "title"]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
"""

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="titanic_complete_pipeline", kind="B", level=1, minutes=40, module=MODULE,
    title="Titanic: a complete pipeline, three variants compared",
    goal="""
        Build with `ColumnTransformer` and `Pipeline` a model on Titanic that handles imputation, scaling and encoding by itself, evaluate it
        in cross-validation, compare three preprocessing variants and read the coefficients with their names.
    """,
    dataset="""
        `ds.titanic()`: 1309 passengers, columns `pclass`, `sex`, `age` (20% missing), `siblings_spouses`, `parents_children`, `fare`, `port`,
        `deck` (77% missing), `title` and the target `survived`. Stratified 75/25 split: `X_tr, X_te, y_tr, y_te`.
    """,
    purpose="""
        Put into practice the standard scheme of a tabular project: one preprocessor per group of columns, a pipeline, evaluation in
        cross-validation on the training set and the test set touched only once.
    """,
    task=[
        "Build `prep` (`ColumnTransformer`) with three groups: `num` (`NUM`: `SimpleImputer(median)` + `StandardScaler`), `cat` (`CAT`: "
        "`SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown=\"ignore\")`) and `pclass` (the column `pclass` with `\"passthrough\"`). "
        "Use `sparse_threshold=0` to get a dense array.",
        "Build `model = make_pipeline(prep, LogisticRegression(max_iter=2000))` and save `cv_auc` (mean AUC in cross-validation `cv` on the "
        "training set).",
        "Train on the training set and save `test_auc` (AUC on the test set). Save in `top5` the five feature names with the largest coefficient "
        "in absolute value (`model[:-1].get_feature_names_out()`).",
        "Build `table` (rows: `base`, `without_deck`, `with_age_indicator`; column `cv_auc`) comparing three variants: the base "
        "one, one in which `deck` is removed from `CAT` and one in which the imputer of `age` has `add_indicator=True`.",
    ],
    setup=_SETUP_TITANIC,
    starter=[
        C("""
        def build(num=NUM, cat=CAT, age_indicator=False):
            num_pipe = ...
            cat_pipe = ...
            prep = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat), ("pclass", "passthrough", ["pclass"])],
                                     sparse_threshold=0)
            return make_pipeline(prep, LogisticRegression(max_iter=2000))
        """),
        C("""
        model = build()
        cv_auc = ...
        model.fit(X_tr, y_tr)
        test_auc = ...
        top5 = ...
        """),
        C("""
        table = pd.DataFrame({"cv_auc": {
            "base": ...,
            "without_deck": ...,
            "with_age_indicator": ...,
        }})
        table.round(4)
        """),
    ],
    checks="""
        check(0.80 < cv_auc < 0.90, "reasonable cross-validated AUC (between 0.80 and 0.90)", "with sex, title and pclass the AUC should exceed 0.8")
        check(0.80 < test_auc < 0.93, "reasonable AUC on the test set", "check that you trained on the training set and evaluated on the test set")
        check(len(top5) == 5 and any("sex" in n or "title" in n for n in top5), "sex or title appear among the five most influential features",
              "top5 = names sorted by |coefficient|, with the names from get_feature_names_out")
        check(set(table.index) == {"base", "without_deck", "with_age_indicator"} and table["cv_auc"].notna().all(), "table with the three variants",
              "rows: base, without_deck, with_age_indicator (in any order)")
        check(close(table.loc["base", "cv_auc"], cv_auc), "the base variant matches cv_auc", "the base variant is the model of step 2")
        check(abs(table.loc["without_deck", "cv_auc"] - table.loc["base", "cv_auc"]) < 0.02,
              "removing the deck (77% missing) changes the AUC little", "the deck is almost all missing: removing it changes the AUC by less than 0.02")
    """,
    hint="""
        `num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=age_indicator), StandardScaler())`. For top5:
        `pd.Series(model[-1].coef_[0], index=model[:-1].get_feature_names_out()).abs().nlargest(5).index.tolist()`. For the variants:
        `cross_val_score(build(cat=[c for c in CAT if c != "deck"]), X_tr, y_tr, cv=cv, scoring="roc_auc").mean()`.
    """,
    solution=[
        C("""
        def build(num=NUM, cat=CAT, age_indicator=False):
            num_pipe = make_pipeline(SimpleImputer(strategy="median", add_indicator=age_indicator), StandardScaler())
            cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
            prep = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat), ("pclass", "passthrough", ["pclass"])],
                                     sparse_threshold=0)
            return make_pipeline(prep, LogisticRegression(max_iter=2000))
        """),
        C("""
        model = build()
        cv_auc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        model.fit(X_tr, y_tr)
        test_auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
        coef = pd.Series(model[-1].coef_[0], index=model[:-1].get_feature_names_out())
        top5 = coef.abs().nlargest(5).index.tolist()
        print(f"cross-validated AUC: {cv_auc:.3f} | test AUC: {test_auc:.3f}")
        print(coef[top5].round(2).to_string())
        """),
        C("""
        def auc(m):
            return cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()


        table = pd.DataFrame({"cv_auc": {
            "base": auc(build()),
            "without_deck": auc(build(cat=[c for c in CAT if c != "deck"])),
            "with_age_indicator": auc(build(age_indicator=True)),
        }})
        table.round(4)
        """),
    ],
    learned="""
        - The scheme is always the same: `ColumnTransformer` per group of columns, every group with its own mini-pipeline (imputation and then
          transformation), a final `Pipeline` with the model, cross-validation on the training set and the test set touched **once**.
        - The variants are compared with the same `cv` and the same function: here the differences (deck, `age` indicator) are minimal, so the
          choice of complexity is not justified by the data. A clean comparison also serves to **not** add useless complications.
        - The names with prefixes (`num__age`, `cat__sex_female`) let you read the model: sex and title dominate, as in the
          story of the Titanic.
    """,
    pitfall="""
        Comparing variants by looking at the test set at every trial. The test set is used once; to choose among variants you use
        cross-validation on the training set.
    """,
    interview=(
        "Why do imputation and scaling sit inside a Pipeline and not in a separate step before the split?",
        """
        Because they are steps that learn from the data (median, mean, standard deviation): if I run them before the split, the test set
        contributes to the statistics. Inside a Pipeline every cross-validation fold refits them on the training part of the fold alone, and in production
        the same object applies exactly the statistics learned during training.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="gridsearch_over_several_models", kind="B", level=2, minutes=45, module=MODULE,
    title="A single GridSearchCV to choose between different models",
    goal="""
        Use a `Pipeline` with a placeholder step and a **list of dictionaries** grid to search at the same time between two model families
        (logistic regression and random forest) with their own hyperparameters, and read `cv_results_` critically.
    """,
    dataset="""
        `ds.titanic()` with the same split and the same `ColumnTransformer` as exercise B1 (already defined as `prep` in the setup).
    """,
    purpose="""
        Learn the advanced `GridSearchCV` syntax: a replaceable step (`clf`), parameters specific to each model, and the computation
        of the number of trainings. Understand how significant a score difference is compared to the variability across folds.
    """,
    task=[
        "Build `pipe = Pipeline([(\"prep\", prep), (\"clf\", LogisticRegression())])`.",
        "Build `grid` as a list of two dictionaries: (1) `clf: [LogisticRegression(max_iter=2000)]` with `clf__C: [0.01, 0.1, 1, 10]`; "
        "(2) `clf: [RandomForestClassifier(random_state=0)]` with `clf__max_depth: [3, 6, None]` and `clf__min_samples_leaf: [1, 5]`.",
        "Run `GridSearchCV(pipe, grid, cv=cv, scoring=\"roc_auc\")` on the training set and save `search`, `n_candidates` (number of combinations), "
        "`n_trainings` (candidates × folds + 1), `best_family` (`\"LogisticRegression\"` or `\"RandomForestClassifier\"`) and `test_auc`.",
        "From `cv_results_` build `r` (DataFrame) and compute `gap`: the difference in `mean_test_score` between the first and the second ranked, "
        "and `best_std`: the `std_test_score` of the best one. Assign to `significant_difference` `True` if `gap > best_std`.",
    ],
    setup=_SETUP_TITANIC + """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import Pipeline

    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CAT),
        ("pclass", "passthrough", ["pclass"]),
    ], sparse_threshold=0)
""",
    starter=[
        C("""
        pipe = ...
        grid = [
            {...},
            {...},
        ]
        """),
        C("""
        search = ...
        n_candidates = ...
        n_trainings = ...
        best_family = ...
        test_auc = ...
        """),
        C("""
        r = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        gap = ...
        best_std = ...
        significant_difference = ...
        """),
    ],
    checks="""
        check(isinstance(grid, list) and len(grid) == 2, "grid = list of two dictionaries", "one dictionary per model family")
        check(n_candidates == 4 + 3 * 2 and n_trainings == 10 * 5 + 1, "10 candidates (4 + 6) and 51 trainings", "candidates x folds + refit")
        check(best_family in ("LogisticRegression", "RandomForestClassifier") and best_family == type(search.best_estimator_[-1]).__name__,
              "best_family consistent with best_estimator_", "use type(search.best_estimator_[-1]).__name__")
        check(0.80 < test_auc < 0.93, "reasonable test AUC", "test_auc = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])")
        check(close(gap, r["mean_test_score"].iloc[0] - r["mean_test_score"].iloc[1]) and close(best_std, r["std_test_score"].iloc[0]),
              "gap and standard deviation of the best are correct", "first two rows of r, sorted by rank")
        check(significant_difference == (gap > best_std), "judgment consistent with the numbers", "significant_difference = gap > best_std")
        check(gap < 0.02, "the gap between the first two is small (less than 0.02): the choice is between nearly equivalent candidates",
              "look at mean_test_score of the first two")
    """,
    hint="""
        Grid: `{"clf": [LogisticRegression(max_iter=2000)], "clf__C": [0.01, 0.1, 1, 10]}` and
        `{"clf": [RandomForestClassifier(random_state=0)], "clf__max_depth": [3, 6, None], "clf__min_samples_leaf": [1, 5]}`.
        `n_candidates = len(search.cv_results_["params"])`.
    """,
    solution=[
        C("""
        pipe = Pipeline([("prep", prep), ("clf", LogisticRegression())])
        grid = [
            {"clf": [LogisticRegression(max_iter=2000)], "clf__C": [0.01, 0.1, 1, 10]},
            {"clf": [RandomForestClassifier(random_state=0)], "clf__max_depth": [3, 6, None], "clf__min_samples_leaf": [1, 5]},
        ]
        """),
        C("""
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        n_candidates = len(search.cv_results_["params"])
        n_trainings = n_candidates * cv.get_n_splits() + 1
        best_family = type(search.best_estimator_[-1]).__name__
        test_auc = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])
        print(f"{n_candidates} candidates, {n_trainings} trainings | best: {best_family} {search.best_params_}")
        print(f"best_score_ {search.best_score_:.3f} | test AUC {test_auc:.3f}")
        """),
        C("""
        r = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        gap = r["mean_test_score"].iloc[0] - r["mean_test_score"].iloc[1]
        best_std = r["std_test_score"].iloc[0]
        significant_difference = bool(gap > best_std)
        print(r[["param_clf", "mean_test_score", "std_test_score", "rank_test_score"]].head(5).round(4).to_string())
        print(f"gap between first and second: {gap:.4f} | standard deviation of the best: {best_std:.4f}")
        """),
    ],
    learned="""
        - With `clf` as a placeholder step and a **list of dictionaries** you can compare model families with different hyperparameters in a
          single search, with the same preprocessing and the same folds: it is a fair comparison.
        - The count of trainings (10 candidates × 5 folds + 1) is what you should do **before** launching the search.
        - The gap between the first and the second ranked is much smaller than the standard deviation across folds: the ranking says
          little. When the candidates are statistically indistinguishable you choose the simplest and the fastest (the standard error rule,
          exercise D1).
    """,
    pitfall="""
        Forgetting `refit` with several metrics: if `scoring` is a dictionary, `refit` must say **which** metric to use to choose the
        `best_estimator_`. Without it, the search works but does not produce the final model.
    """,
    interview=(
        "How do you fairly compare a random forest and a logistic regression on a dataset with common preprocessing?",
        """
        With the same preprocessing pipeline, the same folds and a single `GridSearchCV` with a list of grids (a placeholder step for the model),
        so every family has its own hyperparameters optimized with the same cross-validation budget. Then I evaluate the best on the test set
        once, and look at whether the difference from the second exceeds the variability across folds.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="custom_features_and_saving", kind="B", level=2, minutes=40, module=MODULE,
    title="Custom features inside the pipeline, and saving the model for production",
    goal="""
        Add computed features (family size, logarithmic fare) to the pipeline with `FunctionTransformer`, save
        the whole pipeline with `joblib`, reload it and use it on a **single request** the way a production service would.
    """,
    dataset="""
        `ds.titanic()`: stratified 75/25 split as in the previous exercises. The function `add_features(df)` (to write) receives a
        DataFrame and returns a copy with three more columns.
    """,
    purpose="""
        Understand that the **feature engineering code is part of the model**: if you run it outside the pipeline, in production you have to
        rewrite it (and get it wrong). Inside the pipeline it travels with the model, and saving with `joblib` preserves it.
    """,
    task=[
        "Write `add_features(df)`: it returns a copy of `df` with `family_size = siblings_spouses + parents_children + 1`, "
        "`alone = (family_size == 1).astype(int)` and `log_fare = np.log1p(fare)`. Don't modify `df` in place.",
        "Build `pipe`: `FunctionTransformer(add_features)` as the first step, then a `ColumnTransformer` that handles `NUM + [\"family_size\", "
        "\"log_fare\"]` (median + scaler), `CAT` (mode + one-hot), and passes through `pclass` and `alone` (`\"passthrough\"`), then the logistic.",
        "Train on the training set and save `test_auc`. Check that `names` (`get_feature_names_out` of the preprocessor) contains `num__family_size`.",
        "Save the pipeline with `joblib.dump(pipe, path)` (`path` is a temporary file), reload it in `loaded` and compute "
        "`single_p`: the survival probability for **one** request (a dict turned into a one-row DataFrame, `request`).",
        "Check that `single_p` matches the probability that the original pipeline gives for the same row and save `identical`.",
    ],
    setup=_SETUP_TITANIC + """
    import tempfile
    from pathlib import Path

    import joblib
    from sklearn.preprocessing import FunctionTransformer

    path = Path(tempfile.mkdtemp()) / "titanic_model.joblib"
    request = pd.DataFrame([{"pclass": 3, "sex": "male", "age": 22.0, "siblings_spouses": 1, "parents_children": 0,
                             "fare": 7.25, "port": "S", "deck": None, "title": "Mr"}])
""",
    starter=[
        C("""
        def add_features(df):
            ...
        """),
        C("""
        pipe = make_pipeline(
            FunctionTransformer(add_features),
            ColumnTransformer([...], sparse_threshold=0),
            LogisticRegression(max_iter=2000),
        )
        pipe.fit(X_tr, y_tr)
        test_auc = ...
        names = list(pipe[1].get_feature_names_out())
        """),
        C("""
        joblib.dump(pipe, path)
        loaded = ...
        single_p = ...
        identical = ...
        """),
    ],
    checks="""
        small = pd.DataFrame({"siblings_spouses": [1, 0], "parents_children": [2, 0], "fare": [10.0, 0.0]})
        out = add_features(small)
        check(list(out["family_size"]) == [4, 1] and list(out["alone"]) == [0, 1] and close(out["log_fare"], np.log1p([10.0, 0.0])),
              "add_features computes the three columns", "family_size = siblings + parents + 1; alone = family_size == 1; log_fare = log1p(fare)")
        check("family_size" not in small.columns, "the input is not modified in place", "use df.copy()")
        check(0.80 < test_auc < 0.93, "reasonable test AUC", "test_auc = AUC on the test set of the trained pipeline")
        check("num__family_size" in names and "num__log_fare" in names and any(n.endswith("alone") for n in names),
              "the new features reach the model", "include family_size and log_fare among the numerics and alone in passthrough")
        check(path.exists() and path.stat().st_size > 0, "the joblib file exists", "joblib.dump(pipe, path)")
        check(0 <= float(single_p) <= 1, "single_p is a probability", "predict_proba(request)[0, 1]")
        check(bool(identical) and close(single_p, pipe.predict_proba(request)[0, 1]), "the reloaded pipeline gives the same result as the original",
              "compare with pipe.predict_proba(request)")
    """,
    hint="""
        `add_features`: `d = df.copy(); d["family_size"] = d.siblings_spouses + d.parents_children + 1; ...; return d`. The ColumnTransformer:
        `("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM + ["family_size", "log_fare"])`, `("cat", ..., CAT)`,
        `("pass", "passthrough", ["pclass", "alone"])`. Loading: `joblib.load(path)`.
    """,
    solution=[
        C("""
        def add_features(df):
            d = df.copy()
            d["family_size"] = d["siblings_spouses"] + d["parents_children"] + 1
            d["alone"] = (d["family_size"] == 1).astype(int)
            d["log_fare"] = np.log1p(d["fare"])
            return d
        """),
        C("""
        pipe = make_pipeline(
            FunctionTransformer(add_features),
            ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM + ["family_size", "log_fare"]),
                ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CAT),
                ("pass", "passthrough", ["pclass", "alone"]),
            ], sparse_threshold=0),
            LogisticRegression(max_iter=2000),
        )
        pipe.fit(X_tr, y_tr)
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        names = list(pipe[1].get_feature_names_out())
        print(f"test AUC: {test_auc:.3f} | columns entering the model: {len(names)}")
        """),
        C("""
        joblib.dump(pipe, path)
        loaded = joblib.load(path)                                       # in production: another process, another machine
        single_p = loaded.predict_proba(request)[0, 1]
        identical = bool(np.isclose(single_p, pipe.predict_proba(request)[0, 1]))
        print(f"survival probability for the request: {single_p:.3f} | equal to the original: {identical}")
        """),
    ],
    learned="""
        - The features computed inside the pipeline (`FunctionTransformer`) travel with the model: in production you pass the **raw**
          request and the pipeline does everything. Writing the computation twice (in training and in the service) is the first cause of
          training-serving skew.
        - `joblib.dump` saves the whole pipeline (learned preprocessing included). On loading you need the **same version** of scikit-learn and
          the function code (`add_features`) importable: it is a real limitation, to document next to the file.
        - A single request is a DataFrame with one row and the raw columns: it is the simplest test to discover hidden dependencies
          (for example a column that was always present in training).
    """,
    pitfall="""
        Transformation functions that modify the DataFrame in place. In a pipeline the input DataFrame could be the same object
        used elsewhere: `df.copy()` at the start avoids side effects that are hard to find.
    """,
    interview=(
        "What do you save to put a model in production, and what can break on loading?",
        """
        I save the whole pipeline (preprocessing and model) with `joblib`, together with the version of Python and of the libraries and a small
        set of example requests with their predictions, to use as a regression test on loading. It can break because of the
        scikit-learn version (the internal format changes), because of custom functions not importable in the service, or because of input
        columns different from those of the training set.
        """),
)

EXERCISES = [B1, B2, B3]
