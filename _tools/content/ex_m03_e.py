"""Module 3 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

_SETUP_TITANIC = """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = ds.titanic()
    X, y = df.drop(columns="survived"), df["survived"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    NUM = ["age", "siblings_spouses", "parents_children", "fare"]
    CAT = ["sex", "port", "deck", "title"]
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
"""

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="search_and_evaluation_on_the_same_data", kind="E", level=2, minutes=35, module=MODULE, start="must_fail",
    title="Optimizing and evaluating on the same data",
    goal="""
        Find out why a random forest tuned with `GridSearchCV` "gets" an AUC of 0.95 on the test set on Titanic, fix the
        scheme (split, search on the training set only, test set touched once) and measure how much the number was inflated.
    """,
    dataset="""
        `ds.titanic()` with the usual 75/25 split: `X_tr, X_te, y_tr, y_te`. `X, y` are **all** the data (training + test). The preprocessor
        and the pipeline with the random forest are in the colleague's cell.
    """,
    purpose="""
        Recognize the most common leakage scheme in tuning: after splitting the data, the search is run on the whole dataset "because
        it needs more data" and then you evaluate on a part that the search has already seen.
    """,
    task=[
        "Run the colleague's cell: `auc_bug` is very high (about 0.95) and well above the `best_score_` of its own search (about 0.86).",
        "Compute `test_share_in_training`: the fraction of rows of `X_te` that also appear in the `X` the search trained on (index). Is it 1?",
        "Fix it: run the search **only** on `X_tr, y_tr` and evaluate once on `X_te`. Save `auc_ok` and `best_ok` (the `best_score_` of the correct search).",
        "Save `inflation = auc_bug − auc_ok`: by how much the colleague's method overestimates the performance.",
    ],
    setup=_SETUP_TITANIC + """
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), CAT),
        ("pclass", "passthrough", ["pclass"]),
    ], sparse_threshold=0)
    pipe = make_pipeline(prep, RandomForestClassifier(random_state=0))
    grid = {"randomforestclassifier__max_depth": [4, 8, None], "randomforestclassifier__min_samples_leaf": [1, 3]}
""",
    starter=[
        M("### The colleague's code"),
        C("""
        bug_search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X, y)             # X, y: ALL the data
        auc_bug = roc_auc_score(y_te, bug_search.predict_proba(X_te)[:, 1])
        print(f"test set AUC: {auc_bug:.3f}   (best_score_ of the search: {bug_search.best_score_:.3f})")
        """),
        M("### Your fix"),
        C("""
        test_share_in_training = None
        auc_ok = auc_bug
        best_ok = bug_search.best_score_
        inflation = None
        """),
    ],
    checks="""
        check(auc_bug > 0.93, "the colleague's number is too good for the Titanic: it is leakage", "auc_bug should exceed 0.93; don't modify the colleague's code")
        check(test_share_in_training == 1.0, "all the rows of the test set were in the search", "test_share_in_training = X_te.index.isin(X.index).mean()")
        check(auc_ok < 0.92, "with the search on the training set only the AUC is realistic (below 0.92)", "the search must see only X_tr, y_tr")
        check(close(inflation, auc_bug - auc_ok) and inflation > 0.05, "the colleague overestimated the performance by more than 0.05 of AUC",
              "inflation = auc_bug - auc_ok")
        check(abs(best_ok - auc_ok) < 0.05, "best_score_ (cross-validation) and the correct test AUC are close", "best_ok = best_score_ of the search on the training set alone")
    """,
    hint="""
        `ok_search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)`; then `roc_auc_score(y_te, ok_search.predict_proba(X_te)[:, 1])`
        and `ok_search.best_score_`.
    """,
    solution=[
        C("""
        bug_search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X, y)             # X, y: ALL the data
        auc_bug = roc_auc_score(y_te, bug_search.predict_proba(X_te)[:, 1])
        print(f"test set AUC: {auc_bug:.3f}   (best_score_ of the search: {bug_search.best_score_:.3f})")
        """),
        M("""
        **Where the bug is.** `X, y` also contain the rows of `X_te`: the search trained (with `refit=True` the final model is trained
        on everything) on exactly the passengers on which it is then "evaluated". A random forest with no depth limit memorized them: AUC close to 1.
        It is the evaluation of a model on its own training data.
        """),
        C("""
        test_share_in_training = float(X_te.index.isin(X.index).mean())
        ok_search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)         # the search sees ONLY the training set
        auc_ok = roc_auc_score(y_te, ok_search.predict_proba(X_te)[:, 1])                       # the test set, just once
        best_ok = ok_search.best_score_
        inflation = auc_bug - auc_ok
        print(f"rows of the test set present in the colleague's search: {test_share_in_training:.0%}")
        print(f"correct test AUC: {auc_ok:.3f} | best_score_ (cv on the training set): {best_ok:.3f} | inflation: {inflation:.3f}")
        """),
    ],
    learned="""
        - With `refit=True` the search retrains the best one on **everything** it has seen. If it saw the test set, the model learned it:
          an unregularized random forest memorizes it (AUC close to 1).
        - The correct scheme has three sets with distinct roles: **training** (the search does its cross-validation inside), **test** (a single
          final evaluation), and the inner folds (disposable). The test set must not influence anything, not even the choice of hyperparameters.
        - A symptom of this bug is a score that is "too good": AUC 0.95 on the test set when the cross-validation says 0.86 (and the honest AUC is 0.88) is a warning sign. Comparing
          the test score with `best_score_` and with the baseline is the cheapest check.
    """,
    pitfall="""
        Redoing the search on all the data "after choosing" to produce the final model and then reporting the earlier test score. It is fine only
        if the reported score is the one obtained **before** retraining on everything: afterward, the test set is no longer independent.
    """,
    interview=(
        "After evaluating on the test set, can I retrain the model on training + test before putting it in production?",
        """
        Yes, but the score to report is the test set score obtained **before** retraining: it is the honest estimate of the chosen
        model. The final model trained on more data will probably be a bit better, but I no longer have an independent set to
        measure it. If I have to change something after seeing the test set, the test set is "burned" and I need a new holdout set.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="columntransformer_drops_columns", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="The ColumnTransformer that silently drops the most important columns",
    goal="""
        Find out why a model on Titanic has an AUC of 0.7 instead of 0.85 even though there are no errors: a `ColumnTransformer` with
        `remainder="drop"` (the default) dropped `sex`, `title` and `pclass`. Rebuild the lost columns and fix it.
    """,
    dataset="""
        `ds.titanic()` with the usual 75/25 split. The colleague's preprocessor handles only `["age", "siblings_spouses", "parents_children",
        "fare"]` as numeric and `["port", "deck"]` as categorical.
    """,
    purpose="""
        Recognize one of the most insidious scikit-learn bugs: **columns not listed disappear without warning**. You will learn to inspect
        `transformers_` to see what was dropped, and to put a check in the pipeline.
    """,
    task=[
        "Run the colleague's cell: `cv_auc_bug` is far below what you would expect from Titanic.",
        "Inspect `ct.transformers_` of the trained preprocessor: the last entry (`\"remainder\"`) contains the **indices** of the dropped columns. "
        "Save in `lost_columns` the sorted list of their names (`X.columns[indices]`).",
        "Fix the preprocessor: add the three missing columns to the right group (`sex`, `title` to the categoricals; `pclass` in `passthrough`) and "
        "save `cv_auc_ok` (same `cv` and same model).",
        "Write `check_columns(ct, X)` that raises `ValueError` if any column of `X` ended up in `remainder` without it having been "
        "decided (that is if `ct.transformers_[-1][0] == \"remainder\"` and it has columns). Check that it raises for the colleague's preprocessor and not for the correct one.",
    ],
    setup=_SETUP_TITANIC + """
    ct_bug = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "siblings_spouses", "parents_children", "fare"]),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["port", "deck"]),
    ], sparse_threshold=0)
""",
    starter=[
        M("### The colleague's code"),
        C("""
        bug_model = make_pipeline(ct_bug, LogisticRegression(max_iter=2000))
        cv_auc_bug = cross_val_score(bug_model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        print(f"cross-validated AUC: {cv_auc_bug:.3f}")
        """),
        M("### Your diagnosis"),
        C("""
        ct_bug.fit(X_tr, y_tr)
        lost_columns = None
        """),
        C("""
        ct_ok = None
        cv_auc_ok = cv_auc_bug
        """),
        C("""
        def check_columns(ct, X):
            ...
        """),
    ],
    checks="""
        check(cv_auc_bug < 0.78, "the colleague's model is far below expectations (AUC below 0.78)", "don't modify the colleague's code")
        check(lost_columns == ["pclass", "sex", "title"], "dropped columns: pclass, sex, title", "indices in ct_bug.transformers_[-1][2], names with X.columns[indices]")
        check(cv_auc_ok > cv_auc_bug + 0.08 and cv_auc_ok > 0.82, "including the lost columns the AUC goes up by more than 0.08 (above 0.82)",
              "add sex and title to the categoricals and pclass in passthrough")
        try:
            check_columns(ct_bug, X_tr)
            raises_bug = False
        except ValueError:
            raises_bug = True
        ct_ok.fit(X_tr, y_tr)
        try:
            check_columns(ct_ok, X_tr)
            raises_ok = False
        except ValueError:
            raises_ok = True
        check(raises_bug and not raises_ok, "check_columns raises for the colleague's preprocessor and not for the correct one",
              "raise ValueError if ct.transformers_[-1][0] == 'remainder' and the list of indices is not empty")
    """,
    hint="""
        `name, transformer, indices = ct_bug.transformers_[-1]` (name `"remainder"`); `list(X_tr.columns[indices])`. For the check:
        `if ct.transformers_[-1][0] == "remainder" and len(ct.transformers_[-1][2]) > 0: raise ValueError(...)`.
    """,
    solution=[
        C("""
        bug_model = make_pipeline(ct_bug, LogisticRegression(max_iter=2000))
        cv_auc_bug = cross_val_score(bug_model, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        print(f"cross-validated AUC: {cv_auc_bug:.3f}")
        """),
        M("""
        **Where the bug is.** The colleague's `ColumnTransformer` lists only six columns. With `remainder="drop"` (default) all the others (`pclass`,
        `sex`, `title`) are dropped without warning, and they are exactly the three most predictive features of Titanic: the model works,
        but with a third of the information.
        """),
        C("""
        ct_bug.fit(X_tr, y_tr)
        name, _, indices = ct_bug.transformers_[-1]
        lost_columns = sorted(X_tr.columns[indices])
        print("last entry of transformers_:", name, "| dropped columns:", lost_columns)
        """),
        C("""
        ct_ok = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["age", "siblings_spouses", "parents_children", "fare"]),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), ["port", "deck", "sex", "title"]),
            ("pclass", "passthrough", ["pclass"]),
        ], sparse_threshold=0)
        cv_auc_ok = cross_val_score(make_pipeline(ct_ok, LogisticRegression(max_iter=2000)), X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        print(f"cross-validated AUC: bug {cv_auc_bug:.3f} | fixed {cv_auc_ok:.3f}")
        """),
        C("""
        def check_columns(ct, X):
            name, _, indices = ct.transformers_[-1]
            if name == "remainder" and len(indices) > 0:
                raise ValueError(f"columns silently dropped: {list(X.columns[indices])}")
        """),
    ],
    learned="""
        - `ColumnTransformer` has `remainder="drop"` by default: every column not listed **disappears**, with no error or warning. The colleague's
          AUC (about 0.69) is plausible and stays in a range where nobody gets alarmed.
        - The defense is explicit: (1) list all the columns, or `remainder="passthrough"` if you know they are already numeric; (2) inspect
          `transformers_` or `get_feature_names_out()` after the fit; (3) put an automatic check (like `check_columns`) in the
          training code, which fails loudly when the schema changes.
        - In production the risk is double: if a new column arrives and is not listed, the model ignores it without saying so. A check
          on the expected input columns (`feature_names_in_`) is part of the pipeline of a serious service.
    """,
    pitfall="""
        Thinking that "if a column is not needed, I don't list it": fine, but then dropping it must be a written decision
        (explicit `remainder="drop"`, maybe with a comment), not a side effect of a default.
    """,
    interview=(
        "How do you notice that a Pipeline with ColumnTransformer is ignoring an important column?",
        """
        I look at the output names (`get_feature_names_out()`) and `transformers_`: the `remainder` entry with non-empty indices says which columns were
        dropped. In the training code I add a check that fails if the remainder is not empty, and in the service I verify
        the input columns against `feature_names_in_`. An AUC lower than expected with no errors is often the first signal.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="kfold_on_sorted_data_and_error_score", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="KFold on data sorted by class and `error_score=0` hiding everything",
    goal="""
        Understand why a cross-validation returns an AUC of 0 on all the folds without raising errors, and fix the splitter.
        The colleague's code has two defects that hide each other: data sorted by target and `error_score=0`.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns and the target `defaulted`, but with the rows **sorted by target** (first all the 0s,
        then all the 1s), as happens when two extracts are concatenated. Model: logistic regression with scaler.
    """,
    purpose="""
        Recognize two behaviors that combine badly: `KFold` without shuffling creates **contiguous** folds, and `error_score=0` turns
        errors into scores. The result is a number (0.0) that looks like information but is a hidden error.
    """,
    task=[
        "Run the colleague's cell: `auc_bug`. Is it 0? Why are there no visible errors?",
        "Repeat with `error_score=\"raise\"` inside a `try/except` and save in `message` the first line of the error message (string).",
        "Save `bug_positives_per_fold`: the list of the positives (`y_sorted == 1`) in the **validation** fold of each of the 5 folds of `KFold(5)`.",
        "Fix it with `StratifiedKFold(5, shuffle=True, random_state=0)` **without** `error_score`: save `auc_ok` and `ok_positives_per_fold`.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False).sort_values("defaulted").reset_index(drop=True)      # rows sorted by target
        X_sorted = df[["age", "income", "job_tenure", "amount"]].to_numpy()
        y_sorted = df["defaulted"].to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        bug_scores = cross_val_score(model, X_sorted, y_sorted, cv=KFold(5), scoring="roc_auc", error_score=0)
        auc_bug = bug_scores.mean()
        print("AUC per fold:", bug_scores.round(3), "| mean:", round(auc_bug, 3))
        """),
        M("### Your diagnosis"),
        C("""
        message = None
        bug_positives_per_fold = None
        auc_ok = auc_bug
        ok_positives_per_fold = None
        """),
    ],
    checks="""
        check(auc_bug == 0.0, "the colleague's number is 0.0 on all the folds: they are errors turned into scores", "don't modify the colleague's code")
        check(isinstance(message, str) and ("class" in message.lower() or "one" in message.lower()),
              "the error message talks about a single class", "with error_score='raise' cross_val_score raises a ValueError about the classes")
        check(bug_positives_per_fold is not None and sorted(bug_positives_per_fold)[:4] == [0, 0, 0, 0] and max(bug_positives_per_fold) > 200,
              "four validation folds have no positives at all, the last one almost only positives", "with KFold without shuffle the folds are contiguous blocks of sorted rows")
        check(0.7 < auc_ok < 0.85, "with stratification the AUC is the true one (between 0.7 and 0.85)", "StratifiedKFold(5, shuffle=True, random_state=0) without error_score")
        check(ok_positives_per_fold is not None and max(ok_positives_per_fold) - min(ok_positives_per_fold) <= 1,
              "with stratification the positives are evenly distributed (difference at most 1)", "every fold must have about 19% positives")
    """,
    hint="""
        `try: cross_val_score(model, X_sorted, y_sorted, cv=KFold(5), scoring="roc_auc", error_score="raise") except ValueError as e: message = str(e).splitlines()[0]`.
        Positives: `[int(y_sorted[va].sum()) for _, va in KFold(5).split(X_sorted)]`.
    """,
    solution=[
        C("""
        bug_scores = cross_val_score(model, X_sorted, y_sorted, cv=KFold(5), scoring="roc_auc", error_score=0)
        auc_bug = bug_scores.mean()
        print("AUC per fold:", bug_scores.round(3), "| mean:", round(auc_bug, 3))
        """),
        M("""
        **Where the bug is.** With the rows sorted by target, `KFold(5)` (without shuffling) creates five contiguous blocks: in the first four validation
        folds there is a single class (the AUC is not defined) and in the training set of the fifth almost all the examples are of the same class (the
        model cannot even be trained). Every fold raises an error, and `error_score=0` silently turns it into the score 0.
        """),
        C("""
        try:
            cross_val_score(model, X_sorted, y_sorted, cv=KFold(5), scoring="roc_auc", error_score="raise")
        except ValueError as error:
            message = str(error).splitlines()[0]
        print("real error:", message)

        bug_positives_per_fold = [int(y_sorted[va].sum()) for _, va in KFold(5).split(X_sorted)]
        print("positives in the validation folds (KFold without shuffle):", bug_positives_per_fold)
        """),
        C("""
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        auc_ok = cross_val_score(model, X_sorted, y_sorted, cv=skf, scoring="roc_auc").mean()
        ok_positives_per_fold = [int(y_sorted[va].sum()) for _, va in skf.split(X_sorted, y_sorted)]
        print(f"correct AUC: {auc_ok:.3f} | positives per fold: {ok_positives_per_fold}")
        """),
    ],
    learned="""
        - `KFold` without `shuffle` divides into contiguous blocks: with rows sorted by class, by date or by group the folds are not
          representative. For classification you use `StratifiedKFold(shuffle=True)`, which is also the default of `cross_val_score` when you pass
          an integer (but not if you build `KFold` by hand).
        - `error_score=0` (or `np.nan`) turns errors into numbers. It is useful in a search where some combinations may fail, and
          very dangerous elsewhere: an error you don't see is an error you don't fix. Use `"raise"` while you are developing.
        - A score of exactly 0.0 (or exactly 0.5) on all the folds is almost always a mistake, not a performance.
    """,
    pitfall="""
        Sorting or grouping the data "for convenience" (by class, by customer, by date) before using `KFold` without shuffling. Cross-validation
        assumes the rows are interchangeable: if they are not, the splitter must be chosen deliberately (groups, time, stratification).
    """,
    interview=(
        "When is it legitimate to use `error_score=np.nan` in a hyperparameter search?",
        """
        When some combinations can legitimately fail (for example a solver that does not converge or an incompatible combination of parameters)
        and I don't want to interrupt the whole search. The failed combinations get NaN and are excluded from the ranking. But I always check how many
        failed (`cv_results_`) and investigate: if they all fail, or one particular fold fails, the error has a cause (data, splitter) that I am hiding.
        """),
)

EXERCISES = [E1, E2, E3]
