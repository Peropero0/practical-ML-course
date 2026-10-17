"""Module 3 lessons: pipelines, cross-validation and hyperparameter search."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_03_pipeline_and_validation"
TITLE = "Pipelines, cross-validation and hyperparameter search"
LESSONS = [
    ("01_pipeline_and_columntransformer.ipynb", "3.1", "Pipeline and ColumnTransformer",
     "A single object for preprocessing and model: how to build it, inspect it and modify it."),
    ("02_cross_validation_and_leakage.ipynb", "3.2", "Cross-validation and leakage",
     "K-fold, stratified, grouped and temporal: the splitter must mirror the real use of the model."),
    ("03_hyperparameter_search.ipynb", "3.3", "Hyperparameter search",
     "GridSearchCV, RandomizedSearchCV, `cv_results_` and why `best_score_` is optimistic."),
]
PATH = [
    ("Lesson 3.1", ["A1", "B1", "B3", "E2"]),
    ("Lesson 3.2", ["A2", "E3", "C1", "C2", "F1"]),
    ("Lesson 3.3", ["A3", "B2", "D1", "D2", "E1", "C3", "F2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_3_1() -> List[Cell]:
    """Cells of lesson 3.1."""
    return [
        M("""
        # Lesson 3.1 · Pipeline and ColumnTransformer

        In module 2 you prepared the data one column type at a time and saw how many silent mistakes come from it: a
        `fit` on the test set, columns in the wrong order, an encoder that is not the one from training. The scikit-learn `Pipeline`
        solves the problem at the root: the whole path from raw data to prediction becomes **a single object**.

        **What you will learn**

        - To build a `Pipeline` and understand what happens at every `fit` and every `predict`.
        - To handle columns of different types with a `ColumnTransformer`, even with nested pipelines.
        - To inspect and modify the parameters of a step without rebuilding anything.
        """),
        _setup("""
        from sklearn.compose import ColumnTransformer, make_column_selector
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        df = ds.titanic()
        X, y = df.drop(columns="survived"), df["survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        print(X.shape, "| survival rate:", round(y.mean(), 3))
        X.head(3)
        """),
        M("""
        ## The problem: a dataset with everything you need to go wrong

        Titanic has numerics with missing values (`age`), categoricals with missing values (`deck`, `port`), a discrete feature (`pclass`) and columns with
        different scales (`fare`). To train a model you need to impute, scale, encode, in the right order and with the
        statistics of the training set alone. A `Pipeline` lines everything up.
        """),
        M("""
        ## Pipeline: `fit` and `predict` in a row

        ```
        pipe.fit(X, y):      step1.fit_transform(X, y) → step2.fit_transform(...) → ... → model.fit(...)
        pipe.predict(X_new): step1.transform(X_new)    → step2.transform(...)    → ... → model.predict(...)
        ```

        In `fit` every intermediate step learns from the data and passes it to the next; in `predict` the same steps apply **only**
        `transform`. It is impossible to refit a scaler on the test set by mistake.
        """),
        C("""
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ])
        numeric_only = ["pclass", "age", "siblings_spouses", "parents_children", "fare"]
        pipe.fit(X_tr[numeric_only], y_tr)
        print("test AUC (numerics only):", round(roc_auc_score(y_te, pipe.predict_proba(X_te[numeric_only])[:, 1]), 3))
        print("steps:", list(pipe.named_steps))
        print("median of age learned from the TRAINING set:", pipe.named_steps["imputer"].statistics_[1])
        """),
        M("""
        ## ColumnTransformer: one treatment per group of columns

        Every row of the list is `(name, transformer, columns)`. Columns are given by name, by position or with a *selector*
        (`make_column_selector(dtype_include=...)`). Every transformer can itself be a `Pipeline`: here every group has
        imputation first and then the transformation.
        """),
        C("""
        numeric = ["age", "siblings_spouses", "parents_children", "fare"]
        categorical = ["sex", "port", "deck", "title"]

        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical),
            ("pclass", "passthrough", ["pclass"]),                    # already 1, 2, 3: we leave it as it is
        ], sparse_threshold=0)

        model = make_pipeline(prep, LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
        print("test AUC:", round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 3))
        """),
        M("""
        > **Warning.** The columns **not listed** in a `ColumnTransformer` are silently dropped (`remainder="drop"`, the default).
        > To keep them use `remainder="passthrough"`. It is one of the bugs you will find in exercise E2.

        ## Inspecting and modifying

        A pipeline is an inspectable object: you access the steps by name, take a piece with slicing and change the
        parameters with the `step__parameter` syntax (which is the same one used by `GridSearchCV` in lesson 3.3).
        """),
        C("""
        print("feature names after the preprocessing:", len(model[:-1].get_feature_names_out()))
        print(list(model[:-1].get_feature_names_out())[:6], "...")

        model.set_params(logisticregression__C=0.1)                   # change a parameter of a nested step
        model.fit(X_tr, y_tr)
        coef = pd.Series(model[-1].coef_[0], index=model[:-1].get_feature_names_out())
        coef.abs().nlargest(4).round(2)
        """),
        M("""
        ## Columns by type, without listing them

        `make_column_selector` chooses the columns by data type: useful when the schema changes, less so when you want explicit
        control (in production an explicit list is often preferable, because a badly inferred type changes the model silently).
        """),
        C("""
        prep2 = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), make_column_selector(dtype_include="number")),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
             make_column_selector(dtype_include=object)),
        ], sparse_threshold=0)
        p2 = make_pipeline(prep2, LogisticRegression(max_iter=2000)).fit(X_tr, y_tr)
        print("test AUC with the selectors:", round(roc_auc_score(y_te, p2.predict_proba(X_te)[:, 1]), 3))
        """),
        M("> **Common mistake.** Fitting the scaler or the imputer on the whole dataset **before** splitting or doing the cross-validation: the statistics of the validation fold enter the training. Inside a `Pipeline` it is impossible to do it; outside it is very easy (exercises E1 and E3 of module 1), and a `ColumnTransformer` that silently drops the columns not listed is the twin mistake (exercise E2)."),
        M("## Check your understanding\n\n" + details(
            "Why does the pipeline avoid preprocessing leakage?",
            "Because `fit` calls `fit_transform` on the intermediate steps **only on the training data it receives**, and `predict` calls only `transform`. "
            "Inside `cross_val_score` or `GridSearchCV` the pipeline is cloned and refitted at every fold on the training part of the fold alone: the "
            "statistics of the validation fold never enter the preprocessing.") + "\n\n" + details(
            "When is `make_column_selector` convenient and when the explicit list?",
            "The selector is handy in exploration. In production an explicit list is safer: if a numeric column arrives by "
            "mistake as text, a selector silently moves it to the group of categoricals and the model changes without warning.")),
    ]


def lesson_3_2() -> List[Cell]:
    """Cells of lesson 3.2."""
    return [
        M("""
        # Lesson 3.2 · Cross-validation and leakage

        Cross-validation estimates how well a model generalizes, but **only if the scheme for splitting the data mirrors the real use of the
        model**. If the rows of a validation fold look too much like those of training (same person, same period) the
        estimate is inflated, even if every step looks correct.

        **What you will learn**

        - K-fold, stratified and repeated, and what `cross_val_score`, `cross_validate` and `cross_val_predict` measure.
        - Why `GroupKFold` and `TimeSeriesSplit` are needed, and how much it costs to use the wrong splitter.
        """),
        _setup("""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import (GroupKFold, KFold, RepeatedStratifiedKFold, StratifiedKFold, TimeSeriesSplit,
                                             cross_val_predict, cross_val_score, cross_validate)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        d = ds.loans(with_missing=False)
        X = d[["age", "income", "job_tenure", "amount"]]
        y = d["defaulted"]
        """),
        M("""
        ## K-fold and stratification

        With imbalanced classes (here defaults are about 19%) a plain K-fold can produce folds with different proportions.
        `StratifiedKFold` keeps the proportion of the classes in every fold. For a classification it is the default of `cross_val_score`
        when you pass an integer.
        """),
        C("""
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        for name, cv in [("KFold(5)", KFold(5, shuffle=True, random_state=0)),
                         ("StratifiedKFold(5)", StratifiedKFold(5, shuffle=True, random_state=0)),
                         ("repeated 5x3", RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0))]:
            s = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
            print(f"{name:20s} AUC {s.mean():.3f} +/- {s.std():.3f}  ({len(s)} evaluations)")
        """),
        M("""
        ## More metrics, and the out-of-fold predictions

        `cross_validate` computes several metrics and can also return the scores on the training set (`return_train_score=True`): the gap between
        training and validation is the signal of overfitting. `cross_val_predict` returns for every row the prediction made by the
        model that **did not** see it (out-of-fold predictions), useful for analyzing errors and for stacking.
        """),
        C("""
        r = cross_validate(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           scoring={"auc": "roc_auc", "precision": "precision"}, return_train_score=True)
        print({k: round(v.mean(), 3) for k, v in r.items() if k.startswith(("test", "train"))})
        oof = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
        print("out-of-fold predictions:", oof.shape, "| overall AUC:", round(roc_auc_score(y, oof), 3))
        """),
        M("""
        ## When the random splitter lies: repeated measurements

        Sixty patients, ten measurements each. The measurements of the same patient are almost identical to each other (they are their
        "fingerprint"), and the outcome depends mostly on an individual characteristic **that no measurement explains**. A model that recognizes
        the patient from the fingerprint remembers their outcome.
        """),
        C("""
        p = ds.patient_measurements()
        Xp, yp = p[["m1", "m2", "m3"]], p["outcome"]
        rf = RandomForestRegressor(100, random_state=0)
        random_cv = cross_val_score(rf, Xp, yp, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
        by_group = cross_val_score(rf, Xp, yp, cv=GroupKFold(5), groups=p["patient"], scoring="r2")
        print(f"random KFold : R2 {random_cv.mean():.2f}   <- the same patient is in training and in validation")
        print(f"GroupKFold   : R2 {by_group.mean():.2f}   <- new patients in validation: the real use situation")
        """),
        M("""
        With a new patient the model has nothing to remember and the R² is negative. The correct estimate is the second: if in production
        never-seen patients will arrive, the validation must use never-seen patients. `GroupKFold` guarantees that all the rows of a
        group stay in the same fold.

        ## When the random splitter lies: time
        """),
        C("""
        s = ds.time_series()
        Xs, ys = s[["t", "s", "c"]], s["y"]
        random_cv = cross_val_score(rf, Xs, ys, cv=KFold(5, shuffle=True, random_state=0), scoring="r2")
        temporal = cross_val_score(rf, Xs, ys, cv=TimeSeriesSplit(5), scoring="r2")
        print(f"random KFold    : R2 {random_cv.mean():.2f}   <- interpolates between the past and the immediate future")
        print(f"TimeSeriesSplit : R2 {temporal.mean():.2f}   <- trains on the past, validates on the future")
        """),
        M("""
        With a random KFold every validation month has its neighbors in the training set: the model interpolates. In production it will have to predict
        the future without having seen anything of what comes later. `TimeSeriesSplit` always trains on the past and validates on the future.

        > **Rule of thumb.** Ask yourself "how will this model be used?" and build the splitter so that the validation imitates it:
        > new customers → groups by customer; future → temporal order; rare classes → stratification.

        > **Common mistake.** Reading a cross-validation score as the quality of the model without asking whether the folds look like the real use of the model. With repeated patients or a time order, the random `KFold` above gave R² 0.75 and 0.97 where the right splitters gave -0.77 and 0.23.
        """),
        M("## Check your understanding\n\n" + details(
            "Why is `return_train_score=True` useful?",
            "Because the gap between the training score and the validation score measures overfitting: an AUC of 0.99 in training and 0.70 in "
            "validation says that the model memorizes. It costs a bit more computation time, which is why it is off by default.") + "\n\n" + details(
            "I have 5 folds and I get 5 scores. What is the standard error of my estimate?",
            "It is not the standard deviation divided by the square root of 5: the folds are not independent (they share training data). The "
            "standard deviation across folds is an indicator of stability, not a confidence interval. Repeating the cross-validation with different "
            "splits reduces the noise due to the random partition, but the repetitions still use the same data, so it does not turn that spread into a valid "
            "confidence interval (corrected estimators exist, such as Nadeau and Bengio's). To compare two models, use the same folds and look at the differences fold by fold.")),
    ]


def lesson_3_3() -> List[Cell]:
    """Cells of lesson 3.3."""
    return [
        M("""
        # Lesson 3.3 · Hyperparameter search

        A hyperparameter (the depth of a tree, the strength of the regularization, the number of neighbors) is not learned from the data: it is
        chosen by trying. `GridSearchCV` and `RandomizedSearchCV` automate the trial with cross-validation. This lesson shows
        how to use them on a whole pipeline, how to read the results and why the score of the search is optimistic.

        **What you will learn**

        - The `step__parameter` syntax to search inside a pipeline.
        - To read `cv_results_` instead of trusting only `best_params_`.
        - Why the final score is measured on data that the search has never seen.
        """),
        _setup("""
        from scipy.stats import loguniform
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        df = ds.titanic()
        X, y = df.drop(columns="survived"), df["survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        num = ["pclass", "age", "siblings_spouses", "parents_children", "fare"]
        cat = ["sex", "port", "deck", "title"]
        prep = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
            ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), cat),
        ], sparse_threshold=0)
        pipe = make_pipeline(prep, LogisticRegression(max_iter=2000))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        """),
        M("""
        ## GridSearchCV on a pipeline

        The name of a parameter inside a pipeline is `stepname__parameter` (at several levels: `columntransformer__num__simpleimputer__strategy`).
        `make_pipeline` generates the names in lowercase from the type of the object. With `refit=True` (default) the search retrains the
        best one on the whole training set: `best_estimator_`.
        """),
        C("""
        grid = {
            "logisticregression__C": [0.01, 0.1, 1, 10],
            "columntransformer__num__simpleimputer__strategy": ["mean", "median"],
        }
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        print("best parameters:", search.best_params_)
        print("best_score_ (cross-validation, on the training set):", round(search.best_score_, 3))
        print("AUC on the test set (never seen by the search)    :", round(roc_auc_score(y_te, search.predict_proba(X_te)[:, 1]), 3))
        """),
        M("""
        ## Reading `cv_results_`

        `best_params_` gives a winner; `cv_results_` tells the story of the race. Look at it as a table: small differences compared to the standard
        deviation across folds are not real differences.
        """),
        C("""
        r = pd.DataFrame(search.cv_results_)
        r[["param_logisticregression__C", "param_columntransformer__num__simpleimputer__strategy",
           "mean_test_score", "std_test_score", "rank_test_score"]].sort_values("rank_test_score").head(6).round(4)
        """),
        M("""
        ## Random search

        With many hyperparameters the grid explodes: 5 values for 6 parameters are 15,625 combinations. `RandomizedSearchCV` samples a
        fixed number (`n_iter`) from distributions, much more efficient when only a few parameters really matter. For
        parameters that vary by orders of magnitude you sample on a logarithmic scale (`loguniform`).
        """),
        C("""
        pipe_lr = make_pipeline(prep, LogisticRegression(max_iter=2000))
        search_c = RandomizedSearchCV(pipe_lr, {"logisticregression__C": loguniform(1e-3, 1e2)}, n_iter=12, cv=cv, scoring="roc_auc", random_state=0).fit(X_tr, y_tr)
        tried = search_c.cv_results_["param_logisticregression__C"].astype(float)
        print("values of C tried:", np.sort(tried).round(4))
        print("best C:", round(search_c.best_params_["logisticregression__C"], 4), "| best_score_", round(search_c.best_score_, 3))
        """),
        M("""
        The 12 values of `C` are spread over several orders of magnitude (from 0.08 to 66), a few in every decade: that is what `loguniform` does. A uniform distribution between 0 and 100 would put nine values out of ten above 10
        and would almost never try the small values, where the best one is here (0.44). With several hyperparameters, `RandomizedSearchCV` does the same for each of them.
        """),
        C("""
        pipe_rf = make_pipeline(prep, RandomForestClassifier(random_state=0))
        distributions = {"randomforestclassifier__n_estimators": [100, 200, 400],
                         "randomforestclassifier__max_depth": [3, 5, 8, None],
                         "randomforestclassifier__min_samples_leaf": [1, 3, 5, 10]}
        rnd = RandomizedSearchCV(pipe_rf, distributions, n_iter=12, cv=cv, scoring="roc_auc", random_state=0).fit(X_tr, y_tr)
        print("best:", rnd.best_params_)
        print("best_score_ ", round(rnd.best_score_, 3), "| test", round(roc_auc_score(y_te, rnd.predict_proba(X_te)[:, 1]), 3))
        """),
        M("""
        ## Why `best_score_` is optimistic

        `best_score_` is the **maximum** among many noisy estimates on the same folds: even if all the configurations were equivalent, the
        best one would have a score above the mean by pure luck. The more configurations you try, the more inflated the estimate. The number to
        report is the one on the test set (or a nested cross-validation), touched **once**, after the search.
        In this example the test set is even a bit higher than `best_score_`: with 328 rows the estimate has an uncertainty of about
        ±0.03, so the optimism shows up **on average** over many repetitions (you will measure it in exercise C3), not in every single case.

        > **Common mistake.** Looking at the test set, changing something in the pipeline and looking at it again. After a couple of cycles the test set
        > has become a second validation set and no longer estimates anything.
        """),
        M("## Check your understanding\n\n" + details(
            "Does `refit=True` have an effect on the reported score?",
            "No: `best_score_` is the cross-validated mean and does not change. `refit=True` retrains the best model on the whole training set "
            "(to use it with `predict`): that is the model you evaluate on the test set.") + "\n\n" + details(
            "When do I prefer RandomizedSearchCV to GridSearchCV?",
            "When I have more than two or three hyperparameters or continuous parameters: random search explores many more distinct values of the important "
            "parameters for the same number of trainings, because it does not repeat the same values in every combination.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_pipeline_and_columntransformer.ipynb", lesson_3_1),
                    ("02_cross_validation_and_leakage.ipynb", lesson_3_2),
                    ("03_hyperparameter_search.ipynb", lesson_3_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
