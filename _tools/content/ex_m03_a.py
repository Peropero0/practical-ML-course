"""Module 3 - type A exercises: implement from scratch."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

# ---------------------------------------------------------------------------------------- A1

A1 = dict(
    id="A1", slug="pipeline_from_scratch", kind="A", level=2, minutes=35, module=MODULE,
    title="A Pipeline from scratch: who calls `fit` and who calls `transform`",
    goal="""
        Implement a `MyPipeline` class with `fit`, `predict`, `predict_proba`, `named_steps` and `set_params`, check it against
        `sklearn.pipeline.Pipeline` and show, by counting the calls, that in `predict` no intermediate step is refitted.
    """,
    dataset="""
        `ds.titanic()`: the five numeric columns (`pclass`, `age`, `siblings_spouses`, `parents_children`, `fare`; `age` has missing values)
        and the target `survived`. Stratified 75/25 split. The `Spy` class (defined in the setup) wraps a scaler and counts
        how many times `fit_transform` and `transform` are called.
    """,
    purpose="""
        Take the `Pipeline` apart to understand that it has nothing magic: it is a loop that calls `fit_transform` on the intermediate steps in
        training and `transform` in prediction. All the protection against leakage comes from this simple rule.
    """,
    task=[
        "Write `MyPipeline(steps)`, where `steps` is a list of pairs `(name, object)`. In `__init__` save `steps` and create "
        "`named_steps`, a dictionary name → object.",
        "`fit(X, y)`: for every step **except the last** replace `X` with `step.fit_transform(X, y)`; then call `fit(X, y)` on the last one. "
        "Return `self`.",
        "`predict(X)` and `predict_proba(X)`: for every step except the last apply **only** `transform`; then call the method of the last one.",
        "`set_params(**kw)`: for every key `\"step__parameter\"` call `set_params(parameter=value)` on the right step. Return `self`.",
        "Build `mine` with three steps (`imputer`, `scaler` = `Spy(StandardScaler())`, `model`), train it on `X_tr`, call `predict_proba` "
        "on `X_te` **three times** and save in `spy_fit_transform` and `spy_transform` the counters of the scaler.",
    ],
    setup="""
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.titanic()
        NUM = ["pclass", "age", "siblings_spouses", "parents_children", "fare"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[NUM], df["survived"], test_size=0.25, stratify=df["survived"], random_state=0)


        class Spy:
            \"\"\"Wraps a transformer and counts the calls to fit_transform and transform.\"\"\"

            def __init__(self, t):
                self.t, self.n_fit_transform, self.n_transform = t, 0, 0

            def fit_transform(self, X, y=None):
                self.n_fit_transform += 1
                return self.t.fit_transform(X, y)

            def transform(self, X):
                self.n_transform += 1
                return self.t.transform(X)

            def fit(self, X, y=None):                      # a correct pipeline must never use it
                raise RuntimeError("the pipeline must not call fit on the intermediate steps")
    """,
    starter=[
        C("""
        class MyPipeline:
            def __init__(self, steps):
                self.steps = steps
                self.named_steps = ...

            def fit(self, X, y):
                ...
                return self

            def predict(self, X):
                ...

            def predict_proba(self, X):
                ...

            def set_params(self, **kw):
                ...
                return self
        """),
        C("""
        spy = Spy(StandardScaler())
        mine = MyPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", spy),
                           ("model", LogisticRegression(max_iter=1000))])
        mine.fit(X_tr, y_tr)
        for _ in range(3):
            p = mine.predict_proba(X_te)
        spy_fit_transform = ...
        spy_transform = ...
        """),
    ],
    checks="""
        ref = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                        ("model", LogisticRegression(max_iter=1000))]).fit(X_tr, y_tr)
        check(close(mine.predict_proba(X_te), ref.predict_proba(X_te)), "the probabilities match those of sklearn.Pipeline",
              "MyPipeline.predict_proba must give the same result as Pipeline")
        check(close(mine.predict(X_te), ref.predict(X_te)), "the predicted classes match", "predict must use the same transformations")
        check(list(mine.named_steps) == ["imputer", "scaler", "model"], "named_steps is a dictionary name -> object", "named_steps = dict(steps)")
        check(spy_fit_transform == 1, "in training the scaler did fit_transform just once", "fit_transform must be called once, in fit")
        check(spy_transform == 3, "in the three predictions the scaler did only transform (3 times)", "in predict only transform is called: once per prediction")
        mine.set_params(model__C=0.01)
        check(mine.named_steps["model"].C == 0.01, "set_params reaches the parameters of the last step", "set_params('model__C') must modify the model")
        mine.fit(X_tr, y_tr)
        check(not np.allclose(mine.predict_proba(X_te), ref.predict_proba(X_te)), "after set_params and a new fit the model changes", "small C = strong regularization: the probabilities change")
    """,
    hint="""
        `fit`: `Xt = X; for name, step in self.steps[:-1]: Xt = step.fit_transform(Xt, y)`; then `self.steps[-1][1].fit(Xt, y)`. For `set_params`:
        `name, parameter = key.split("__", 1)` and `self.named_steps[name].set_params(**{parameter: value})`.
    """,
    solution=[
        C("""
        class MyPipeline:
            def __init__(self, steps):
                self.steps = steps
                self.named_steps = dict(steps)

            def _transform(self, X):
                for _, step in self.steps[:-1]:
                    X = step.transform(X)                       # in prediction: ONLY transform
                return X

            def fit(self, X, y):
                for _, step in self.steps[:-1]:
                    X = step.fit_transform(X, y)                # in training: fit_transform on the intermediate steps
                self.steps[-1][1].fit(X, y)
                return self

            def predict(self, X):
                return self.steps[-1][1].predict(self._transform(X))

            def predict_proba(self, X):
                return self.steps[-1][1].predict_proba(self._transform(X))

            def set_params(self, **kw):
                for key, value in kw.items():
                    name, parameter = key.split("__", 1)
                    self.named_steps[name].set_params(**{parameter: value})
                return self
        """),
        C("""
        spy = Spy(StandardScaler())
        mine = MyPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", spy),
                           ("model", LogisticRegression(max_iter=1000))])
        mine.fit(X_tr, y_tr)
        for _ in range(3):
            p = mine.predict_proba(X_te)
        spy_fit_transform = spy.n_fit_transform
        spy_transform = spy.n_transform
        print("fit_transform:", spy_fit_transform, "| transform:", spy_transform)
        """),
    ],
    learned="""
        - The rule is just one: **`fit_transform` in training, `transform` in prediction**. It is enough to guarantee that no statistic
          of the test set enters the preprocessing, because in prediction nothing gets computed.
        - `GridSearchCV` and `cross_val_score` use the same rule: they clone the pipeline and refit it on the training part of every fold alone. Preprocessing
          leakage becomes impossible by construction.
        - `set_params` with the `step__parameter` syntax is what makes the hyperparameter search inside the pipeline possible (lesson 3.3).
    """,
    pitfall="""
        Calling `fit` (instead of `fit_transform`) on the intermediate steps and then `transform` on the same data is equivalent. Calling
        `fit_transform` in **prediction** is not: it recomputes the statistics on the batch it receives, the bug of exercise E3 of module 2.
    """,
    interview=(
        "What happens, step by step, when you call `cross_val_score(pipeline, X, y, cv=5)`?",
        """
        At every fold `cross_val_score` clones the pipeline (parameters, not state), trains it on the training part of the fold (`fit`: the intermediate
        steps do `fit_transform`, the last one `fit`) and evaluates it on the validation fold (`predict`: only `transform` and then
        `predict`). The preprocessing statistics are therefore recomputed 5 times, each only on the training data of the fold: it is the
        reason why there is no leakage.
        """),
)

# ---------------------------------------------------------------------------------------- A2

A2 = dict(
    id="A2", slug="kfold_and_cross_val_score_from_scratch", kind="A", level=2, minutes=40, module=MODULE,
    title="K-fold, stratified K-fold and `cross_val_score` from scratch",
    goal="""
        Implement the generation of folds (identical to scikit-learn's `KFold`), a stratified version and a
        `my_cross_val_score` function, and verify that with the same folds they produce the same scores.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns and the target `defaulted` (about 19% positives). Model: logistic
        regression with a scaler in a pipeline.
    """,
    purpose="""
        Understand what cross-validation really does (partition, train from scratch on each training set, evaluate on the rest) and how
        stratification keeps the class proportions.
    """,
    task=[
        "Write `kfold_indices(n, k, seed)`: use `np.random.RandomState(seed).shuffle` on `np.arange(n)`; the folds have size `n//k` and the "
        "first `n % k` have one more. Return the list of pairs `(train_idx, val_idx)` with **sorted** indices. It must match "
        "`KFold(k, shuffle=True, random_state=seed)`.",
        "Write `strat_indices(y, k, seed)`: for every class shuffle its indices (`np.random.RandomState(seed)`) and assign them to the folds in turn "
        "(`i % k`). Return the list of pairs `(train_idx, val_idx)` sorted. In every fold the number of elements of each class must not "
        "differ by more than 1 from that of the other folds.",
        "Write `my_cross_val_score(model, X, y, folds)`: for every pair clone the model with `sklearn.base.clone`, train it on the training set, "
        "compute the AUC (`roc_auc_score` with `predict_proba`) on the validation fold and return the array of scores.",
        "Compute `my_scores = my_cross_val_score(model, X, y, kfold_indices(len(y), 5, 0))` and compare them with scikit-learn's `cross_val_score` "
        "passing it the **same** folds as `cv`.",
    ],
    setup="""
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]].to_numpy()
        y = df["defaulted"].to_numpy()
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    """,
    starter=[
        C("""
        def kfold_indices(n, k, seed):
            ...


        def strat_indices(y, k, seed):
            ...


        def my_cross_val_score(model, X, y, folds):
            ...
        """),
        C("""
        folds = kfold_indices(len(y), 5, 0)
        my_scores = ...
        sk_scores = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
        print(my_scores.round(4), sk_scores.round(4))
        """),
    ],
    checks="""
        ref = list(KFold(5, shuffle=True, random_state=0).split(np.arange(len(y))))
        check(len(folds) == 5 and all(np.array_equal(a[1], b[1]) and np.array_equal(a[0], b[0]) for a, b in zip(folds, ref)),
              "kfold_indices matches scikit-learn's KFold(shuffle=True)", "same shuffle (RandomState(seed)), folds of size n//k with the remainder on the first ones")
        everyone = np.sort(np.concatenate([v for _, v in folds]))
        check(np.array_equal(everyone, np.arange(len(y))), "the validation folds partition the data", "every row in exactly one validation fold")
        sf = strat_indices(y, 5, 0)
        counts = np.array([[(y[v] == c).sum() for c in (0, 1)] for _, v in sf])
        check(np.all(counts.max(axis=0) - counts.min(axis=0) <= 1), "stratification balances every class across the folds (difference at most 1)",
              "for every class, the counts in the folds must differ by at most 1")
        check(all(len(np.intersect1d(t, v)) == 0 and len(t) + len(v) == len(y) for t, v in sf), "training and validation are disjoint and complete",
              "every pair must partition all the rows")
        check(close(my_scores, sk_scores), "my_cross_val_score matches cross_val_score on the same folds",
              "clone the model at every fold, train on the training part, AUC on the validation fold")
    """,
    hint="""
        Sizes: `size = np.full(k, n // k); size[: n % k] += 1`. With `pos = np.cumsum(np.r_[0, size])` fold i is `idx[pos[i]:pos[i+1]]`
        (then sort it). Training: `np.setdiff1d(np.arange(n), val)`. Stratification: `fold_of = np.empty(n, int)`, for every class
        `ind = np.flatnonzero(y == c); r.shuffle(ind); fold_of[ind] = np.arange(len(ind)) % k`.
    """,
    solution=[
        C("""
        def kfold_indices(n, k, seed):
            idx = np.arange(n)
            np.random.RandomState(seed).shuffle(idx)
            size = np.full(k, n // k)
            size[: n % k] += 1                                  # the first n % k folds have one more element
            pos = np.cumsum(np.r_[0, size])
            folds = []
            for i in range(k):
                val = np.sort(idx[pos[i]:pos[i + 1]])
                folds.append((np.setdiff1d(np.arange(n), val), val))
            return folds


        def strat_indices(y, k, seed):
            r = np.random.RandomState(seed)
            fold_of = np.empty(len(y), dtype=int)
            for c in np.unique(y):
                ind = np.flatnonzero(y == c)
                r.shuffle(ind)
                fold_of[ind] = np.arange(len(ind)) % k          # in turn, class by class
            return [(np.flatnonzero(fold_of != i), np.flatnonzero(fold_of == i)) for i in range(k)]


        def my_cross_val_score(model, X, y, folds):
            scores = []
            for tr, va in folds:
                m = clone(model).fit(X[tr], y[tr])              # from scratch on every fold: no memory
                scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
            return np.array(scores)
        """),
        C("""
        folds = kfold_indices(len(y), 5, 0)
        my_scores = my_cross_val_score(model, X, y, folds)
        sk_scores = cross_val_score(model, X, y, cv=folds, scoring="roc_auc")
        print("mine        :", my_scores.round(4))
        print("scikit-learn:", sk_scores.round(4))
        sf = strat_indices(y, 5, 0)
        print("positives per fold (stratified):", [int(y[v].sum()) for _, v in sf], "| (plain K-fold):", [int(y[v].sum()) for _, v in folds])
        """),
    ],
    learned="""
        - A cross-validation is a four-line loop: partition, clone the model, train on the training set, evaluate on the rest. There is no other
          magic; the value lies in the choice of **how** to partition (random, stratified, grouped, temporal).
        - `clone` is essential: every fold must start from a model with no state. Reusing the same object would make fold 2 learn starting
          from fold 1 (a subtle form of leakage).
        - Stratification guarantees the class proportions in every fold: with a rare class it avoids folds with no positives, where the AUC is not
          even defined.
    """,
    pitfall="""
        Not shuffling the data (`shuffle=False`) when the rows are sorted by class, by time or by group: the folds become contiguous blocks
        with different distributions. It is the bug of exercise E3.
    """,
    interview=(
        "Why must the model be cloned and not reused from one fold to the next in cross-validation?",
        """
        Because `fit` does not reset the state of models that train incrementally (initial coefficients with `warm_start`, some models
        with `partial_fit`), and even where it does reset it the cost of getting it wrong is silent. Reusing a model already trained on other folds
        contaminates the evaluation with information from the past. `clone` creates a model with the same hyperparameters and no state.
        """),
)

# ---------------------------------------------------------------------------------------- A3

A3 = dict(
    id="A3", slug="grid_search_from_scratch", kind="A", level=2, minutes=35, module=MODULE,
    title="GridSearchCV from scratch",
    goal="""
        Implement a grid search with cross-validation and a final refit, and verify that it gives the same scores as `GridSearchCV`
        on the same folds, including the `step__parameter` syntax.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns and the target `defaulted`. Model: pipeline with `StandardScaler` and
        `KNeighborsClassifier`. Grid: `n_neighbors` in `[5, 15, 30, 60]` and `weights` in `["uniform", "distance"]` (8 combinations).
    """,
    purpose="""
        See `GridSearchCV` for what it is: a double loop (combinations × folds) with a `set_params` and a mean. You will also understand
        why `refit` exists and why it costs so many evaluations.
    """,
    task=[
        "Write `my_grid_search(estimator, grid, X, y, folds)`: build all the combinations with `itertools.product` on the keys of the "
        "grid; for each one clone the estimator, apply `set_params(**combination)`, compute the mean AUC over the `folds` (train/val pairs) and "
        "collect the results in a DataFrame `results` (one column per parameter and `mean_score`, sorted from the best).",
        "The function returns `(results, best_params, best_model)`, where `best_model` is the estimator with the best "
        "parameters **retrained on all the data** (refit).",
        "Call `my_grid_search` and compare with `GridSearchCV(estimator, grid, cv=folds, scoring=\"roc_auc\")`: same scores, "
        "same best parameters.",
        "Save `n_trainings`: the number of times the model is trained in total (combinations × folds, plus the refit).",
    ],
    setup="""
        import itertools

        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]].to_numpy()
        y = df["defaulted"].to_numpy()
        estimator = make_pipeline(StandardScaler(), KNeighborsClassifier())
        grid = {"kneighborsclassifier__n_neighbors": [5, 15, 30, 60], "kneighborsclassifier__weights": ["uniform", "distance"]}
        folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    """,
    starter=[
        C("""
        def my_grid_search(estimator, grid, X, y, folds):
            keys = list(grid)
            rows = []
            for values in itertools.product(*grid.values()):
                params = dict(zip(keys, values))
                ...
            results = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
            best = ...
            best_model = ...
            return results, best, best_model
        """),
        C("""
        results, best, best_model = my_grid_search(estimator, grid, X, y, folds)
        sk = GridSearchCV(estimator, grid, cv=folds, scoring="roc_auc").fit(X, y)
        n_trainings = ...
        results.round(4)
        """),
    ],
    checks="""
        check(len(results) == 8 and "mean_score" in results.columns, "8 combinations with the mean score", "one row per combination")
        check(results["mean_score"].is_monotonic_decreasing, "results sorted from the best", "sort by mean_score descending")
        check(best == sk.best_params_, "the same best parameters as GridSearchCV", "best = dictionary of the parameters of the first row")
        check(close(np.sort(results["mean_score"].to_numpy()), np.sort(sk.cv_results_["mean_test_score"])),
              "the same mean scores as GridSearchCV", "mean of the AUC over the folds, for every combination")
        check(close(best_model.predict_proba(X), sk.best_estimator_.predict_proba(X)), "the refit matches best_estimator_",
              "the final model must be retrained on ALL the data with the best parameters")
        check(n_trainings == 8 * 5 + 1, "41 trainings: 8 combinations x 5 folds + the refit", "combinations x folds, plus one")
    """,
    hint="""
        Inside the loop: `m = clone(estimator).set_params(**params)`; for every `(tr, va)`: `m2 = clone(m).fit(X[tr], y[tr])` and the AUC on `va`.
        `rows.append({**params, "mean_score": np.mean(scores)})`. For the best parameters: `{k: results.loc[0, k] for k in keys}`
        (mind the types: `int(...)` for `n_neighbors`).
    """,
    solution=[
        C("""
        def my_grid_search(estimator, grid, X, y, folds):
            keys = list(grid)
            rows = []
            for values in itertools.product(*grid.values()):
                params = dict(zip(keys, values))
                scores = []
                for tr, va in folds:
                    m = clone(estimator).set_params(**params).fit(X[tr], y[tr])
                    scores.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
                rows.append({**params, "mean_score": np.mean(scores)})
            results = pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)
            best = {k: grid[k][grid[k].index(results.loc[0, k])] for k in keys}     # takes back the original value
            best_model = clone(estimator).set_params(**best).fit(X, y)              # refit on all the data
            return results, best, best_model
        """),
        C("""
        results, best, best_model = my_grid_search(estimator, grid, X, y, folds)
        sk = GridSearchCV(estimator, grid, cv=folds, scoring="roc_auc").fit(X, y)
        n_trainings = len(results) * len(folds) + 1
        print("best (mine):", best)
        print("best (sk)  :", sk.best_params_)
        print("total trainings:", n_trainings)
        results.round(4)
        """),
    ],
    learned="""
        - `GridSearchCV` is a double loop: for every combination, a complete cross-validation. The cost is combinations × folds + 1
          (the refit): with a grid of 5 parameters of 5 values and 5 folds that is 15,626 trainings.
        - `cv_results_` is nothing more than the `results` table: the mean and the standard deviation across folds for every combination. Reading it
          in full (lesson 3.3 and exercise D1) is much more informative than the winner alone.
        - `refit=True` retrains the best one on the **whole** training set: that model is the one to evaluate on the test set and put in production.
    """,
    pitfall="""
        Reporting `best_score_` as the final estimate. It is the maximum of many estimates on the same folds, so it is biased upward: you measure it
        in exercise C3.
    """,
    interview=(
        "How many trainings does GridSearchCV do with 3 hyperparameters of 4 values each and 5 folds?",
        """
        4 × 4 × 4 = 64 combinations, for 5 folds: 320 trainings, plus one refit on the full training set: 321. It is the reason why with many
        hyperparameters you move to RandomizedSearchCV or to methods that discard poor combinations early (halving).
        """),
)

EXERCISES = [A1, A2, A3]
