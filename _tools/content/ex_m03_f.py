"""Module 3 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="validation_strategies_bias_and_variance", kind="F", level=3, minutes=50, module=MODULE,
    title="Hold-out, 5-fold, 10-fold, repeated and leave-one-out: bias and variance of the estimate",
    goal="""
        Compare five validation strategies on 30 simulated datasets of 200 rows, measuring for each how much the accuracy estimate departs
        from the **true** value (computed on 4000 new rows): bias, variability and mean squared error.
    """,
    dataset="""
        `generate(seed)`: with `make_classification` (10 features, 4 informative) a dataset of 200 rows and a **truth** one of 4000. The true performance
        is the accuracy of the model trained on the 200 rows and evaluated on the 4000. Model: logistic regression with scaler.
    """,
    purpose="""
        Replace "10-fold is better" with numbers. Every strategy is a compromise between how much data it uses to train (bias), how many folds it
        evaluates (variance) and how much it costs (trainings).
    """,
    task=[
        "Write `estimate(strategy, X, y, seed)`: apply the strategy to `(X, y)` and return the estimated accuracy. Use `SCHEMES`, a dictionary "
        "name → function `seed → splitter` (already defined): a `ShuffleSplit` that generates a single split is a hold-out.",
        "For each `seed` from 0 to 29: generate the data, train the model on the 200 rows and compute `true` (accuracy on the 4000); then compute the estimate of "
        "each strategy with `cross_val_score(model, X, y, cv=splitter, scoring=\"accuracy\").mean()`. Collect in `errors` (30 × 5 DataFrame) "
        "the error `estimate − true`.",
        "Build `table` (rows: strategies; columns `bias` = mean of the errors, `std_dev` = standard deviation of the errors, `rmse` = root of the mean of the "
        "squares) and `n_trainings` (number of fits per strategy on one dataset).",
        "Assign to `most_variable` the name of the strategy with the largest `std_dev` and to `best_rmse` the one with the smallest `rmse`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import (LeaveOneOut, RepeatedStratifiedKFold, ShuffleSplit, StratifiedKFold, cross_val_score)
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))


        def generate(seed):
            X, y = make_classification(n_samples=4200, n_features=10, n_informative=4, class_sep=0.8, flip_y=0.05, random_state=seed)
            return X[:200], y[:200], X[200:], y[200:]


        SCHEMES = {
            "hold-out 25%": lambda s: ShuffleSplit(1, test_size=0.25, random_state=s),
            "5-fold": lambda s: StratifiedKFold(5, shuffle=True, random_state=s),
            "10-fold": lambda s: StratifiedKFold(10, shuffle=True, random_state=s),
            "repeated 5x5": lambda s: RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=s),
            "leave-one-out": lambda s: LeaveOneOut(),
        }
    """,
    starter=[
        C("""
        rows = []
        for seed in range(30):
            X, y, X_true, y_true = generate(seed)
            true = ...
            seed_errors = {}
            for name, make in SCHEMES.items():
                ...
            rows.append(seed_errors)
        errors = pd.DataFrame(rows)
        errors.round(3).head(3)
        """),
        C("""
        table = pd.DataFrame({"bias": ..., "std_dev": ..., "rmse": ...})
        n_trainings = {"hold-out 25%": ..., "5-fold": ..., "10-fold": ..., "repeated 5x5": ..., "leave-one-out": ...}
        most_variable = ...
        best_rmse = ...
        table.round(4)
        """),
    ],
    checks="""
        check(errors.shape == (30, 5) and list(errors.columns) == list(SCHEMES), "30 datasets x 5 strategies", "one error estimate - true for every dataset and strategy")
        check(list(table.columns) == ["bias", "std_dev", "rmse"] and set(table.index) == set(SCHEMES), "table with the three metrics per strategy", "bias, std_dev, rmse")
        check(close(table["bias"].to_numpy(), errors.mean().reindex(table.index).to_numpy()) and close(table["rmse"].to_numpy(), np.sqrt((errors ** 2).mean()).reindex(table.index).to_numpy()),
              "bias and rmse computed correctly", "bias = mean, rmse = root of the mean of the squares")
        check(n_trainings == {"hold-out 25%": 1, "5-fold": 5, "10-fold": 10, "repeated 5x5": 25, "leave-one-out": 200},
              "correct count of trainings", "hold-out 1, 5-fold 5, 10-fold 10, repeated 25, leave-one-out 200")
        check(most_variable == table["std_dev"].idxmax() and most_variable == "hold-out 25%", "the hold-out is the most variable strategy", "a single split with 50 validation rows is very noisy")
        check(best_rmse == table["rmse"].idxmin() and best_rmse != "hold-out 25%", "the strategy with the smallest error is not the hold-out", "look at the rmse column")
        check(table.loc["hold-out 25%", "rmse"] > 1.3 * table.loc["repeated 5x5", "rmse"],
              "the hold-out has a mean squared error over 30% larger than the repeated validation", "repeating the validation reduces the variability a lot")
    """,
    hint="""
        `true = model.fit(X, y).score(X_true, y_true)` (accuracy). For every strategy: `cross_val_score(model, X, y, cv=make(seed), scoring="accuracy").mean() - true`.
        `table = pd.DataFrame({"bias": errors.mean(), "std_dev": errors.std(), "rmse": np.sqrt((errors ** 2).mean())})`.
    """,
    solution=[
        C("""
        rows = []
        for seed in range(30):
            X, y, X_true, y_true = generate(seed)
            true = model.fit(X, y).score(X_true, y_true)                      # the true performance of the model trained on 200 rows
            seed_errors = {}
            for name, make in SCHEMES.items():
                estimate = cross_val_score(model, X, y, cv=make(seed), scoring="accuracy").mean()
                seed_errors[name] = estimate - true
            rows.append(seed_errors)
        errors = pd.DataFrame(rows)
        errors.round(3).head(3)
        """),
        C("""
        table = pd.DataFrame({"bias": errors.mean(), "std_dev": errors.std(), "rmse": np.sqrt((errors ** 2).mean())})
        n_trainings = {"hold-out 25%": 1, "5-fold": 5, "10-fold": 10, "repeated 5x5": 25, "leave-one-out": 200}
        most_variable = table["std_dev"].idxmax()
        best_rmse = table["rmse"].idxmin()
        print("most variable:", most_variable, "| smallest error:", best_rmse)
        table.round(4)
        """),
    ],
    learned="""
        - The **hold-out** is the most variable strategy: with 50 validation rows the estimate is noisy (standard deviation of the error about 0.064, against
          0.045 for 5-fold and 0.038 for the repeated one) and its RMSE is about 1.7 times that of the repeated validation. It can turn out well or badly by pure luck of the split.
        - **5-fold, 10-fold and repeated** reduce the variability: every row is used to validate once and with repetition the split is averaged out too.
          The difference between 5 and 10 folds is small; repetition matters more than the number of folds, and costs 25 trainings instead of 5.
        - The **bias** of the estimates is small (one percentage point or less): the hold-out is a bit pessimistic (-0.012) because the model is trained on 150 rows
          instead of 200; with 5 and 10 folds the bias is practically zero.
        - **Leave-one-out** has zero bias like 10-fold but a variance between those of 10-fold and the repeated one (0.040), and costs 200 trainings: there is no
          gain that justifies the cost, except with very little data.
    """,
    pitfall="""
        Choosing the validation scheme by the "magic number" (10-fold is the standard) without looking at the size of the dataset and the
        cost of the model. With small datasets it is better to repeat; with huge data a single hold-out is more than enough.
    """,
    interview=(
        "Why 5 or 10 folds and not leave-one-out, which uses more data for training?",
        """
        Because leave-one-out has the lowest bias but not the lowest variance: the 200 models are almost identical (they differ by one row) and
        the errors are highly correlated, so the mean has less information than it seems; moreover it costs n trainings. With 5 or 10 folds the
        compromise between bias (slightly smaller training set) and variance and cost is better, and with repetition the variance goes down further.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="grid_random_halving_compared", kind="F", level=3, minutes=45, module=MODULE,
    title="GridSearchCV, RandomizedSearchCV and HalvingRandomSearchCV compared",
    goal="""
        Compare three hyperparameter search strategies for a gradient boosting by number of trainings, time, validation score and AUC on the
        test set, and understand when random search and halving save effort without losing quality.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns and the target `defaulted`, 1500 rows (stratified 75/25 split). Model:
        `GradientBoostingClassifier` with a search space on `learning_rate`, `max_depth`, `n_estimators` and `subsample` (54 combinations).
    """,
    purpose="""
        Know what you buy with each strategy: the grid is exhaustive but costly, random search explores a lot with little, halving throws away poor
        combinations early using little data at the start. The comparison is done on the numbers.
    """,
    task=[
        "Build three searches with 3 folds each and `scoring=\"roc_auc\"`: `GridSearchCV` on the full grid (`grid`), `RandomizedSearchCV` with "
        "`n_iter=12` and `random_state=0` on the same space, `HalvingRandomSearchCV` with `n_candidates=27`, `factor=3`, `random_state=0`.",
        "For each one measure the time with `time.time()` (seconds) while you run `fit(X_tr, y_tr)`.",
        "Fill `table` (rows: `grid`, `random`, `halving`; columns `n_trainings` (number of `fit` = rows of `cv_results_` × 3 folds; for halving, "
        "sum over the iterations: use `len(cv_results_[\"params\"]) * 3`), `time_s`, `best_score`, `test_auc`).",
        "Save `random_saving = 1 − n_trainings(random)/n_trainings(grid)` and `random_loss = test_auc(grid) − test_auc(random)`.",
    ],
    setup="""
        import time

        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.experimental import enable_halving_search_cv              # noqa: F401
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]]
        y = df["defaulted"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        cv = StratifiedKFold(3, shuffle=True, random_state=0)
        est = GradientBoostingClassifier(random_state=0)
        grid = {"learning_rate": [0.03, 0.1, 0.3], "max_depth": [2, 3, 4], "n_estimators": [50, 100, 200], "subsample": [0.6, 1.0]}
    """,
    starter=[
        C("""
        searches = {
            "grid": GridSearchCV(est, grid, cv=cv, scoring="roc_auc"),
            "random": ...,
            "halving": ...,
        }
        table = pd.DataFrame(index=list(searches), columns=["n_trainings", "time_s", "best_score", "test_auc"], dtype=float)
        for name, search in searches.items():
            ...
        table.round(3)
        """),
        C("""
        random_saving = ...
        random_loss = ...
        """),
    ],
    checks="""
        check(set(table.index) == {"grid", "random", "halving"} and table.notna().all().all(), "complete table", "three rows and four columns with no NaN")
        check(table.loc["grid", "n_trainings"] == 54 * 3, "the grid does 54 x 3 = 162 trainings", "54 combinations x 3 folds")
        check(table.loc["random", "n_trainings"] == 12 * 3, "random search does 12 x 3 = 36", "12 combinations x 3 folds")
        check(table.loc["random", "time_s"] < table.loc["grid", "time_s"] / 2, "random search costs less than half of the grid in time", "36 trainings against 162")
        check(random_saving > 0.7 and abs(random_loss) < 0.03, "random search saves over 70% of the trainings without losing more than 0.03 of test AUC",
              "compare n_trainings and test_auc between grid and random")
        check(table.loc["halving", "n_trainings"] > table.loc["random", "n_trainings"] and table["test_auc"].min() > 0.72,
              "halving tries more combinations than random search (but on less data for the first ones), and all the strategies have a decent test AUC",
              "halving starts with 27 candidates and reduces them at every iteration")
    """,
    hint="""
        `RandomizedSearchCV(est, grid, n_iter=12, cv=cv, scoring="roc_auc", random_state=0)`; `HalvingRandomSearchCV(est, grid, n_candidates=27, factor=3,
        cv=cv, scoring="roc_auc", random_state=0)`. Loop: `t0 = time.time(); search.fit(X_tr, y_tr); elapsed = time.time() - t0`; `len(search.cv_results_["params"]) * 3`.
    """,
    solution=[
        C("""
        searches = {
            "grid": GridSearchCV(est, grid, cv=cv, scoring="roc_auc"),
            "random": RandomizedSearchCV(est, grid, n_iter=12, cv=cv, scoring="roc_auc", random_state=0),
            "halving": HalvingRandomSearchCV(est, grid, n_candidates=27, factor=3, cv=cv, scoring="roc_auc", random_state=0),
        }
        table = pd.DataFrame(index=list(searches), columns=["n_trainings", "time_s", "best_score", "test_auc"], dtype=float)
        for name, search in searches.items():
            t0 = time.time()
            search.fit(X_tr, y_tr)
            table.loc[name, "time_s"] = time.time() - t0
            table.loc[name, "n_trainings"] = len(search.cv_results_["params"]) * 3
            table.loc[name, "best_score"] = search.best_score_
            table.loc[name, "test_auc"] = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])
        table.round(3)
        """),
        C("""
        random_saving = 1 - table.loc["random", "n_trainings"] / table.loc["grid", "n_trainings"]
        random_loss = table.loc["grid", "test_auc"] - table.loc["random", "test_auc"]
        print(f"random search saves {random_saving:.0%} of the trainings and changes the test AUC by {-random_loss:+.4f}")
        """),
    ],
    learned="""
        - The full grid does 162 trainings and random search 36 (−78%): yet the test AUC is practically the same. With few hyperparameters
          that really matter, a random sample of 12 combinations almost always finds a good region: it is the classic result of Bergstra and Bengio.
        - **Halving** starts from many candidates (27) on little data and keeps only the best for successive iterations with more data: the count of
          trainings includes those done on subsamples, so it costs less than the apparent counts; it is particularly useful with slow models.
        - The `best_score_` values are not comparable across strategies (halving evaluates the last candidates on more data): the right comparison is
          always the AUC on the test set, touched once.
    """,
    pitfall="""
        Concluding that random search "is always better" from a single comparison. With many interacting hyperparameters, or a narrow and
        badly defined search space, a fine grid can make the difference. The point is knowing how much budget you need, not believing a rule.
    """,
    interview=(
        "You have a budget of 100 trainings and 6 hyperparameters. Grid or random search?",
        """
        Random search: with 6 hyperparameters a grid of 100 combinations would have about 2 values per parameter, that is almost no exploration, while 100
        random samples test 100 different values of every parameter. If a few parameters matter much more than the others (as is almost always the case) random
        search finds the good region with the same budget. Then I may refine with a narrow grid around the best, or use a
        method with early discarding like halving.
        """),
)

EXERCISES = [F1, F2]
