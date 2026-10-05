"""Module 1 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_01_linear_regression"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="leakage_in_feature_selection", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="The model that finds signal in noise",
    goal="""
        A colleague evaluated a model on pure noise data and got a positive cross-validated R². Find
        the mistake, fix it with a `Pipeline`, and measure by how much the evaluation was optimistic.
    """,
    dataset="""
        `ds.pure_noise(seed)`: 100 observations, 1000 features and a target, all **independent** Gaussian numbers. There is
        no signal to find, so an honest model must have an R² that does not exceed zero.
    """,
    purpose="""
        Recognize leakage from feature selection, one of the most common and costly mistakes: the cross-validation
        is run correctly, but on data that have already "seen" the target. You will also learn to quantify it.
    """,
    task=[
        "Run the colleague's cell: `r2_bug` is positive on data that contain nothing. It is not a miracle.",
        "Fix `experiment(seed)` so that `r2_ok` is an honest evaluation: the feature selection must happen "
        "**inside** every training fold (use a `Pipeline`).",
        "Repeat the experiment over 10 seeds (from 0 to 9) in `results` (10 × 2 array, columns `r2_bug`, `r2_ok`) and save in "
        "`mean_optimism` the mean of `r2_bug − r2_ok`.",
        "Choose in `answer` (\"a\", \"b\" or \"c\") why `r2_ok` comes out **negative** and not simply zero:\n"
        "   - **a**: there is a mistake in the call to `cross_val_score`;\n"
        "   - **b**: the model learns random relationships from the training set (10 features chosen among 1000 of noise) and on new data "
        "it predicts worse than the plain mean;\n"
        "   - **c**: `SelectKBest` always makes any model worse.",
    ],
    setup="""
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import KFold, cross_val_score
        from sklearn.pipeline import Pipeline

        cv = KFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        M("""
        ### The colleague's code

        It runs without errors and prints an encouraging number. Find what is wrong and fix `r2_ok`.
        """),
        C("""
        def experiment(seed):
            \"\"\"Returns (r2_bug, r2_ok) on a pure noise dataset.\"\"\"
            X, y = ds.pure_noise(seed=seed)          # X and y are independent: no signal

            # --- the colleague's code ---
            X_sel = SelectKBest(f_regression, k=10).fit_transform(X, y)
            r2_bug = cross_val_score(LinearRegression(), X_sel, y, cv=cv, scoring="r2").mean()

            # --- to fix ---
            r2_ok = r2_bug
            return r2_bug, r2_ok


        r2_bug_0, r2_ok_0 = experiment(0)
        print(f"colleague's R2: {r2_bug_0:.3f} | fixed R2: {r2_ok_0:.3f}")
        """),
        C("""
        results = np.array([experiment(s) for s in range(10)])
        mean_optimism = np.mean(results[:, 0] - results[:, 1])
        answer = None
        """),
    ],
    checks="""
        check(r2_bug_0 > 0.15, "the colleague's number is positive and looks like a good model",
              "r2_bug should be about 0.26 on this dataset: don't modify the colleague's code")
        check(r2_ok_0 < 0, "the fixed R2 is negative: in noise there is nothing to learn",
              "r2_ok must come from a Pipeline with SelectKBest INSIDE the cross-validation")
        check(results.shape == (10, 2) and bool(np.all(results[:, 0] > results[:, 1])),
              "in all 10 seeds the colleague's evaluation is more optimistic",
              "results must have 10 rows and r2_bug > r2_ok in every row")
        check(mean_optimism > 0.4, "the mean optimism is more than 0.4 points of R2",
              "mean_optimism should be the mean of r2_bug - r2_ok over 10 seeds")
        check(answer == "b", "correct explanation", "reread: what does a model learn from 10 random features on 80 observations?")
    """,
    hint="""
        `Pipeline([("sel", SelectKBest(f_regression, k=10)), ("lr", LinearRegression())])` passed directly to
        `cross_val_score(..., X, y, ...)` with the **complete** matrix X: the selection is redone on every training fold.
    """,
    solution=[
        M("""
        **Where the bug is.** `SelectKBest(...).fit_transform(X, y)` is run on the *whole* dataset, before the
        cross-validation. To choose the 10 features with the highest correlation with `y`, it looked at the `y` of all the
        observations, including those that then end up in the validation folds. Among 1000 noise columns, the 10 best
        have, by pure luck, a correlation with `y` of about 0.3, and the subsequent cross-validation "confirms" it because
        those correlations also hold on the validation folds: it already used them to be chosen.
        """),
        C("""
        def experiment(seed):
            X, y = ds.pure_noise(seed=seed)

            # --- the colleague's code: the selection sees ALL of y ---
            X_sel = SelectKBest(f_regression, k=10).fit_transform(X, y)
            r2_bug = cross_val_score(LinearRegression(), X_sel, y, cv=cv, scoring="r2").mean()

            # --- fixed: the selection is part of the pipeline, so it is redone on every training fold ---
            pipe = Pipeline([("sel", SelectKBest(f_regression, k=10)), ("lr", LinearRegression())])
            r2_ok = cross_val_score(pipe, X, y, cv=cv, scoring="r2").mean()
            return r2_bug, r2_ok


        r2_bug_0, r2_ok_0 = experiment(0)
        print(f"colleague's R2: {r2_bug_0:.3f} | fixed R2: {r2_ok_0:.3f}")
        """),
        C("""
        results = np.array([experiment(s) for s in range(10)])
        mean_optimism = np.mean(results[:, 0] - results[:, 1])
        print("colleague's R2 per seed:", results[:, 0].round(2))
        print("fixed R2 per seed      :", results[:, 1].round(2))
        print(f"mean optimism: {mean_optimism:.2f} points of R2")
        answer = "b"
        """),
    ],
    learned="""
        - Any step that **learns something from the target** (feature selection, choice of hyperparameters, target-based
          imputations, target encoding) must sit inside the cross-validation. Outside, the evaluation is contaminated.
        - The optimism here is huge: from an R² of about 0.2-0.3 (a "decent" model) to one of about -0.5 (worse than the
          mean) on data that contain nothing. The leakage grows when p is large compared to n, which is exactly the
          situation in which you do feature selection.
        - A negative R² on data with no signal is not a bug: the model estimated 10 coefficients from 80 points and all of them are
          noise, so on new data it does worse than the mean.
    """,
    pitfall="""
        Thinking that "doing the cross-validation" is enough. Cross-validation only protects from evaluating on data seen in
        training: it does not protect from preprocessing and selection run beforehand. Rule of thumb: if a transformation has a
        `fit`, it must sit in a `Pipeline`.
    """,
    interview=(
        "You have 5000 features and 200 patients. You selected the 20 most correlated with the outcome and then did the cross-validation, with "
        "AUC 0.85. Do you trust it?",
        """
        No: the selection saw the labels of all the patients, including those of the validation folds, so the AUC is
        inflated. With p much greater than n it is easy to find 20 features that correlate with the outcome by chance. I would redo the evaluation
        with the selection inside a Pipeline (or a nested cross-validation), and compare with the AUC obtained by permuting
        the labels: if that one is also high, the number means nothing.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="metrics_computed_wrongly", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="Regression metrics computed wrongly: four bugs",
    goal="""
        Fix an evaluation function that runs without errors and prints plausible numbers, but computes the metrics the
        wrong way, and rewrite it as a reusable function.
    """,
    dataset="""
        `ds.house_prices()`: 400 houses with `area`, `rooms`, `distance_center_km` and `price` in euros (skewed
        distribution). The colleague trained the regression on the **logarithm** of the price, and the split is already done.
    """,
    purpose="""
        A wrong model gets noticed, a wrong metric does not. The kind of bug here is the kind that passes a review at a
        glance: arguments in the wrong order, wrong units of measure, RMSE without the root, baseline computed with the
        test data.
    """,
    task=[
        "Run the colleague's cell. There are no errors, only numbers.",
        "The true target to evaluate is the **price in euros** of the test set. Fix the computation so that `r2`, `rmse`, "
        "`rmse_baseline` and `improvement_pct` are all in euros, with the baseline predicting the mean of the **training set**.",
        "Rewrite everything in `evaluate(y_true, y_pred, y_train)`, which returns a dictionary with the keys `r2`, `rmse`, "
        "`rmse_baseline`, `improvement_pct` (as a percentage, on the RMSE).",
        "Use `evaluate` for two models: `log_metrics` (the colleague's model, with the predictions brought back to euros) and "
        "`direct_metrics` (a `LinearRegression` trained directly on `y_tr`). Assign to `best_by_rmse` the string "
        "`\"log\"` or `\"direct\"`.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        df = ds.house_prices()
        X = df[["area", "rooms", "distance_center_km"]]
        y = df["price"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

        # the colleague trained the model on the logarithm of the price: there is no mistake here
        model = LinearRegression().fit(X_tr, np.log(y_tr))
    """,
    starter=[
        M("### The colleague's evaluation"),
        C("""
        pred_log = model.predict(X_te)
        y_te_log = np.log(y_te)

        r2 = r2_score(pred_log, y_te_log)
        rmse = mean_squared_error(y_te_log, pred_log)
        rmse_baseline = mean_squared_error(y_te_log, np.full(len(y_te), y_te_log.mean()))
        improvement_pct = 100 * (1 - rmse / rmse_baseline)

        print(f"R2 {r2:.3f} | RMSE {rmse:.4f} | baseline RMSE {rmse_baseline:.4f} | improvement {improvement_pct:.1f}%")
        """),
        M("### Your version"),
        C("""
        def evaluate(y_true, y_pred, y_train):
            ...
            return {"r2": ..., "rmse": ..., "rmse_baseline": ..., "improvement_pct": ...}
        """),
        C("""
        log_metrics = ...
        direct = LinearRegression().fit(X_tr, y_tr)
        direct_metrics = ...
        best_by_rmse = ...
        """),
    ],
    checks="""
        check(close(r2, 0.92547, rtol=1e-4), "correct r2 (in euros, arguments in the right order)",
              "wrong r2: is it computed on the prices in euros? r2_score wants (y_true, y_pred) in this order")
        check(close(rmse, 32853.88, rtol=1e-4), "correct rmse (in euros, with the root)",
              "wrong rmse: it must be the root of the MSE, computed on the prices in euros")
        check(close(rmse_baseline, 120444.62, rtol=1e-4), "correct baseline (mean of the training set, in euros)",
              "wrong rmse_baseline: the baseline predicts the mean of y_tr, not that of the test set")
        check(close(improvement_pct, 72.72, rtol=1e-3), "correct improvement_pct", "improvement_pct = 100 * (1 - rmse / rmse_baseline)")
        check(set(log_metrics) == {"r2", "rmse", "rmse_baseline", "improvement_pct"}, "evaluate returns the four keys",
              "the dictionary must have r2, rmse, rmse_baseline, improvement_pct")
        check(close(log_metrics["rmse"], rmse), "evaluate reproduces the correct computation for the log model",
              "evaluate(y_te, np.exp(pred_log), y_tr) should give the same rmse")
        check(best_by_rmse == ("log" if log_metrics["rmse"] < direct_metrics["rmse"] else "direct"),
              "choice consistent with the numbers", "compare log_metrics['rmse'] and direct_metrics['rmse']")
    """,
    hint="""
        The four bugs are in the colleague's cell alone: (1) the order of the arguments of `r2_score`; (2) the predictions and the target
        are on a logarithmic scale; (3) `mean_squared_error` returns the MSE, not the RMSE (in scikit-learn 1.1: `squared=False`);
        (4) the baseline uses `y_te_log.mean()`, that is information from the test set. Remember `np.exp` to go back to euros.
    """,
    solution=[
        M("""
        Four bugs in five lines:

        1. `r2_score(pred_log, y_te_log)` has the arguments swapped. R² is not symmetric: the denominator uses the
           variance of the first argument, which must be the **true** value.
        2. Everything is computed on a logarithmic scale, while the business problem is in euros: "RMSE 0.16" means nothing
           to whoever looks at the prices.
        3. `mean_squared_error` returns the **MSE**, not the RMSE: the "improvement" ratio is thus computed on the squares.
        4. The baseline uses the mean of the **test set** (`y_te_log.mean()`): the test set is not touched to build the reference.
        """),
        C("""
        def evaluate(y_true, y_pred, y_train):
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            baseline = np.full(len(y_true), np.mean(y_train))               # the mean of the TRAINING set
            rmse_base = mean_squared_error(y_true, baseline, squared=False)
            return {
                "r2": r2_score(y_true, y_pred),                              # (true, predicted), in this order
                "rmse": rmse,
                "rmse_baseline": rmse_base,
                "improvement_pct": 100 * (1 - rmse / rmse_base),
            }

        pred_log = model.predict(X_te)
        pred_euro = np.exp(pred_log)                                         # back to the unit of the target
        result = evaluate(y_te, pred_euro, y_tr)
        r2, rmse = result["r2"], result["rmse"]
        rmse_baseline, improvement_pct = result["rmse_baseline"], result["improvement_pct"]
        print(f"R2 {r2:.4f} | RMSE {rmse:,.0f} euros | baseline RMSE {rmse_baseline:,.0f} euros | improvement {improvement_pct:.1f}%")
        """),
        C("""
        log_metrics = evaluate(y_te, np.exp(model.predict(X_te)), y_tr)
        direct = LinearRegression().fit(X_tr, y_tr)
        direct_metrics = evaluate(y_te, direct.predict(X_te), y_tr)
        best_by_rmse = "log" if log_metrics["rmse"] < direct_metrics["rmse"] else "direct"

        print(pd.DataFrame({"log": log_metrics, "direct": direct_metrics}).round(3).to_string())
        print("best by RMSE:", best_by_rmse)
        """),
    ],
    learned="""
        - The colleague's numbers (R² 0.941, "RMSE" 0.0251, "improvement" of 93.8%) look excellent and are all wrong:
          the right ones, in euros, are R² 0.925, RMSE 32,854 euros and an improvement of 72.7% over the baseline.
        - The scikit-learn metrics want `(y_true, y_pred)` in this order. With MAE and MSE the error is hidden by the
          symmetry, with R² it is visible only by comparing with the correct value.
        - An `evaluate(y_true, y_pred, y_train)` function reused for all the models eliminates the class of bugs "I computed the
          metrics in a different way for the two models".
    """,
    pitfall="""
        Computing the baseline with the mean of the test set. The improvement looks a bit smaller or larger than the true one and the
        difference is small: it is a bug that survives because it does not make anything collapse. The test set must not influence in
        any way any reference quantity.
    """,
    interview=(
        "The model on the logarithm of the price has an R² of 0.94 computed by a script. The direct model has 0.92. Which one do you choose?",
        """
        First I check where each R² is computed: if the one of the log model is computed on the logarithm of the price while
        the other on the price, they are not comparable, they are two different targets. I would recompute both on the same target (the
        price in euros, after bringing the predictions back with exp) and with the same function, and choose based on the metric
        that reflects the cost of the error for the business.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="scaler_refitted_on_test", kind="E", level=2, minutes=30, module=MODULE, start="must_fail",
    title="The scaler refitted on the test set",
    goal="""
        Find out why a model with excellent training results has a test error ten times larger, fix
        the preprocessing and verify that a `Pipeline` produces exactly the same correct result.
    """,
    dataset="""
        `ds.trending_series()`: 400 chronological rows with two features `x1`, `x2` that **grow over time** and a target `y`.
        The split is chronological: the first 300 rows for training, the last 100 for the test (you predict the future).
    """,
    purpose="""
        Remember what a scaler does: `fit` computes mean and standard deviation, `transform` applies them. If you call `fit` on
        the test set too, you compute new ones and re-center the data, erasing exactly the information (the level reached
        by the features) that the model had learned to use.
    """,
    task=[
        "Run the colleague's cell: look at the test RMSE (`rmse_bug`), enormously higher than the training one.",
        "Compute `test_mean_std_right`: the means of the two features of the test set standardized **with the statistics of the "
        "training set**. They are not zero: it is what the model should see.",
        "Fix the preprocessing (`X_te_ok`) and compute `rmse_ok` with the same model.",
        "Build `pipe = make_pipeline(StandardScaler(), LinearRegression())`, train it on `X_tr` and save the predictions "
        "in `pred_pipe`. They must match those of the correct manual preprocessing (`pred_manual`).",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        d = ds.trending_series()
        X, y = d[["x1", "x2"]], d["y"]
        cut = int(0.75 * len(d))
        X_tr, X_te, y_tr, y_te = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.fit_transform(X_te)

        model = LinearRegression().fit(X_tr_s, y_tr)
        rmse_train = mean_squared_error(y_tr, model.predict(X_tr_s), squared=False)
        rmse_bug = mean_squared_error(y_te, model.predict(X_te_s), squared=False)
        print(f"RMSE train {rmse_train:.2f} | RMSE test {rmse_bug:.2f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        test_mean_std_right = ...
        print(test_mean_std_right)
        """),
        C("""
        X_te_ok = ...
        pred_manual = ...
        rmse_ok = rmse_bug
        """),
        C("""
        pipe = ...
        pred_pipe = ...
        """),
    ],
    checks="""
        check(rmse_bug > 3 * rmse_ok, "the wrong preprocessing makes the error more than 3 times worse",
              "rmse_ok must come from X_te standardized with scaler.transform (training statistics)")
        check(np.all(np.abs(np.asarray(test_mean_std_right)) > 1),
              "in the test, the features are more than one standard deviation above the training level",
              "test_mean_std_right = scaler.transform(X_te).mean(axis=0): the trend takes the features far above the training means")
        check(close(pred_manual, model.predict(scaler.transform(X_te))), "pred_manual uses the scaler fitted on the training set",
              "pred_manual must be model.predict(scaler.transform(X_te))")
        check(close(pred_pipe, pred_manual), "the Pipeline produces exactly the same predictions",
              "the Pipeline trained on X_tr must give the same predictions as the correct manual preprocessing")
        check(rmse_ok < 2, "the fixed RMSE is close to the noise of the target", "rmse_ok should be about 1")
    """,
    hint="""
        The right method is `scaler.transform(X_te)`, without `fit`. `test_mean_std_right` is `scaler.transform(X_te).mean(axis=0)`.
        `pipe.fit(X_tr, y_tr)` fits the scaler on the training set alone and then applies `transform` (not `fit`) to the test.
    """,
    solution=[
        M("""
        **Where the bug is.** `scaler.fit_transform(X_te)` recomputes mean and standard deviation **on the test set** and
        re-centers it. But the model was trained on features centered with respect to the training set: in the last 100 rows the
        features have risen (trend), so in "training" units they are worth about +2 standard deviations. By re-centering the test,
        the model sees values around 0 and predicts as if we were back at the average level of the past.
        """),
        C("""
        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        model = LinearRegression().fit(X_tr_s, y_tr)

        # what the model should see on the test set
        test_mean_std_right = scaler.transform(X_te).mean(axis=0)
        print("test means, in training standard deviations:", test_mean_std_right.round(2))
        """),
        C("""
        X_te_bug = StandardScaler().fit_transform(X_te)              # the colleague's code
        rmse_bug = mean_squared_error(y_te, model.predict(X_te_bug), squared=False)

        X_te_ok = scaler.transform(X_te)                             # transform only, with the training statistics
        pred_manual = model.predict(X_te_ok)
        rmse_ok = mean_squared_error(y_te, pred_manual, squared=False)
        print(f"test RMSE: with the bug {rmse_bug:.2f} | fixed {rmse_ok:.2f}")
        """),
        C("""
        pipe = make_pipeline(StandardScaler(), LinearRegression()).fit(X_tr, y_tr)
        pred_pipe = pipe.predict(X_te)
        print("the Pipeline matches the correct manual preprocessing:", np.allclose(pred_pipe, pred_manual))
        """),
    ],
    learned="""
        - `fit` (computes the statistics) is done **only** on the training set; `transform` (applies them) is used on training, validation
          and test with the same statistics. `fit_transform(X_te)` is almost always a bug.
        - Here the bug does not make you "cheat" (the test is not at an advantage) but it **breaks** the model: it erases the trend and produces a
          test error of about 8 against 0.9. With non-stationary data (with a drift) it is the normal consequence, not a
          strange case.
        - A `Pipeline` makes the bug impossible by construction: `fit` on the training set calls `fit_transform` of the intermediate
          steps, `predict` calls only `transform`.
    """,
    pitfall="""
        Standardizing "per fold" with `fit_transform` on every validation fold (it happens when you write the
        cross-validation by hand). The scaler must be fitted on the training part of the fold and applied to the validation: it is exactly what
        `cross_val_score` does with a Pipeline.
    """,
    interview=(
        "In production your model receives one request at a time. How do you apply the scaler to a single request?",
        """
        With `transform`, using the mean and standard deviation saved in training: it is also the reason why the scaler object
        (or better the whole Pipeline) must be serialized with the model. With a single row `fit_transform` would give a standard
        deviation of zero and would center the value on itself, that is no information.
        """),
)

EXERCISES = [E1, E2, E3]
