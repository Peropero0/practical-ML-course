"""Module 1 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_01_linear_regression"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="california_baseline_and_log_target", kind="B", level=2, minutes=35, module=MODULE,
    title="California housing: baseline, linear model and logarithmic target",
    goal="""
        Build a linear model for house prices in California, evaluate it against a baseline, and find out that
        transforming the target into a logarithm changes which model comes out best **depending on the metric**.
    """,
    dataset="""
        California housing (`ds.california()`): 20,640 districts, 8 numeric features. The target `y` is the median
        house value in hundreds of thousands of dollars. On first use it is downloaded and kept in `data/`.
        The target is **capped at 5.0**: all the most expensive houses appear with the same value.
    """,
    purpose="""
        Practice the complete scikit-learn workflow (split, baseline, fit, metrics) and learn to use
        `TransformedTargetRegressor`. The most important point is that "best" is not a property of the model: it depends
        on what you measure.
    """,
    task=[
        "Split into training and test (80/20, `random_state=0`) and call the pieces `X_tr, X_te, y_tr, y_te`.",
        "Train a baseline `DummyRegressor(strategy=\"mean\")` and save in `rmse_base` its RMSE on the test set.",
        "Train `linear = LinearRegression()` and compute `rmse_lin`, `mae_lin`, `mape_lin` (mean percentage error) and "
        "`r2_lin` on the test set. The predictions are called `pred_lin`.",
        "Build `log_linear`, a `TransformedTargetRegressor` that trains the same linear regression on the "
        "logarithm of the target and brings the predictions back to the original unit. Compute `rmse_log`, `mae_log`, `mape_log` "
        "**in the original units** (predictions in `pred_log`).",
        "Assign to `best_by_rmse` and to `best_by_mape` the string `\"linear\"` or `\"log\"`, based on the numbers "
        "you obtained.",
        "The price is capped at 5.0: build the mask `capped = (y_te >= 5.0)`, compute `capped_share` (the fraction "
        "of capped houses in the test set) and compare the RMSE of the linear model on the capped houses (`rmse_capped`) with the one "
        "on the others (`rmse_others`).",
    ],
    setup="""
        from sklearn.compose import TransformedTargetRegressor
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                                     mean_squared_error, r2_score)
        from sklearn.model_selection import train_test_split

        X, y = ds.california()
        print(X.shape, "| min/max price:", y.min(), y.max())
    """,
    starter=[
        C("""
        X_tr, X_te, y_tr, y_te = ...
        """),
        C("""
        base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        rmse_base = ...
        """),
        C("""
        linear = ...
        pred_lin = ...
        rmse_lin, mae_lin, mape_lin, r2_lin = ..., ..., ..., ...
        """),
        C("""
        log_linear = TransformedTargetRegressor(regressor=..., func=..., inverse_func=...)
        log_linear.fit(X_tr, y_tr)
        pred_log = ...
        rmse_log, mae_log, mape_log = ..., ..., ...
        """),
        C("""
        print(f"linear : RMSE {rmse_lin:.3f} | MAE {mae_lin:.3f} | MAPE {mape_lin:.3f}")
        print(f"log    : RMSE {rmse_log:.3f} | MAE {mae_log:.3f} | MAPE {mape_log:.3f}")
        best_by_rmse = ...
        best_by_mape = ...
        """),
        C("""
        capped = ...
        capped_share = ...
        rmse_capped = ...
        rmse_others = ...
        """),
    ],
    checks="""
        check(len(X_te) == round(0.2 * len(X)), "the split is 80/20", "X_te should have 20% of the rows")
        check(rmse_lin < rmse_base, "the linear model beats the baseline",
              "the linear model should have a lower RMSE than the baseline")
        check(close(rmse_lin, mean_squared_error(y_te, linear.predict(X_te), squared=False)),
              "rmse_lin is computed correctly", "rmse_lin does not match the predictions of `linear`")
        check(close(rmse_log, mean_squared_error(y_te, log_linear.predict(X_te), squared=False)),
              "rmse_log is computed in the original units",
              "rmse_log does not match log_linear.predict(X_te): the predictions must already be in the original unit")
        check(log_linear.predict(X_te).min() > 0, "TransformedTargetRegressor returns positive prices",
              "with func=np.log and inverse_func=np.exp the predictions must be positive")
        check(best_by_rmse == ("linear" if rmse_lin < rmse_log else "log"), "choice consistent with the RMSE",
              "best_by_rmse is not consistent with rmse_lin and rmse_log")
        check(best_by_mape == ("linear" if mape_lin < mape_log else "log"), "choice consistent with the MAPE",
              "best_by_mape is not consistent with mape_lin and mape_log")
        check(abs(capped_share - (y_te >= 5.0).mean()) < 1e-9, "correct share of capped houses",
              "capped_share must be the mean of the mask (y_te >= 5.0)")
        if not ds.CALIFORNIA_SYNTHETIC:
            check(best_by_rmse != best_by_mape,
                  "the two criteria choose DIFFERENT models: that is the point of the exercise",
                  "with RMSE and MAPE you should find two different winners: review the calculations, all in the original unit")
            check(rmse_capped > 1.5 * rmse_others, "on the capped houses the model is much more wrong",
                  "the RMSE on the capped houses should be much higher")
    """,
    hint="""
        `TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)`. The MAPE is
        `mean_absolute_percentage_error(y_te, pred)`. To apply the mask: `mean_squared_error(y_te[capped], pred_lin[capped], squared=False)`.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

        base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        rmse_base = mean_squared_error(y_te, base.predict(X_te), squared=False)
        print("baseline RMSE:", round(rmse_base, 3))
        """),
        C("""
        linear = LinearRegression().fit(X_tr, y_tr)
        pred_lin = linear.predict(X_te)
        rmse_lin = mean_squared_error(y_te, pred_lin, squared=False)
        mae_lin = mean_absolute_error(y_te, pred_lin)
        mape_lin = mean_absolute_percentage_error(y_te, pred_lin)
        r2_lin = r2_score(y_te, pred_lin)
        """),
        M("""
        `TransformedTargetRegressor` applies `func` to the target before the fit and `inverse_func` to the predictions: the metrics
        therefore stay in the original unit (dollars) and are comparable with those of the linear model. Comparing the R²
        computed on the logarithm with the one computed on the prices would be a mistake: they are two different targets.
        """),
        C("""
        log_linear = TransformedTargetRegressor(regressor=LinearRegression(), func=np.log, inverse_func=np.exp)
        log_linear.fit(X_tr, y_tr)
        pred_log = log_linear.predict(X_te)
        rmse_log = mean_squared_error(y_te, pred_log, squared=False)
        mae_log = mean_absolute_error(y_te, pred_log)
        mape_log = mean_absolute_percentage_error(y_te, pred_log)

        print(f"linear : RMSE {rmse_lin:.3f} | MAE {mae_lin:.3f} | MAPE {mape_lin:.3f}")
        print(f"log    : RMSE {rmse_log:.3f} | MAE {mae_log:.3f} | MAPE {mape_log:.3f}")
        best_by_rmse = "linear" if rmse_lin < rmse_log else "log"
        best_by_mape = "linear" if mape_lin < mape_log else "log"
        print("best by RMSE:", best_by_rmse, "| best by MAPE:", best_by_mape)
        """),
        C("""
        capped = (y_te >= 5.0)
        capped_share = capped.mean()
        rmse_capped = mean_squared_error(y_te[capped], pred_lin[capped], squared=False)
        rmse_others = mean_squared_error(y_te[~capped], pred_lin[~capped], squared=False)
        print(f"capped houses: {capped_share:.1%} | RMSE on those {rmse_capped:.2f} versus {rmse_others:.2f} on the others")
        """),
    ],
    learned="""
        - The model on the logarithm has a **better MAPE** (it is less wrong in percentage terms) but a **worse RMSE**. There is no
          contradiction: the squared RMSE weighs mostly the errors on expensive houses, where a small
          percentage error is worth many dollars; the MAPE treats a 100 thousand house and a 500 thousand one the same way.
        - `exp(mean of the logarithms)` is about the **median**, not the mean: bringing the predictions back to the original unit
          introduces a systematic bias (retransformation bias), whose sign and size depend on the sample.
        - The capping of the target at 5.0 is not visible in any aggregate metric, but the model is enormously
          wrong on those houses: it is a limit of the data, not of the model.
    """,
    pitfall="""
        Comparing two models with metrics computed in different units (one R² on the log, another on the prices) or choosing the
        target transformation by looking at the metric that the transformation itself optimizes. The right question is always:
        which error costs more in the real problem?
    """,
    interview=(
        "The model on the logarithm of the price has a better MAPE but a worse RMSE. Which one do you put in production?",
        """
        It depends on what the error costs the company. If the relative error matters (for example an estimate shown to the customer,
        where being off by 30% is bad on both a small and a large house) I choose the model on the logarithm. If
        the absolute error in euros matters (for example the committed capital) and expensive houses weigh more, I look at the RMSE or
        the error weighted by value. In any case I would check the bias of the predictions brought back to the original unit.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="lasso_and_noise_features", kind="B", level=2, minutes=35, module=MODULE,
    title="Lasso and noise features: automatic selection and scale",
    goal="""
        Add 30 pure noise features to the diabetes dataset, use `LassoCV` to see which ones survive, and find out
        how the result changes if you forget the scaler.
    """,
    dataset="""
        Diabetes (`ds.diabetes()`): 442 patients, 10 features **already standardized by scikit-learn** (standard deviation
        about 0.05) and as target the progression of the disease. In the setup cell we put next to them 30 columns
        `noise_i` of random numbers with standard deviation 1, which by construction have no relationship with `y`.
    """,
    purpose="""
        Use `Pipeline`, `StandardScaler` and `LassoCV` the right way and get a hands-on feel for the most common mistake with
        regularization: the penalty depends on the scale of the features. Here the scale is deliberately unbalanced in
        favor of the noise.
    """,
    task=[
        "Split into training and test (75/25, `random_state=0`). Compute `rmse_ols_real` (OLS on the 10 true features only) and "
        "`rmse_ols_all` (OLS on all 40).",
        "Build `lasso_scaled = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000))`, train it "
        "and save `rmse_lasso`. Count how many **true** features (`n_real_kept`) and how many **noise** ones (`n_noise_kept`) have a "
        "nonzero coefficient.",
        "Repeat without the scaler: `lasso_raw = LassoCV(cv=5, random_state=0, max_iter=20000)`. Save `n_real_kept_raw` and "
        "`n_noise_kept_raw`.",
        "Look at `X_tr.std()` and choose in `answer` (\"a\", \"b\" or \"c\") the correct explanation of the result of step 3:\n"
        "   - **a**: without the scaler the Lasso penalizes more the features with a large standard deviation, so it discards the noise;\n"
        "   - **b**: the features with a small standard deviation need large coefficients to have the same effect, "
        "so the penalty hits them harder; the noise, with standard deviation 1, has small coefficients and escapes;\n"
        "   - **c**: `LassoCV` without the scaler does not converge because it requires zero-mean data.",
    ],
    setup="""
        from sklearn.linear_model import LassoCV, LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X_real, y = ds.diabetes()
        rng = np.random.default_rng(0)
        noise = pd.DataFrame(rng.normal(size=(len(X_real), 30)), columns=[f"noise_{i}" for i in range(30)])
        X = pd.concat([X_real, noise], axis=1)          # 10 true features + 30 of pure noise
        real_columns = list(X_real.columns)
        X.shape
    """,
    starter=[
        C("""
        X_tr, X_te, y_tr, y_te = ...

        def rmse(model, columns):
            \"\"\"Test-set RMSE of an already trained model, using only the given `columns`.\"\"\"
            ...

        rmse_ols_real = ...
        rmse_ols_all = ...
        """),
        C("""
        lasso_scaled = ...
        rmse_lasso = ...
        coef = lasso_scaled[-1].coef_
        n_real_kept = ...
        n_noise_kept = ...
        """),
        C("""
        lasso_raw = ...
        n_real_kept_raw = ...
        n_noise_kept_raw = ...
        """),
        C("""
        print(X_tr.std().round(3).to_string())
        answer = ...
        """),
    ],
    checks="""
        check(rmse_ols_all > rmse_ols_real, "adding noise makes OLS worse",
              "OLS on 40 features should have a higher RMSE than the one on 10")
        check(rmse_lasso < rmse_ols_all, "the scaled Lasso beats OLS on the 40 features",
              "with scaler and Lasso you should do better than OLS on all the features")
        check(n_real_kept >= 3, "the Lasso keeps several true features",
              "the scaled Lasso should keep at least a few true features")
        check(n_noise_kept < 10, "the scaled Lasso discards most of the noise",
              "with the scaler the Lasso should discard almost all of the 30 noise features")
        check(n_noise_kept_raw >= 25, "without the scaler the Lasso keeps almost all the noise",
              "without the scaler almost all of the 30 noise features should survive: did you remove the StandardScaler?")
        check(n_real_kept_raw + n_noise_kept_raw > 0, "the raw model has been trained",
              "count the nonzero coefficients of lasso_raw.coef_")
        check(answer == "b", "correct explanation", "reread the standard deviation of the true features and of the noise")
    """,
    hint="""
        To count the nonzero coefficients of the true features: `(coef[:10] != 0).sum()`; for the noise: `(coef[10:] != 0).sum()`
        (the column order is 10 true, then 30 of noise). A model inside a `Pipeline` is indexed with `[-1]` to
        take the last step.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)

        def rmse(model, columns):
            return mean_squared_error(y_te, model.predict(X_te[columns]), squared=False)

        ols_real = LinearRegression().fit(X_tr[real_columns], y_tr)
        ols_all = LinearRegression().fit(X_tr, y_tr)
        rmse_ols_real = rmse(ols_real, real_columns)
        rmse_ols_all = rmse(ols_all, list(X.columns))
        print(f"OLS on 10 true features: {rmse_ols_real:.2f} | OLS on all 40: {rmse_ols_all:.2f}")
        """),
        C("""
        lasso_scaled = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0, max_iter=20000)).fit(X_tr, y_tr)
        rmse_lasso = rmse(lasso_scaled, list(X.columns))
        coef = lasso_scaled[-1].coef_
        n_real_kept = int((coef[:10] != 0).sum())
        n_noise_kept = int((coef[10:] != 0).sum())
        print(f"Lasso with scaler: RMSE {rmse_lasso:.2f} | true kept {n_real_kept}/10 | noise kept {n_noise_kept}/30")
        """),
        C("""
        lasso_raw = LassoCV(cv=5, random_state=0, max_iter=20000).fit(X_tr, y_tr)
        coef_r = lasso_raw.coef_
        n_real_kept_raw = int((coef_r[:10] != 0).sum())
        n_noise_kept_raw = int((coef_r[10:] != 0).sum())
        print(f"Lasso without scaler: true kept {n_real_kept_raw}/10 | noise kept {n_noise_kept_raw}/30 "
              f"| chosen alpha {lasso_raw.alpha_:.3f}")
        """),
        C("""
        print("standard deviation of the true features :", X_tr[real_columns].std().mean().round(3))
        print("standard deviation of the noise features:", X_tr.drop(columns=real_columns).std().mean().round(3))
        answer = "b"
        """),
    ],
    learned="""
        - The scaler makes the coefficients **comparable**, so the L1 penalty hits the features according to how much they
          matter and not to how they are measured. With the scaler the Lasso keeps 5 true features and 4 noise ones; without the scaler
          it keeps *all 30* noise features.
        - The mismatch here is extreme (0.05 against 1), but in real data it happens all the time: an income in euros next to
          an age in years.
        - The Lasso improves on OLS with 40 features but does not bring it to the level of OLS on the 10 true ones alone: automatic
          selection has a cost, because it chooses based on the data and does not know the truth.
    """,
    pitfall="""
        Letting `LassoCV` choose `alpha` does not fix a scale problem: it chooses the best alpha **for those
        features as they were**. The scaler goes in the pipeline (so it is fitted only on the training fold at every
        round of cross-validation), not applied by hand beforehand.
    """,
    interview=(
        "You used the Lasso to select features and only 6 out of 200 survived. Can you say that the other 194 "
        "don't matter?",
        """
        No. The Lasso chooses based on that sample: with correlated features it keeps one per group arbitrarily and discards
        the others, and the selection can change by repeating the fit on another bootstrap sample. I can say that those six
        are enough, together, to predict well *in that dataset*; not that the others have no relationship with the target. For a
        substantive conclusion I would check the stability of the selection or use an ElasticNet.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="polynomial_ridge_gridsearch_pipeline", kind="B", level=2, minutes=40, module=MODULE,
    title="Polynomial pipeline with Ridge and GridSearchCV",
    goal="""
        Build a `Pipeline` (PolynomialFeatures, StandardScaler, Ridge), choose degree and regularization with
        `GridSearchCV`, and measure what happens outside the range of the data with a degree that is too high.
    """,
    dataset="""
        `ds.energy_consumption()`: 160 days, outside temperature in degrees Celsius (from −5 to 35) and energy consumption in kWh.
        The relationship is U-shaped: more is used in the cold (heating) and in the heat (air conditioning).
    """,
    purpose="""
        Use `Pipeline` and `GridSearchCV` with nested parameters (`step__parameter`), and understand that **a good test error
        does not guarantee a good model when you extrapolate**.
    """,
    task=[
        "Split into training and test (70/30, `random_state=0`).",
        "Build `pipe = Pipeline([(\"poly\", PolynomialFeatures(include_bias=False)), (\"scaler\", StandardScaler()), "
        "(\"ridge\", Ridge())])`.",
        "With `GridSearchCV` search `poly__degree` from 1 to 10 and `ridge__alpha` in `np.logspace(-4, 2, 7)`, with `cv=KFold(5, shuffle=True, "
        "random_state=0)` and `scoring=\"neg_root_mean_squared_error\"`. Save `search`, `chosen_degree`, `chosen_alpha`, `rmse_cv` "
        "(positive!) and `rmse_test` (only once, at the end).",
        "Train two comparison models: `linear` (`LinearRegression`, degree 1) and `poly10` (degree 10 with "
        "`LinearRegression`, without regularization, with the scaler). Compute `rmse_test_linear` and `rmse_test_poly10`.",
        "Predict at 45 degrees (`t45`, outside the training range) with the three models: `prediction_45_chosen`, "
        "`prediction_45_linear`, `prediction_45_poly10`. The value of the generating curve is `true_45`.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV, KFold, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler

        d = ds.energy_consumption()
        X, y = d[["temperature"]], d["consumption"]
        t45 = pd.DataFrame({"temperature": [45.0]})
        true_45 = 50 + 0.16 * (45 - 17) ** 2               # the true curve, which the model does not know
        print("true value at 45 degrees:", round(true_45, 1), "kWh")
    """,
    starter=[
        C("""
        X_tr, X_te, y_tr, y_te = ...
        pipe = ...
        """),
        C("""
        grid = {"poly__degree": ..., "ridge__alpha": ...}
        search = GridSearchCV(pipe, grid, cv=..., scoring=...)
        search.fit(X_tr, y_tr)

        chosen_degree = ...
        chosen_alpha = ...
        rmse_cv = ...
        rmse_test = ...
        print("degree", chosen_degree, "| alpha", chosen_alpha, "| RMSE cv", round(rmse_cv, 2), "| RMSE test", round(rmse_test, 2))
        """),
        C("""
        linear = ...
        poly10 = ...
        rmse_test_linear = ...
        rmse_test_poly10 = ...
        """),
        C("""
        prediction_45_chosen = ...
        prediction_45_linear = ...
        prediction_45_poly10 = ...
        """),
    ],
    checks="""
        check(len(X_te) == 48, "70/30 split", "X_te should have 48 rows (30% of 160)")
        check(2 <= chosen_degree <= 5, "the search chooses a low degree (the true curve is quadratic)",
              "the chosen degree should be between 2 and 5: check the scoring and the grid")
        check(rmse_cv > 0, "rmse_cv is positive", "use the minus sign: best_score_ is a negative score")
        check(close(rmse_test, mean_squared_error(y_te, search.predict(X_te), squared=False)),
              "rmse_test is computed on the test set", "rmse_test must be the RMSE of search.predict(X_te)")
        check(rmse_test_linear > 2 * rmse_test, "the line is much worse than the chosen model (the trend is U-shaped)",
              "the linear model should have an RMSE at least double")
        check(rmse_test_poly10 < 1.3 * rmse_test,
              "the degree 10 polynomial has an almost equal test error: INSIDE the range you don't see the problem",
              "degree 10 with the scaler should have a comparable test error: check that you used the scaler")
        check(abs(prediction_45_chosen - true_45) < 20, "the chosen model extrapolates reasonably",
              "the prediction at 45 degrees of the chosen model should be close to the true value")
        check(abs(prediction_45_poly10) > 1000, "degree 10 explodes outside the range",
              "the degree 10 polynomial should give an absurd prediction (thousands of kWh) at 45 degrees")
    """,
    hint="""
        To access the results: `search.best_params_["poly__degree"]`, `search.best_score_` (negative: the RMSE is
        `-search.best_score_`). The unregularized degree 10 polynomial is
        `Pipeline([("poly", PolynomialFeatures(10, include_bias=False)), ("scaler", StandardScaler()), ("lr", LinearRegression())])`.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
        pipe = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                         ("scaler", StandardScaler()),
                         ("ridge", Ridge())])
        """),
        M("""
        The step names serve to address the parameters: `poly__degree` means "the `degree` parameter of the `poly`
        step". The scaler sits **after** `PolynomialFeatures`, because it is the high powers that have enormous scales.
        """),
        C("""
        grid = {"poly__degree": list(range(1, 11)), "ridge__alpha": np.logspace(-4, 2, 7)}
        search = GridSearchCV(pipe, grid, cv=KFold(5, shuffle=True, random_state=0),
                              scoring="neg_root_mean_squared_error")
        search.fit(X_tr, y_tr)

        chosen_degree = search.best_params_["poly__degree"]
        chosen_alpha = search.best_params_["ridge__alpha"]
        rmse_cv = -search.best_score_
        rmse_test = mean_squared_error(y_te, search.predict(X_te), squared=False)
        print("degree", chosen_degree, "| alpha", chosen_alpha, "| RMSE cv", round(rmse_cv, 2), "| RMSE test", round(rmse_test, 2))
        """),
        C("""
        linear = LinearRegression().fit(X_tr, y_tr)
        poly10 = Pipeline([("poly", PolynomialFeatures(10, include_bias=False)),
                           ("scaler", StandardScaler()),
                           ("lr", LinearRegression())]).fit(X_tr, y_tr)
        rmse_test_linear = mean_squared_error(y_te, linear.predict(X_te), squared=False)
        rmse_test_poly10 = mean_squared_error(y_te, poly10.predict(X_te), squared=False)
        print(f"RMSE test: chosen {rmse_test:.2f} | linear {rmse_test_linear:.2f} | degree 10 {rmse_test_poly10:.2f}")
        """),
        C("""
        prediction_45_chosen = float(search.predict(t45)[0])
        prediction_45_linear = float(linear.predict(t45)[0])
        prediction_45_poly10 = float(poly10.predict(t45)[0])
        print(f"at 45 degrees -> true {true_45:.0f} | chosen {prediction_45_chosen:.0f} | "
              f"linear {prediction_45_linear:.0f} | degree 10 {prediction_45_poly10:.0f}")
        """),
    ],
    learned="""
        - The test error of the degree 10 polynomial is **practically equal** to that of the chosen model: if you stop
          at that number you notice nothing. Outside the range of the data it predicts thousands of kWh at 45 degrees.
        - A high-degree polynomial oscillates between the points (few constraints) and diverges at the edges: the high powers dominate as soon as
          you leave the training interval. The model chosen by the search, of low degree, stays close to the true curve.
        - `GridSearchCV` chooses hyperparameters based on the performance **inside** the range of the data; extrapolation has to be
          judged with a separate argument (what range do I need? what behavior is physically plausible?).
    """,
    pitfall="""
        Reporting the RMSE of `search.best_score_` as the final estimate of the performance: it is the maximum among many
        combinations tried on the same folds, so it is optimistic. For the honest estimate you need the test set, touched only
        once (or a nested cross-validation).
    """,
    interview=(
        "Your degree 10 polynomial model has an excellent test error. Why don't you trust it yet?",
        """
        Because the test set has the same distribution as the training set: it measures the ability to interpolate, not to
        extrapolate. A high-degree polynomial can follow the points well inside the range and diverge right outside it. I would check
        whether values outside the range will arrive in production (drift), I would look at the predictions on plausible extreme inputs and
        I would prefer a low-degree model with regularization.
        """),
)

EXERCISES = [B1, B2, B3]
