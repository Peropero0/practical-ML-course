"""Module 1 lessons: linear regression."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_01_linear_regression"
TITLE = "Linear regression"

LESSONS = [
    ("01_idea_and_least_squares.ipynb", "1.1", "The idea and least squares",
     "OLS as a matrix product, the normal equation, the `LinearRegression` API."),
    ("02_evaluating_the_model.ipynb", "1.2", "Evaluating the model",
     "MAE, RMSE, R², baseline, cross-validation, order of the arguments."),
    ("03_assumptions_and_diagnostics.ipynb", "1.3", "Assumptions and diagnostics",
     "Residual plots, VIF, leverage and Cook's distance."),
    ("04_regularization_and_nonlinearity.ipynb", "1.4", "Regularization and nonlinearity",
     "Ridge, Lasso, ElasticNet, `PolynomialFeatures`, log target."),
]

PATH = [
    ("Lesson 1.1", ["A1", "A2"]),
    ("Lesson 1.2", ["B1", "E2", "D1"]),
    ("Lesson 1.3", ["C1", "C2", "D2", "D3", "A3"]),
    ("Lesson 1.4", ["B2", "B3", "C3", "E1", "E3", "F2", "F1", "F3"]),
]



def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_1_1() -> List[Cell]:
    """Cells of lesson 1.1."""
    return [
        M("""
        # Lesson 1.1 · Linear regression: the idea and least squares

        This lesson is about the code, not the derivation. We don't rederive the theory: we look at what the model
        really does when you run it, and we build the tools you will use in the exercises.

        **What you will learn**

        - To read the prediction as a matrix product, ŷ = A·β, and to compute the coefficients by hand.
        - To use `LinearRegression` and to read `coef_` and `intercept_` without getting the dimensions wrong.
        - To understand why scikit-learn does not solve the normal equation, even though it is the textbook formula.
        - To recognize when a straight line is the wrong choice.
        """),
        _setup("from sklearn.linear_model import LinearRegression"),
        M("""
        ## Definition

        Linear regression models the target as a weighted sum of the features, plus a constant term called the
        **intercept**. The weights, the **coefficients**, are chosen by minimizing the sum of squares of the **residuals**, that is
        the differences between true value and predicted value (OLS criterion, Ordinary Least Squares).

        ```
        ŷ = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ          minimizes      Σᵢ (yᵢ − ŷᵢ)²
        ```
        """),
        M("""
        ## Intuition: the line that minimizes the squares

        Every vertical segment in the plot is a residual. Squaring them means that one error of 10 costs a hundred,
        while ten errors of 1 cost ten in total: the line gets "pulled" toward the far points much more than
        toward the near ones. That is the reason why outliers weigh so much (you will measure it in exercise C2).
        """),
        C("""
        rng = np.random.default_rng(0)
        x = rng.uniform(0, 10, 25)
        y = 2 + 1.5 * x + rng.normal(0, 2, 25)

        model = LinearRegression().fit(x.reshape(-1, 1), y)
        y_hat = model.predict(x.reshape(-1, 1))

        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(x, y, s=25)
        grid = np.linspace(0, 10, 100)
        ax.plot(grid, model.predict(grid.reshape(-1, 1)), color="#1F6E5C")
        ax.vlines(x, y, y_hat, color="#9C4530", alpha=0.6, lw=1)        # the residuals
        ax.set(xlabel="x", ylabel="y", title="The OLS line and its residuals (red segments)")

        def ssr(b0, b1):
            return ((y - (b0 + b1 * x)) ** 2).sum()

        print("SSR of the OLS line      :", round(ssr(model.intercept_, model.coef_[0]), 2))
        print("SSR of an eyeballed line :", round(ssr(2, 1.3), 2))
        print("SSR of another line      :", round(ssr(0, 1.8), 2))
        """),
        M("""
        ## From the line to matrices

        With p features it is convenient to write everything in matrix form. You build the matrix **A** by adding to X a
        column of ones, which represents the intercept: so ŷ = A·β and the problem becomes
        minimizing ‖y − A·β‖². Setting the gradient to zero gives the **normal equation**:

        ```
        (AᵀA) β = Aᵀy        ⇒        β = (AᵀA)⁻¹ Aᵀy
        ```
        """),
        C("""
        X, y = ds.diabetes()                                       # 442 patients, 10 features
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])      # column of ones + features
        beta = np.linalg.solve(A.T @ A, A.T @ y.to_numpy())       # normal equation

        sk = LinearRegression().fit(X, y)
        pd.DataFrame({"by hand": beta, "sklearn": np.r_[sk.intercept_, sk.coef_]},
                     index=["intercept", *X.columns]).round(3)
        """),
        M("""
        The two columns coincide. With a difference worth knowing: scikit-learn does **not** solve the normal
        equation. It uses `scipy.linalg.lstsq`, which goes through the SVD decomposition, because computing AᵀA squares the
        numerical conditioning of the matrix and, with nearly collinear features, loses almost all the
        significant digits. You will see it with your own eyes in exercise A1.
        """),
        M("""
        ## The scikit-learn API

        The cycle is always the same: you create the object, call `fit(X, y)`, read the learned attributes (the ones
        with a trailing underscore, by convention) and call `predict`.
        """),
        C("""
        model = LinearRegression()
        model.fit(X, y)

        print("intercept:", round(model.intercept_, 2))
        print("R2 on the training set:", round(model.score(X, y), 3))
        pd.Series(model.coef_, index=X.columns).round(1)
        """),
        C("""
        model.predict(X.iloc[:3])          # three patients, three predictions
        """),
        M("""
        > **Common mistake.** `fit` wants `X` to be **two-dimensional** (n_observations, n_features) and `y` one-dimensional. With a
        > single feature a 1D array raises an error: you use `X[["column"]]` (double brackets) or
        > `x.reshape(-1, 1)`.
        """),
        C("""
        try:
            LinearRegression().fit(X["bmi"].to_numpy(), y)         # 1D array: wrong
        except ValueError as error:
            print("ValueError:", str(error).splitlines()[0])

        LinearRegression().fit(X[["bmi"]], y).coef_                # one-column DataFrame: correct
        """),
        M("""
        ## When to use it and when to avoid it

        - **Use it** as the baseline of every regression problem, when you need to interpret the effect of each variable
          and when the data are few compared to the number of features.
        - **Avoid it** (or transform the features) when the relationship is clearly curved, when there are outliers you cannot
          remove, or when you must predict outside the range of the data seen: a line goes on forever.
        """),
        M("## Check your understanding\n\n" + details(
            "Why is there a column of ones in the matrix A?",
            "Because the intercept is a coefficient like the others, multiplied by a feature that is always 1. "
            "By adding that column, the line is no longer forced to pass through the origin and the whole problem stays a "
            "single matrix product.") + "\n\n" + details(
            "What changes with `fit_intercept=False` if the data are not centered?",
            "The line is forced to pass through the origin. If the mean value of the target is far from zero, the model "
            "compensates by distorting the coefficients and the residuals no longer have mean zero: it is almost always a mistake.")),
    ]


def lesson_1_2() -> List[Cell]:
    """Cells of lesson 1.2."""
    return [
        M("""
        # Lesson 1.2 · Evaluating a regression model

        A model is not good because it has a high R² on the data you trained it with: it is good if it generalizes. Here you see the
        three fundamental metrics, the role of the baseline and cross-validation, all applied to a real dataset.

        **What you will learn**

        - To compute MAE, RMSE and R² by hand and with scikit-learn, without swapping the order of the arguments.
        - To judge a number by comparing it with a baseline that knows nothing.
        - To estimate the uncertainty of the performance with cross-validation.
        """),
        _setup("""
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        """),
        M("""
        ## The dataset

        California housing: 20,640 districts, 8 features (median income, age of the houses, latitude...) and as target the
        median house value, in hundreds of thousands of dollars. On first use scikit-learn downloads it and
        keeps it in the `data/` folder.
        """),
        C("""
        X, y = ds.california()
        print(X.shape)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        X.head(3)
        """),
        M("""
        ## The three metrics

        ```
        MAE  = mean of |y − ŷ|                     same unit as the target; every error counts in proportion
        RMSE = square root of the mean of (y − ŷ)²  same unit as the target; large errors count more
        R²   = 1 − SSR / SST                        fraction of the error of the mean that the model removes
        ```

        where SSR is the sum of squares of the model residuals and SST the sum of squares of the deviations from the mean of y.
        """),
        C("""
        model = LinearRegression().fit(X_tr, y_tr)
        pred = model.predict(X_te)

        mae = np.abs(y_te - pred).mean()
        rmse = np.sqrt(((y_te - pred) ** 2).mean())
        r2 = 1 - ((y_te - pred) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
        print(f"by hand: MAE {mae:.4f} | RMSE {rmse:.4f} | R2 {r2:.4f}")

        print(f"sklearn: MAE {mean_absolute_error(y_te, pred):.4f} | "
              f"RMSE {mean_squared_error(y_te, pred, squared=False):.4f} | R2 {r2_score(y_te, pred):.4f}")
        """),
        M("""
        > **Common mistake.** All the functions in `sklearn.metrics` want the **true** values first and then the predictions:
        > `r2_score(y_true, y_pred)`. With MAE and MSE swapping the arguments does not change the result because they are symmetric;
        > with R² it does, and it is a silent bug (you will hunt for it in exercise E2).
        >
        > Version note: `squared=False` computes the root in scikit-learn 1.1. In recent versions it has been replaced
        > by `root_mean_squared_error`.
        """),
        M("""
        ## The baseline: what a model that knows nothing does

        An RMSE of 0.73 means nothing until you know what the dumbest possible model scores, the one that always predicts
        the mean of the training set. R² is exactly this comparison made explicit.
        """),
        C("""
        base = DummyRegressor(strategy="mean").fit(X_tr, y_tr)
        pred_base = base.predict(X_te)

        print(f"baseline: RMSE {mean_squared_error(y_te, pred_base, squared=False):.4f} | R2 {r2_score(y_te, pred_base):.4f}")
        print(f"model   : RMSE {rmse:.4f} | R2 {r2:.4f}")
        """),
        M("""
        The R² of the baseline on the test set is slightly **negative**, not zero: the mean of the training set does not match
        that of the test set, so it does a bit worse than predicting the test mean. A negative R² is not a calculation
        error: it means the model does worse than the mean.
        """),
        M("""
        ## Cross-validation

        A single split gives you one number. Cross-validation gives you five, and their spread tells you how far you can
        trust it. In scikit-learn the metrics "to minimize" are passed with the sign flipped (`neg_...`), because the
        internal convention is that the highest score is the best.
        """),
        C("""
        cv = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = -cross_val_score(LinearRegression(), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")

        print("RMSE per fold:", scores.round(4))
        print(f"mean RMSE {scores.mean():.4f} +/- {scores.std():.4f}")
        """),
        M("""
        ## A useful indicator: the RMSE / MAE ratio

        If the errors were all the same size the two values would coincide. The RMSE is always greater than or equal
        to the MAE, and the gap between the two tells you how heavy the tails of the errors are.
        """),
        C("""
        print("RMSE / MAE:", round(rmse / mae, 2))
        cap = 5.0     # in this dataset the price is capped at 5.0: the most expensive houses are all "5"
        capped = (y_te >= cap).to_numpy()
        print("share of houses with a capped price:", round(capped.mean(), 3))
        print("RMSE on the capped ones:", round(np.sqrt(((y_te[capped] - pred[capped]) ** 2).mean()), 3),
              "| on the others:", round(np.sqrt(((y_te[~capped] - pred[~capped]) ** 2).mean()), 3))
        """),
        M("## Check your understanding\n\n" + details(
            "An R² of 0.60 on the training set and 0.58 in cross-validation: overfitting?",
            "No. The gap is small, so the model generalizes almost as well as it fits. Overfitting shows up as a "
            "high training R² and a much lower validation score.") + "\n\n" + details(
            "Why shouldn't the test set be used to choose between two models?",
            "Because choosing the best one by looking at the test set turns it into part of the selection process: the score "
            "you will report for the chosen model is optimistically biased. To choose you use cross-validation "
            "on the training set, and you touch the test set only once at the end.")),
    ]


def lesson_1_3() -> List[Cell]:
    """Cells of lesson 1.3."""
    return [
        M("""
        # Lesson 1.3 · Assumptions and diagnostics

        The coefficients of a linear regression have a meaning only if the model is reasonable for that data. This
        lesson gives you the tools to check it: residual plots, the multicollinearity index (VIF) and the measures
        of leverage and influence.

        **What you will learn**

        - Which assumptions matter for predicting and which for interpreting.
        - To read residual plots and to recognize nonlinearity and heteroscedasticity.
        - To compute VIF, leverage and Cook's distance with a few lines of numpy.
        """),
        _setup("from sklearn.linear_model import LinearRegression\nfrom common.plots import residual_panel"),
        M("""
        ## The assumptions, and what they break

        | Assumption | If it is violated | It compromises |
        |---|---|---|
        | Linearity of the relationship | systematic error | predictions and interpretation |
        | Independence of the errors | wrong standard errors | mostly inference |
        | Constant variance (homoscedasticity) | wrong standard errors | mostly inference |
        | Normal residuals | inaccurate t and F tests on small samples | inference in small samples |
        | No strong multicollinearity | unstable coefficients | interpretation, not predictions |

        If you only care about predicting, linearity is the assumption that really matters. If you want to explain the effect of a
        variable, they all matter.
        """),
        M("""
        ## Looking at the residuals

        The most informative plot is the one of the residuals against the predicted values. With an adequate model it must look like a
        cloud with no structure, centered on zero and of constant height. Any recognizable shape is a signal.
        """),
        C("""
        rng = np.random.default_rng(1)
        n = 300
        x = rng.uniform(1, 10, n)
        cases = {
            "assumptions respected": 2 + 1.5 * x + rng.normal(0, 1.5, n),
            "curved relationship":   2 + 0.6 * (x - 5) ** 2 + rng.normal(0, 1.5, n),
            "growing variance":      2 + 1.5 * x + rng.normal(0, 1, n) * 0.4 * x,
        }
        for name, y in cases.items():
            m = LinearRegression().fit(x.reshape(-1, 1), y)
            residual_panel(y, m.predict(x.reshape(-1, 1)), name)
        """),
        M("""
        In the first case the residuals are a horizontal cloud and the Q-Q plot follows the diagonal. In the second the residuals draw
        a **U**: the line errs low in the middle and high at the sides. In the third they form a **funnel**: the error
        grows with the predicted value. The two pathologies have different remedies (transforming the features or the target in the first
        case, robust estimators or robust standard errors in the second).
        """),
        M("""
        ## Multicollinearity: the VIF

        The **Variance Inflation Factor** of a feature answers one question: how well can the other features
        reconstruct it? It is computed as 1 / (1 − R²), where R² is the one from the regression of that feature on all the others.
        A VIF of 10 means that the variance of its coefficient is inflated tenfold compared to the case of
        uncorrelated features. The usual alarm thresholds are 5 and 10.
        """),
        C("""
        def vif(X):
            \"\"\"VIF of every column: 1 / (1 - R2) of the regression of that column on the others.\"\"\"
            X = pd.DataFrame(X)
            values = {}
            for col in X.columns:
                others = X.drop(columns=col)
                r2 = LinearRegression().fit(others, X[col]).score(others, X[col])
                values[col] = 1 / (1 - r2)
            return pd.Series(values)

        rng = np.random.default_rng(2)
        a = rng.normal(size=200)
        df = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.1, size=200), "c": rng.normal(size=200)})
        print("correlation a-b:", round(df.a.corr(df.b), 3))
        vif(df).round(1)
        """),
        M("""
        ## Leverage and influence

        Not all points have the same power over the model. The **leverage** of an observation measures how far its
        x value is from the center of the data: it is the diagonal element hᵢᵢ of the *hat matrix*
        **H = A(AᵀA)⁻¹Aᵀ**, the one that turns y into ŷ. The mean leverage is p/n, where p is the number of columns of A.

        A high-leverage point has the *potential* to move the line; it really does so only if it also has a large residual.
        **Cook's distance** combines the two things:

        ```
        Dᵢ = eᵢ² / (p · s²) · hᵢᵢ / (1 − hᵢᵢ)²         with  s² = SSR / (n − p)
        ```

        A rule of thumb: Dᵢ > 4/n deserves a look, Dᵢ > 1 is almost always a problem.
        """),
        C("""
        rng = np.random.default_rng(3)
        x = np.r_[rng.uniform(0, 10, 30), 25.0]                 # the last point is far from the others
        y = 2 + 1.5 * x + rng.normal(0, 1.5, len(x))
        y[-1] -= 20                                             # and we move it a lot vertically

        A = np.column_stack([np.ones_like(x), x])
        H = A @ np.linalg.inv(A.T @ A) @ A.T                    # hat matrix: y_hat = H y
        leverage = np.diag(H)

        beta = np.linalg.lstsq(A, y, rcond=None)[0]
        e = y - A @ beta
        s2 = e @ e / (len(x) - 2)
        cook = e ** 2 / (2 * s2) * leverage / (1 - leverage) ** 2

        print("expected mean leverage (p/n):", round(2 / len(x), 3))
        print("leverage of the far point   :", round(leverage[-1], 3))
        print("maximum Cook's distance     :", round(cook.max(), 2), "on point", cook.argmax())
        """),
        M("> **Common mistake.** Judging a linear model by R² and p-value without looking at the residuals: four datasets with the same line and the same R² (Anscombe's quartet) have completely different shapes, and a single point with a lot of leverage can move everything (exercises D2 and D3)."),
        M("## Check your understanding\n\n" + details(
            "Two features have correlation 0.98. Are the model predictions worse?",
            "Almost never. Multicollinearity makes the individual coefficients unstable and hard to interpret, but their "
            "combination, that is the prediction, stays stable. The problem concerns whoever wants to explain the effect of a "
            "single variable, not whoever wants to predict.") + "\n\n" + details(
            "A point has high leverage but a residual of almost zero. Is it a problem?",
            "Usually not: it is a point far from the center that nevertheless sits on the line the others were already drawing. "
            "It increases the precision of the slope estimate. It becomes a problem only if it also has a large residual, and that is "
            "exactly what Cook's distance measures.")),
    ]


def lesson_1_4() -> List[Cell]:
    """Cells of lesson 1.4."""
    return [
        M("""
        # Lesson 1.4 · Regularization and nonlinearity

        Two ways to extend linear regression without changing model family: **penalizing** the coefficients to
        reduce variance (Ridge, Lasso, ElasticNet), and **transforming the features** to capture curved relationships
        (polynomials). Both require scaling the features, for reasons you will see in the cells.

        **What you will learn**

        - How Ridge and Lasso shrink the coefficients as α grows, and why only Lasso sets them to zero.
        - To choose α with `RidgeCV` and `LassoCV`, inside a `Pipeline` with the scaler.
        - To use `PolynomialFeatures` to fit a curve with a model that is linear in the parameters.
        """),
        _setup("""
        from sklearn.linear_model import LassoCV, LinearRegression, Ridge, Lasso, RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        """),
        M("""
        ## Ridge and Lasso in two lines

        A penalty on the size of the coefficients is added to the cost function: the sum of squares for
        Ridge (L2), the sum of absolute values for Lasso (L1). The `alpha` parameter sets the strength: with α = 0 you are back
        to OLS, with a very large α the coefficients are squashed toward zero.

        ```
        Ridge:  minimizes  Σ (y − ŷ)² + α · Σ βⱼ²
        Lasso:  minimizes  Σ (y − ŷ)² + α · Σ |βⱼ|
        ```

        These are the textbook forms. scikit-learn writes the Lasso objective as 1/(2n) · Σ (y − ŷ)² + α · Σ |βⱼ|, dividing the squared error by 2n (Ridge does not divide), so the same `alpha`
        means different things in the two models: an α chosen for the Lasso cannot be reused for the Ridge.

        > **Common mistake.** The penalty hits the numeric value of the coefficients, which depends on the unit of measure of the
        > feature. Without `StandardScaler` the regularization punishes the features based on how you measured them, not on how much they
        > matter. The scaler goes **inside** the pipeline, so it is fitted only on the training fold.
        """),
        C("""
        X, y = ds.diabetes()

        ridge, lasso = {}, {}
        for alpha in [0.01, 1, 10, 100, 1000]:
            ridge[alpha] = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X, y)[-1].coef_
        for alpha in [0.1, 1, 5, 10, 20]:
            lasso[alpha] = make_pipeline(StandardScaler(), Lasso(alpha=alpha)).fit(X, y)[-1].coef_

        print("Ridge: the coefficients shrink but never reach zero")
        display(pd.DataFrame(ridge, index=X.columns).round(1))
        print("Lasso: some coefficients become exactly zero")
        display(pd.DataFrame(lasso, index=X.columns).round(1))
        """),
        M("""
        ## Choosing α

        α is a hyperparameter: you don't choose it by looking at the training error (which always improves as α decreases) but with
        cross-validation. `RidgeCV` and `LassoCV` do it in one go, trying a grid of values. The logarithmic
        scale is the natural choice, because the effects of α show up by orders of magnitude.
        """),
        C("""
        ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25))).fit(X, y)
        lasso_cv = make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=0)).fit(X, y)

        print("alpha chosen by RidgeCV:", round(ridge_cv[-1].alpha_, 3))
        print("alpha chosen by LassoCV:", round(lasso_cv[-1].alpha_, 3))
        print("features kept by the Lasso:", list(X.columns[lasso_cv[-1].coef_ != 0]))
        """),
        M("""
        ## Nonlinearity with a linear model

        "Linear" means linear **in the parameters**, not in the features. If you add x², x³, ... as columns the model
        stays a linear regression, but the curve it draws in the (x, y) plane is no longer a line.
        `PolynomialFeatures` does exactly this.
        """),
        C("""
        d = ds.energy_consumption()                   # energy consumption against temperature: U-shaped relationship
        X_t, y_t = d[["temperature"]], d["consumption"]

        linear = LinearRegression().fit(X_t, y_t)
        quadratic = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()).fit(X_t, y_t)
        print("linear R2   :", round(linear.score(X_t, y_t), 3))
        print("quadratic R2:", round(quadratic.score(X_t, y_t), 3))

        grid = pd.DataFrame({"temperature": np.linspace(-5, 35, 200)})
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(d.temperature, d.consumption, s=12, alpha=0.6)
        ax.plot(grid, linear.predict(grid), label="degree 1", color="#9C4530")
        ax.plot(grid, quadratic.predict(grid), label="degree 2", color="#1F6E5C")
        ax.set(xlabel="temperature (C)", ylabel="consumption (kWh)"); ax.legend();
        """),
        M("""
        > **Warning.** High powers explode: with x between 0 and 30, x¹⁰ reaches 10¹⁴. With polynomials you scale **after**
        > `PolynomialFeatures` (`make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge())`), and the degree must be treated
        > as a hyperparameter to choose in cross-validation, like α.

        ## Transforming the target

        When the target is skewed or the errors grow with the level, it is often better to model the logarithm.
        `TransformedTargetRegressor` applies the transformation in fit and inverts it automatically in predict, so the
        metrics stay in the original unit. It has a price that you will measure in exercise B1: the exponential of a mean of
        logarithms is not the mean.
        """),
        M("## Which one to choose\n\n"
          "| Situation | Reasonable choice |\n|---|---|\n"
          "| Many features, almost all useful | Ridge |\n"
          "| Many features, you suspect that few matter | Lasso |\n"
          "| Groups of highly correlated features | ElasticNet |\n"
          "| Curved relationship in one or two variables | PolynomialFeatures + Ridge, with the degree chosen in CV |\n"
          "| Skewed target or errors proportional to the level | TransformedTargetRegressor with log |"),
        M("## Check your understanding\n\n" + details(
            "Why does the Lasso set coefficients to zero and Ridge doesn't?",
            "Because of the geometry of the penalty. The allowed region for L1 is a diamond with its corners on the axes: the "
            "level curves of the error tend to touch it at a corner, where some coordinates are exactly zero. With L2 "
            "the region is a circle, with no corners, and the contact point is almost always generic.") + "\n\n" + details(
            "Why does `LinearRegression` not need the scaler and `Ridge` does?",
            "Because OLS is scale invariant: rescaling a feature rescales its coefficient in the opposite direction and the "
            "prediction does not change. The Ridge penalty, instead, depends on the value of the coefficients, so it depends on the "
            "units of measure. You will verify it in exercise F2.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_idea_and_least_squares.ipynb", lesson_1_1),
                    ("02_evaluating_the_model.ipynb", lesson_1_2),
                    ("03_assumptions_and_diagnostics.ipynb", lesson_1_3),
                    ("04_regularization_and_nonlinearity.ipynb", lesson_1_4)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
