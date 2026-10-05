"""Module 2 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="manual_preprocessing_mixed_dataset", kind="B", level=1, minutes=35, module=MODULE,
    title="Preprocessing a mixed dataset, one column type at a time",
    goal="""
        Turn a dataset with numeric, nominal and ordinal columns into a numeric matrix, using the right scaler and the two
        encoders for each group, keeping the column names, and train a logistic regression.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: four numeric columns (`age`, `income`, `job_tenure`, `amount`), two nominal ones
        (`city`, 12 levels, and `contract_type`, 4), one ordinal (`education`, with order `ds.EDUCATION_ORDER`) and the target
        `defaulted`. In module 3 you will see how to do it in one line with `ColumnTransformer`: here you do it by hand to understand what happens.
    """,
    purpose="""
        Practice `fit` on the training set and `transform` on the test set for three different transformers, and keep track of which column
        of the final matrix corresponds to which variable: without the names, the coefficients of a model are unreadable.
    """,
    task=[
        "Split into training and test (75/25, `stratify=y`, `random_state=0`).",
        "Standardize the numeric columns: `Z_num_tr`, `Z_num_te` (scaler fitted only on the training set).",
        "Encode the nominal ones with `OneHotEncoder(sparse=False, handle_unknown=\"ignore\")`: `Z_nom_tr`, `Z_nom_te`.",
        "Encode the ordinal one with `OrdinalEncoder(categories=[ds.EDUCATION_ORDER])`: `Z_ord_tr`, `Z_ord_te`.",
        "Join the three blocks with `np.hstack` in `Z_tr` and `Z_te`, and build `column_names` (the names of the 21 columns, in order: "
        "numeric, one-hot with `get_feature_names_out`, ordinal).",
        "Train `LogisticRegression(max_iter=1000)` on `Z_tr`, compute `test_auc` and save in `top3` the names of the three columns with "
        "the largest coefficient **in absolute value**.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        df = ds.loans(with_missing=False)
        NUM = ["age", "income", "job_tenure", "amount"]
        NOMINAL = ["city", "contract_type"]
        ORDINAL = ["education"]
        X = df[NUM + NOMINAL + ORDINAL]
        y = df["defaulted"]
    """,
    starter=[
        C("""
        X_tr, X_te, y_tr, y_te = ...
        """),
        C("""
        scaler = StandardScaler()
        Z_num_tr = ...
        Z_num_te = ...
        """),
        C("""
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        Z_nom_tr = ...
        Z_nom_te = ...

        oe = OrdinalEncoder(categories=[ds.EDUCATION_ORDER])
        Z_ord_tr = ...
        Z_ord_te = ...
        """),
        C("""
        Z_tr = ...
        Z_te = ...
        column_names = ...
        Z_tr.shape
        """),
        C("""
        model = LogisticRegression(max_iter=1000)
        test_auc = ...
        top3 = ...
        """),
    ],
    checks="""
        check(len(X_te) == 375 and abs(y_te.mean() - y.mean()) < 0.01, "stratified 75/25 split", "use test_size=0.25 and stratify=y")
        check(Z_tr.shape[1] == 4 + 12 + 4 + 1 and Z_te.shape[1] == Z_tr.shape[1], "21 columns in both training and test",
              "4 numeric + 12 cities + 4 contracts + 1 ordinal = 21")
        check(np.abs(Z_tr[:, :4].mean(axis=0)).max() < 1e-9 and np.abs(Z_te[:, :4].mean(axis=0)).max() > 1e-3,
              "the numeric columns are centered on the training set, not on the test set", "fit the scaler only on X_tr, then transform X_te")
        check(sorted(np.unique(Z_tr[:, -1])) == [0, 1, 2, 3, 4], "the ordinal uses the codes 0-4 in the right order",
              "use OrdinalEncoder(categories=[ds.EDUCATION_ORDER])")
        check(set(np.unique(Z_tr[:, 4:20])) <= {0.0, 1.0}, "the one-hot columns contain only 0 and 1", "columns 4-19 must be one-hot")
        check(len(column_names) == 21 and column_names[:4] == NUM and column_names[-1] == "education" and "contract_type_unemployed" in column_names,
              "column_names is correct", "numeric, then one-hot (get_feature_names_out), then 'education'")
        check(test_auc > 0.65, "the model has a reasonable test AUC", "test_auc should exceed 0.65")
        check(len(top3) == 3 and set(top3) <= set(column_names), "top3 is a list of three column names", "top3 = names of the three columns with the largest |coef|")
    """,
    hint="""
        For the one-hot names: `list(ohe.get_feature_names_out(NOMINAL))`. For top3: `pd.Series(model.coef_[0], index=column_names).abs().nlargest(3).index.tolist()`.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)

        scaler = StandardScaler().fit(X_tr[NUM])
        Z_num_tr, Z_num_te = scaler.transform(X_tr[NUM]), scaler.transform(X_te[NUM])

        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore").fit(X_tr[NOMINAL])
        Z_nom_tr, Z_nom_te = ohe.transform(X_tr[NOMINAL]), ohe.transform(X_te[NOMINAL])

        oe = OrdinalEncoder(categories=[ds.EDUCATION_ORDER]).fit(X_tr[ORDINAL])
        Z_ord_tr, Z_ord_te = oe.transform(X_tr[ORDINAL]), oe.transform(X_te[ORDINAL])
        """),
        C("""
        Z_tr = np.hstack([Z_num_tr, Z_nom_tr, Z_ord_tr])
        Z_te = np.hstack([Z_num_te, Z_nom_te, Z_ord_te])
        column_names = NUM + list(ohe.get_feature_names_out(NOMINAL)) + ORDINAL
        print("shape:", Z_tr.shape, "| first columns:", column_names[:6], "...")
        """),
        C("""
        model = LogisticRegression(max_iter=1000).fit(Z_tr, y_tr)
        test_auc = roc_auc_score(y_te, model.predict_proba(Z_te)[:, 1])
        coef = pd.Series(model.coef_[0], index=column_names)
        top3 = coef.abs().nlargest(3).index.tolist()
        print(f"test AUC: {test_auc:.3f}")
        print(coef[top3].round(2).to_string())
        """),
    ],
    learned="""
        - For every group of columns the cycle is the same: `fit` on the training set, `transform` on training and test. If you get the test
          set wrong (`fit_transform`) you have a bug that exercises E1-E3 make you hunt for.
        - The transformed matrices lose the names: they must be rebuilt in the order in which you concatenated the blocks. An order
          mistake produces coefficients attributed to the wrong variable (exercise E2).
        - A linear model gets a decent AUC with a few lines of preprocessing: the cost of doing it by hand is the effort
          of keeping three `fit` calls aligned, and it is the reason why module 3 introduces `ColumnTransformer` and `Pipeline`.
    """,
    pitfall="""
        Concatenating the blocks in a different order between training and test (for example `np.hstack([Z_nom_te, Z_num_te, ...])`). The
        model does not complain because the number of columns matches, but it applies every coefficient to the wrong column.
    """,
    interview=(
        "Why do you encode `education` with an OrdinalEncoder and not with a one-hot?",
        """
        Because it has a natural order (middle school < ... < phd) and a single column with those codes lets a linear model
        exploit the progression with a single coefficient, with fewer parameters and less noise. A one-hot ignores the
        order and adds columns. If I suspected a non-monotonic effect (for example the risk does not grow linearly with
        the degree), the one-hot would leave the model more freedom.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="imputers_compared", kind="B", level=2, minutes=40, module=MODULE,
    title="Imputers compared: mean, median, KNN and iterative, with known truth",
    goal="""
        Insert random missing values into complete data, impute them with four methods and measure the error with respect to the true
        values **only on the removed cells**, to establish which method reconstructs them best and why.
    """,
    dataset="""
        Synthetic: 800 rows and 5 standardized features, built from two latent factors so that they are correlated with each other
        (`X_true`). 20% of the cells are removed at random (MCAR): `X_miss` and the boolean mask `mask` (True where the value is missing).
    """,
    purpose="""
        Compare the imputers with a **measurable truth**, impossible with real data. You will see why the mean has an error
        equal to the standard deviation and how much is gained by whoever exploits the correlation between the columns.
    """,
    task=[
        "Build the dictionary `imputers` with four objects: `\"mean\"` (`SimpleImputer(strategy=\"mean\")`), `\"median\"`, `\"knn\"` "
        "(`KNNImputer(n_neighbors=5)`) and `\"iterative\"` (`IterativeImputer(random_state=0, max_iter=10)`).",
        "For each one compute `X_imp = fit_transform(X_miss)` and save in `rmse[name]` the RMSE between `X_imp[mask]` and `X_true[mask]`: "
        "only on the missing cells.",
        "Assign to `best` the name of the method with the lowest RMSE.",
        "With `SimpleImputer(strategy=\"mean\", add_indicator=True)` save in `shape_with_indicator` the shape of the output. Why "
        "does it have more columns than the input?",
        "Write in `relative_rmse` a dictionary with the RMSE of each method divided by that of the mean: how much each one gains "
        "compared to the simplest method.",
    ],
    setup="""
        from sklearn.experimental import enable_iterative_imputer          # noqa: F401
        from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

        rng = np.random.default_rng(0)
        n = 800
        X_true = rng.normal(size=(n, 2)) @ rng.normal(size=(2, 5)) + 0.5 * rng.normal(size=(n, 5))
        X_true = (X_true - X_true.mean(axis=0)) / X_true.std(axis=0)                     # 5 standardized, correlated features
        mask = rng.random(X_true.shape) < 0.20                                           # 20% of the cells missing, at random
        X_miss = X_true.copy()
        X_miss[mask] = np.nan
        print("missing cells:", mask.mean().round(3), "| mean correlation between the features:",
              np.abs(np.corrcoef(X_true.T)[np.triu_indices(5, 1)]).mean().round(2))
    """,
    starter=[
        C("""
        imputers = {
            "mean": ...,
            "median": ...,
            "knn": ...,
            "iterative": ...,
        }
        """),
        C("""
        rmse = {}
        for name, imp in imputers.items():
            ...
        best = ...
        rmse
        """),
        C("""
        shape_with_indicator = ...
        relative_rmse = ...
        """),
    ],
    checks="""
        check(set(imputers) == {"mean", "median", "knn", "iterative"}, "four imputers", "you need the keys mean, median, knn, iterative")
        check(set(rmse) == set(imputers) and all(np.isfinite(v) for v in rmse.values()), "one RMSE per method", "rmse must have a key for every imputer")
        check(0.9 < rmse["mean"] < 1.1, "the RMSE of the mean is about 1: the standard deviation of a standardized feature",
              "imputing with the mean the error is the standard deviation of the feature, that is about 1")
        check(rmse["knn"] < rmse["mean"] and rmse["iterative"] < rmse["mean"], "exploiting the other columns beats the mean",
              "knn and iterative should have a lower RMSE than the mean")
        check(best == min(rmse, key=rmse.get), "best consistent with rmse", "best = min(rmse, key=rmse.get)")
        check(best == "iterative", "with linear relationships between the columns the iterative imputer wins",
              "with data built from linear factors, IterativeImputer should win")
        check(shape_with_indicator == (800, 10), "add_indicator adds one column for every column with missing values (5 + 5)",
              "expected shape (800, 10)")
        check(abs(relative_rmse["mean"] - 1.0) < 1e-9 and relative_rmse["iterative"] < 0.8, "the iterative imputer reduces the error by more than 20%",
              "relative_rmse = rmse[name] / rmse['mean']")
    """,
    hint="""
        `X_imp = imp.fit_transform(X_miss)` and then `np.sqrt(np.mean((X_imp[mask] - X_true[mask]) ** 2))`; indexing with the
        boolean mask extracts only the removed cells.
    """,
    solution=[
        C("""
        imputers = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=5),
            "iterative": IterativeImputer(random_state=0, max_iter=10),
        }
        rmse = {}
        for name, imp in imputers.items():
            X_imp = imp.fit_transform(X_miss)
            rmse[name] = np.sqrt(np.mean((X_imp[mask] - X_true[mask]) ** 2))         # error ONLY on the missing cells
        best = min(rmse, key=rmse.get)
        print({k: round(v, 3) for k, v in rmse.items()}, "-> best:", best)
        """),
        C("""
        shape_with_indicator = SimpleImputer(strategy="mean", add_indicator=True).fit_transform(X_miss).shape
        relative_rmse = {k: v / rmse["mean"] for k, v in rmse.items()}
        print("shape with the indicator:", shape_with_indicator)
        print({k: round(v, 2) for k, v in relative_rmse.items()})
        """),
    ],
    learned="""
        - With the mean the error is the standard deviation of the feature (about 1): the imputed value says nothing about the true value,
          so it is wrong on average by one standard deviation.
        - KNN and Iterative use the other columns: when the features are correlated they reduce the error sharply. Where the
          columns are independent there is nothing to exploit and they go back to the level of the mean.
        - `add_indicator=True` adds a binary column for every column that had missing values (here 5 + 5 = 10). With MCAR missing values
          it is not needed, with informative missing values (`income` in the loans) it can be the most useful signal.
    """,
    pitfall="""
        Evaluating an imputer on the RMSE of all the cells (including the non-missing ones, which are unchanged) instead of only on the
        removed ones: the result looks better than it is. And forgetting that the error is computed against a truth that in real
        data you don't have: the simulation serves to choose the method, not to estimate the error in production.
    """,
    interview=(
        "When would you use imputation with the mean instead of KNN or iterative?",
        """
        When the columns are barely correlated (KNN and iterative have nothing to exploit), when the dataset is huge and the
        computation cost matters, or when I need a robust method that is easy to explain and to put in production. With few missing values
        the difference in error is negligible. With many missing values and correlated columns I lose information by not using KNN or iterative.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="transformations_of_skewed_variables", kind="B", level=2, minutes=40, module=MODULE,
    title="Skewed variables: log, Yeo-Johnson and quantiles, inside and outside the range",
    goal="""
        Compare three transformations of skewed variables (log, `PowerTransformer`, `QuantileTransformer`) on the skewness and
        on the performance of a linear model, and verify that the `QuantileTransformer` **cannot extrapolate**.
    """,
    dataset="""
        Synthetic: 1200 rows, two very skewed features `x1` and `x2` (lognormal) and a normal one `x3`, with
        `y = 2·log(x1) − 1.5·log(x2) + x3 + noise`. The linear model on the raw features is therefore wrong, and on log(x) it is correct.
        `X_tr`/`X_te` (random 50/50 split) and `X_extreme`: the same points as `X_te` with `x1` and `x2` multiplied by 20, that is
        **outside the range** seen in training.
    """,
    purpose="""
        Understand that the transformations are not interchangeable: the quantile transformer produces perfect distributions but
        **saturates outside the range of the data**, while a continuous function like the log keeps working (and Yeo-Johnson degrades gradually).
    """,
    task=[
        "Compute `skewness`, a DataFrame with `x1` and `x2` as rows and as columns `raw`, `log`, `yeo_johnson`, `quantile` "
        "(`scipy.stats.skew` on the training set after each transformation; transformers fitted on the training set).",
        "For each transformation build a pipeline (`FunctionTransformer(np.log)` on x1 and x2 only: use "
        "`ColumnTransformer`) with a final `Ridge(alpha=1)`, train it on `X_tr` and save in `r2_normal` and `r2_extreme` the R² on the "
        "normal test set and on the out-of-range test set.",
        "Assign to `best_normal` and `best_extreme` the name of the transformation with the highest R² in the two cases.",
        "Explain in `answer` (\"a\", \"b\" or \"c\") why the `QuantileTransformer` does badly out of range:\n"
        "   - **a**: the quantiles are computed on the training set: any value beyond the training maximum gets the same "
        "value (saturation), so the model does not see the difference between 3 and 20 times the maximum;\n"
        "   - **b**: the QuantileTransformer returns non-finite values out of range;\n"
        "   - **c**: Ridge does not work with Gaussian data.",
    ],
    setup="""
        from scipy import stats
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, PowerTransformer, QuantileTransformer

        rng = np.random.default_rng(0)
        n = 1200
        X = pd.DataFrame({"x1": rng.lognormal(0, 1, n), "x2": rng.lognormal(1, 0.8, n), "x3": rng.normal(size=n)})
        y = 2 * np.log(X.x1) - 1.5 * np.log(X.x2) + X.x3 + rng.normal(0, 0.5, n)
        X_tr, X_te = X.iloc[:600], X.iloc[600:]
        y_tr, y_te = y.iloc[:600], y.iloc[600:]
        X_extreme = X_te.copy()
        X_extreme[["x1", "x2"]] *= 20                                   # outside the training range
        y_extreme = 2 * np.log(X_extreme.x1) - 1.5 * np.log(X_extreme.x2) + X_extreme.x3 + (y_te - (2 * np.log(X_te.x1) - 1.5 * np.log(X_te.x2) + X_te.x3))
        COL = ["x1", "x2"]
    """,
    starter=[
        C("""
        transforms = {
            "raw": "passthrough",
            "log": FunctionTransformer(np.log),
            "yeo_johnson": PowerTransformer(),
            "quantile": QuantileTransformer(n_quantiles=200, output_distribution="normal"),
        }
        skewness = pd.DataFrame(index=COL, columns=list(transforms), dtype=float)
        for name, t in transforms.items():
            ...
        skewness.round(2)
        """),
        C("""
        r2_normal, r2_extreme = {}, {}
        for name, t in transforms.items():
            model = ...
            ...
        best_normal = ...
        best_extreme = ...
        """),
        C("""
        answer = ...
        """),
    ],
    checks="""
        check(list(skewness.columns) == ["raw", "log", "yeo_johnson", "quantile"], "correct columns", "raw, log, yeo_johnson, quantile")
        check((skewness["raw"] > 2).all() and (skewness.drop(columns="raw").abs() < 0.3).all().all(),
              "the three transformations reduce the skewness (from above 2 to below 0.3 in absolute value)", "the skewness of the transformed ones must be close to zero")
        check(r2_normal["log"] > r2_normal["raw"] + 0.2, "the log improves the linear model a lot compared to the raw",
              "r2 with the log must exceed the raw one by more than 0.2")
        check(abs(r2_normal["yeo_johnson"] - r2_normal["log"]) < 0.05, "Yeo-Johnson is equivalent to the log on the normal data", "on the data inside the range they must almost coincide")
        check(r2_extreme["log"] > r2_extreme["quantile"] + 0.3, "out of range the log stays good and the quantiles collapse",
              "on the extreme data the quantile transformer should be much worse than the log")
        check(best_extreme in ("log", "yeo_johnson") and best_extreme != "quantile", "the best out of range is not the quantile transformer", "look at r2_extreme")
        check(answer == "a", "correct explanation", "reread: what does QuantileTransformer do with a value beyond the maximum seen?")
    """,
    hint="""
        Pipeline: `make_pipeline(ColumnTransformer([("t", t, COL)], remainder="passthrough"), Ridge(alpha=1))`; for `"passthrough"` it works
        the same (`ColumnTransformer` accepts the string). Skewness: `stats.skew(t.fit_transform(X_tr[[c]]).ravel())`; for `"passthrough"` use `X_tr[c]`.
    """,
    solution=[
        C("""
        transforms = {
            "raw": "passthrough",
            "log": FunctionTransformer(np.log),
            "yeo_johnson": PowerTransformer(),
            "quantile": QuantileTransformer(n_quantiles=200, output_distribution="normal"),
        }
        skewness = pd.DataFrame(index=COL, columns=list(transforms), dtype=float)
        for name, t in transforms.items():
            for c in COL:
                z = X_tr[[c]].to_numpy() if isinstance(t, str) else t.fit_transform(X_tr[[c]])
                skewness.loc[c, name] = stats.skew(np.asarray(z).ravel())
        skewness.round(2)
        """),
        C("""
        r2_normal, r2_extreme = {}, {}
        for name, t in transforms.items():
            model = make_pipeline(ColumnTransformer([("t", t, COL)], remainder="passthrough"), Ridge(alpha=1)).fit(X_tr, y_tr)
            r2_normal[name] = r2_score(y_te, model.predict(X_te))
            r2_extreme[name] = r2_score(y_extreme, model.predict(X_extreme))
        best_normal = max(r2_normal, key=r2_normal.get)
        best_extreme = max(r2_extreme, key=r2_extreme.get)
        print("R2 on the normal test set :", {k: round(v, 3) for k, v in r2_normal.items()})
        print("R2 on the extreme test set:", {k: round(v, 3) for k, v in r2_extreme.items()})
        """),
        C("""
        answer = "a"
        """),
    ],
    learned="""
        - Log and Yeo-Johnson bring the skewness close to zero and let the linear model find the true relationship again
          (log): R² goes up from about 0.71 to about 0.96 compared to the raw features.
        - The `QuantileTransformer` produces perfect Gaussian distributions but it is a step function computed on the training set:
          out of range it **saturates**. A customer with an income 20 times the maximum seen in training is indistinguishable from the maximum.
          It is a real limit of all the transformations based on quantiles. Out of range the log stays at 0.96 because it coincides with the true relationship, Yeo-Johnson drops to about 0.56 (its power approximates the log only inside the range) and the quantiles to about 0.46.
        - Yeo-Johnson generalizes the log (it also accepts zero and negative values) and estimates the power by itself: it is a good default choice
          for skewed features, but it remains an approximation: if you know the true relationship (here the log), using it beats the automatic estimate.
    """,
    pitfall="""
        Choosing the transformation by looking only at the skewness (`skew` close to zero) or only at the in-range test set. Neither
        tells you what happens when the production data leave the range: it is a common case for incomes, prices and counts.
    """,
    interview=(
        "You have a feature with a long tail. Log or QuantileTransformer?",
        """
        I start from the model: for trees and boosting nothing is needed (monotonic transformations do not change the splits). For a linear
        or distance-based model I prefer the log or Yeo-Johnson: they are continuous, monotonic functions that keep
        working out of range and keep an interpretation (percentage changes). I use the quantile transformer when the
        distribution is very irregular (bimodal) and I don't expect values outside the range seen.
        """),
)

EXERCISES = [B1, B2, B3]
