"""Module 2 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_02_preprocessing"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="scaler_per_model_with_and_without_outliers", kind="F", level=3, minutes=50, module=MODULE,
    title="Which scaler for which model, with and without outliers",
    goal="""
        Compare five preprocessing choices (none, Standard, MinMax, Robust, Quantile) on four models (KNN, logistic
        regression, tree, SVM) first on clean data and then with a small number of severe outliers, and answer with numbers "which scaler
        is needed by whom".
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns (`age`, `income`, `job_tenure`, `amount`) and the target `defaulted`.
        `X` is the clean version. In `X_out` 2% of the rows (30 out of 1500, chosen at random) have `income` and `amount` multiplied by 100: a
        unit-of-measure error (cents instead of euros), typical of real data.
    """,
    purpose="""
        Stop applying `StandardScaler` out of habit. Every model reacts differently to scaling, and every scaler reacts
        differently to outliers: the right choice is a **pair** (model, scaler), and you evaluate it in cross-validation.
    """,
    task=[
        "Write `evaluate_all(X, y)`: for every model and every scaler build `make_pipeline(scaler, model)` and compute the mean AUC in "
        "cross-validation (`cv` provided, `scoring=\"roc_auc\"`). Return a DataFrame with the scalers on the rows and the models on the columns. "
        "Use the dictionaries `MODELS` and `SCALERS` already defined.",
        "Save `clean_table = evaluate_all(X, y)` and `outlier_table = evaluate_all(X_out, y)`.",
        "Save in `tree_invariant` `True` if, in **both** tables, the AUC of the tree varies by less than 0.01 across all the scalers.",
        "Compute `outlier_loss`: a DataFrame (`clean_table − outlier_table`) and find `most_fragile_scaler` (the scaler with the largest "
        "mean loss over the three scale-sensitive models: `knn`, `logistic`, `svm`) and `most_robust_scaler` (the one with the smallest loss).",
        "Assign to `best_knn_out` the name of the scaler with the highest AUC for the KNN in `outlier_table`.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        df = ds.loans(with_missing=False)
        NUM = ["age", "income", "job_tenure", "amount"]
        X = df[NUM].to_numpy()
        y = df["defaulted"].to_numpy()
        X_out = X.copy()
        outlier_rows = np.random.default_rng(0).choice(len(X), 30, replace=False)
        X_out[np.ix_(outlier_rows, [1, 3])] *= 100                 # unit error on income and amount
        cv = StratifiedKFold(5, shuffle=True, random_state=0)

        MODELS = {"knn": lambda: KNeighborsClassifier(15), "logistic": lambda: LogisticRegression(max_iter=2000),
                  "tree": lambda: DecisionTreeClassifier(max_depth=5, random_state=0), "svm": lambda: SVC(C=1.0)}
        SCALERS = {"none": lambda: "passthrough", "standard": lambda: StandardScaler(), "minmax": lambda: MinMaxScaler(),
                   "robust": lambda: RobustScaler(), "quantile": lambda: QuantileTransformer(n_quantiles=200, output_distribution="normal")}
    """,
    starter=[
        C("""
        def evaluate_all(X, y):
            tab = pd.DataFrame(index=list(SCALERS), columns=list(MODELS), dtype=float)
            for scaler_name, make_scaler in SCALERS.items():
                for model_name, make_model in MODELS.items():
                    ...
            return tab
        """),
        C("""
        clean_table = evaluate_all(X, y)
        outlier_table = evaluate_all(X_out, y)
        display(clean_table.round(3)); display(outlier_table.round(3))
        """),
        C("""
        tree_invariant = ...
        outlier_loss = ...
        most_fragile_scaler = ...
        most_robust_scaler = ...
        best_knn_out = ...
        """),
    ],
    checks="""
        check(clean_table.shape == (5, 4) and outlier_table.shape == (5, 4), "two tables of 5 scalers x 4 models", "rows: scalers, columns: models")
        check(bool(tree_invariant) and np.ptp(clean_table["tree"]) < 0.01 and np.ptp(outlier_table["tree"]) < 0.01,
              "the tree does not depend on the scaler (variation below 0.01 in both tables)", "monotonic transformations do not change the splits")
        check(abs(clean_table.loc["standard", "knn"] - clean_table.loc["none", "knn"]) < 0.02,
              "on clean data scaling does not change the KNN: the informative features are also the ones on the larger scale",
              "on clean data KNN with and without StandardScaler should be close (less than 0.02)")
        check(outlier_table.loc["standard", "knn"] < outlier_table.loc["none", "knn"] - 0.05,
              "with outliers StandardScaler makes the KNN WORSE than not scaling at all",
              "with outliers StandardScaler should be worse than no scaler for the KNN (by more than 0.05)")
        check(outlier_table.loc["minmax", "knn"] < clean_table.loc["minmax", "knn"] - 0.03,
              "with outliers MinMax makes the KNN lose more than 0.03 of AUC", "outliers squash all the normal data with MinMax")
        check(most_fragile_scaler in ("minmax", "standard"), "the most fragile scaler is based on minimum/maximum or mean/deviation", "the most fragile is the one with the largest mean loss")
        check(most_robust_scaler in ("robust", "quantile"), "the most robust scaler is based on quantiles", "the most robust is robust or quantile")
        check(best_knn_out in ("robust", "quantile"), "with outliers the best scaler for the KNN is based on quantiles", "look at the knn column of outlier_table")
    """,
    hint="""
        Inside the double loop: `pipe = make_pipeline(make_scaler(), make_model())` and `tab.loc[scaler_name, model_name] = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()`.
        The mean loss over the sensitive models: `outlier_loss[["knn", "logistic", "svm"]].mean(axis=1)`; `idxmax()` and `idxmin()`.
    """,
    solution=[
        C("""
        def evaluate_all(X, y):
            tab = pd.DataFrame(index=list(SCALERS), columns=list(MODELS), dtype=float)
            for scaler_name, make_scaler in SCALERS.items():
                for model_name, make_model in MODELS.items():
                    pipe = make_pipeline(make_scaler(), make_model())
                    tab.loc[scaler_name, model_name] = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc").mean()
            return tab
        """),
        C("""
        clean_table = evaluate_all(X, y)
        outlier_table = evaluate_all(X_out, y)
        print("AUC with clean data"); display(clean_table.round(3))
        print("AUC with 2% of the rows having a unit error"); display(outlier_table.round(3))
        """),
        C("""
        tree_invariant = bool(np.ptp(clean_table["tree"]) < 0.01 and np.ptp(outlier_table["tree"]) < 0.01)
        outlier_loss = clean_table - outlier_table
        sensitive = outlier_loss[["knn", "logistic", "svm"]].mean(axis=1)
        most_fragile_scaler = sensitive.idxmax()
        most_robust_scaler = sensitive.idxmin()
        best_knn_out = outlier_table["knn"].idxmax()
        print("mean loss (scale-sensitive models):", sensitive.round(3).to_dict())
        print("most fragile:", most_fragile_scaler, "| most robust:", most_robust_scaler, "| best scaler for the KNN with outliers:", best_knn_out)
        """),
    ],
    learned="""
        - **Tree**: nothing changes, with any scaler and with or without outliers (AUC between 0.715 and 0.719). Scaling is wasted time.
        - **On clean data** scaling makes no difference for the KNN (0.756 against 0.755): here the most informative features (income and
          amount) are also the ones with the larger scale, so the raw distance favors them by chance. Scaling is not
          always an improvement: it is a **protection** against the cases where the scale does not coincide with importance.
        - **With outliers** the scalers based on mean or minimum/maximum become harmful: for the KNN StandardScaler drops to 0.657 and
          MinMaxScaler to 0.617, worse than not scaling (0.751); the logistic goes from 0.78 to 0.64 and 0.62. The normal data get squashed
          into a tiny interval and the distances between customers become almost equal.
        - **RobustScaler and QuantileTransformer** use quantiles and resist: the QuantileTransformer reaches 0.758 (KNN) and 0.788
          (logistic) even with outliers. At the cost of losing the magnitude of the outlier, and with the extrapolation limits seen in B3.
    """,
    pitfall="""
        Comparing scalers on clean data and concluding that "they are the same", or that scaling is not needed. The differences show up with real
        data, which has outliers. When in doubt you measure in cross-validation, you don't choose out of habit.
    """,
    interview=(
        "You have a dataset with outliers and a KNN. Which preprocessing do you propose, and how would you validate it?",
        """
        I would start from a quantile-based scaler (`RobustScaler`, or `QuantileTransformer` if the distribution is very irregular),
        because minimum/maximum and mean/standard deviation get dragged by the outliers. I would compare it with `StandardScaler` in cross-validation inside
        a Pipeline, for the final model, and I would also check whether the outliers are errors (to correct) or real cases (to keep).
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="encoders_for_high_cardinality", kind="F", level=3, minutes=50, module=MODULE,
    title="Six ways to encode `occupation` (40 levels), with two models",
    goal="""
        Compare six strategies for a high-cardinality categorical variable (none, one-hot, one-hot with rare categories
        grouped, ordinal, frequency, out-of-fold target encoding) with a logistic regression and a gradient boosting, in cross-validation.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: four numeric columns, `contract_type` (one-hot), `occupation` (40 levels, 6 of which with fewer than 20
        rows; each one has a true effect on the risk) and the target `defaulted`. `build(occ_encoder, model)` builds the
        complete pipeline: standardized numerics, one-hot contract, `occupation` with the given encoder (or left out if `None`).
    """,
    purpose="""
        Understand that the choice of the encoder depends on the **model**: the logistic needs an encoding that makes the effect
        linear, boosting can isolate the categories by itself. And quantify, with numbers, the value of target encoding.
    """,
    task=[
        "Fill the dictionary `encoders` with six entries: `\"no occupation\"` (`None`), `\"one-hot\"` (`OneHotEncoder(handle_unknown=\"ignore\")`), "
        "`\"one-hot rare grouped\"` (`OneHotEncoder(handle_unknown=\"infrequent_if_exist\", min_frequency=30)`), `\"ordinal\"` "
        "(`OrdinalEncoder(handle_unknown=\"use_encoded_value\", unknown_value=-1)`), `\"frequency\"` (`FrequencyEncoder()`) and `\"target OOF\"` "
        "(`TargetEncoderOOF(m=10)`).",
        "For each entry and for each of the two models (`logistic`, `boosting`) compute the mean AUC in cross-validation (`cv`, "
        "`scoring=\"roc_auc\"`) of `build(enc, model)`. Save `table` (rows: encoders, columns: `logistic`, `boosting`).",
        "Save `best_logistic` and `best_boosting`: the names of the encoders with the highest AUC for each model.",
        "Compute `logistic_te_gain`: AUC of the logistic with `target OOF` minus AUC with `no occupation`.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
        from common.transformers import FrequencyEncoder, TargetEncoderOOF

        df = ds.loans(with_missing=False)
        NUM = ["age", "income", "job_tenure", "amount"]
        NOM, OCC = ["contract_type"], ["occupation"]
        X = df[NUM + NOM + OCC]
        y = df["defaulted"]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)


        def build(occ_encoder, model):
            parts = [("num", StandardScaler(), NUM), ("nom", OneHotEncoder(handle_unknown="ignore"), NOM)]
            if occ_encoder is not None:
                parts.append(("occ", occ_encoder, OCC))
            return make_pipeline(ColumnTransformer(parts, sparse_threshold=0), model)


        MODELS = {"logistic": lambda: LogisticRegression(max_iter=2000),
                  "boosting": lambda: HistGradientBoostingClassifier(max_iter=100, random_state=0)}
        print("occupations with fewer than 20 rows:", int((X.occupation.value_counts() < 20).sum()))
    """,
    starter=[
        C("""
        encoders = {
            "no occupation": ...,
            "one-hot": ...,
            "one-hot rare grouped": ...,
            "ordinal": ...,
            "frequency": ...,
            "target OOF": ...,
        }
        """),
        C("""
        table = pd.DataFrame(index=list(encoders), columns=list(MODELS), dtype=float)
        for enc_name, enc in encoders.items():
            for model_name, make in MODELS.items():
                ...
        table.round(3)
        """),
        C("""
        best_logistic = ...
        best_boosting = ...
        logistic_te_gain = ...
        """),
    ],
    checks="""
        check(list(encoders) == ["no occupation", "one-hot", "one-hot rare grouped", "ordinal", "frequency", "target OOF"] and encoders["no occupation"] is None,
              "six encoders, the first one is None", "respect the order and the names required")
        check(table.shape == (6, 2) and table.notna().all().all(), "complete 6 x 2 table", "rows: encoders, columns: logistic, boosting")
        check(table.loc["ordinal", "logistic"] < table.loc["one-hot", "logistic"], "with the logistic the ordinal is worse than the one-hot",
              "a linear model cannot use an integer code with no order")
        check(table.loc["target OOF", "logistic"] > table.loc["no occupation", "logistic"] + 0.005,
              "out-of-fold target encoding adds useful information for the logistic", "target encoding should improve the AUC compared to not using the occupation")
        check(best_logistic == table["logistic"].idxmax() and best_boosting == table["boosting"].idxmax(),
              "best encoders consistent with the table", "use idxmax() on the columns of the table")
        check(close(logistic_te_gain, table.loc["target OOF", "logistic"] - table.loc["no occupation", "logistic"]),
              "target encoding gain computed", "gain = AUC(target OOF) - AUC(no occupation)")
        check(best_logistic != "ordinal" and best_boosting != "no occupation", "the best encoders are not the two trivial cases", "review the table")
    """,
    hint="""
        Inside the double loop: `table.loc[enc_name, model_name] = cross_val_score(build(enc, make()), X, y, cv=cv, scoring="roc_auc").mean()`.
        `make` is a function that returns a new model: call it with `make()`.
    """,
    solution=[
        C("""
        encoders = {
            "no occupation": None,
            "one-hot": OneHotEncoder(handle_unknown="ignore"),
            "one-hot rare grouped": OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30),
            "ordinal": OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            "frequency": FrequencyEncoder(),
            "target OOF": TargetEncoderOOF(m=10),
        }
        """),
        C("""
        table = pd.DataFrame(index=list(encoders), columns=list(MODELS), dtype=float)
        for enc_name, enc in encoders.items():
            for model_name, make in MODELS.items():
                table.loc[enc_name, model_name] = cross_val_score(build(enc, make()), X, y, cv=cv, scoring="roc_auc").mean()
        table.round(3)
        """),
        C("""
        best_logistic = table["logistic"].idxmax()
        best_boosting = table["boosting"].idxmax()
        logistic_te_gain = table.loc["target OOF", "logistic"] - table.loc["no occupation", "logistic"]
        print("best for the logistic:", best_logistic, "| for boosting:", best_boosting)
        print(f"gain of target encoding on the logistic: {logistic_te_gain:+.3f} of AUC")
        """),
    ],
    learned="""
        - With the **logistic** ordinal (0.786) and frequency (0.786) add nothing compared to dropping the column (0.787): a numeric
          code imposes a linear effect across occupations that have no order. One-hot (0.805), one-hot with rare grouped
          (0.803) and target encoding (0.802) give every occupation its own weight: about +0.02 of AUC.
        - With **boosting** the **ordinal** wins (0.771 against 0.760 for the one-hot and 0.744 for target encoding): trees isolate a
          category with two splits, even from a code with no order, and target encoding compresses the occupation into a single number
          that the trees cannot decompose again. The best encoder **depends on the model**.
        - Among one-hot, rare grouped and target encoding with the logistic the differences (0.802-0.805) are in the noise: there is no clear
          winner. Target encoding has the advantage of producing a single column (with 500 categories it is decisive).
        - Target encoding works **only** if it is out-of-fold and inside the pipeline (exercises A3 and E3); here the pipeline guarantees it.
    """,
    pitfall="""
        Choosing the encoder on the AUC of a single model, or on the value of a single fold. The differences between encoders are often of the order
        of 0.01: they must be read with the standard deviation across folds and repeated with different seeds before concluding.
    """,
    interview=(
        "For a variable with 500 categories and a gradient boosting model, which encoding do you use?",
        """
        I would start from the native handling of categoricals (LightGBM, CatBoost, or `HistGradientBoostingClassifier` with `categorical_features`),
        which avoids the explosion of columns and does not require choosing an order. Alternatively out-of-fold target encoding with smoothing, or
        frequency. One-hot with 500 columns slows things down and fragments the data; ordinal works with trees but with no quality guarantees.
        I would compare the options in cross-validation inside the pipeline.
        """),
)

EXERCISES = [F1, F2]
