"""Module 10 - type B projects: a complete problem with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_10_projects"

# ---------------------------------------------------------------------------------------- P1

P1 = dict(
    id="P1", slug="credit_risk_scoring_from_data_to_decision", kind="B", level=3, minutes=110, module=MODULE,
    title="Credit risk scoring: from raw data to the decision",
    goal="""
        Build on your own a model to decide whether to grant a loan, following the steps of lesson 10.1: closed test set, baseline, pipeline with dirty data (missing values, high-cardinality categories, an ordinal), comparison between a linear model and XGBoost, ablation of the missingness indicators, tuning, motivated choice, one-time check on the test set, threshold
        decided with the **costs** (a missed defaulter costs 8 times a wrong refusal) and explanation.
    """,
    dataset="""
        `ds.loans(n=3000, seed=4)`: 8 columns (age, income, job tenure, amount, city, occupation with 40 levels, ordinal education, contract type) and `defaulted` (about 20%). There are missing values with different mechanisms: `income` is missing **more often among defaulters**, `job_tenure` is missing for the unemployed.
    """,
    purpose="""
        Put the whole course together in a single flow, respecting the rules you learned the hard way: no test data in the decisions, preprocessing only in the pipeline, comparison with the baseline and with the variability, choice guided by the standard error (the simplest model if the difference is in the noise), decision with the costs.
    """,
    task=[
        "**Split:** `X_tr, X_te, y_tr, y_te` with `test_size=0.25`, `stratify=y`, `random_state=0`. From here on the test set is touched **only at step 7**.",
        "**Preprocessing:** write `linear_prep()` (ColumnTransformer: numeric with median and **missingness indicator** then scaling; `education` with `OrdinalEncoder(categories=[ds.EDUCATION_ORDER], handle_unknown=\"use_encoded_value\", unknown_value=-1)`, median with indicator, scaling; nominal `city`, `contract_type`, `occupation` with `OneHotEncoder(handle_unknown=\"ignore\", min_frequency=15)`) and `tree_prep()` (numeric and `education` without imputation: XGBoost handles NaN; same one-hots).",
        "**Comparison in CV (training set only):** `cv_table` (`pd.DataFrame`, index `baseline`, `logistic`, `xgb`; columns `auc`, `auc_sd`, `logloss`) with `cross_validate` on the 5 folds `cv` (given). Baseline: `DummyClassifier(strategy=\"prior\")`; logistic: `LogisticRegression(max_iter=2000)`; xgb: `XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, eval_metric=\"logloss\")`. All inside `make_pipeline`.",
        "**Ablation:** `auc_without_indicators` (CV AUC of the logistic with `add_indicator=False` in both imputations) and `indicator_loss` (`cv_table.loc[\"logistic\", \"auc\"] − auc_without_indicators`).",
        "**Tuning:** `search = RandomizedSearchCV(xgb_pipeline, space, n_iter=12, scoring=\"neg_log_loss\", cv=cv, random_state=0)` (`space` is given; the XGBoost step of `make_pipeline` is called `xgbclassifier`) and `tuned_logloss = −search.best_score_`.",
        "**Choice:** `logistic_logloss_sd` (standard deviation of the log-loss across the 5 folds of the logistic) and `chosen` (`\"xgb\"` only if `tuned_logloss` is better than that of the logistic by **more** than `logistic_logloss_sd`; otherwise `\"logistic\"`). `chosen_model` is the corresponding pipeline (`search.best_estimator_` or the logistic).",
        "**One-time check:** train `chosen_model` on the whole training set and save `test_auc` and `test_logloss` on the test set.",
        "**Decision:** with `cross_val_predict(..., method=\"predict_proba\")` on the training set choose `threshold` (grid `np.linspace(0.02, 0.9, 89)`) that minimizes `cost` (given). On the test set save `threshold_cost`, `cost_05` (threshold 0.5) and `refuse_all_cost` (refusing all the applicants).",
        "**Explanation:** `top3` (list of the three original columns with the largest permutation importance on the **test set**, scoring `roc_auc`, `n_repeats=10`, `random_state=0`, in descending order).",
        "**Report:** `report` (dictionary with `model`, `cv_auc`, `test_auc`, `threshold`, `threshold_cost`, `refuse_all_cost`, `top3` and `notes` (a string with two limits of the model that you would point out to whoever uses it)).",
    ],
    setup="""
        import xgboost as xgb
        from scipy.stats import randint
        from sklearn.base import clone
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_validate, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        df = ds.loans(n=3000, seed=4)
        y = df.pop("defaulted")
        X = df
        NUMERIC = ["age", "income", "job_tenure", "amount"]
        NOMINAL = ["city", "contract_type", "occupation"]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        space = {
            "xgbclassifier__max_depth": randint(2, 6), "xgbclassifier__learning_rate": [0.02, 0.05, 0.1], "xgbclassifier__n_estimators": [100, 200, 400],
            "xgbclassifier__min_child_weight": [1, 5, 10], "xgbclassifier__subsample": [0.7, 1.0], "xgbclassifier__colsample_bytree": [0.6, 1.0], "xgbclassifier__reg_lambda": [1, 5, 20],
        }


        def cost(y_true, p, threshold):
            \"\"\"A wrong refusal (false positive) costs 1, an approved defaulter (false negative) costs 8.\"\"\"
            y_true = np.asarray(y_true)
            return int(((p >= threshold) & (y_true == 0)).sum() * 1 + ((p < threshold) & (y_true == 1)).sum() * 8)
    """,
    starter=[
        M("### 1 · Split"),
        C("""
        X_tr = X_te = y_tr = y_te = None
        """),
        M("### 2 · Preprocessing"),
        C("""
        def linear_prep(add_indicator=True):
            ...


        def tree_prep():
            ...
        """),
        M("### 3-4 · Comparison and ablation"),
        C("""
        cv_table = pd.DataFrame(index=["baseline", "logistic", "xgb"], columns=["auc", "auc_sd", "logloss"], dtype=float)
        auc_without_indicators = None
        indicator_loss = None
        """),
        M("### 5-6 · Tuning and choice"),
        C("""
        search = None
        tuned_logloss = None
        logistic_logloss_sd = None
        chosen = None
        chosen_model = None
        """),
        M("### 7-8 · One-time check and decision"),
        C("""
        test_auc = test_logloss = None
        threshold = threshold_cost = cost_05 = refuse_all_cost = None
        """),
        M("### 9 · Explanation and report"),
        C("""
        top3 = []
        report = {}
        """),
    ],
    checks="""
        check(len(X_tr) == 2250 and len(X_te) == 750 and abs(y_tr.mean() - y_te.mean()) < 0.005, "stratified 75/25 split (2250 and 750 rows, same prevalence)", "train_test_split(..., stratify=y)")
        Z = linear_prep().fit_transform(X_tr)
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
        A = tree_prep().fit_transform(X_tr)
        A = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
        check(Z.shape[0] == 2250 and not np.isnan(Z.astype(float)).any() and A.shape[0] == 2250 and np.isnan(A.astype(float)).any() and Z.shape[1] > A.shape[1] - 5, "linear_prep leaves no NaN; tree_prep leaves them (for XGBoost)", "SimpleImputer only in the linear version")
        check(cv_table.notna().all().all() and abs(cv_table.loc["baseline", "auc"] - 0.5) < 0.02 and cv_table.loc["logistic", "auc"] > 0.8 and cv_table.loc["xgb", "auc"] > 0.79 and abs(cv_table.loc["logistic", "auc"] - cv_table.loc["xgb", "auc"]) < 0.03, "baseline at 0.5, logistic above 0.80, xgb above 0.79 and the two models within 0.03 of each other", "cross_validate with scoring=['roc_auc', 'neg_log_loss']")
        check(cv_table.loc["baseline", "logloss"] > cv_table.loc["logistic", "logloss"] + 0.1 and (cv_table.loc[["logistic", "xgb"], "auc_sd"] > 0).all(), "the baseline log-loss exceeds that of the logistic by over 0.1; positive standard deviations for the two models", "-test_neg_log_loss")
        check(auc_without_indicators is not None and indicator_loss > 0.02, "without the missingness indicators the CV AUC drops by over 0.02: the missingness carries information", "SimpleImputer(add_indicator=False)")
        check(search is not None and len(search.cv_results_["params"]) == 12 and tuned_logloss < 0.375, "12 configurations tried; log-loss of the best below 0.375", "RandomizedSearchCV(..., n_iter=12)")
        check(logistic_logloss_sd is not None and 0.005 < logistic_logloss_sd < 0.05 and chosen == "logistic", "the difference between tuned xgb and logistic is smaller than the standard deviation across folds: the logistic is chosen", "logloss(logistic) - tuned_logloss > sd?")
        check(test_auc is not None and 0.77 < test_auc < 0.86 and abs(test_auc - cv_table.loc["logistic", "auc"]) < 0.05, "test AUC between 0.77 and 0.86 and consistent with the CV one (within 0.05)", "chosen_model.fit(X_tr, y_tr) and then evaluate on the test set")
        check(threshold is not None and 0.05 <= threshold <= 0.4 and threshold_cost < 0.75 * cost_05 and threshold_cost < refuse_all_cost, "threshold between 0.05 and 0.4; cost on the test set at least 25% below that of threshold 0.5 and below refusing everyone", "argmin of cost(y_tr, p_cv, s)")
        check(len(top3) == 3 and top3[0] == "income" and "amount" in top3, "`income` is the most important feature and `amount` is among the top three", "permutation_importance on X_te, y_te")
        check(set(report) >= {"model", "cv_auc", "test_auc", "threshold", "threshold_cost", "refuse_all_cost", "top3", "notes"} and isinstance(report["notes"], str) and len(report["notes"]) > 40, "complete report, with two limits in the notes", "a dictionary with the required keys")
    """,
    hint="""
        Linear preprocessing: `ColumnTransformer([("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=ind), StandardScaler()), NUMERIC), ("edu", make_pipeline(OrdinalEncoder(...), SimpleImputer(strategy="median", add_indicator=ind), StandardScaler()), ["education"]), ("nom", OneHotEncoder(...), NOMINAL)])`.
        For `cv_table`: `r = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["roc_auc", "neg_log_loss"])`, `r["test_roc_auc"].mean()`, `.std()`, `-r["test_neg_log_loss"].mean()`. For `top3`: `pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False).index[:3].tolist()`.
    """,
    solution=[
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        print(f"training {len(X_tr)}, test {len(X_te)} | prevalence {y_tr.mean():.3f} / {y_te.mean():.3f}")
        """),
        C("""
        def _ordinal():
            return OrdinalEncoder(categories=[ds.EDUCATION_ORDER], handle_unknown="use_encoded_value", unknown_value=-1)


        def _nominal():
            return OneHotEncoder(handle_unknown="ignore", min_frequency=15)


        def linear_prep(add_indicator=True):
            return ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=add_indicator), StandardScaler()), NUMERIC),
                ("edu", make_pipeline(_ordinal(), SimpleImputer(strategy="median", add_indicator=add_indicator), StandardScaler()), ["education"]),
                ("nom", _nominal(), NOMINAL),
            ])


        def tree_prep():
            return ColumnTransformer([("num", "passthrough", NUMERIC), ("edu", _ordinal(), ["education"]), ("nom", _nominal(), NOMINAL)])
        """),
        C("""
        def new_xgb():
            return xgb.XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, eval_metric="logloss", n_jobs=-1, random_state=0)


        models = {
            "baseline": make_pipeline(linear_prep(), DummyClassifier(strategy="prior")),
            "logistic": make_pipeline(linear_prep(), LogisticRegression(max_iter=2000)),
            "xgb": make_pipeline(tree_prep(), new_xgb()),
        }
        cv_table = pd.DataFrame(index=list(models), columns=["auc", "auc_sd", "logloss"], dtype=float)
        results = {}
        for name, model in models.items():
            results[name] = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["roc_auc", "neg_log_loss"])
            r = results[name]
            cv_table.loc[name] = [r["test_roc_auc"].mean(), r["test_roc_auc"].std(), -r["test_neg_log_loss"].mean()]
        cv_table.round(3)
        """),
        M("""
        **Ablation of the missingness indicators.** `income` is missing 30% of the time among the defaulters and 6% among the others: **the fact that it is missing is a signal**. An imputer that replaces the value without flagging it throws it away.
        """),
        C("""
        r_without = cross_validate(make_pipeline(linear_prep(add_indicator=False), LogisticRegression(max_iter=2000)), X_tr, y_tr, cv=cv, scoring="roc_auc")
        auc_without_indicators = float(r_without["test_score"].mean())
        indicator_loss = cv_table.loc["logistic", "auc"] - auc_without_indicators
        print(f"AUC of the logistic: with indicators {cv_table.loc['logistic', 'auc']:.3f} | without {auc_without_indicators:.3f} | loss {indicator_loss:.3f}")
        """),
        C("""
        search = RandomizedSearchCV(make_pipeline(tree_prep(), new_xgb()), space, n_iter=12, scoring="neg_log_loss", cv=cv, random_state=0, n_jobs=1)
        search.fit(X_tr, y_tr)
        tuned_logloss = float(-search.best_score_)
        print(f"CV log-loss: base xgb {cv_table.loc['xgb', 'logloss']:.4f} | tuned xgb {tuned_logloss:.4f} | logistic {cv_table.loc['logistic', 'logloss']:.4f}")
        print({k.replace('xgbclassifier__', ''): v for k, v in search.best_params_.items()})
        """),
        C("""
        logistic_logloss_sd = float((-results["logistic"]["test_neg_log_loss"]).std())
        improvement = cv_table.loc["logistic", "logloss"] - tuned_logloss
        chosen = "xgb" if improvement > logistic_logloss_sd else "logistic"
        chosen_model = clone(search.best_estimator_) if chosen == "xgb" else clone(models["logistic"])
        print(f"improvement of tuned xgb over the logistic: {improvement:.4f} | standard deviation across folds: {logistic_logloss_sd:.4f} -> I choose: {chosen}")
        """),
        M("""
        **Step 7: the test set is opened once.** The choice is made; here you only measure.
        """),
        C("""
        chosen_model.fit(X_tr, y_tr)
        p_te = chosen_model.predict_proba(X_te)[:, 1]
        test_auc, test_logloss = float(roc_auc_score(y_te, p_te)), float(log_loss(y_te, p_te))
        print(f"test: AUC {test_auc:.3f} (in CV {cv_table.loc['logistic', 'auc']:.3f}) | log-loss {test_logloss:.3f}")
        """),
        C("""
        p_cv = cross_val_predict(clone(chosen_model), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        thresholds = np.linspace(0.02, 0.9, 89)
        threshold = float(thresholds[int(np.argmin([cost(y_tr, p_cv, s) for s in thresholds]))])
        threshold_cost, cost_05 = cost(y_te, p_te, threshold), cost(y_te, p_te, 0.5)
        refuse_all_cost = int((y_te == 0).sum())
        print(f"threshold chosen on the training set: {threshold:.2f} | cost on the test set: threshold {threshold:.2f} -> {threshold_cost} | threshold 0.5 -> {cost_05} | refuse everyone -> {refuse_all_cost}")
        """),
        C("""
        pi = permutation_importance(chosen_model, X_te, y_te, scoring="roc_auc", n_repeats=10, random_state=0)
        importances = pd.Series(pi.importances_mean, index=X_te.columns).sort_values(ascending=False)
        top3 = importances.index[:3].tolist()
        print(importances.round(3).to_string())
        """),
        C("""
        report = {
            "model": chosen, "cv_auc": float(cv_table.loc["logistic", "auc"]), "test_auc": test_auc, "threshold": threshold,
            "threshold_cost": threshold_cost, "refuse_all_cost": refuse_all_cost, "top3": top3,
            "notes": "1) The model is trained on a single period and a single population: it has to be recalibrated if they change. 2) The threshold depends on the assumed 8:1 cost ratio: with other costs it has to be redone.",
        }
        report
        """),
    ],
    learned="""
        - **The correct flow is a sequence of decisions, not of computations:** test set closed after the split, all the preprocessing inside the pipeline, comparison with a baseline on the same folds, choice motivated by the numbers, check only once. The test AUC (0.81) is consistent with the CV one (0.83): it is the sign that no information leaked.
        - **With data of linear structure, the simple model ties XGBoost:** logistic and XGBoost are within 0.01 of AUC (0.825 and 0.818), and the tuning takes the log-loss from 0.364 to 0.363: an improvement of 0.002, much smaller than the standard deviation across folds (about 0.02). The standard error rule picks the logistic: interpretable, fast, no tuning.
        - **Missingness is information:** removing the indicators from the preprocessing takes the AUC of the logistic from 0.825 to 0.788 (−0.037), because `income` is missing mostly among the defaulters. It is the kind of effect that a "clean" imputation hides. XGBoost captures it on its own with native NaN.
        - **The threshold is worth more than the model:** with costs 8:1 the optimal threshold is about 0.09, far below the 0.5 of habit: the cost on the test set drops from 800 to 446 (−44%) and also beats refusing everyone (603). With threshold 0.5 the model costs **more** than granting no loans at all.
        - **The honest explanation is done on the test set:** `income` alone is worth 0.13 of AUC, followed by `amount` and `occupation` (0.04 each). The report also has to include the limits: dependence on the assumed costs and on a single training population.
    """,
    pitfall="""
        Choosing the model by looking at the test set, or choosing the most complex one because "it has the lowest log-loss" even when the difference is much smaller than the variability across folds. If the difference is in the noise, simplicity wins.
    """,
    interview=(
        "How do you choose between a linear model and XGBoost when the CV results are almost equal?",
        """
        I look at the difference compared with the standard deviation across the folds (or I use a repeated CV): if it is within the noise, there is no evidence that the complex one is better, so I choose the simple one for interpretability, training speed and stability. I choose the complex one only if the advantage exceeds the standard error or if there is a domain reason
        (known interactions, a lot of data). The decision is then evaluated with the costs, not with the metric alone.
        """),
)

# ---------------------------------------------------------------------------------------- P2

P2 = dict(
    id="P2", slug="house_prices_from_prediction_to_trust_in_the_prediction", kind="B", level=3, minutes=100, module=MODULE,
    title="House prices: from the prediction to trust in the prediction",
    goal="""
        Build a regression model with XGBoost on California housing and then answer two questions a customer would really ask: **how reliable** is the number you estimate if the houses to predict are in an area never seen (spatial validation) and **with what margin** (90% prediction intervals with quantile regression and their coverage).
    """,
    dataset="""
        `ds.california()`: 20640 districts, 8 features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`); target: median value (in hundreds of thousands of dollars, capped at 5). 80/20 split (`X_tr, X_te, y_tr, y_te`, given). `new_xgb()`: `XGBRegressor(400 trees, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8)`.
    """,
    purpose="""
        Learn that the estimated error depends on the **validation protocol**: with a random partition neighboring districts are in both the training and the validation set, and the error comes out much lower than the one you will have with new areas. And how to give an honest margin of uncertainty, checking that the interval really covers 90% of the cases (and not only where it was calibrated).
    """,
    task=[
        "**Feature engineering:** `enrich(X)` (returns a copy with `rooms_per_person = AveRooms / AveOccup`, `bedroom_share = AveBedrms / AveRooms`, `d_LA`, `d_SF`, `d_SD` (Euclidean distance `np.hypot(Latitude − lat, (Longitude − lon) * 0.8)` from Los Angeles, San Francisco and San Diego: coordinates in `CITIES`) and `d_city`, the minimum of the three). Save `base_rmse` (XGBoost on `X_tr`, RMSE on the test set) and `fe_rmse` (with `enrich`).",
        "**Random and spatial validation:** `blocks` (for each row of `X_tr` the string `\"{floor(Latitude)}_{floor(Longitude)}\"`: cells of 1° by 1°) and `n_blocks`; with `enrich(X_tr)` and `new_xgb()` compute `random_rmse` (`KFold(5, shuffle=True, random_state=0)`) and `spatial_fold_rmse` (list of the 5 RMSEs of `GroupKFold(5)` with `groups=blocks`, scoring `neg_root_mean_squared_error`), `spatial_rmse` (the mean) and `overestimate = spatial_rmse / random_rmse − 1`.",
        "**Prediction intervals:** on a subsample of 10000 training rows (`idx = np.random.default_rng(0).permutation(len(X_tr))[:10000]`, `enrich` features) train two `GradientBoostingRegressor(loss=\"quantile\", alpha=..., n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=0)` with `alpha = 0.05` and `0.95`. Save `test_coverage` (fraction of `y_te` in the interval `[q05, q95]`) and `mean_width`.",
        "**Spatial coverage:** for each of the 5 folds of `GroupKFold(5)` (groups `blocks`) train the same two models on the training part of the fold (subsample of 10000 rows with `np.random.default_rng(0).permutation(len(a))[:10000]`) and measure the coverage on the validation part: `spatial_coverages` (list), `mean_spatial_coverage` and `min_coverage`.",
        "**Report:** `report` with `test_rmse` (=`fe_rmse`), `random_rmse`, `spatial_rmse`, `test_coverage`, `mean_spatial_coverage` and `message` (a sentence for the customer: how much you can expect to be wrong in a new area and with what margin).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GroupKFold, KFold, cross_val_score, train_test_split

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
        CITIES = {"LA": (34.05, -118.24), "SF": (37.77, -122.42), "SD": (32.72, -117.16)}


        def new_xgb():
            return xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=0)


        def quantile_model(alpha):
            return GradientBoostingRegressor(loss="quantile", alpha=alpha, n_estimators=150, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=0)
    """,
    starter=[
        M("### 1 · Feature engineering"),
        C("""
        def enrich(X):
            ...


        base_rmse = fe_rmse = None
        """),
        M("### 2 · Random and spatial validation"),
        C("""
        blocks = None
        n_blocks = None
        random_rmse = None
        spatial_fold_rmse = []
        spatial_rmse = None
        overestimate = None
        """),
        M("### 3-4 · Intervals and coverage"),
        C("""
        test_coverage = None
        mean_width = None
        spatial_coverages = []
        mean_spatial_coverage = None
        min_coverage = None
        """),
        M("### 5 · Report"),
        C("""
        report = {}
        """),
    ],
    checks="""
        Fe = enrich(X_tr)
        check({"rooms_per_person", "bedroom_share", "d_LA", "d_SF", "d_SD", "d_city"} <= set(Fe.columns) and len(Fe) == len(X_tr) and np.isfinite(Fe.to_numpy()).all() and (Fe["d_city"] <= Fe[["d_LA", "d_SF", "d_SD"]].max(axis=1)).all(), "six new columns, with no NaN or infinities, d_city is the minimum", "np.hypot(Latitude - lat, (Longitude - lon) * 0.8)")
        check(base_rmse is not None and fe_rmse < base_rmse - 0.008 and fe_rmse < 0.44, "the new features reduce the test RMSE by over 0.008 (below 0.44)", "XGBoost on enrich(X_tr), test on enrich(X_te)")
        check(blocks is not None and len(blocks) == len(X_tr) and 40 <= n_blocks <= 70, "between 40 and 70 blocks of 1 degree by 1 degree", "np.floor(Latitude), np.floor(Longitude)")
        check(0.42 < random_rmse < 0.48 and len(spatial_fold_rmse) == 5 and close(spatial_rmse, np.mean(spatial_fold_rmse)), "RMSE with random CV between 0.42 and 0.48; five spatial folds", "scoring='neg_root_mean_squared_error' (flip the sign)")
        check(overestimate > 0.25 and max(spatial_fold_rmse) > 1.4 * min(spatial_fold_rmse), "the spatial error exceeds the random one by over 25% and the spatial folds differ by over 40% among themselves", "spatial_rmse / random_rmse - 1")
        check(test_coverage is not None and 0.80 < test_coverage < 0.93 and mean_width > 1.0, "coverage on the test set between 0.80 and 0.93 (nominal 0.90); mean width over 1.0 (100 thousand dollars)", "(y_te >= q05) & (y_te <= q95)")
        check(len(spatial_coverages) == 5 and mean_spatial_coverage < test_coverage - 0.03 and min_coverage < 0.8, "in new areas the mean coverage drops by over 3 points below that of the test set and in the worst fold it is below 0.80", "GroupKFold(5) with the blocks")
        check(set(report) >= {"test_rmse", "random_rmse", "spatial_rmse", "test_coverage", "mean_spatial_coverage", "message"} and isinstance(report["message"], str) and len(report["message"]) > 40, "complete report", "a dictionary with the required keys")
    """,
    hint="""
        `blocks = (np.floor(X_tr["Latitude"]).astype(int).astype(str) + "_" + np.floor(X_tr["Longitude"]).astype(int).astype(str)).to_numpy()`. CV: `cross_val_score(new_xgb(), Fe, y_tr, cv=GroupKFold(5), groups=blocks, scoring="neg_root_mean_squared_error")` and then `-s`. Coverage: `((y_val >= q05.predict(Xv)) & (y_val <= q95.predict(Xv))).mean()`.
    """,
    solution=[
        C("""
        def enrich(X):
            F = X.copy()
            F["rooms_per_person"] = F["AveRooms"] / F["AveOccup"]
            F["bedroom_share"] = F["AveBedrms"] / F["AveRooms"]
            for name, (lat, lon) in CITIES.items():
                F[f"d_{name}"] = np.hypot(F["Latitude"] - lat, (F["Longitude"] - lon) * 0.8)
            F["d_city"] = F[[f"d_{n}" for n in CITIES]].min(axis=1)
            return F


        rmse = lambda a, b: mean_squared_error(a, b, squared=False)
        base_rmse = rmse(y_te, new_xgb().fit(X_tr, y_tr).predict(X_te))
        fe_rmse = rmse(y_te, new_xgb().fit(enrich(X_tr), y_tr).predict(enrich(X_te)))
        print(f"test RMSE: original features {base_rmse:.4f} | with the new features {fe_rmse:.4f} ({1 - fe_rmse / base_rmse:.1%} less)")
        """),
        M("""
        **Spatial validation.** In random cross-validation a district has, in the training set, almost identical neighbors (same neighborhood, same prices). With `GroupKFold` on the **geographic blocks** the districts of the validation fold are in areas the model has never seen: it is the scenario of someone who uses the model for a new city.
        """),
        C("""
        Fe = enrich(X_tr)
        blocks = (np.floor(X_tr["Latitude"]).astype(int).astype(str) + "_" + np.floor(X_tr["Longitude"]).astype(int).astype(str)).to_numpy()
        n_blocks = len(set(blocks))
        random_rmse = float(-cross_val_score(new_xgb(), Fe, y_tr, cv=KFold(5, shuffle=True, random_state=0), scoring="neg_root_mean_squared_error").mean())
        spatial_fold_rmse = (-cross_val_score(new_xgb(), Fe, y_tr, cv=GroupKFold(5), groups=blocks, scoring="neg_root_mean_squared_error")).tolist()
        spatial_rmse = float(np.mean(spatial_fold_rmse))
        overestimate = spatial_rmse / random_rmse - 1
        print(f"{n_blocks} blocks | RMSE with random CV {random_rmse:.3f} | with spatial CV {spatial_rmse:.3f} ({overestimate:+.0%}) | per fold: {np.round(spatial_fold_rmse, 3)}")
        """),
        M("""
        **Intervals with quantile regression.** A model with `loss="quantile"` and `alpha=0.05` estimates the 5% quantile of the price given `X`; with `alpha=0.95` the 95% quantile. The interval `[q05, q95]` should contain the true value in 90% of the cases: we check it.
        """),
        C("""
        idx = np.random.default_rng(0).permutation(len(X_tr))[:10000]
        Fte = enrich(X_te)
        q05 = quantile_model(0.05).fit(Fe.iloc[idx], y_tr.iloc[idx])
        q95 = quantile_model(0.95).fit(Fe.iloc[idx], y_tr.iloc[idx])
        low, high = q05.predict(Fte), q95.predict(Fte)
        test_coverage = float(((y_te >= low) & (y_te <= high)).mean())
        mean_width = float((high - low).mean())
        print(f"coverage on the test set (nominal 0.90): {test_coverage:.3f} | mean width of the interval: {mean_width:.2f} (that is {mean_width * 100:.0f} thousand dollars)")
        """),
        C("""
        spatial_coverages = []
        for a, b in GroupKFold(5).split(Fe, y_tr, blocks):
            sub = a[np.random.default_rng(0).permutation(len(a))[:10000]]
            lo = quantile_model(0.05).fit(Fe.iloc[sub], y_tr.iloc[sub])
            hi = quantile_model(0.95).fit(Fe.iloc[sub], y_tr.iloc[sub])
            yv = y_tr.iloc[b]
            spatial_coverages.append(float(((yv >= lo.predict(Fe.iloc[b])) & (yv <= hi.predict(Fe.iloc[b]))).mean()))
        mean_spatial_coverage, min_coverage = float(np.mean(spatial_coverages)), float(min(spatial_coverages))
        print(f"coverage per spatial fold: {np.round(spatial_coverages, 3)} | mean {mean_spatial_coverage:.3f} | minimum {min_coverage:.3f}")
        """),
        C("""
        report = {
            "test_rmse": float(fe_rmse), "random_rmse": random_rmse, "spatial_rmse": spatial_rmse,
            "test_coverage": test_coverage, "mean_spatial_coverage": mean_spatial_coverage,
            "message": f"In an area never seen the typical error is about {spatial_rmse * 100:.0f} thousand dollars (not {random_rmse * 100:.0f}) and the 90% interval covers on average only {mean_spatial_coverage:.0%} of the districts.",
        }
        report
        """),
    ],
    learned="""
        - **Geographic feature engineering pays:** distances from three big cities and rooms per person take the test RMSE from 0.446 to 0.428 (−4%): a modest gain but almost free, because you add information that trees struggle to build on their own (a distance is a combination of two coordinates).
        - **Random validation underestimates the error in new areas by 40%:** RMSE 0.447 with `KFold` against 0.625 with `GroupKFold` on the geographic blocks. Neighboring districts look alike and the random partition puts them in both the training and the validation set. If the model will be used for a new area, the number to report is the second.
        - **The risk depends on the area:** the five spatial folds give RMSE from 0.48 to 0.81: the estimate with a single validation area can be very different from the mean. That is why you report the mean and the dispersion of the folds, not a single number.
        - **An interval has to be verified, not only built:** the 5%-95% quantiles cover 86% of the cases on the test set (nominal 90%, a bit narrow) and only about 80% in new areas, with one fold at 75%. The "declared" uncertainty is that of the data the model has seen; in new areas a more cautious calibration is needed (wider intervals, calibration per area).
    """,
    pitfall="""
        Reporting the error of a random cross-validation for a model that will be used in new areas (or periods, or on new people): the validation protocol must look like the real use.
    """,
    interview=(
        "How would you estimate the error of a price model to be used in a city not present in the training set?",
        """
        With a geographic block validation (GroupKFold on cells or cities): I keep entire areas out and measure the error on those. The random partition is optimistic because neighbors look alike. I report the mean and the dispersion across the areas, and I check that the prediction intervals keep the declared coverage outside the area too.
        """),
)

EXERCISES = [P1, P2]
