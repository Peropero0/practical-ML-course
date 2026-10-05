"""Module 4 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="loans_pipeline_and_ablation", kind="B", level=2, minutes=50, module=MODULE,
    title="A complete logistic on the loans: pipeline and ablation study",
    goal="""
        Build a classification pipeline on mixed data with missing values (numerics with missingness indicators, nominals of low and high cardinality, an ordinal),
        measure in cross-validation the **contribution of each piece** with an ablation study and read the coefficients as odds ratios.
    """,
    dataset="""
        `ds.loans()` with the missing values (1500 applications, 19% defaulters): stratified 75/25 split. The missing values of `income` are **linked to the target**
        (whoever did not repay the loan more often has the income missing), those of `job_tenure` to the contract type.
    """,
    purpose="""
        Put together what you learned in modules 2-4 and learn the **ablation** reasoning: remove one piece at a time and measure what it costs. You will discover
        that the most useful information is not in the values, but in **which** values are missing.
    """,
    task=[
        "Write `build(indicators, categoricals)` that returns a `Pipeline`: `FunctionTransformer(add_ratio)` (given) → `ColumnTransformer` → `LogisticRegression(C=1, max_iter=2000)`. "
        "Numerics `age, income, job_tenure, amount, ratio`: median imputation (with `add_indicator=indicators`) and standardization. If `categoricals` is true add: `city`, "
        "`contract_type`, `occupation` with most-frequent imputation and `OneHotEncoder(handle_unknown=\"ignore\")`, and `education` with imputation, `OrdinalEncoder` "
        "in the order of `EDUCATION_ORDER` and scaler.",
        "With `StratifiedKFold(5, shuffle=True, random_state=0)` on the training set compute the mean AUC of four variants and save it in `cv_auc` (a `pd.Series`, index: "
        "`numerics only`, `+ indicators`, `+ categoricals`, `all`), where `+ categoricals` means categoricals **without** indicators.",
        "Save `indicators_gain` (difference between `+ indicators` and `numerics only`) and `categoricals_gain` (between `+ categoricals` and `numerics only`).",
        "Train `all` on the training set and save `test_auc`. Then read the coefficients (the names: `final[1].get_feature_names_out()`): save in `missing_income_odds` the odds ratio of 'income missing versus present' "
        "**in original units** (exponential of the coefficient of `num__missingindicator_income` divided by the standard deviation with which the `StandardScaler` scaled that column) and in "
        "`top4_numeric` the names of the 4 features of the `num__` block with the largest coefficient in absolute value.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

        df = ds.loans()
        X, y = df.drop(columns="defaulted"), df["defaulted"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        NUM = ["age", "income", "job_tenure", "amount", "ratio"]


        def add_ratio(X):
            X = X.copy()
            X["ratio"] = X["amount"] / X["income"]              # NaN where income is missing: the imputer will handle it
            return X
    """,
    starter=[
        C("""
        def build(indicators, categoricals):
            ...
        """),
        C("""
        cv_auc = pd.Series(dtype=float)
        indicators_gain = None
        categoricals_gain = None
        cv_auc
        """),
        C("""
        final = build(True, True).fit(X_tr, y_tr)
        test_auc = None
        missing_income_odds = None
        top4_numeric = None
        """),
    ],
    checks="""
        check(list(cv_auc.index) == ["numerics only", "+ indicators", "+ categoricals", "all"], "four variants in the right order", "index: numerics only, + indicators, + categoricals, all")
        check(cv_auc["all"] > 0.80 and cv_auc["all"] == cv_auc.max(), "the complete variant is the best (AUC > 0.80)", "all = indicators + categoricals")
        check(close(indicators_gain, cv_auc["+ indicators"] - cv_auc["numerics only"]) and indicators_gain > 0.04,
              "the missingness indicators bring over 0.04 of AUC", "add_indicator=True in the SimpleImputer of the numerics")
        check(close(categoricals_gain, cv_auc["+ categoricals"] - cv_auc["numerics only"]) and 0.01 < categoricals_gain < indicators_gain,
              "the categoricals contribute less than the indicators", "categoricals_gain must be positive but lower than indicators_gain")
        check(abs(test_auc - cv_auc["all"]) < 0.06 and close(test_auc, roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])), "test AUC consistent with the cross-validated one", "roc_auc_score on the test set with predict_proba")
        check(missing_income_odds is not None and missing_income_odds > 2, "having the income missing more than doubles the default odds", "exp(coef / scale) for the indicator column; the scale is final[1].named_transformers_['num'][-1].scale_")
        check(top4_numeric is not None and set(top4_numeric) >= {"num__ratio", "num__missingindicator_income"}, "ratio and the income-missing indicator are among the four heaviest numerics", "sort by |coef| the names starting with 'num__'")
    """,
    hint="""
        Numerics: `make_pipeline(SimpleImputer(strategy="median", add_indicator=indicators), StandardScaler())`. Names and coefficients:
        `names = final[1].get_feature_names_out()` (the names are given by the `ColumnTransformer`, second step of the pipeline; `FunctionTransformer` does not expose them); `coef = pd.Series(final[-1].coef_[0], index=names)`.
        The scale: `scale = final[1].named_transformers_["num"][-1].scale_` (one per column of the `num__` block, in the same order as the names); position of the indicator: `list(names).index("num__missingindicator_income")`.
    """,
    solution=[
        C("""
        def build(indicators, categoricals):
            groups = [("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=indicators), StandardScaler()), NUM)]
            if categoricals:
                groups.append(("nom", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")),
                               ["city", "contract_type", "occupation"]))
                groups.append(("ord", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder(categories=[ds.EDUCATION_ORDER]), StandardScaler()),
                               ["education"]))
            return make_pipeline(FunctionTransformer(add_ratio), ColumnTransformer(groups, sparse_threshold=0), LogisticRegression(C=1, max_iter=2000))
        """),
        C("""
        variants = {"numerics only": (False, False), "+ indicators": (True, False), "+ categoricals": (False, True), "all": (True, True)}
        cv_auc = pd.Series({name: cross_val_score(build(*args), X_tr, y_tr, cv=cv, scoring="roc_auc").mean() for name, args in variants.items()})
        indicators_gain = cv_auc["+ indicators"] - cv_auc["numerics only"]
        categoricals_gain = cv_auc["+ categoricals"] - cv_auc["numerics only"]
        print(cv_auc.round(4).to_string())
        print(f"gain of the missingness indicators: {indicators_gain:+.4f} | gain of the categoricals: {categoricals_gain:+.4f}")
        """),
        C("""
        final = build(True, True).fit(X_tr, y_tr)
        test_auc = roc_auc_score(y_te, final.predict_proba(X_te)[:, 1])
        names = final[1].get_feature_names_out()
        coef = pd.Series(final[-1].coef_[0], index=names)
        scale = final[1].named_transformers_["num"][-1].scale_                                     # standard deviation used for every column of the numeric block
        k = list(names).index("num__missingindicator_income")
        missing_income_odds = float(np.exp(coef.iloc[k] / scale[k]))                               # from "per 1 std dev" to "missing versus present"
        num = coef[[n for n in names if n.startswith("num__")]]
        top4_numeric = list(num.abs().nlargest(4).index)
        print(f"test AUC: {test_auc:.4f} | odds ratio of income missing (versus present): {missing_income_odds:.2f}")
        print(num.round(3).to_string())
        """),
    ],
    learned="""
        - Ablation answers the question "what is this piece worth?" with a number. Here **the missingness indicators are worth about 0.06 of AUC, the categoricals about 0.04**:
          the most informative fact is that the income **was not declared**. If you just imputed, you would throw away the strongest signal in the dataset.
        - A missing value is not always "noise to fill in": when it is missing for a reason linked to the target (`MNAR`), the absence is a feature. That is why `add_indicator=True`
          is almost free and almost always worth trying.
        - Odds ratios are readable only if you know **in which units** they are: the coefficient of a standardized feature is "per one standard deviation", that of a 0/1 indicator must be reported in original
          units (dividing by the scale of the scaler) to say "income missing versus present". Comparing coefficients of standardized features and of dummies without this step (the occupation dummies
          end up in the top positions) is the classic way of telling false stories: it is the theme of D1.
        - Look at the table of the numeric coefficients: `missingindicator_ratio` and `missingindicator_income` have the same coefficient (0.315) because they are **the same column** (the ratio is missing exactly when the
          income is missing). With a regularized logistic it does no harm, but with an unregularized linear model the perfect collinearity would be a problem: it is a case where `add_indicator` must be used with judgment.
        - The whole construction sits in a single object that lives inside `cross_val_score`: imputation, scaler and one-hot are learned inside every fold, with no leakage.
    """,
    pitfall="""
        Filling the missing values with the median **without indicators** and concluding that "income matters little": with the imputation you replaced the signal (whoever does not declare) with a
        typical value equal for everyone. The AUC without indicators is noticeably lower, and no error tells you so.
    """,
    interview=(
        "You have columns with missing values: how do you decide how to treat them?",
        """
        First I understand **why** they are missing (random, linked to another column, linked to the target). If the mechanism is informative I add a missingness indicator
        on top of a simple imputation (median or most frequent) inside the pipeline, and measure with an ablation in cross-validation how much it is worth. More
        sophisticated imputations (KNN, iterative) are justified only if they really gain over the median plus the indicator.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="optimal_threshold_with_costs", kind="B", level=3, minutes=55, module=MODULE,
    title="Choosing the threshold with the real costs",
    goal="""
        Given a cost for each kind of error (an unpaid loan costs 10, a good customer turned down costs 1), choose the decision threshold **using only the training set**
        (out-of-sample predictions with `cross_val_predict`), compare it with the theoretical threshold $c_{FP}/(c_{FP}+c_{FN})$ and with `class_weight="balanced"`, and measure the cost on the test set.
    """,
    dataset="""
        `ds.loans(with_missing=False)` with `ratio = amount/income`: features `ratio, age, job_tenure, income`, stratified 70/30 split. Model: logistic with scaler.
    """,
    purpose="""
        Take classification from the metric to **money**: threshold 0.5 is not a neutral default but an implicit choice that assumes equal costs. You will learn
        a clean procedure (threshold chosen on the training set, test set touched once) and the theoretical rule for calibrated probabilities.
    """,
    task=[
        "Define `cost(y, p, threshold)`: total cost if you reject (`p >= threshold`) and do not grant the loan. A defaulter accepted (false negative) costs 10; a non-defaulter turned down (false positive) costs 1.",
        "Get with `cross_val_predict(model, X_tr, y_tr, cv=cv, method=\"predict_proba\")` the **out-of-sample** probabilities on the training set and compute `oof_costs`: the cost for each threshold of `thresholds` "
        "(already defined). Save `best_threshold` (minimum-cost threshold) and `theoretical_threshold = 1/11`.",
        "On the **test set** (model trained on the training set) compute `test_costs`, a dictionary with: `threshold 0.5`, `best threshold`, `theoretical threshold`, `class_weight balanced (0.5)`, "
        "`approve all`, `reject all`.",
        "Save `saving = 1 − cost(best threshold) / cost(threshold 0.5)` on the test set.",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.3, stratify=df["defaulted"], random_state=0)
        y_tr, y_te = y_tr.to_numpy(), y_te.to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    """,
    starter=[
        C("""
        def cost(y, p, threshold):
            ...
        """),
        C("""
        oof = None
        oof_costs = None
        best_threshold = None
        theoretical_threshold = None
        """),
        C("""
        test_costs = {}
        saving = None
        pd.Series(test_costs)
        """),
    ],
    checks="""
        check(cost(np.array([1, 0]), np.array([0.2, 0.9]), 0.5) == 11, "cost: a defaulter not caught (10) + a good one turned down (1)", "FN weighs 10, FP weighs 1")
        check(oof is not None and oof.shape == (len(y_tr),) and len(oof_costs) == len(thresholds), "out-of-sample predictions and one cost per threshold", "predict_proba[:, 1] from cross_val_predict")
        check(0.05 <= best_threshold <= 0.2 and abs(best_threshold - theoretical_threshold) < 0.06, "the best threshold is close to the theoretical 1/11 (about 0.09)", "with calibrated probabilities the best threshold is c_FP / (c_FP + c_FN)")
        check(test_costs["best threshold"] < 0.7 * test_costs["threshold 0.5"], "with the best threshold the cost drops by over 30% compared to 0.5", "use the threshold found on the training set, applied to the test set")
        check(test_costs["best threshold"] < test_costs["class_weight balanced (0.5)"] and test_costs["best threshold"] < min(test_costs["approve all"], test_costs["reject all"]),
              "it beats class_weight balanced and the two trivial policies", "compare the six costs")
        check(close(saving, 1 - test_costs["best threshold"] / test_costs["threshold 0.5"]) and saving > 0.3, "correct saving", "1 - best cost / cost at 0.5")
        check(test_costs["threshold 0.5"] > test_costs["reject all"], "at threshold 0.5 the model costs more than rejecting everyone: the default threshold can lose value", "compare threshold 0.5 and reject all")
    """,
    hint="""
        `cost`: `pred = p >= threshold; 10 * ((y == 1) & ~pred).sum() + 1 * ((y == 0) & pred).sum()`. Best threshold: `thresholds[np.argmin(oof_costs)]`. For `balanced`
        train a second model with `LogisticRegression(class_weight="balanced")` and use threshold 0.5. "Approve all" = threshold 1.1 (nobody rejected).
    """,
    solution=[
        C("""
        def cost(y, p, threshold):
            reject = p >= threshold
            return int(10 * ((y == 1) & ~reject).sum() + 1 * ((y == 0) & reject).sum())
        """),
        C("""
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]         # predictions never seen by the model that produces them
        oof_costs = np.array([cost(y_tr, oof, t) for t in thresholds])
        best_threshold = float(thresholds[np.argmin(oof_costs)])
        theoretical_threshold = 1 / 11
        print(f"minimum-cost threshold (training set, out-of-sample): {best_threshold} | theoretical: {theoretical_threshold:.3f}")
        print("the five best:", thresholds[np.argsort(oof_costs)[:5]])
        """),
        C("""
        model.fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        balanced = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")).fit(X_tr, y_tr)
        test_costs = {
            "threshold 0.5": cost(y_te, p_te, 0.5),
            "best threshold": cost(y_te, p_te, best_threshold),
            "theoretical threshold": cost(y_te, p_te, theoretical_threshold),
            "class_weight balanced (0.5)": cost(y_te, balanced.predict_proba(X_te)[:, 1], 0.5),
            "approve all": cost(y_te, p_te, 1.1),
            "reject all": cost(y_te, p_te, 0.0),
        }
        saving = 1 - test_costs["best threshold"] / test_costs["threshold 0.5"]
        print(pd.Series(test_costs).to_string())
        print(f"saving compared to threshold 0.5: {saving:.0%}")
        """),
    ],
    learned="""
        - With asymmetric costs the right threshold is far from 0.5: here about **0.09**, which coincides with the theoretical rule $c_{FP}/(c_{FP}+c_{FN}) = 1/11$. It is a
          general result: if the probabilities are calibrated, you should act whenever the expected cost of not acting exceeds that of acting.
        - The threshold is chosen on the **training set** with out-of-sample probabilities (`cross_val_predict`), and the test set evaluates it once. Choosing it on the test set
          would be optimizing on the grade that judges you.
        - `class_weight="balanced"` moves the model in the right direction but "blindly": it does not know the costs (here 10 to 1, not the ratio between the classes). The cost you get
          is intermediate. The explicit threshold is more flexible and leaves the probabilities calibrated.
        - At threshold 0.5 the model costs **more** than rejecting all the applicants: a model "with AUC 0.78" can lose money if the threshold is wrong. Always compare with
          the trivial policies.
    """,
    pitfall="""
        Choosing the threshold by looking at the ROC curve "at the point closest to the top left corner": it is the threshold that assumes equal costs and 50% prevalence. With
        real costs and prevalences the right point of the curve is another one.
    """,
    interview=(
        "How do you choose the threshold of a classifier in production?",
        """
        I start from the costs (or from the operational constraint: for example "I can check at most 100 cases a day"), not from the metric. With calibrated probabilities the theoretical threshold
        is c_FP/(c_FP+c_FN); in practice I search for it on a grid with out-of-sample predictions on the training set and validate it once on the test set. I recheck it when the
        prevalence or the costs change, because the threshold depends on both.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="multiclass_digits_and_choice_of_c", kind="B", level=2, minutes=45, module=MODULE,
    title="Multiclass classification: handwritten digits",
    goal="""
        Solve a 10-class problem with a logistic: choose `C` with `LogisticRegressionCV`, compare the multinomial approach with **one-vs-rest**, summarize the
        performance with the right averages (macro, per class) and identify the main confusions.
    """,
    dataset="""
        `load_digits`: 1797 8×8 images (64 pixels) of digits from 0 to 9, stratified 70/30 split: `X_tr, X_te, y_tr, y_te`.
    """,
    purpose="""
        Bring to the multiclass case everything you saw in the binary one: what changes in `coef_`, in `predict_proba`, in the metrics and how a 10×10 confusion matrix is read.
    """,
    task=[
        "Build `cv_model`: pipeline `StandardScaler` + `LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, scoring=\"neg_log_loss\", max_iter=3000)`. Train it and save `chosen_C` (float).",
        "With that `C` train `multinomial` (`LogisticRegression`) and `ovr` (`OneVsRestClassifier(LogisticRegression(...))`), both in a pipeline with a scaler. "
        "Save in `table` (DataFrame, index `multinomial` and `ovr`) the columns `accuracy`, `macro_f1` and `log_loss` on the test set.",
        "From the multinomial model get `per_class_recall` (array of 10) and `hardest_class` (the digit with the lowest recall).",
        "Save `main_confusion`: the pair `(true, predicted)` with the largest number of errors (off the diagonal) in the confusion matrix of the test set.",
        "Save `coef_shape` (shape of `coef_` of the multinomial) and `prob_sum` (sum of the predicted probabilities for the first test image).",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
        from sklearn.metrics import confusion_matrix, f1_score, log_loss, recall_score
        from sklearn.model_selection import train_test_split
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        digits = load_digits()
        X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.3, stratify=digits.target, random_state=0)
    """,
    starter=[
        C("""
        cv_model = ...
        chosen_C = None
        """),
        C("""
        multinomial = ...
        ovr = ...
        table = pd.DataFrame(index=["multinomial", "ovr"], columns=["accuracy", "macro_f1", "log_loss"], dtype=float)
        table
        """),
        C("""
        per_class_recall = None
        hardest_class = None
        main_confusion = None
        coef_shape = None
        prob_sum = None
        """),
    ],
    checks="""
        check(isinstance(chosen_C, float) and any(np.isclose(chosen_C, c) for c in np.logspace(-4, 1, 6)), "C chosen among the grid values", "chosen_C = float(cv_model[-1].C_[0])")
        check(table.notna().all().all() and table.loc["multinomial", "accuracy"] > 0.95, "multinomial accuracy above 0.95", "use the same C for both models")
        check(table.loc["multinomial", "log_loss"] < table.loc["ovr", "log_loss"], "the multinomial has a better log-loss than one-vs-rest", "the multinomial probabilities sum to 1 by construction, the ovr ones are normalized afterwards")
        check(close(table.loc["multinomial", "macro_f1"], f1_score(y_te, multinomial.predict(X_te), average="macro")), "correct macro F1", "average='macro'")
        check(len(per_class_recall) == 10 and hardest_class == int(np.argmin(per_class_recall)) and hardest_class == 8, "the hardest digit is the 8", "recall_score(..., average=None)")
        check(main_confusion == (8, 1), "the most frequent error is 8 mistaken for 1", "confusion matrix with the diagonal zeroed, then argmax")
        check(tuple(coef_shape) == (10, 64) and close(prob_sum, 1.0), "coef_ has one row per class and the probabilities sum to 1", "shape (10, 64)")
    """,
    hint="""
        `chosen_C = float(cv_model[-1].C_[0])`. `cm = confusion_matrix(y_te, pred); np.fill_diagonal(cm, 0); np.unravel_index(cm.argmax(), cm.shape)`.
        `recall_score(y_te, pred, average=None)`.
    """,
    solution=[
        C("""
        cv_model = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=np.logspace(-4, 1, 6), cv=5, scoring="neg_log_loss", max_iter=3000)).fit(X_tr, y_tr)
        chosen_C = float(cv_model[-1].C_[0])
        print("C chosen in cross-validation:", chosen_C, "| values tried:", cv_model[-1].Cs_)
        """),
        C("""
        multinomial = make_pipeline(StandardScaler(), LogisticRegression(C=chosen_C, max_iter=3000)).fit(X_tr, y_tr)
        ovr = make_pipeline(StandardScaler(), OneVsRestClassifier(LogisticRegression(C=chosen_C, max_iter=3000))).fit(X_tr, y_tr)
        table = pd.DataFrame(index=["multinomial", "ovr"], columns=["accuracy", "macro_f1", "log_loss"], dtype=float)
        for name, m in {"multinomial": multinomial, "ovr": ovr}.items():
            table.loc[name] = [m.score(X_te, y_te), f1_score(y_te, m.predict(X_te), average="macro"), log_loss(y_te, m.predict_proba(X_te))]
        table.round(4)
        """),
        C("""
        pred = multinomial.predict(X_te)
        per_class_recall = recall_score(y_te, pred, average=None)
        hardest_class = int(np.argmin(per_class_recall))
        cm = confusion_matrix(y_te, pred)
        errors = cm.copy(); np.fill_diagonal(errors, 0)
        main_confusion = tuple(int(i) for i in np.unravel_index(errors.argmax(), errors.shape))
        coef_shape = multinomial[-1].coef_.shape
        prob_sum = float(multinomial.predict_proba(X_te[:1]).sum())
        print("recall per class:", per_class_recall.round(3), "| the hardest:", hardest_class)
        print("most frequent error (true, predicted):", main_confusion, "with", errors.max(), "cases")
        """),
    ],
    learned="""
        - In the multiclass case `coef_` has one row per class and `predict_proba` is a softmax: the probabilities sum to 1 **by construction**. `OneVsRestClassifier` trains 10 independent
          binary models and normalizes afterwards: its probabilities are less coherent (here the log-loss is about 35% worse, 0.14 against 0.10).
        - `LogisticRegressionCV` chooses `C` on a grid with cross-validation and refits the model on the whole training set: it is the equivalent of `GridSearchCV` for a logistic, much faster
          because along the grid it reuses the previous solutions (warm start).
        - **The accuracy (0.97) hides the differences between classes**: the recall goes from 0.92 (the 8) to 1.0 (0 and 7). Looking at `average=None` and the confusion matrix tells you where to intervene;
          the macro F1 treats all the classes alike, which is what you want when the rare classes matter.
        - With only 540 test examples, one error more or less moves the accuracy by 0.2 points: differences between models of this order of magnitude are noise.
    """,
    pitfall="""
        Reporting only the accuracy on a multiclass problem: a class with recall 0.6 can vanish in the average if it is rare. For every multiclass problem always look at
        recall and precision **per class** before trusting a single number.
    """,
    interview=(
        "What is the difference between macro, micro and weighted average?",
        """
        Macro: simple mean of the per-class metrics, every class weighs the same (sensitive to rare classes). Micro: counts TP, FP, FN over all the classes together
        (for single-label multiclass it coincides with the accuracy). Weighted: mean weighted by the number of examples of each class. I use macro when the rare
        classes matter, micro or accuracy when the total is what matters.
        """),
)

EXERCISES = [B1, B2, B3]
