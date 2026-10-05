"""Module 4 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="coefficients_odds_ratios_and_omitted_variables", kind="D", level=3, minutes=55, module=MODULE,
    title="Reading the coefficients: odds ratios, collinearity and omitted variables",
    goal="""
        Interpret the coefficients of a logistic correctly: tell odds from probability, measure collinearity (correlation and VIF), see
        how the same coefficient changes **sign and size** depending on the other variables in the model, and verify it with a bootstrap interval.
    """,
    dataset="""
        `ds.loans(with_missing=False)` with `ratio = amount/income`, `age`, `job_tenure` and `unemployed` (1 if `contract_type` is "unemployed"). The features are
        standardized inside `coefficients(columns, d, yy)` (given), which returns the coefficients as a `pd.Series` (logistic with `C=1e4`, almost unregularized).
    """,
    purpose="""
        Dismantle the reflex "the coefficient is the effect of the variable". A coefficient is the effect **all else being equal among the variables in the model**: change the model, the coefficient changes.
        You will learn to see the signals (correlation, VIF), to verify the stability (bootstrap) and to suspect an omitted variable.
    """,
    task=[
        "Compute `corr_age_tenure` (Pearson correlation between `age` and `job_tenure`) and `vif` (`pd.Series` with the variance inflation factor of each of the three features "
        "`ratio, age, job_tenure`: $1/(1-R^2)$ where $R^2$ comes from the linear regression of that feature on the other two).",
        "With the function `coefficients` compute `coef_a` (columns `ratio, age, job_tenure`), `coef_b` (`ratio, age`) and `coef_c` (`ratio, job_tenure`). "
        "Answer with two booleans: `age_matters_alone` (does the coefficient of `age` in `coef_b` exceed 0.1 in absolute value?) and `age_matters_given_tenure` (in `coef_a` does it exceed 0.25?).",
        "**Bootstrap** (200 resamplings with `rng = np.random.default_rng(0)`, indices `rng.integers(0, n, n)`): compute the 95% interval (2.5 and 97.5 percentiles) of the coefficient of `age` "
        "in model A (`interval_a`) and in model B (`interval_b`), as tuples `(low, high)`.",
        "**Omitted variable.** Add `unemployed` to model A: save `coef_e`. By how much (in absolute value) does the coefficient of `age` and the one of `job_tenure` shrink compared to A? "
        "Save the two ratios `|coef_e| / |coef_a|` in `shrinkage` (dictionary with keys `age` and `job_tenure`).",
        "**Odds versus probability.** With model A, for an average customer (all features standardized to 0) save `base_prob` (default probability) and `after_prob` (after +1 std dev of `ratio`), "
        "`ratio_or` (odds ratio of `ratio`) and `prob_delta = after_prob - base_prob`. Check that the odds get multiplied by `ratio_or` but the probability does **not** get multiplied.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        df["unemployed"] = (df["contract_type"] == "unemployed").astype(int)
        y = df["defaulted"].to_numpy()


        def coefficients(columns, d=df, yy=y):
            \"\"\"Coefficients (on standardized features) of an almost unregularized logistic.\"\"\"
            m = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000)).fit(d[columns], yy)
            return pd.Series(m[-1].coef_[0], index=columns)


        cols_a = ["ratio", "age", "job_tenure"]
        print(f"{len(df)} customers, {y.mean():.1%} defaulters")
    """,
    starter=[
        M("### 1. Collinearity"),
        C("""
        corr_age_tenure = None
        vif = None
        """),
        M("### 2. The same coefficient in three models"),
        C("""
        coef_a = None
        coef_b = None
        coef_c = None
        age_matters_alone = None
        age_matters_given_tenure = None
        """),
        M("### 3. Bootstrap"),
        C("""
        interval_a = None
        interval_b = None
        """),
        M("### 4. An omitted variable"),
        C("""
        coef_e = None
        shrinkage = None
        """),
        M("### 5. Odds versus probability"),
        C("""
        model_a = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000)).fit(df[cols_a], y)
        base_prob = None
        after_prob = None
        ratio_or = None
        prob_delta = None
        """),
    ],
    checks="""
        check(0.8 < corr_age_tenure < 0.9, "age and job_tenure are highly correlated (about 0.85)", "np.corrcoef or df.corr()")
        check(vif is not None and vif["ratio"] < 1.5 and vif["age"] > 3 and vif["job_tenure"] > 3, "the VIF flags the collinearity between age and tenure (above 3), not for ratio", "VIF = 1 / (1 - R^2) from the linear regression of each feature on the other two")
        check(close(coef_a["age"], coefficients(cols_a)["age"]) and close(coef_b["age"], coefficients(["ratio", "age"])["age"]), "coefficients of the three models computed with the given function", "coefficients(columns)")
        check(age_matters_alone is False and age_matters_given_tenure is True, "alone age does not matter (coef close to 0); given tenure it does, and with a positive sign", "compare |coef_b['age']| with 0.1 and coef_a['age'] with 0.25")
        check(interval_a[0] > 0, "in model A the bootstrap interval of age lies entirely above zero", "2.5 and 97.5 percentiles of the 200 coefficients")
        check(interval_b[0] < 0 < interval_b[1], "in model B the interval of age contains zero", "age alone has no effect distinguishable from zero")
        check(shrinkage["age"] < 0.5 and shrinkage["job_tenure"] < 0.5, "by adding `unemployed`, both coefficients shrink to less than half", "|coef_e| / |coef_a| for age and job_tenure")
        check(close((after_prob / (1 - after_prob)) / (base_prob / (1 - base_prob)), ratio_or) and prob_delta < (ratio_or - 1) * base_prob,
              "the odds get multiplied by the odds ratio; the probability goes up far less than 'x times'", "prob = sigmoid(intercept) for the average customer; after_prob = sigmoid(intercept + coefficient of ratio)")
    """,
    hint="""
        `vif[c] = 1 / (1 - LinearRegression().fit(df[others], df[c]).score(df[others], df[c]))`. Bootstrap: `i = rng.integers(0, n, n); coefficients(cols, df.iloc[i], y[i])`.
        Average customer: `z0 = model_a[-1].intercept_[0]`; `base_prob = 1 / (1 + np.exp(-z0))`; `z1 = z0 + model_a[-1].coef_[0][0]`.
    """,
    solution=[
        C("""
        corr_age_tenure = df["age"].corr(df["job_tenure"])
        vif = pd.Series({c: 1 / (1 - LinearRegression().fit(df[[a for a in cols_a if a != c]], df[c]).score(df[[a for a in cols_a if a != c]], df[c])) for c in cols_a})
        print(f"age-tenure correlation: {corr_age_tenure:.3f}\\nVIF:\\n{vif.round(2).to_string()}")
        """),
        C("""
        coef_a = coefficients(cols_a)
        coef_b = coefficients(["ratio", "age"])
        coef_c = coefficients(["ratio", "job_tenure"])
        age_matters_alone = bool(abs(coef_b["age"]) > 0.1)
        age_matters_given_tenure = bool(coef_a["age"] > 0.25)
        print(pd.DataFrame({"A (age + tenure)": coef_a, "B (age only)": coef_b, "C (tenure only)": coef_c}).round(3))
        """),
        M("""
        Alone, age has no effect (coefficient 0.02), but **given tenure** it raises the risk (0.34). The same holds, with the opposite sign, for tenure. The pair behaves like
        two variables working together: what matters are their **differences**, not the single values. See what happens with the bootstrap.
        """),
        C("""
        rng = np.random.default_rng(0)
        n = len(df)
        boot_a, boot_b = [], []
        for _ in range(200):
            i = rng.integers(0, n, n)
            boot_a.append(coefficients(cols_a, df.iloc[i], y[i])["age"])
            boot_b.append(coefficients(["ratio", "age"], df.iloc[i], y[i])["age"])
        interval_a = tuple(np.percentile(boot_a, [2.5, 97.5]))
        interval_b = tuple(np.percentile(boot_b, [2.5, 97.5]))
        print("95% bootstrap interval of the coefficient of age: model A", np.round(interval_a, 3), "| model B", np.round(interval_b, 3))
        """),
        C("""
        coef_e = coefficients(cols_a + ["unemployed"])
        shrinkage = {c: abs(coef_e[c]) / abs(coef_a[c]) for c in ["age", "job_tenure"]}
        print(pd.DataFrame({"A": coef_a, "A + unemployed": coef_e}).round(3))
        print("ratio between the coefficients with and without `unemployed`:", {k: round(v, 2) for k, v in shrinkage.items()})
        """),
        M("""
        **What happened.** In the data whoever is unemployed has by construction `job_tenure = 0` and a much higher risk, and does not appear in model A. All else being equal in tenure, an
        older customer is more often an unemployed one (so more at risk) and a low `job_tenure` serves to "detect" the unemployed. It is not an effect of age: it is an **omitted variable** that the two coefficients
        split between them. When you add it, the two effects shrink to about a quarter of the previous value.
        """),
        C("""
        model_a = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000)).fit(df[cols_a], y)
        z0, w_ratio = model_a[-1].intercept_[0], model_a[-1].coef_[0][0]
        base_prob = 1 / (1 + np.exp(-z0))
        after_prob = 1 / (1 + np.exp(-(z0 + w_ratio)))
        ratio_or = float(np.exp(w_ratio))
        prob_delta = after_prob - base_prob
        print(f"average customer: probability {base_prob:.3f} -> {after_prob:.3f} after +1 std dev of ratio (odds ratio {ratio_or:.2f})")
        print(f"the odds get multiplied by {(after_prob / (1 - after_prob)) / (base_prob / (1 - base_prob)):.2f}, the probability by {after_prob / base_prob:.2f} (and goes up by {prob_delta:.3f}, not by {(ratio_or - 1) * base_prob:.3f})")
        """),
    ],
    learned="""
        - A coefficient is the effect **all else being equal among the variables in the model**. With `age` and `job_tenure` (correlation 0.85, VIF about 4) the coefficient of age goes from 0.02 (alone)
          to +0.34 (with tenure): same data, different "truths" depending on the model.
        - The bootstrap tells you how **stable** the coefficients are on the sample, not whether they are **right**: in model A the interval of `age` is above zero (it is well estimated), and yet it comes from an omitted variable.
          Stable does not mean causal.
        - With `unemployed` in the model the two coefficients collapse (to about a quarter of the previous value: 0.28 and 0.27). The rule: when two correlated coefficients have opposite signs and similar sizes,
          suspect a missing variable or a difference effect, and don't tell the story "age raises the risk".
        - **Odds ratios** are multiplicative on the odds, not on the probabilities. For a customer with a 16% risk an odds ratio of 3.5 does not take the probability from 16% to 56% (3.5 times): it takes it to about 40%. With very small probabilities
          the two readings coincide (odds ≈ probability), with medium probabilities they don't.
        - The VIF is an alarm, not a sentence: with VIF below 5 the estimates remain usable; the problem is interpretive, not numerical.
    """,
    pitfall="""
        Reporting "the effect of age on default" by reading a coefficient of a model that also contains a highly correlated variable. A table of coefficients is a description
        **of the model**, not of reality, and it is not causal even if the model is well estimated.
    """,
    interview=(
        "Two highly correlated variables have coefficients of opposite sign in your regression. How do you explain it to the team?",
        """
        I say the coefficient measures the effect of one variable holding the other fixed, and when they are correlated the common part is split arbitrarily: their differences matter, not the single
        values. I check the collinearity (VIF), try the model with only one at a time and with a constructed feature (a difference, a ratio), look for omitted variables that explain the opposite
        sign, and if the goal is prediction I don't interpret the individual coefficients. For a causal effect you would need a different design, not a better model.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="calibration_of_five_models", kind="D", level=3, minutes=50, module=MODULE,
    title="Calibration: who tells the truth when they say 30%?",
    goal="""
        Compare five models on the same data for **discrimination** (AUC) and **calibration** (expected calibration error ECE, Brier score, log-loss), read the calibration curves
        and verify that a post-hoc calibration (`CalibratedClassifierCV`) fixes a random forest.
    """,
    dataset="""
        `ds.loans(n=6000, with_missing=False)`, stratified 50/50 split: `X_tr, X_te, y_te`. The predicted probabilities on the test set of the five models are in `probabilities`
        (dictionary name → array): logistic, logistic with `class_weight="balanced"`, random forest, random forest calibrated with sigmoid and with isotonic.
    """,
    purpose="""
        Tell apart two different questions: "does it rank well?" (AUC) and "are the probabilities true?" (calibration). A model can have the same AUC as another and unusable probabilities.
    """,
    task=[
        "Write `ece(y, p, k=10)`: **expected calibration error** with `k` groups of equal size (`pd.qcut(p, k, duplicates=\"drop\")`): sum over the groups of `weight · |mean(p) − observed frequency|`, with `weight = share of examples in the group`.",
        "Build `table` (index: the five models; columns `auc`, `brier`, `logloss`, `ece`, `mean_p`) with `roc_auc_score`, `brier_score_loss`, `log_loss`, your `ece` and the mean of the probabilities.",
        "Answer with the data: `same_discrimination` (True if the logistic and the `balanced` logistic have AUCs differing by less than 0.005); `balanced_overestimation` (difference between `mean_p` of `balanced` and the true frequency of positives); "
        "`best_calibration` (name of the model with minimum ECE); `best_auc` (name of the model with maximum AUC).",
        "Compute `isotonic_gain = ece(forest) / ece(isotonic_forest)` and `sigmoid_gain` (the same with the sigmoid): by how many times does the calibration error of the forest drop?",
        "Question: `why_balanced_is_skewed = 'c'` (choose `'a'`, `'b'` or `'c'`): a) the model ranks badly; b) the model is overfitting; c) the weights shift the probabilities upward without changing (almost) the ordering.",
    ],
    setup="""
        from sklearn.calibration import CalibratedClassifierCV, calibration_curve
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(n=6000, with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income", "amount"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.5, stratify=df["defaulted"], random_state=0)
        y_te = y_te.to_numpy()

        models = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
            "balanced": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
            "forest": RandomForestClassifier(300, random_state=0, n_jobs=-1),
            "forest_sigmoid": CalibratedClassifierCV(RandomForestClassifier(300, random_state=0, n_jobs=-1), method="sigmoid", cv=5),
            "forest_isotonic": CalibratedClassifierCV(RandomForestClassifier(300, random_state=0, n_jobs=-1), method="isotonic", cv=5),
        }
        probabilities = {name: m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for name, m in models.items()}
        print("frequency of positives in the test set:", y_te.mean().round(3))
    """,
    starter=[
        C("""
        def ece(y, p, k=10):
            ...
        """),
        C("""
        table = pd.DataFrame(index=list(probabilities), columns=["auc", "brier", "logloss", "ece", "mean_p"], dtype=float)
        table
        """),
        C("""
        same_discrimination = None
        balanced_overestimation = None
        best_calibration = None
        best_auc = None
        isotonic_gain = None
        sigmoid_gain = None
        why_balanced_is_skewed = None
        """),
        M("### The plot (to look at, not to be graded)"),
        C("""
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for name, p in probabilities.items():
            freq, mean = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
            ax.plot(mean, freq, "o-", label=name, ms=4)
        ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="predicted probability", ylabel="observed frequency"); ax.legend(); plt.show()
        """),
    ],
    checks="""
        def reference(y, p, k=10):
            d = pd.DataFrame({"y": y, "p": p, "g": pd.qcut(p, k, duplicates="drop")}).groupby("g", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
            return float((d["n"] / d["n"].sum() * (d["p"] - d["y"]).abs()).sum())
        check(close(ece(y_te, probabilities["balanced"]), reference(y_te, probabilities["balanced"])) and close(ece(y_te, probabilities["forest"]), reference(y_te, probabilities["forest"])), "correct ECE", "groups of equal size with pd.qcut, then a weighted sum of the absolute gaps")
        check(table.notna().all().all() and close(table.loc["logistic", "auc"], roc_auc_score(y_te, probabilities["logistic"])), "complete table", "five rows and five columns")
        check(same_discrimination is True and balanced_overestimation > 0.2, "AUC practically equal, but `balanced` overestimates the frequency by over 0.2", "compare AUC and mean_p of the two models")
        check(best_calibration == "logistic" and best_auc in ("logistic", "balanced"), "the plain logistic is the best calibrated, and the two logistics beat the forests on AUC", "idxmin of the ECE, idxmax of the AUC")
        check(isotonic_gain > 2 and sigmoid_gain > 1.5, "calibration reduces the ECE of the forest by over 2 times (isotonic) and 1.5 times (sigmoid)", "ece(forest) / ece(calibrated version)")
        check(table.loc["forest_isotonic", "logloss"] < table.loc["forest", "logloss"] and table.loc["forest_isotonic", "brier"] < table.loc["forest", "brier"], "calibrating improves log-loss and Brier too", "look at the logloss and brier columns")
        check(why_balanced_is_skewed == "c", "correct answer: the weights shift the probabilities", "the AUC is almost equal, so the ordering is the same")
    """,
    hint="""
        `groups = pd.qcut(p, k, duplicates="drop"); df_ = pd.DataFrame({"y": y, "p": p, "g": groups}); s = df_.groupby("g").agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))`;
        `ece = (s["n"] / s["n"].sum() * (s["p"] - s["y"]).abs()).sum()`.
    """,
    solution=[
        C("""
        def ece(y, p, k=10):
            d = pd.DataFrame({"y": y, "p": p, "g": pd.qcut(p, k, duplicates="drop")})
            s = d.groupby("g", observed=True).agg(n=("y", "size"), y=("y", "mean"), p=("p", "mean"))
            return float((s["n"] / s["n"].sum() * (s["p"] - s["y"]).abs()).sum())
        """),
        C("""
        table = pd.DataFrame(index=list(probabilities), columns=["auc", "brier", "logloss", "ece", "mean_p"], dtype=float)
        for name, p in probabilities.items():
            table.loc[name] = [roc_auc_score(y_te, p), brier_score_loss(y_te, p), log_loss(y_te, p), ece(y_te, p), p.mean()]
        table.round(4)
        """),
        C("""
        same_discrimination = bool(abs(table.loc["logistic", "auc"] - table.loc["balanced", "auc"]) < 0.005)
        balanced_overestimation = table.loc["balanced", "mean_p"] - y_te.mean()
        best_calibration = table["ece"].idxmin()
        best_auc = table["auc"].idxmax()
        isotonic_gain = table.loc["forest", "ece"] / table.loc["forest_isotonic", "ece"]
        sigmoid_gain = table.loc["forest", "ece"] / table.loc["forest_sigmoid", "ece"]
        why_balanced_is_skewed = "c"
        print(f"balanced: equal AUC ({same_discrimination}), but mean probability {table.loc['balanced', 'mean_p']:.3f} against a true frequency of {y_te.mean():.3f}")
        print(f"ECE of the forest: {table.loc['forest', 'ece']:.3f} -> sigmoid {table.loc['forest_sigmoid', 'ece']:.3f} (x{sigmoid_gain:.1f}) | isotonic {table.loc['forest_isotonic', 'ece']:.3f} (x{isotonic_gain:.1f})")
        """),
        C("""
        fig, ax = plt.subplots(figsize=(5.5, 5))
        for name, p in probabilities.items():
            freq, mean = calibration_curve(y_te, p, n_bins=8, strategy="quantile")
            ax.plot(mean, freq, "o-", label=name, ms=4)
        ax.plot([0, 1], [0, 1], "k:"); ax.set(xlabel="predicted probability", ylabel="observed frequency"); ax.legend(); plt.show()
        """),
    ],
    learned="""
        - **Discrimination and calibration are different properties.** The logistic and its `balanced` version have the same AUC (0.764 against 0.763) but the second has mean probabilities of 0.43 when the true frequency is 0.19:
          the calibration curve sits clearly above the diagonal (ECE 0.24 against 0.01). For a decision with costs (B2) the model with weights is unusable without recalibrating.
        - The random forest has a lower AUC (0.715) **and** less calibrated probabilities (ECE 0.059): the mean of the votes of the trees is not a probability. `CalibratedClassifierCV` fixes it (isotonic: ECE 0.018, sigmoid: 0.027) without changing
          the ordering, and improves log-loss and Brier too.
        - The plain logistic is calibrated **by construction** (it maximizes a Bernoulli likelihood): if you need probabilities, it is an excellent starting point. If your model is a tree/forest/boosting,
          always check the calibration before reading "0.3" as "30%".
        - Isotonic is more flexible than sigmoid but requires more data (with few examples it tends to overfit); sigmoid corrects only "S-shaped" distortions. With a few hundred examples prefer the sigmoid.
    """,
    pitfall="""
        Choosing the model with the best AUC and using its probabilities as risk estimates. The AUC knows nothing about the scale of the probabilities: a model with AUC 0.92 can say "90%" for cases that
        are positive 60 times out of 100.
    """,
    interview=(
        "Your model says 0.3: does it mean that 30% of these cases are positive?",
        """
        Only if the model is calibrated. I verify it with a calibration curve (observed frequency against predicted probability for groups of cases) and an indicator like the ECE or the Brier score, on data not used for
        training. The logistic is usually calibrated; forests, boosting and SVMs are not, and neither are models trained with weights or on rebalanced data: in that case I recalibrate (sigmoid or isotonic) with `CalibratedClassifierCV`.
        """),
)

# ---------------------------------------------------------------------------------------- D3

D3 = dict(
    id="D3", slug="multiclass_metrics_with_rare_classes", kind="D", level=2, minutes=45, module=MODULE,
    title="Reading a 10x10 confusion matrix with two rare classes",
    goal="""
        Interpret the metrics of a multiclass classifier when two classes are rare in the training set: precision, recall and F1 **per class**, micro, macro and weighted average, and understand where
        the errors of the rare classes end up. Then fix with `class_weight="balanced"`.
    """,
    dataset="""
        `load_digits` (60/40 split) where in the training set the digits **3** and **8** were reduced by 85% (about 16 examples each remain against 107 for the others). The test set is balanced like the original:
        `X_s, y_s` (imbalanced training set), `X_te, y_te`.
    """,
    purpose="""
        Learn to read a multiclass confusion matrix and to recognize the classic sign of rare classes: **high precision and low recall** (the model predicts them rarely, but when it does it is right). See what
        micro and macro hide.
    """,
    task=[
        "Train `model` = `StandardScaler` + `LogisticRegression(C=0.1, max_iter=3000)` on `X_s, y_s` and save `pred`, `cm` (confusion matrix of the test set) and `class_table` (DataFrame index 0..9, columns `precision`, `recall`, `f1`, `support` with `precision_recall_fscore_support`).",
        "Save `accuracy`, `micro_f1`, `macro_f1`, `weighted_f1` and `micro_equals_accuracy` (True if micro F1 coincides with the accuracy: it is a general fact for single-label multiclass).",
        "Find `silent_classes`: the (sorted) list of the classes with `precision − recall > 0.15`. For each one find in `where_they_go` (dictionary) the **wrong** class into which its digits are most often classified "
        "(the column with the highest count, excluding the diagonal).",
        "Redo with `class_weight=\"balanced\"`: save `rare_recall_before` and `rare_recall_after` (mean recall of the two rare classes), `balanced_macro_f1` and `balanced_accuracy`.",
    ],
    setup="""
        from sklearn.datasets import load_digits
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        digits = load_digits()
        X_tr, X_te, y_tr, y_te = train_test_split(digits.data, digits.target, test_size=0.4, stratify=digits.target, random_state=0)
        rng = np.random.default_rng(0)
        keep = np.ones(len(y_tr), dtype=bool)
        for c in (3, 8):
            idx = np.where(y_tr == c)[0]
            keep[rng.choice(idx, int(0.85 * len(idx)), replace=False)] = False
        X_s, y_s = X_tr[keep], y_tr[keep]
        print("examples per class in the training set:", np.bincount(y_s))
    """,
    starter=[
        C("""
        model = None
        pred = None
        cm = None
        class_table = None
        """),
        C("""
        accuracy = None
        micro_f1 = None
        macro_f1 = None
        weighted_f1 = None
        micro_equals_accuracy = None
        """),
        C("""
        silent_classes = None
        where_they_go = None
        """),
        C("""
        rare_recall_before = None
        rare_recall_after = None
        balanced_macro_f1 = None
        balanced_accuracy = None
        """),
    ],
    checks="""
        check(cm.shape == (10, 10) and cm.sum() == len(y_te) and list(class_table.columns) == ["precision", "recall", "f1", "support"], "10x10 matrix and per-class table", "confusion_matrix(y_te, pred), precision_recall_fscore_support")
        check(close(accuracy, (pred == y_te).mean()) and close(macro_f1, f1_score(y_te, pred, average="macro")) and close(weighted_f1, f1_score(y_te, pred, average="weighted")), "correct accuracy, macro and weighted F1", "average='macro' and 'weighted'")
        check(micro_equals_accuracy is True and close(micro_f1, accuracy), "micro F1 = accuracy", "in multiclass with a single label per example, micro-F1 = accuracy")
        check(silent_classes == [3, 8], "the silent classes are 3 and 8: high precision, low recall", "precision - recall > 0.15")
        check(all(where_they_go[c] == int(np.argmax(np.where(np.eye(10, dtype=bool)[c], -1, cm[c]))) and where_they_go[c] != c for c in silent_classes), "for each silent class, the most frequent wrong class is given", "row of the class in the confusion matrix, without the diagonal")
        check(rare_recall_after > rare_recall_before + 0.05, "with class_weight='balanced' the recall of the rare classes goes up by over 0.05", "mean recall of classes 3 and 8")
        check(balanced_macro_f1 > macro_f1 and balanced_accuracy > accuracy, "and macro F1 and accuracy improve too", "compare the values before and after")
        check(macro_f1 < accuracy, "macro F1 is lower than the accuracy: the rare classes weigh more in the macro", "compare macro_f1 and accuracy")
    """,
    hint="""
        `precision, recall, f1, support = precision_recall_fscore_support(y_te, pred)`. For class c: `errors = cm[c].copy(); errors[c] = 0; np.argmax(errors)`.
    """,
    solution=[
        C("""
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(X_s, y_s)
        pred = model.predict(X_te)
        cm = confusion_matrix(y_te, pred)
        p, r, f, s = precision_recall_fscore_support(y_te, pred)
        class_table = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s})
        class_table.round(3).T
        """),
        C("""
        accuracy = accuracy_score(y_te, pred)
        micro_f1 = f1_score(y_te, pred, average="micro")
        macro_f1 = f1_score(y_te, pred, average="macro")
        weighted_f1 = f1_score(y_te, pred, average="weighted")
        micro_equals_accuracy = bool(np.isclose(micro_f1, accuracy))
        print(f"accuracy {accuracy:.3f} | micro F1 {micro_f1:.3f} | macro {macro_f1:.3f} | weighted {weighted_f1:.3f}")
        """),
        C("""
        silent_classes = [int(c) for c in class_table.index[(class_table["precision"] - class_table["recall"]) > 0.15]]
        where_they_go = {}
        for c in silent_classes:
            errors = cm[c].copy(); errors[c] = 0
            where_they_go[c] = int(np.argmax(errors))
        print("classes with high precision and low recall:", silent_classes, "| their digits end up mostly in:", where_they_go)
        """),
        C("""
        balanced = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced")).fit(X_s, y_s)
        pred_balanced = balanced.predict(X_te)
        rare_recall_before = class_table.loc[[3, 8], "recall"].mean()
        rare_recall_after = precision_recall_fscore_support(y_te, pred_balanced)[1][[3, 8]].mean()
        balanced_macro_f1 = f1_score(y_te, pred_balanced, average="macro")
        balanced_accuracy = accuracy_score(y_te, pred_balanced)
        print(f"mean recall of the rare classes: {rare_recall_before:.3f} -> {rare_recall_after:.3f} | macro F1 {macro_f1:.3f} -> {balanced_macro_f1:.3f} | accuracy {accuracy:.3f} -> {balanced_accuracy:.3f}")
        """),
    ],
    learned="""
        - The signature of the rare classes is **high precision, low recall**: the model predicts few of them, but when it does it is right. Looking only at the precision or the accuracy does not show it; the per-class recall shows it right away.
        - The errors of the rare classes are not random: they end up in the "nearby" classes that the model has seen more often. The confusion matrix tells you where to intervene (more data, features to tell them apart) better than a number.
        - **Micro F1 = accuracy** for single-label multiclass: it is a way of saying that micro counts the examples, not the classes. **The macro** weighs every class the same and for that reason drops with rare classes;
          the weighted one comes back close to the accuracy. Choose the average according to whom you care about (the rare classes or the total).
        - `class_weight="balanced"` recovers recall on the rare classes and here it improves the overall accuracy too, because the model was wrong on two classes with few examples. It is not always so: often it is a compromise between
          the recall of the rare ones and the precision of the others (D2/B2).
    """,
    pitfall="""
        Reporting the accuracy of a multiclass problem with imbalanced classes: the rare classes can have recall 0.5 and not move the total. Always report the per-class table.
    """,
    interview=(
        "You have a macro F1 much lower than the accuracy. What does it tell you?",
        """
        That there are (rare) classes on which the model does badly and that the accuracy hides them because they count little in the total. I look at precision and recall per class and the confusion matrix to see where the errors go, then
        I evaluate interventions: more data or weights for the rare classes, features that tell them apart, different thresholds per class if the decisions allow it.
        """),
)

EXERCISES = [D1, D2, D3]
