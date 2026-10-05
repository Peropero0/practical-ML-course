"""Module 4 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_04_logistic_regression"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="regularization_without_scaling", kind="E", level=2, minutes=35, module=MODULE, start="must_fail",
    title="The income in euros that wipes out the amount/income ratio",
    goal="""
        Understand why a regularized logistic changes its predictions if you change the **unit of measure** of a column (euros versus thousands of euros), find out which coefficient gets sacrificed and fix it with
        a scaler inside the pipeline. Verify that after the fix the model is **invariant** to the units.
    """,
    dataset="""
        `ds.loans(with_missing=False)` with `ratio = amount/income` (scale about 0.5), `age`, `job_tenure` and `income` (in euros, scale about 30,000): stratified 70/30 split. `evaluate(model, k)` (given) trains a copy
        of the model dividing `income` by `k` and returns the test AUC.
    """,
    purpose="""
        Recognize a bug that gives no errors or warnings: `C=0.1` penalizes all the coefficients the same way, but a column with an enormous scale needs a tiny coefficient and is in fact penalized
        very little, while a column with a small scale (the ratio) gets squashed. The result changes with the unit of measure: it is a model that depends on an arbitrary choice.
    """,
    task=[
        "Run the colleague's cell: `auc_bug` (income in euros) and `auc_bug_k_euro` (income in thousands of euros). The same model on the same data should give the same result: save `unit_difference = auc_bug_k_euro − auc_bug`.",
        "Train the colleague's model on the training set (income in euros) and save `bug_ratio_coef` (coefficient of `ratio`).",
        "Fix it: `ok_model` = `StandardScaler` + `LogisticRegression(C=0.1, max_iter=1000)`. Save `auc_ok` and `auc_ok_k_euro` (with `evaluate`) and `invariance = abs(auc_ok − auc_ok_k_euro)`.",
        "Train `ok_model` on the training set and save `ok_ratio_coef`: the coefficient of `ratio` **in original units** (coefficient of the model divided by the standard deviation used by the scaler: `scaler.scale_[0]`).",
    ],
    setup="""
        from sklearn.base import clone
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income"]
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], df["defaulted"], test_size=0.3, stratify=df["defaulted"], random_state=0)


        def evaluate(model, k=1.0):
            \"\"\"Test AUC of a copy of the model, with `income` divided by k (k = 1000: thousands of euros).\"\"\"
            a, b = X_tr.copy(), X_te.copy()
            a["income"] /= k
            b["income"] /= k
            return roc_auc_score(y_te, clone(model).fit(a, y_tr).predict_proba(b)[:, 1])
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        bug_model = LogisticRegression(C=0.1, max_iter=1000)
        auc_bug = evaluate(bug_model)                 # income in euros
        auc_bug_k_euro = evaluate(bug_model, 1000)    # same data, income in thousands of euros
        print(f"test AUC: {auc_bug:.3f} with the income in euros | {auc_bug_k_euro:.3f} in thousands of euros")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        unit_difference = None
        bug_ratio_coef = None
        """),
        C("""
        ok_model = None
        auc_ok = None
        auc_ok_k_euro = None
        invariance = None
        ok_ratio_coef = None
        """),
    ],
    checks="""
        check(auc_bug < 0.75, "the colleague's model with the income in euros is poor (AUC below 0.75)", "don't modify the colleague's code")
        check(close(unit_difference, auc_bug_k_euro - auc_bug) and unit_difference > 0.04, "changing only the unit of measure the AUC changes by over 0.04", "auc_bug_k_euro - auc_bug")
        check(bug_ratio_coef is not None and abs(bug_ratio_coef) < 0.02, "in the colleague's model the coefficient of the ratio has been squashed (below 0.02)", "coef_[0][0] of the model trained on the training set in euros")
        check(invariance is not None and invariance < 1e-3 and auc_ok > 0.77, "with the scaler the AUC does not depend on the unit of measure (gap below 0.001) and is 0.77 or more", "StandardScaler before LogisticRegression")
        check(ok_ratio_coef > 50 * abs(bug_ratio_coef), "in original units the right coefficient of the ratio is hundreds of times larger than the colleague's", "coef of the model / scaler.scale_[0]")
    """,
    hint="""
        `ok_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))`; `ok_model.fit(X_tr, y_tr)`; `ok_model[-1].coef_[0][0] / ok_model[0].scale_[0]`.
        The colleague's coefficient: `bug_model.fit(X_tr, y_tr).coef_[0][0]` (on the copy, not on the one used by `evaluate`).
    """,
    solution=[
        C("""
        bug_model = LogisticRegression(C=0.1, max_iter=1000)
        auc_bug = evaluate(bug_model)                 # income in euros
        auc_bug_k_euro = evaluate(bug_model, 1000)    # same data, income in thousands of euros
        print(f"test AUC: {auc_bug:.3f} with the income in euros | {auc_bug_k_euro:.3f} in thousands of euros")
        """),
        M("""
        **Where the bug is.** The L2 penalty sums the squares of the coefficients. With the income in euros (scale 30,000) its coefficient can be tiny and still weigh what it needs to, so it is barely
        penalized; the ratio (scale 0.5) needs a large coefficient to count and the penalty **squashes** it (`C = 0.1` is very strong). The model relies almost only on the income. With the income in
        thousands of euros the balance changes: **same data, different model**.
        """),
        C("""
        unit_difference = auc_bug_k_euro - auc_bug
        bug_ratio_coef = clone(bug_model).fit(X_tr, y_tr).coef_[0][0]
        print(f"difference between the two units: {unit_difference:.3f} | coefficient of the ratio in the colleague's model: {bug_ratio_coef:.4f}")
        """),
        C("""
        ok_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
        auc_ok = evaluate(ok_model)
        auc_ok_k_euro = evaluate(ok_model, 1000)
        invariance = abs(auc_ok - auc_ok_k_euro)
        ok_model.fit(X_tr, y_tr)
        ok_ratio_coef = ok_model[-1].coef_[0][0] / ok_model[0].scale_[0]
        print(f"AUC with the scaler: {auc_ok:.4f} (euros) | {auc_ok_k_euro:.4f} (thousands of euros) | gap {invariance:.1e}")
        print(f"coefficient of the ratio in original units: {ok_ratio_coef:.2f} (against {bug_ratio_coef:.4f} for the colleague)")
        """),
    ],
    learned="""
        - **A regularized logistic is not invariant to the scale of the features.** The penalty treats the coefficients as if they were comparable: they are so only if the features have the same scale. Here the coefficient of the ratio
          in the colleague's model is 0.008; in original units, with the scaler, it is 1.96: the penalty had reduced it by over 200 times.
        - The bug raises no errors: the colleague's AUC (0.716) is "plausible" and sits below the correct one (0.778) by 0.06. You find it only with an **invariance** test: change the unit of a column and check that the predictions
          do not change. With the scaler the invariance is exact (gap below 1e-3), because standardization erases the unit.
        - The scaler goes in the `Pipeline`: that way it is learned on the training part of every fold alone (M3) and you apply it identically to the test set. Scaling is not only for solvers: it serves to make the penalty meaningful.
        - To interpret the coefficients in original units, divide them by the standard deviation of the scaler (`coef / scaler.scale_`), as you did here.
    """,
    pitfall="""
        Regularizing (`Ridge`, `Lasso`, `LogisticRegression`, `SVC`, ...) without standardizing, "because the features are all positive anyway" or "because the model converges all the same". Convergence has nothing to do with it: it is the penalty
        that depends on the scale.
    """,
    interview=(
        "Which models require feature scaling and which don't?",
        """
        Those that use distances (KNN, kernel SVM, k-means), PCA and the **regularized** linear ones or those trained by gradient descent (logistic, Ridge, Lasso, networks). Trees and their ensembles (random forest, boosting)
        use thresholds and are invariant to monotonic transformations of the individual features, so they don't need it. When in doubt I run an invariance test: I change the unit of a column and see whether the predictions change.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="predict_proba_columns_and_classes", kind="E", level=2, minutes=35, module=MODULE, start="must_fail",
    title="`predict_proba[:, 1]` is not always the probability you think",
    goal="""
        Find out why a model that flags at-risk customers "finds" 94% of the customers (far too many) with a precision of 14%, despite an excellent AUC: the columns of `predict_proba` are
        in **alphabetical** order of the classes, and with text labels column 1 is not the one the colleague thinks. Fix it and write a robust function.
    """,
    dataset="""
        `ds.loans(with_missing=False)` with the target turned into text: `"defaulted"` or `"regular"` (`y_tr`, `y_te` are arrays of strings). Features: `ratio, age, job_tenure, income`, logistic with scaler.
    """,
    purpose="""
        Recognize one of the most insidious bugs with non-numeric labels: the AUC computed with `roc_auc_score` on the strings is **right by chance** (the positive class is the last one alphabetically, like column 1),
        and hides the error, which shows up only when you use the probability to decide.
    """,
    task=[
        "Run the colleague's cell: `auc_bug` is good, but `bug_flagged_share` (share of customers with `p >= 0.3`) is very high and `bug_precision` very low.",
        "Inspect `model.classes_`: save `classes` (list) and `defaulted_index` (position of `\"defaulted\"` in the list).",
        "Recompute with the right column `p_def`: save `ok_flagged_share`, `ok_precision` and `ok_recall` (with threshold 0.3 and `\"defaulted\"` as the positive class).",
        "Write `class_proba(model, X, cls)` that returns the probability of the requested class by looking up its column in `model.classes_`; try it on a model trained with `1/0` labels (class `1`) and on one with "
        "reversed text labels (`\"yes\"` for defaulted and `\"no\"`, class `\"yes\"`).",
    ],
    setup="""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        cols = ["ratio", "age", "job_tenure", "income"]
        y_txt = np.where(df["defaulted"] == 1, "defaulted", "regular")
        X_tr, X_te, y_tr, y_te = train_test_split(df[cols], y_txt, test_size=0.3, stratify=y_txt, random_state=0)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        p = model.predict_proba(X_te)[:, 1]                                         # "default probability"
        auc_bug = roc_auc_score(y_te, p)
        flagged = p >= 0.3                                                          # flag as defaulter whoever has p >= 0.3
        pred = np.where(flagged, "defaulted", "regular")
        bug_flagged_share = flagged.mean()
        bug_precision = precision_score(y_te, pred, pos_label="defaulted")
        print(f"AUC {auc_bug:.3f} | customers flagged {bug_flagged_share:.1%} | precision {bug_precision:.3f}")
        """),
        M("### Your diagnosis"),
        C("""
        classes = None
        defaulted_index = None
        """),
        C("""
        p_def = None
        ok_flagged_share = None
        ok_precision = None
        ok_recall = None
        """),
        C("""
        def class_proba(model, X, cls):
            ...
        """),
    ],
    checks="""
        check(auc_bug > 0.78, "the colleague's AUC is good: the bug hides", "don't modify the colleague's code")
        check(bug_flagged_share > 0.9 and bug_precision < 0.2, "the colleague flags over 90% of the customers with a precision below 20%", "don't modify the colleague's code")
        check(classes == ["defaulted", "regular"] and defaulted_index == 0, "the classes are in alphabetical order: 'defaulted' is column 0", "model.classes_")
        check(close(p_def, model.predict_proba(X_te)[:, 0]), "the default probability is column 0", "use defaulted_index")
        check(ok_flagged_share < 0.25 and ok_precision > 0.5 and ok_recall > 0.5, "with the right column: about 18% flagged, precision and recall above 0.5", "threshold 0.3 on p_def, positive class 'defaulted'")
        m1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, (y_tr == "defaulted").astype(int))
        m2 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, np.where(y_tr == "defaulted", "yes", "no"))
        check(close(class_proba(m1, X_te, 1), m1.predict_proba(X_te)[:, 1]) and close(class_proba(m2, X_te, "yes"), m2.predict_proba(X_te)[:, 1]) and close(class_proba(m2, X_te, "yes"), 1 - class_proba(m2, X_te, "no")),
              "class_proba works with numeric and text labels, in any order", "look up the column with list(model.classes_).index(cls)")
    """,
    hint="""
        `p_def = model.predict_proba(X_te)[:, defaulted_index]`. In the pipeline `classes_` is exposed by the final estimator: `model.classes_` works, or `model[-1].classes_`.
        `class_proba`: `return model.predict_proba(X)[:, list(model.classes_).index(cls)]`.
    """,
    solution=[
        C("""
        p = model.predict_proba(X_te)[:, 1]                                         # "default probability"
        auc_bug = roc_auc_score(y_te, p)
        flagged = p >= 0.3                                                          # flag as defaulter whoever has p >= 0.3
        pred = np.where(flagged, "defaulted", "regular")
        bug_flagged_share = flagged.mean()
        bug_precision = precision_score(y_te, pred, pos_label="defaulted")
        print(f"AUC {auc_bug:.3f} | customers flagged {bug_flagged_share:.1%} | precision {bug_precision:.3f}")
        """),
        M("""
        **Where the bug is.** `predict_proba` orders the columns like `classes_`, that is in **alphabetical order**: `["defaulted", "regular"]`. Column 1 is the probability of `"regular"`, not of defaulted. The colleague flags whoever has
        `P(regular) >= 0.3`, that is almost everyone. The AUC comes out right because `roc_auc_score` with text labels takes the last class in alphabetical order (`"regular"`) as positive and column 1 is exactly its probability: two
        errors that cancel in the metric and not in the decision.
        """),
        C("""
        classes = list(model.classes_)
        defaulted_index = classes.index("defaulted")
        print("classes_:", classes, "-> index of 'defaulted':", defaulted_index)
        """),
        C("""
        p_def = model.predict_proba(X_te)[:, defaulted_index]
        flagged = p_def >= 0.3
        pred = np.where(flagged, "defaulted", "regular")
        ok_flagged_share = flagged.mean()
        ok_precision = precision_score(y_te, pred, pos_label="defaulted")
        ok_recall = recall_score(y_te, pred, pos_label="defaulted")
        print(f"customers flagged {ok_flagged_share:.1%} | precision {ok_precision:.3f} | recall {ok_recall:.3f}")
        """),
        C("""
        def class_proba(model, X, cls):
            \"\"\"Probability of the requested class, looked up by name (never by position).\"\"\"
            return model.predict_proba(X)[:, list(model.classes_).index(cls)]
        """),
    ],
    learned="""
        - **The columns of `predict_proba` follow `classes_`, which is sorted.** With a 0/1 target column 1 is the positive class; with text labels (`"no"/"yes"`, `"defaulted"/"regular"`) it can be the opposite one. Never trust
          the position: look it up by name.
        - One bug can be hidden by another: the AUC computed on the strings comes out **right by chance** (positive = last alphabetical class, like column 1) and the check everybody does first does not see it. The symptom
          is elsewhere: here 94% of the customers flagged and a precision of 14% (the prevalence, like flagging everyone).
        - For the metrics that depend on the positive class (`precision_score`, `recall_score`, `f1_score`) with text labels **always** pass `pos_label`, otherwise scikit-learn uses the one it likes (and with more than two values it raises an error).
        - The remedy is simple: convert the target to 0/1 with the positive class explicit (`y == "defaulted"`) at the start of the data pipeline, or use a function like `class_proba`.
    """,
    pitfall="""
        Writing `predict_proba(X)[:, 1]` as if it were an always-valid idiom. It is valid with a 0/1 target and `classes_ = [0, 1]`; with booleans, text or classes not ordered the way you expect, it is not. Check `classes_` at least once per project.
    """,
    interview=(
        "How do you make sure you use the right column of predict_proba?",
        """
        I encode the target as 0/1 with the class of interest as 1 at the start, and check `model.classes_`. If the target has text labels I look up the column by name and not by position. In the automated tests I verify
        that the mean predicted probability is close to the frequency of the positive class: it is a very simple check that immediately catches swapped columns.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="oversampling_before_the_split", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="100% recall with a random forest: the rebalancing done before the split",
    goal="""
        Find out why a random forest on a problem with 7% positives gets recall, precision and AUC close to 1: the colleague **duplicated the positives to balance the classes before** splitting training and test, so
        the same examples are in both. Fix it (split first, rebalancing only on the training set) and measure the true performance.
    """,
    dataset="""
        `generate()` (given): 3000 examples, 10 features (4 informative), about 7% positives (`df` with a `y` column). Model: `RandomForestClassifier(200, random_state=0)`.
    """,
    purpose="""
        Recognize rebalancing leakage, one of the most common in production: any transformation that **duplicates or synthesizes** rows (oversampling, SMOTE, augmentation) must happen after the split and only on the training set.
        An identical copy in the test set is an example the model has already seen.
    """,
    task=[
        "Run the colleague's cell: `bug_recall`, `bug_precision`, `bug_auc` are almost perfect. Be suspicious.",
        "Compute `test_duplicates_share`: the fraction of the positives of the colleague's test set (`Xt_bug`, `yt_bug`) that appear identical (same values in all the features) in `Xa_bug`.",
        "Fix it: split **first** (stratified `train_test_split` 70/30, `random_state=0`) the original data, duplicate the positives **only in the training set** (`sample(..., replace=True, random_state=0)`), train and evaluate on the original "
        "test set. Save `ok_recall`, `ok_precision`, `ok_auc` and `ok_ap` (average precision).",
        "Compare with the solution without duplicates: forest with `class_weight=\"balanced\"` on the non-duplicated training set. Save `weights_ap` and `weights_recall`.",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
        from sklearn.model_selection import train_test_split


        def generate():
            X, y = make_classification(n_samples=3000, n_features=10, n_informative=4, n_redundant=0, weights=[0.94], class_sep=0.8, flip_y=0.02, random_state=0)
            df = pd.DataFrame(X, columns=[f"x{i}" for i in range(10)])
            df["y"] = y
            return df


        df = generate()
        rf = lambda **kw: RandomForestClassifier(200, random_state=0, n_jobs=-1, **kw)
        print("fraction of positives:", df["y"].mean().round(3))
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        majority, minority = df[df["y"] == 0], df[df["y"] == 1]
        balanced = pd.concat([majority, minority.sample(len(majority), replace=True, random_state=0)], ignore_index=True)     # 50% / 50%
        Xa_bug, Xt_bug, ya_bug, yt_bug = train_test_split(balanced.drop(columns="y"), balanced["y"], test_size=0.3, stratify=balanced["y"], random_state=0)
        m_bug = rf().fit(Xa_bug, ya_bug)
        pred_bug, score_bug = m_bug.predict(Xt_bug), m_bug.predict_proba(Xt_bug)[:, 1]
        bug_recall, bug_precision, bug_auc = recall_score(yt_bug, pred_bug), precision_score(yt_bug, pred_bug), roc_auc_score(yt_bug, score_bug)
        print(f"recall {bug_recall:.3f} | precision {bug_precision:.3f} | AUC {bug_auc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        test_duplicates_share = None
        """),
        C("""
        ok_recall = None
        ok_precision = None
        ok_auc = None
        ok_ap = None
        """),
        C("""
        weights_ap = None
        weights_recall = None
        """),
    ],
    checks="""
        check(bug_recall > 0.95 and bug_precision > 0.95 and bug_auc > 0.98, "the colleague's numbers are too good to be true", "don't modify the colleague's code")
        check(test_duplicates_share is not None and test_duplicates_share > 0.9, "over 90% of the positives of the test set appear identical in the training set", "merge between Xt_bug (positives only) and Xa_bug.drop_duplicates()")
        check(ok_recall < 0.4 and 0.75 < ok_auc < 0.92, "on the true test set: low recall (below 0.4) and AUC 0.75-0.92", "the test set must have the original prevalence (7%) and no duplicates")
        check(ok_ap < 0.6, "average precision below 0.6: the problem is hard", "average_precision_score on the original test set")
        check(weights_ap is not None and abs(weights_ap - ok_ap) < 0.15 and weights_recall < 0.4, "class_weight='balanced' gives comparable results, with no duplicates and no leakage", "RandomForestClassifier(class_weight='balanced') on the original training set")
    """,
    hint="""
        Duplicates: `Xa_bug.drop_duplicates()` and `Xt_bug[yt_bug == 1].merge(Xa_bug.drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").mean()`.
        Fix: `tr = pd.concat([X_tr, y_tr], axis=1)`; then `minority.sample(len(majority), replace=True, random_state=0)` **only on `tr`**.
    """,
    solution=[
        C("""
        majority, minority = df[df["y"] == 0], df[df["y"] == 1]
        balanced = pd.concat([majority, minority.sample(len(majority), replace=True, random_state=0)], ignore_index=True)     # 50% / 50%
        Xa_bug, Xt_bug, ya_bug, yt_bug = train_test_split(balanced.drop(columns="y"), balanced["y"], test_size=0.3, stratify=balanced["y"], random_state=0)
        m_bug = rf().fit(Xa_bug, ya_bug)
        pred_bug, score_bug = m_bug.predict(Xt_bug), m_bug.predict_proba(Xt_bug)[:, 1]
        bug_recall, bug_precision, bug_auc = recall_score(yt_bug, pred_bug), precision_score(yt_bug, pred_bug), roc_auc_score(yt_bug, score_bug)
        print(f"recall {bug_recall:.3f} | precision {bug_precision:.3f} | AUC {bug_auc:.3f}")
        """),
        M("""
        **Where the bug is.** The positives (about 200) were duplicated until they equaled the negatives (about 2,800): every positive appears on average 13-14 times. Then the random split puts some copies in the training set and others in the test set. For the model,
        recognizing a positive of the test set means **remembering an identical copy** seen in training: a random forest memorizes them.
        """),
        C("""
        test_duplicates_share = Xt_bug[yt_bug == 1].merge(Xa_bug.drop_duplicates(), how="left", indicator=True)["_merge"].eq("both").mean()
        print(f"positives of the test set that appear identical in the training set: {test_duplicates_share:.1%}")
        """),
        C("""
        X_tr, X_te, y_tr, y_te = train_test_split(df.drop(columns="y"), df["y"], test_size=0.3, stratify=df["y"], random_state=0)
        tr = pd.concat([X_tr, y_tr], axis=1)
        maj, mino = tr[tr["y"] == 0], tr[tr["y"] == 1]
        tr_balanced = pd.concat([maj, mino.sample(len(maj), replace=True, random_state=0)], ignore_index=True)       # the duplication happens ONLY in the training set
        m_ok = rf().fit(tr_balanced.drop(columns="y"), tr_balanced["y"])
        pred_ok, score_ok = m_ok.predict(X_te), m_ok.predict_proba(X_te)[:, 1]
        ok_recall, ok_precision = recall_score(y_te, pred_ok), precision_score(y_te, pred_ok)
        ok_auc, ok_ap = roc_auc_score(y_te, score_ok), average_precision_score(y_te, score_ok)
        print(f"original test set ({y_te.mean():.1%} positives): recall {ok_recall:.3f} | precision {ok_precision:.3f} | AUC {ok_auc:.3f} | AP {ok_ap:.3f}")
        """),
        C("""
        m_weights = rf(class_weight="balanced").fit(X_tr, y_tr)
        weights_ap = average_precision_score(y_te, m_weights.predict_proba(X_te)[:, 1])
        weights_recall = recall_score(y_te, m_weights.predict(X_te))
        print(f"class_weight='balanced': AP {weights_ap:.3f} | recall {weights_recall:.3f}")
        """),
    ],
    learned="""
        - With the rebalancing **before** the split every positive of the test set has an identical copy in the training set: the random forest remembers it and gets recall 1.0, precision 0.99 and AUC 1.0. On the true test set, with the
          original prevalence, the recall is 0.21 and the AUC 0.83. A test set with copies of the training set measures nothing.
        - Rule: **split first, always.** Oversampling, SMOTE, augmentation, imputation, scaling, feature selection: anything that looks at the data and modifies it goes after the split and only on the training set (in a `Pipeline` or with `imblearn.pipeline.Pipeline`
          if you use samplers). And the test set keeps the **real prevalence**.
        - Duplicated data does not move the problem: with the correct rebalancing the random forest reaches an average precision of 0.42, practically that of a model with `class_weight` (0.415). Rebalancing changes the working point (more recall, less precision), it does not add information.
        - Warning sign: a model with **all** the metrics close to 1 on a problem with rare classes and noise (`flip_y`) is almost always leakage. To check it, look for duplicated rows between training and test.
    """,
    pitfall="""
        Balancing the dataset "to prepare it" in a data-cleaning notebook and then doing the split in the modeling notebook. The balanced dataset must never exist outside a training pipeline.
    """,
    interview=(
        "Why must SMOTE be applied after the split and inside the cross-validation?",
        """
        SMOTE creates synthetic examples by interpolating between nearby positives: if I apply it before the split, a synthetic example in the training set is almost identical to the real positive that ends up in the test set, so the model "recognizes" it. You need
        a pipeline that applies the sampling only on the training part of every fold (with `imblearn.pipeline.Pipeline`), and the test/validation must keep the real distribution.
        """),
)

EXERCISES = [E1, E2, E3]
