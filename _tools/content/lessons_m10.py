"""Module 10 lessons: final projects."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_10_projects"
TITLE = "Final projects"
LESSONS = [
    ("01_the_life_of_a_project.ipynb", "10.1", "The life of a project",
     "From the problem to the report: closed test set, baseline, pipeline, comparison, tuning, threshold and cost."),
    ("02_sanity_checks.ipynb", "10.2", "Sanity checks",
     "How to notice a mistake that raises no errors: permuted labels, learning curves, validation that respects the structure of the data."),
]
NOTE = ("> **This module is different from the others.** It introduces no new tools and it does not have six types of exercise: it has **three long projects** (about an hour and a half each) "
        "that put the whole course together. The checks verify the numbers and the decisions, not the way you write the code: no solution is the only possible one.")
PATH = [
    ("Lesson 10.1", ["P1", "P2"]),
    ("Lesson 10.2", ["P3"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_10_1() -> List[Cell]:
    """Cells of lesson 10.1."""
    return [
        M("""
        # Lesson 10.1 · The life of a project

        In modules 1-9 you learned **tools**. In a real project most of the mistakes are not in the tools but in the **order** in which you use them and in what you take for granted. This lesson is a checklist with a runnable example; the three projects of this module put it to the test.

        **What you will learn**

        - The nine steps of a project, and at which step each typical mistake of the previous modules is born.
        - How to build a correct flow in a few cells: closed test set, baseline, pipeline, comparison, choice and final check.
        - Why the comparison with a "dumb" baseline is the first sanity check.
        """),
        M("""
        ## The nine steps

        | # | Step | Question | Typical mistake (where you saw it) |
        |---|---|---|---|
        | 1 | **Problem and metric** | What does a mistake cost? What is the prevalence? Which metric do I decide with? | Accuracy with imbalanced classes (M4) |
        | 2 | **Split and closed test set** | The test set is touched **once**, at the end. Does the split respect the structure of the data (time, groups, areas)? | Random split on grouped or spatial data (M3, M10) |
        | 3 | **Baseline** | What does the dumbest model do (`DummyClassifier`, the mean, a linear model)? | Not having a reference point (M1, M9 F2) |
        | 4 | **Pipeline** | Imputation, scaling, encoding **inside** the pipeline, fitted only on the training part of each fold. | Leakage from preprocessing on the whole dataset (M2, M3) |
        | 5 | **Comparison** | Same folds, same metric, with the variability (mean and deviation). | Comparison with a single seed or a single partition (M8 E1) |
        | 6 | **Tuning** | Only on the training set, with nested validation or a separate validation set. | Selection optimism (M3, M9 B3) |
        | 7 | **Final check** | A single shot on the test set, with an uncertainty interval. | Choosing by looking at the test (M9 R9) |
        | 8 | **Decision** | Threshold and costs, calibration, prediction intervals. | Threshold 0.5 out of habit (M4 B2) |
        | 9 | **Explanation and report** | Importances on unseen data, single cases, limits and risks. | Importance computed on the training set (M8 E3) |
        """),
        _setup("""
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

        df = ds.loans(n=2000, seed=4)
        y = df.pop("defaulted")
        X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, stratify=y, random_state=0)      # step 2: the test set is closed here
        print(f"training {len(X_tr)} rows, test {len(X_te)} | prevalence of defaulters: {y_tr.mean():.1%}")
        """),
        M("""
        ## Steps 3-5: baseline, pipeline, comparison

        The `DummyClassifier` that always predicts the most frequent class has an accuracy equal to the prevalence (about 80%) and an AUC of 0.5: **it is the score to beat**. Then a linear pipeline and a tree-based one, on the **same folds**.
        """),
        C("""
        numeric = ["age", "income", "job_tenure", "amount"]
        nominal = ["city", "contract_type", "occupation"]
        order = [ds.EDUCATION_ORDER] if hasattr(ds, "EDUCATION_ORDER") else None


        def preprocessor():
            return ColumnTransformer([
                ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numeric),
                ("edu", make_pipeline(OrdinalEncoder(categories=order, handle_unknown="use_encoded_value", unknown_value=-1),
                                      SimpleImputer(strategy="median"), StandardScaler()), ["education"]),
                ("nom", OneHotEncoder(handle_unknown="ignore", min_frequency=15), nominal),
            ])


        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        candidates = {
            "baseline (most frequent class)": make_pipeline(preprocessor(), DummyClassifier(strategy="prior")),
            "logistic regression": make_pipeline(preprocessor(), LogisticRegression(max_iter=2000)),
        }
        rows = {}
        for name, model in candidates.items():
            r = cross_validate(model, X_tr, y_tr, cv=cv, scoring=["roc_auc", "accuracy"])
            rows[name] = {"mean AUC": r["test_roc_auc"].mean(), "AUC std": r["test_roc_auc"].std(), "accuracy": r["test_accuracy"].mean()}
        pd.DataFrame(rows).T.round(3)
        """),
        M("""
        The accuracy of the model (about 0.85) is only slightly above that of the baseline (0.80): looking only at that you would say the model "is not needed". The **AUC** (0.82 against 0.50) says the opposite: the model ranks people well by risk, even if at threshold 0.5 it does not change
        many decisions. With imbalanced classes the right metric is almost always probabilistic or a ranking one, then the threshold is chosen with the costs (step 8).

        ## Steps 7-8: final check and decision

        Only after choosing **one** model do you open the test set, once. The threshold is chosen with the **out-of-sample predictions on the training set** (`cross_val_predict`), not on the test.
        """),
        C("""
        from sklearn.model_selection import cross_val_predict

        model = candidates["logistic regression"]
        p_cv = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        cost = lambda y_true, p, threshold: int(((p >= threshold) & (y_true == 0)).sum() * 1 + ((p < threshold) & (y_true == 1)).sum() * 8)      # a missed defaulter costs 8 times a wrong refusal
        thresholds = np.linspace(0.02, 0.9, 89)
        threshold = thresholds[int(np.argmin([cost(y_tr.to_numpy(), p_cv, s) for s in thresholds]))]

        p_te = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
        print(f"AUC on the test (only once): {roc_auc_score(y_te, p_te):.3f} | threshold chosen on the training set: {threshold:.2f}")
        print(f"cost on the test: threshold 0.5 -> {cost(y_te.to_numpy(), p_te, 0.5)} | threshold {threshold:.2f} -> {cost(y_te.to_numpy(), p_te, threshold)} | refuse everyone -> {int((y_te == 0).sum())}")
        """),
        M("""
        > **Common mistake.** Choosing the threshold (or the model, or the hyperparameters) by looking at the test set "just to see". From that moment the test set has become a validation set, and the number you report is optimistic without any warning stopping you.

        ## Check your understanding

        """ + details(
            "Why must the baseline be computed with the same folds and the same metric as the model?",
            "Because it serves to answer the question \"how much is the model worth compared with doing nothing intelligent?\": with different folds or metrics the difference also contains the difference in protocol. With imbalanced classes the baseline has high accuracy and AUC 0.5: it immediately shows which metric is informative.") + "\n\n" + details(
            "Why is the threshold chosen on the out-of-sample predictions of the training set and not on the test?",
            "Because the test set must simulate data never used for any decision. The threshold is a hyperparameter: chosen on the test, the reported cost is optimistic. With `cross_val_predict` every prediction comes from a model that has not seen that example, so it is an honest estimate to choose with.")),
    ]


def lesson_10_2() -> List[Cell]:
    """Cells of lesson 10.2."""
    return [
        M("""
        # Lesson 10.2 · Sanity checks

        The worst mistakes in a project do not raise exceptions: they produce **plausible numbers**. This lesson collects three checks that run in a few minutes and that expose most of the defects you met in the type E exercises.

        **What you will learn**

        - The **permuted labels test**: a correct flow on random labels must give AUC 0.5.
        - The **learning curve**: understanding whether you need more data, more capacity or more features.
        - Validation that **respects the structure** of the data (groups, time, areas).
        """),
        _setup("""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
        from sklearn.pipeline import make_pipeline

        df = ds.loans(n=1500, seed=4)
        y = df.pop("defaulted").to_numpy()
        numeric = ["age", "income", "job_tenure", "amount"]
        X = pd.get_dummies(df[numeric + ["city", "contract_type"]], columns=["city", "contract_type"], dtype=float)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        forest = lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0)
        """),
        M("""
        ## Check 1 · permuted labels

        If you shuffle the labels at random **before** the whole flow, there is no signal left to find: a correct flow must give an AUC close to 0.5. If you get much more, information is coming in through the wrong door (leakage, duplicates between training and validation, feature selection on the whole dataset...).

        Here the **correct** flow is a pipeline evaluated in cross-validation; the **faulty** flow duplicates the rare class *before* the cross-validation, so the same rows end up both in the training and in the validation set.
        """),
        C("""
        rng = np.random.default_rng(0)
        y_perm = rng.permutation(y)                                    # no signal: X and y_perm are independent

        correct_pipeline = make_pipeline(SimpleImputer(strategy="median"), forest())
        correct = cross_val_score(correct_pipeline, X, y_perm, cv=cv, scoring="roc_auc").mean()


        def cv_with_duplicates_first(X, y):
            rare = np.where(y == 1)[0]
            extra = np.random.default_rng(0).choice(rare, (y == 0).sum() - len(rare), replace=True)       # 'balance' by duplicating the rare class
            idx = np.r_[np.arange(len(y)), extra]
            Xd = SimpleImputer(strategy="median").fit_transform(X)[idx]
            return cross_val_score(forest(), Xd, y[idx], cv=cv, scoring="roc_auc").mean()


        faulty = cv_with_duplicates_first(X, y_perm)
        print(f"AUC on permuted labels | correct flow: {correct:.3f} | duplicates before the CV: {faulty:.3f}")
        """),
        M("""
        The correct flow gives about 0.5, as it should. The faulty one gives **over 0.9 on labels with no meaning at all**: the forest memorized the duplicated rows and recognizes them in the validation fold. It is the cheapest test there is against leakage: **if your flow beats chance on data with no signal, it
        is not a valid flow**.

        ## Check 2 · learning curve

        Train on the same problem with growing amounts of training data and look at what the training score and the validation score do.
        """),
        C("""
        sizes, tr_scores, va_scores = learning_curve(make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=0)),
                                                     X, y, cv=cv, scoring="roc_auc", train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0], random_state=0)
        curve = pd.DataFrame({"training": tr_scores.mean(1), "validation": va_scores.mean(1)}, index=sizes.astype(int))
        curve.index.name = "training examples"
        curve.round(3)
        """),
        M("""
        How to read it:

        - **Validation still climbing** as the data grow: collecting more examples helps.
        - **Validation flat and training much higher**: overfitting; you need regularization or a simpler model.
        - **The two curves close together and low**: underfitting; you need better features or a more capable model.

        Here the validation climbs slowly (from 0.74 with 120 examples to 0.79 with 1200) while the training stays at 0.98: the gap of 0.2 is typical of a deep forest that memorizes the training set, and doubling the examples would give a few thousandths. You need better features or regularization (`min_samples_leaf`), not fine tuning.

        ## Check 3 · the validation must look like the real use

        If in production you predict **new patients**, **future months** or **areas never seen**, the validation must simulate exactly that: `GroupKFold` for groups, a temporal split for series, geographic blocks for maps. With a random partition the "neighboring" examples (same patient, same month, same neighborhood) are both in the training and in the validation set and the error
        comes out much lower than the one you will really see. In project P2 you measure this effect on California housing: the gap is 40%.

        > **Common mistake.** Attributing a good validation score to the quality of the model, without asking whether the validation protocol looks like how the model will be used.

        ## Check your understanding

        """ + details(
            "What does an AUC of 0.9 on the permuted labels test tell you?",
            "That information about the label is entering the flow through a path that should not exist: duplicates between training and validation, preprocessing or feature selection fitted on the whole dataset, features built with the target. The model has not 'learned' anything: it has seen the answers again.") + "\n\n" + details(
            "In what order do you run the checks?",
            "First the cheap and general ones (baseline, permuted labels), then those on the protocol (validation that respects the structure of the data), then the learning curves and the error analyses. A check that fails has to be fixed before tuning the model: optimizing a faulty flow is wasted time.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_the_life_of_a_project.ipynb", lesson_10_1),
                    ("02_sanity_checks.ipynb", lesson_10_2)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
