"""Module 10 - type E project: reviewing a notebook."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_10_projects"

# ---------------------------------------------------------------------------------------- P3

P3 = dict(
    id="P3", slug="reviewing_a_notebook_which_flaw_matters", kind="E", level=3, minutes=75, module=MODULE, start="must_fail",
    title="Reviewing a notebook: which flaw really matters?",
    goal="""
        A colleague presents a risk model with "AUC 0.97 and accuracy 0.91, great". Their notebook has **three flaws** (duplicates before the cross-validation, imputation and encoding fitted on the whole dataset, encoding of `occupation` with arbitrary numbers) and a **misleading metric** (accuracy with 80% prevalence). Measure the effect
        of each **on its own**, rank them by severity with the numbers and show with the permuted labels test that the colleague's flow is not valid.
    """,
    dataset="""
        `ds.loans(1500)` with missing values (about 19% defaulters). The colleague evaluates with `StratifiedKFold(5, shuffle=True, random_state=0)` a `RandomForestClassifier(300, min_samples_leaf=3)`. The "lab" `lab(flaws, given_y)` (given, in the setup cell) contains the correct flow and, switchable one at a time, the three flaws: `"duplicates"`, `"global_scaler"`,
        `"occupation_codes"`.
    """,
    purpose="""
        Learn to **measure** a flaw instead of guessing: some methodological mistakes are catastrophic (`duplicates`: +0.14 of AUC), others cost little (`occupation_codes`: −0.01) and others are correct principles with no measurable effect in this case (`global_scaler`: +0.002). You need to recognize them all, but ranking them by effect decides where to spend the time, and the permuted labels test
        is the check that finds the leakage without knowing where it is.
    """,
    task=[
        "Run the colleague's cell: `lab` with the three flaws active. Save `flawed_auc`, `flawed_acc` and, with `lab(set())`, `honest_auc` and `honest_acc`.",
        "**Isolated effects:** `effects` (dictionary key → AUC with **only** that flaw minus `honest_auc`), `order` (the three keys sorted by absolute value of the effect, largest first) and `worst_flaw` (the key with the largest absolute effect).",
        "**Permuted labels test:** with `y_perm = np.random.default_rng(0).permutation(y)` save `honest_perm` (`lab(set(), y_perm)[0]`), `duplicates_perm` (only `\"duplicates\"`) and `scaler_perm` (only `\"global_scaler\"`).",
        "**The metric:** `dummy_acc` (CV accuracy of `DummyClassifier(strategy=\"most_frequent\")` with the same `cv`), `acc_gain` (`honest_acc − dummy_acc`) and `auc_gain` (`honest_auc − 0.5`, the AUC of the baseline).",
        "Write `recommendations` (dictionary with the three flaw keys → a sentence: how to fix it and why, citing the number you measured).",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import StratifiedKFold, cross_validate
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        df = ds.loans(n=1500, seed=4)
        y = df.pop("defaulted").to_numpy()
        X = df
        NUM = ["age", "income", "job_tenure", "amount"]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)


        def lab(flaws=(), given_y=None):
            \"\"\"Cross-validation evaluation of a forest. With `flaws` empty the flow is correct.
            Returns (mean AUC, mean accuracy). Switchable flaws: 'duplicates', 'global_scaler', 'occupation_codes'.\"\"\"
            flaws = set(flaws)
            yy = y if given_y is None else np.asarray(given_y)
            codes = "occupation_codes" in flaws

            def preprocessor():
                transformers = [
                    ("num", SimpleImputer(strategy="median", add_indicator=True), NUM),
                    ("edu", make_pipeline(OrdinalEncoder(categories=[ds.EDUCATION_ORDER], handle_unknown="use_encoded_value", unknown_value=-1),
                                          SimpleImputer(strategy="median")), ["education"]),
                    ("nom", OneHotEncoder(handle_unknown="ignore", min_frequency=10), ["city", "contract_type"] + ([] if codes else ["occupation"])),
                ]
                if codes:                                                      # flaw: 'occupation' as arbitrary numbers (the order is alphabetical!)
                    transformers.append(("occ", make_pipeline(SimpleImputer(strategy="constant", fill_value="unknown"),
                                                              OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)), ["occupation"]))
                return ColumnTransformer(transformers)

            model = RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0)
            Xl, yl = X, yy
            if "duplicates" in flaws:                                          # flaw: 'balance' by duplicating the rare class BEFORE the cross-validation
                rare = np.where(yl == 1)[0]
                extra = np.random.default_rng(0).choice(rare, (yl == 0).sum() - len(rare), replace=True)
                idx = np.r_[np.arange(len(yl)), extra]
                Xl, yl = X.iloc[idx].reset_index(drop=True), yl[idx]
            if "global_scaler" in flaws:                                       # flaw: imputation and encoding fitted on the whole dataset
                Z = preprocessor().fit_transform(Xl)
                Xl, pipe = (Z.toarray() if hasattr(Z, "toarray") else Z), model
            else:
                pipe = make_pipeline(preprocessor(), model)
            r = cross_validate(pipe, Xl, yl, cv=cv, scoring=["roc_auc", "accuracy"])
            return float(r["test_roc_auc"].mean()), float(r["test_accuracy"].mean())
    """,
    starter=[
        M("### The colleague's notebook"),
        C("""
        all_flaws = {"duplicates", "global_scaler", "occupation_codes"}
        auc_c, acc_c = lab(all_flaws)
        print(f"AUC {auc_c:.3f}, accuracy {acc_c:.3f} -> 'great, let's put it in production'")
        """),
        M("### Your review"),
        C("""
        flawed_auc = None
        flawed_acc = None
        honest_auc = None
        honest_acc = None
        """),
        C("""
        effects = {}
        order = []
        worst_flaw = None
        """),
        C("""
        honest_perm = None
        duplicates_perm = None
        scaler_perm = None
        """),
        C("""
        dummy_acc = None
        acc_gain = None
        auc_gain = None
        recommendations = {}
        """),
    ],
    checks="""
        check(auc_c > 0.95 and acc_c > 0.88, "the colleague gets AUC over 0.95 and accuracy over 0.88", "don't modify the colleague's code")
        check(flawed_auc is not None and flawed_auc > 0.95 and 0.80 < honest_auc < 0.86 and close(flawed_auc, auc_c), "flawed AUC over 0.95, honest between 0.80 and 0.86", "lab(set()) and lab(all_flaws)")
        check(len(effects) == 3 and effects["duplicates"] > 0.1 and abs(effects["global_scaler"]) < 0.01 and -0.05 < effects["occupation_codes"] < -0.005, "duplicates: over +0.1; global scaler: below 0.01 in absolute value; arbitrary codes: between -0.05 and -0.005", "lab({key})[0] - honest_auc")
        check(order == ["duplicates", "occupation_codes", "global_scaler"] and worst_flaw == "duplicates", "order by severity: duplicates, arbitrary codes, global scaler", "sorted(..., key=lambda k: abs(effects[k]), reverse=True)")
        check(honest_perm is not None and 0.45 < honest_perm < 0.6 and duplicates_perm > 0.9 and scaler_perm < 0.6, "on permuted labels the correct flow stays near 0.5, with the duplicates it goes over 0.9; the global scaler alone does not explain anything (below 0.6)", "lab(..., y_perm)")
        check(dummy_acc is not None and 0.79 < dummy_acc < 0.83 and acc_gain < 0.05 and auc_gain > 0.3, "the baseline has accuracy 0.79-0.83 (the prevalence of the non-defaulters): the model gains less than 5 points of accuracy but over 0.3 of AUC", "DummyClassifier('most_frequent')")
        check(set(recommendations) == {"duplicates", "global_scaler", "occupation_codes"} and all(isinstance(v, str) and len(v) > 25 for v in recommendations.values()), "one recommendation for each flaw", "one sentence each")
    """,
    hint="""
        `effects = {k: lab({k})[0] - honest_auc for k in ["duplicates", "global_scaler", "occupation_codes"]}`. For the order: `sorted(effects, key=lambda k: abs(effects[k]), reverse=True)`. Dummy: `cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring="accuracy")["test_score"].mean()`.
    """,
    solution=[
        C("""
        all_flaws = {"duplicates", "global_scaler", "occupation_codes"}
        auc_c, acc_c = lab(all_flaws)
        print(f"AUC {auc_c:.3f}, accuracy {acc_c:.3f} -> 'great, let's put it in production'")
        """),
        C("""
        flawed_auc, flawed_acc = lab(all_flaws)
        honest_auc, honest_acc = lab(set())
        print(f"colleague's flow: AUC {flawed_auc:.3f}, accuracy {flawed_acc:.3f} | correct flow: AUC {honest_auc:.3f}, accuracy {honest_acc:.3f}")
        """),
        M("""
        **Method.** There is no point in reasoning "by feel": you start from the correct flow and switch on **one flaw at a time**, measuring the difference in AUC. It is the ablation in reverse: what each mistake costs (or gives away).
        """),
        C("""
        effects = {k: lab({k})[0] - honest_auc for k in ["duplicates", "global_scaler", "occupation_codes"]}
        order = sorted(effects, key=lambda k: abs(effects[k]), reverse=True)
        worst_flaw = order[0]
        print({k: round(v, 4) for k, v in effects.items()})
        print("order by severity:", order)
        """),
        M("""
        **The permuted labels test.** With labels that have no link with `X` a valid flow must give AUC 0.5. The flow with the duplicates does **not**, and without knowing where the mistake is: it is a generic alarm that does not require knowing the flaw.
        """),
        C("""
        y_perm = np.random.default_rng(0).permutation(y)
        honest_perm = lab(set(), y_perm)[0]
        duplicates_perm = lab({"duplicates"}, y_perm)[0]
        scaler_perm = lab({"global_scaler"}, y_perm)[0]
        print(f"AUC on permuted labels | correct {honest_perm:.3f} | duplicates only {duplicates_perm:.3f} | global scaler only {scaler_perm:.3f}")
        """),
        C("""
        r = cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring=["accuracy", "roc_auc"])
        dummy_acc = float(r["test_accuracy"].mean())
        acc_gain = honest_acc - dummy_acc
        auc_gain = honest_auc - float(r["test_roc_auc"].mean())
        print(f"baseline: accuracy {dummy_acc:.3f} | the honest model gains {acc_gain:.3f} of accuracy but {auc_gain:.3f} of AUC")
        """),
        C("""
        recommendations = {
            "duplicates": f"Balance (or duplicate) only inside each training fold, or use class_weight: with the duplicates before the CV the AUC rises by {effects['duplicates']:+.2f} and stays at 0.96 even on random labels.",
            "global_scaler": f"Put imputation and encoding inside the Pipeline: here the measured effect is only {effects['global_scaler']:+.3f}, but the principle matters and with other models or small samples it can cost much more.",
            "occupation_codes": f"Encode with one-hot (or out-of-fold target encoding): numeric codes are a meaningless alphabetical order and cost the forest {effects['occupation_codes']:+.3f} of AUC.",
        }
        recommendations
        """),
    ],
    learned="""
        - **The AUC of 0.97 is the result of the most serious flaw, not of the model:** with **only** the duplicates before the cross-validation the AUC goes from 0.83 to 0.97 (+0.14). The same rows are in the training and in the validation set, and the forest recognizes them. It is an ordering mistake (split first, then process) that no warning flags.
        - **The permuted labels test finds it without knowing where it is:** on random labels the correct flow gives 0.53 (about chance); with the duplicates 0.96. If a flow beats chance on data **with no signal**, it is not valid. It is the first check to run on any "too good" pipeline.
        - **Not all the flaws weigh the same:** `global_scaler` (imputation and encoding on the whole dataset) moves the AUC by +0.002, within the noise: for a forest and for this sample it changes nothing. But it is a flaw **of principle**: on small samples, with models sensitive to scale or with extreme statistics, it can matter. It is fixed anyway, at the cost of two lines (`Pipeline`).
        - **Arbitrary codes cost little but are free to avoid:** −0.014 of AUC for the forest. A one-hot respects the fact that `occupation_17` is not "more than" `occupation_16`. The ranking by effect (duplicates ≫ codes > scaler) says where to spend the time of a review: first the flaws that invalidate the number, then those that worsen it, then the theoretical ones.
        - **Accuracy hides the value of the model:** the baseline "nobody is a defaulter" has accuracy 0.81; the honest model has 0.84 (+0.03), while the AUC goes from 0.5 to 0.83 (+0.33). With imbalanced classes the reference metric is a ranking or probabilistic one, and the decision is evaluated with the costs.
    """,
    pitfall="""
        Fixing the flaws in the order in which you notice them instead of by measured effect, or trusting a score that is too high without a check that does not require knowing where the mistake is (permuted labels, baseline, learning curve).
    """,
    interview=(
        "You receive a notebook with AUC 0.97 on a problem where you expect 0.8. What do you do?",
        """
        First I doubt the number: I run the permuted labels test on the whole flow (it must give 0.5), I check that the preprocessing and any rebalancing are inside the validation (after the split), I look for duplicate rows between training and validation and for features built with the target or with later information. Then I measure the effect of each suspect on its own
        and report the honest result, with the baseline next to it.
        """),
)

EXERCISES = [P3]
