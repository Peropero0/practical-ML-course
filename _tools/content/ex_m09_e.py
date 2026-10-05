"""Module 9 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="the_early_stopping_that_never_stops", kind="E", level=2, minutes=45, module=MODULE, start="must_fail",
    title="The early stopping that never stops",
    goal="""
        Understand why a colleague, who turned on `early_stopping_rounds=30` and "monitors both curves" by passing training and validation to `eval_set`, always gets `best_iteration = 499` (the model uses all the rounds) and concludes that "early stopping is not needed, the model always improves". Find **which set XGBoost uses
        to stop**, fix it and measure the difference on the test set.
    """,
    dataset="""
        `make_classification(5000, 12 features, 5 informative, flip_y=0.1)`; split 50% training, 25% validation, 25% test. Model: `XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric="logloss", early_stopping_rounds=30)`.
    """,
    purpose="""
        Know a rule that is easy to miss: with several sets in `eval_set`, early stopping looks at **the last one**. Passing `[(X_val, y_val), (X_tr, y_tr)]` means stopping on the training curve, which always drops: the model never stops and overfits, with no error or warning.
    """,
    task=[
        "Run the colleague's cell and observe `best_col` (the number of rounds used) and `bug_test_ll` (log-loss on the test set).",
        "Diagnosis: save `which_set` (`\"training\"` or `\"validation\"`: the set on which XGBoost evaluates the stopping criterion) and `why_it_never_stops` (a short string). Check it with `evals_result()`: the `validation_0` curve is the validation one (first element), `validation_1` the training one (second); save `val_min_round` (round, from 1, at which the validation curve is at its minimum **in the colleague's run**).",
        "Fix: put the validation set **last** in `eval_set` (`[(X_tr, y_tr), (X_val, y_val)]`, or only the validation set): save `ok_round` (rounds used) and `ok_test_ll` (log-loss on the test set).",
        "Check: in the corrected run, the `argmin` of the validation curve (the last key of `evals_result()`) matches `best_iteration`; save `matches` (True/False). Save `gain` (`bug_test_ll − ok_test_ll`).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        X, y = make_classification(5000, n_features=12, n_informative=5, class_sep=0.7, flip_y=0.1, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.5, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)


        def new_model():
            return xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1, random_state=0)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        m_bug = new_model()
        m_bug.fit(X_tr, y_tr, eval_set=[(X_val, y_val), (X_tr, y_tr)], verbose=False)      # 'I monitor validation and training'
        best_col = m_bug.best_iteration + 1
        bug_test_ll = log_loss(y_te, m_bug.predict_proba(X_te)[:, 1])
        print(f"rounds used: {best_col} out of 500 | test log-loss {bug_test_ll:.4f} -> 'early stopping is not needed, the model always improves'")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        which_set = None
        why_it_never_stops = None
        val_min_round = None
        """),
        C("""
        ok_round = None
        ok_test_ll = None
        matches = None
        gain = None
        """),
    ],
    checks="""
        check(best_col >= 490, "the colleague sees that the model uses almost all the 500 rounds", "don't modify the colleague's code")
        check(which_set == "training" and isinstance(why_it_never_stops, str) and len(why_it_never_stops) > 10, "XGBoost uses the last set of eval_set, that is the training set, and you explain why", "XGBoost stops on eval_set[-1]")
        check(val_min_round is not None and 20 <= val_min_round <= 80, "the validation, in the colleague's run, had its minimum between 20 and 80 rounds: from there on the model was getting worse", "argmin of evals_result()['validation_0']['logloss'] + 1")
        check(ok_round is not None and ok_round < 100 and ok_test_ll < 0.42, "with the validation set last the model stops under 100 rounds and the test log-loss drops below 0.42", "eval_set=[(X_tr, y_tr), (X_val, y_val)]")
        check(matches is True and gain is not None and gain > 0.05, "the minimum of the validation curve matches best_iteration; the gain on the test log-loss exceeds 0.05", "bug_test_ll - ok_test_ll")
    """,
    hint="""
        `res = m_bug.evals_result()`; the key `"validation_0"` is the first set of `eval_set`. In the fix the last key is the validation set: `list(m.evals_result())[-1]`.
    """,
    solution=[
        C("""
        m_bug = new_model()
        m_bug.fit(X_tr, y_tr, eval_set=[(X_val, y_val), (X_tr, y_tr)], verbose=False)      # 'I monitor validation and training'
        best_col = m_bug.best_iteration + 1
        bug_test_ll = log_loss(y_te, m_bug.predict_proba(X_te)[:, 1])
        print(f"rounds used: {best_col} out of 500 | test log-loss {bug_test_ll:.4f} -> 'early stopping is not needed, the model always improves'")
        """),
        M("""
        **Where the bug is.** With several sets in `eval_set`, XGBoost applies `early_stopping_rounds` **to the last one**. Here the last one is the training set, whose log-loss drops at every round: the criterion "no improvement in 30 rounds" never triggers and the model goes on up to 500 rounds, well past the validation minimum. No error,
        no warning.
        """),
        C("""
        which_set = "training"                                                     # eval_set[-1] is (X_tr, y_tr)
        why_it_never_stops = "Early stopping looks at the last eval_set, that is the training set, whose log-loss always drops"
        val = np.array(m_bug.evals_result()["validation_0"]["logloss"])              # validation_0 = first set = validation
        val_min_round = int(val.argmin()) + 1
        print(f"in the colleague's run the validation has its minimum at round {val_min_round}, then rises up to {val[-1]:.3f} (minimum {val.min():.3f})")
        """),
        C("""
        m_ok = new_model()
        m_ok.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)     # the validation set is the last one
        ok_round = m_ok.best_iteration + 1
        ok_test_ll = log_loss(y_te, m_ok.predict_proba(X_te)[:, 1])
        val_ok = np.array(m_ok.evals_result()[list(m_ok.evals_result())[-1]]["logloss"])
        matches = bool(int(val_ok.argmin()) == m_ok.best_iteration)
        gain = bug_test_ll - ok_test_ll
        print(f"rounds used: {ok_round} | test log-loss {ok_test_ll:.4f} (before {bug_test_ll:.4f}, gain {gain:.4f}) | argmin of the validation = best_iteration: {matches}")
        """),
    ],
    learned="""
        - **Early stopping applies to the last set of `eval_set`:** with `[(val), (train)]` XGBoost looks at the training set, which always drops, and never stops (500 rounds out of 500). With the validation set last it stops at 38 rounds. There is no error or warning: the only symptom is `best_iteration` equal to `n_estimators - 1`.
        - **The cost is real:** with 500 rounds the test log-loss is 0.463, with the correct early stopping 0.374: a difference of 0.09, larger than almost any improvement you would get with tuning. The validation in the wrong run had its minimum (0.396) at round 38 and then rose to 0.486 without anyone looking at it.
        - **A signal to always check:** `best_iteration` very close to `n_estimators` (with learning rate 0.1) is almost always a sign that early stopping is not looking at the right curve, or that you need more `n_estimators`. In both cases it is worth plotting `evals_result()`.
        - **Practical rule:** put the validation set alone or last, and use `evals_result()` to check that the minimum of its curve matches `best_iteration`. If you also want to see the training curve, put it **first**.
    """,
    pitfall="""
        Adding the training set to `eval_set` "to see both curves" and putting it last: the curve under control becomes the training one and early stopping stops working.
    """,
    interview=(
        "How does XGBoost choose the stopping round when `eval_set` has several sets?",
        """
        It uses the last set (and, if there are several metrics, the last metric): so the validation set has to go last. If I put the training set there the stop never triggers, because its metric always improves; the sign is `best_iteration` close to `n_estimators`. I check with `evals_result()` that the minimum of the validation curve matches
        `best_iteration`.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="the_booster_ignores_early_stopping", kind="E", level=3, minutes=50, module=MODULE, start="must_fail",
    title="The production model does not match the validated one",
    goal="""
        Find out why the probabilities computed with `model.get_booster().predict(DMatrix)` (used "in production", for explanations and for export) differ from `model.predict_proba` (used in validation) and why the per-example contributions **do not sum** to the margin of the scikit-learn wrapper: after early stopping the booster contains
        **all** the trained trees, including the 30 added after the minimum, and the wrapper ignores them automatically while the native booster does not.
    """,
    dataset="""
        `make_classification(3000, 12 features, 5 informative, flip_y=0.1)`, 60/20/20 split (training / validation / test). `model = XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, early_stopping_rounds=30)` trained with the validation set in `eval_set`.
    """,
    purpose="""
        Know that `best_iteration` is an attribute of the model, **not** a truncation of the booster: after `fit` with early stopping the booster has `best_iteration + 31` trees. The wrapper `predict`/`predict_proba` uses only the first `best_iteration + 1`; `Booster.predict`, `pred_contribs` and (partly) saving do not, unless you tell them to.
    """,
    task=[
        "Run the colleague's cell: `max_diff` (maximum absolute gap between `production_p` = `booster.predict(DMatrix(X_te))` and `validated_p` = `model.predict_proba(X_te)[:, 1]`) and `production_ll`, `validated_ll`.",
        "Diagnosis: save `best` (`model.best_iteration`), `total_trees` (`booster.num_boosted_rounds()`) and `extra_trees` (`total_trees - (best + 1)`).",
        "Fix: `ok_p = booster.predict(DMatrix(X_te), iteration_range=(0, best + 1))` and `ok_p_matches` (True if `ok_p` matches `validated_p`). Save `contribs` (`pred_contribs=True`, with the same `iteration_range`) and `sum_ok` (True if `contribs.sum(1)` matches `model.predict(X_te, output_margin=True)`).",
        "Measure the damage of the wrong explanation: `bug_sum_gap` (maximum absolute gap between the sum of the contributions computed **without** `iteration_range` and the margin of the wrapper) and `top_changes` (True if the ranking of the most important feature by mean of the absolute values of the contributions changes between the wrong version and the correct one: it may also not change: answer with what you measure).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.metrics import log_loss
        from sklearn.model_selection import train_test_split

        X, y = make_classification(3000, n_features=12, n_informative=5, class_sep=0.7, flip_y=0.1, random_state=0)
        X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
        X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
        model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.1, max_depth=5, eval_metric="logloss", early_stopping_rounds=30, n_jobs=-1, random_state=0)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        booster = model.get_booster()
        dte = xgb.DMatrix(X_te)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        validated_p = model.predict_proba(X_te)[:, 1]                        # in validation
        production_p = booster.predict(dte)                                  # 'in production we use the booster: it is faster'
        max_diff = float(np.abs(production_p - validated_p).max())
        validated_ll, production_ll = log_loss(y_te, validated_p), log_loss(y_te, production_p)
        print(f"maximum gap between the probabilities: {max_diff:.4f} | validated log-loss {validated_ll:.4f}, in production {production_ll:.4f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        best = None
        total_trees = None
        extra_trees = None
        """),
        C("""
        ok_p = None
        ok_p_matches = None
        contribs = None
        sum_ok = None
        """),
        C("""
        bug_sum_gap = None
        top_changes = None
        """),
    ],
    checks="""
        check(max_diff > 0.03 and production_ll > validated_ll, "the colleague sees a gap of over 0.03 between the probabilities and a worse log-loss in production", "don't modify the colleague's code")
        check(best is not None and total_trees is not None and total_trees == best + 31 and extra_trees == 30, "the booster has best_iteration + 31 trees: the 30 rounds of patience are inside the model", "booster.num_boosted_rounds()")
        check(ok_p is not None and ok_p_matches is True, "with iteration_range=(0, best + 1) the booster reproduces predict_proba", "iteration_range=(0, best_iteration + 1)")
        check(contribs is not None and contribs.shape == (len(X_te), 13) and sum_ok is True, "13 columns (12 features and the bias) and sum equal to the margin of the wrapper", "pred_contribs=True with the same iteration_range")
        check(bug_sum_gap is not None and bug_sum_gap > 0.3 and top_changes in (True, False), "without iteration_range the sum of the contributions departs by over 0.3 from the validated margin", "sum of the contributions minus margin")
    """,
    hint="""
        `booster.num_boosted_rounds()` counts the trees (one tree per round for binary classification). Margin of the wrapper: `model.predict(X_te, output_margin=True)`. Contributions with `iteration_range`: `booster.predict(dte, pred_contribs=True, iteration_range=(0, best + 1))`.
    """,
    solution=[
        C("""
        validated_p = model.predict_proba(X_te)[:, 1]                        # in validation
        production_p = booster.predict(dte)                                  # 'in production we use the booster: it is faster'
        max_diff = float(np.abs(production_p - validated_p).max())
        validated_ll, production_ll = log_loss(y_te, validated_p), log_loss(y_te, production_p)
        print(f"maximum gap between the probabilities: {max_diff:.4f} | validated log-loss {validated_ll:.4f}, in production {production_ll:.4f}")
        """),
        M("""
        **Where the bug is.** `fit` with early stopping **does not remove** the trees after the minimum: the booster contains `best_iteration + 1` good trees plus the 30 of "patience". `model.predict_proba` automatically uses only the first `best_iteration + 1`; `booster.predict` (the native API) uses all of them, as does `pred_contribs`. You need
        `iteration_range=(0, best_iteration + 1)`.
        """),
        C("""
        best = int(model.best_iteration)
        total_trees = booster.num_boosted_rounds()
        extra_trees = total_trees - (best + 1)
        print(f"best_iteration = {best} -> good trees {best + 1}, trees in the booster {total_trees} ({extra_trees} more)")
        """),
        C("""
        ok_p = booster.predict(dte, iteration_range=(0, best + 1))
        ok_p_matches = bool(np.allclose(ok_p, validated_p, atol=1e-6))
        contribs = booster.predict(dte, pred_contribs=True, iteration_range=(0, best + 1))
        margin = model.predict(X_te, output_margin=True)
        sum_ok = bool(np.allclose(contribs.sum(1), margin, atol=1e-4))
        print("the booster with iteration_range reproduces predict_proba:", ok_p_matches, "| the contributions sum to the margin of the wrapper:", sum_ok)
        """),
        C("""
        bug_contribs = booster.predict(dte, pred_contribs=True)                       # without iteration_range: all the trees
        bug_sum_gap = float(np.abs(bug_contribs.sum(1) - margin).max())
        top_bug = np.abs(bug_contribs[:, :-1]).mean(0).argmax()
        top_ok = np.abs(contribs[:, :-1]).mean(0).argmax()
        top_changes = bool(top_bug != top_ok)
        print(f"maximum gap between the sum of the contributions (without iteration_range) and the margin: {bug_sum_gap:.3f} | the most important feature changes: {top_changes} (bug: f{top_bug}, correct: f{top_ok})")
        """),
    ],
    learned="""
        - **`best_iteration` is an attribute, not a truncation:** the booster always contains all the trained trees (here 126, against the 96 good ones). `XGBClassifier.predict` and `predict_proba` know how to use `best_iteration`, the native API (`Booster.predict`, `pred_contribs`) does not unless you specify `iteration_range`.
        - **Effect on the probabilities:** maximum gap of 0.11 between the two versions and log-loss 0.385 against 0.380 on the test set: a small but systematic difference, which in production shows up as "the model got worse compared with the validation" without anyone understanding why.
        - **Effect on the explanations:** the sum of the contributions computed with all the trees departs by 0.86 from the validated margin: the explanation concerns a different model from the one that predicts. The mean importance per feature often does not change (the top features stay the same), but the decomposition of the single case is no longer reliable.
        - **Practical rule:** either always pass `iteration_range=(0, best_iteration + 1)` to the native APIs, or retrain with `n_estimators = best_iteration + 1` without early stopping before exporting. A useful regression test: `np.allclose(booster.predict(...), wrapper.predict_proba(...))` before going to production.
    """,
    pitfall="""
        Extracting the booster (`get_booster()`), saving it or using it for SHAP/contributions believing it contains only the "best" model: it also contains the patience rounds. Check `num_boosted_rounds()` and pass `iteration_range`.
    """,
    interview=(
        "After a fit with early stopping, does the booster contain the trees of the best round?",
        """
        No, it contains all the trained trees (best round plus the patience rounds). The scikit-learn wrapper uses only the first best_iteration + 1 in predict, but the native API (Booster.predict, pred_contribs, SHAP) uses everything unless you pass iteration_range=(0, best_iteration + 1). Before exporting I compare the predictions of the booster with those
        of the wrapper, or I retrain with the optimal number of rounds.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="monotone_constraints_in_the_wrong_order", kind="E", level=3, minutes=50, module=MODULE, start="must_fail",
    title="Monotonic constraints ruin the model (or is it the order?)",
    goal="""
        A colleague adds `monotone_constraints=(-1, 0, 1, 0)` to a loan repayment model ("more income, more probability of repayment; more debt, less") and finds that the AUC collapses from 0.83 to 0.65: they conclude that "monotonic constraints do not work". Find the bug (the tuple follows a **different order** from the columns), fix it with a dictionary by name and
        **check** that the model really is monotonic and that `income` has an effect again.
    """,
    dataset="""
        Synthetic data: 4000 applicants with `income`, `debt`, `age`, `noise`; repayment probability increasing with `income`, decreasing with `debt`, almost independent of `age`. Columns in the order `income, debt, age, noise`; 60/40 split.
    """,
    purpose="""
        Learn that `monotone_constraints` as a tuple is **positional** and does not check the names: a constraint in the wrong order is applied to another feature, forcing an effect opposite to the real one. And how to verify a constraint: a `profile` function that varies one feature keeping the others fixed.
    """,
    task=[
        "Run the colleague's cell: `free_auc` (without constraints), `bug_auc` (with the colleague's tuple).",
        "Diagnosis: `applied` (dictionary column → constraint actually applied, with `zip(X.columns, tuple)`); `real_signs` (dictionary column → sign of the correlation with `y`, `-1` or `+1`, for `income`, `debt`, `age`); `wrong` (sorted list of the columns whose applied constraint is `+1` or `-1` **and** has the opposite sign to the real one).",
        "Fix: `ok_constraints = {\"income\": 1, \"debt\": -1}` (dictionary by name, accepted by XGBoost 1.6) and `ok_auc` with the constrained model.",
        "Check with `profile(model, X_te, column)` (given: probability as one column varies, the others at the median): `bug_swing` and `ok_swing` (maximum minus minimum of the `income` profile in the two models: how much the probability changes as `income` varies), `free_violations` (number of grid points where, in the model **without** constraints, `income` makes the probability decrease) and `monotone_ok` (True if the corrected model has no violations for `income` (+) and `debt` (−)).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split

        rng = np.random.default_rng(0)
        n = 4000
        X = pd.DataFrame({"income": rng.normal(size=n), "debt": rng.normal(size=n), "age": rng.normal(size=n), "noise": rng.normal(size=n)})
        y = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * X["income"] - 1.0 * X["debt"] + 0.3 * X["age"])))).astype(int)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)


        def model(**kw):
            return xgb.XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, eval_metric="auc", n_jobs=-1, random_state=0, **kw)


        def profile(m, X_ref, column, points=30):
            \"\"\"Probability as one column varies (from the 5th to the 95th percentile), with the other columns at the median.\"\"\"
            grid = np.linspace(X_ref[column].quantile(0.05), X_ref[column].quantile(0.95), points)
            base = pd.DataFrame([X_ref.median()] * points).reset_index(drop=True)
            base[column] = grid
            return m.predict_proba(base)[:, 1]
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        constraints = (-1, 0, 1, 0)            # debt (-), age, income (+), noise: 'I wrote them in the order I thought of the features'
        free_model = model().fit(X_tr, y_tr)
        bug_model = model(monotone_constraints=constraints).fit(X_tr, y_tr)
        free_auc = roc_auc_score(y_te, free_model.predict_proba(X_te)[:, 1])
        bug_auc = roc_auc_score(y_te, bug_model.predict_proba(X_te)[:, 1])
        print(f"AUC without constraints {free_auc:.3f} | with the colleague's constraints {bug_auc:.3f} -> 'monotonic constraints ruin the model'")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        applied = {}
        real_signs = {}
        wrong = []
        """),
        C("""
        ok_constraints = None
        ok_auc = None
        """),
        C("""
        bug_swing = None
        ok_swing = None
        free_violations = None
        monotone_ok = None
        """),
    ],
    checks="""
        check(free_auc > 0.8 and bug_auc < free_auc - 0.1, "without constraints the AUC goes over 0.8; with the colleague's it loses more than 0.1", "don't modify the colleague's code")
        check(applied == {"income": -1, "debt": 0, "age": 1, "noise": 0}, "the tuple applied to the columns: income -1, debt 0, age +1, noise 0", "dict(zip(X.columns, tuple))")
        check(real_signs.get("income") == 1 and real_signs.get("debt") == -1 and real_signs.get("age") == 1, "real signs: income +, debt -, age +", "np.sign(np.corrcoef(X[c], y)[0, 1])")
        check(wrong == ["income"], "the only column with a constraint of the opposite sign to the real one is income (for age the sign matches by chance)", "applied constraint ≠ real sign, both different from 0")
        check(ok_constraints == {"income": 1, "debt": -1} and ok_auc is not None and ok_auc > free_auc - 0.01, "with the dictionary by name the AUC goes back to the free one (or exceeds it)", "monotone_constraints={'income': 1, 'debt': -1}")
        check(bug_swing is not None and bug_swing < 0.02 and ok_swing > 0.3, "in the colleague's model the probability is flat with respect to income (swing below 0.02); in the corrected one it changes by over 0.3", "profile(m, X_te, 'income').max() - .min()")
        check(free_violations is not None and free_violations > 0 and monotone_ok is True, "the free model has points where income makes the probability drop (spurious oscillations); the corrected one is monotonic for income (+) and debt (−)", "np.diff(profile(...))")
    """,
    hint="""
        `applied = dict(zip(X_tr.columns, constraints))`; signs: `np.sign(np.corrcoef(X_tr[c], y_tr)[0, 1])`. Swing: `p = profile(m, X_te, "income"); p.max() - p.min()`. Violations: `(np.diff(p) < -1e-9).sum()`. The corrected model: `model(monotone_constraints={"income": 1, "debt": -1}).fit(X_tr, y_tr)`.
    """,
    solution=[
        C("""
        constraints = (-1, 0, 1, 0)            # debt (-), age, income (+), noise: 'I wrote them in the order I thought of the features'
        free_model = model().fit(X_tr, y_tr)
        bug_model = model(monotone_constraints=constraints).fit(X_tr, y_tr)
        free_auc = roc_auc_score(y_te, free_model.predict_proba(X_te)[:, 1])
        bug_auc = roc_auc_score(y_te, bug_model.predict_proba(X_te)[:, 1])
        print(f"AUC without constraints {free_auc:.3f} | with the colleague's constraints {bug_auc:.3f} -> 'monotonic constraints ruin the model'")
        """),
        M("""
        **Where the bug is.** The tuple is **positional**: the first value goes to the first column of the DataFrame (`income`), not to `debt`. The colleague wrote the constraints in a different order from the columns, which are `income, debt, age, noise`, so `income` receives `-1` ("the probability cannot increase with income") and `age` receives `+1`.
        Since income really has an increasing effect, every useful split is forbidden: the model cannot use the most important feature and ignores it. It is not a flaw of monotonic constraints.
        """),
        C("""
        applied = dict(zip(X_tr.columns, constraints))
        real_signs = {c: int(np.sign(np.corrcoef(X_tr[c], y_tr)[0, 1])) for c in ["income", "debt", "age"]}
        wrong = sorted(c for c in ["income", "debt", "age"] if applied[c] != 0 and applied[c] != real_signs[c])
        print("constraints applied:", applied)
        print("real signs:", real_signs, "| constraints with the opposite sign:", wrong)
        """),
        C("""
        ok_constraints = {"income": 1, "debt": -1}
        ok_model = model(monotone_constraints=ok_constraints).fit(X_tr, y_tr)
        ok_auc = roc_auc_score(y_te, ok_model.predict_proba(X_te)[:, 1])
        print(f"AUC with the correct constraints {ok_auc:.3f} (free {free_auc:.3f})")
        """),
        C("""
        def n_violations(m, column, sign=1):
            return int(((sign * np.diff(profile(m, X_te, column))) < -1e-9).sum())


        def swing(m, column):
            p = profile(m, X_te, column)
            return float(p.max() - p.min())


        bug_swing, ok_swing = swing(bug_model, "income"), swing(ok_model, "income")
        free_violations = n_violations(free_model, "income")
        monotone_ok = bool(n_violations(ok_model, "income", 1) == 0 and n_violations(ok_model, "debt", -1) == 0)
        print(f"swing of the probability as income varies: colleague's model {bug_swing:.3f}, corrected {ok_swing:.3f}")
        print(f"points (out of 29) where income makes the probability drop: free model {free_violations}, corrected {n_violations(ok_model, 'income')}")
        print("the corrected model is monotonic for income (+) and debt (-):", monotone_ok)
        """),
    ],
    learned="""
        - **`monotone_constraints` as a tuple is positional:** the i-th value goes to the i-th column and XGBoost does not check the names. With the wrong order the constraint is applied to another feature: here `income` receives `-1`, but the real effect is increasing, so the model cannot use it (its profile is flat) and the AUC drops from 0.83 to 0.65. The colleague blamed the constraints.
        - **The dictionary by name is safer:** `{"income": 1, "debt": -1}` (XGBoost 1.6 accepts it) does not depend on the order of the columns, and a ColumnTransformer that shuffles them no longer breaks anything. In pipelines, where the output names change, the tuple is a time bomb.
        - **A right constraint costs almost nothing:** the constrained AUC is equal to the free one (0.827 against 0.827). When the constraint is consistent with reality, it only removes the spurious oscillations and it is a gift for explainability (and for regulators, in credit/insurance).
        - **Always verify the constraint, do not trust it:** with `profile` (one feature varying, the others at the median) you measure two things: the monotonicity violations (the free model has 5 out of 29 points, the corrected one 0 by construction) and the swing, that is how much the probability really changes. A wrong-sign constraint is not seen from the violations (a flat profile is monotonic!) but from the null swing. It is a test to put in the validation pipeline.
    """,
    pitfall="""
        Passing `monotone_constraints` as a tuple after a preprocessing step that changes the order of the columns (ColumnTransformer, one-hot): the constraints shift position without errors. Use the dictionary by name or build the tuple from the column names.
    """,
    interview=(
        "What are monotonic constraints in XGBoost for and how do you verify them?",
        """
        They force the prediction to be nondecreasing (or nonincreasing) with respect to a feature: useful when the domain requires it (credit, prices, insurance) and as a regularization. They cost little if the constraint is consistent with the data. I specify them by name (not by position) and verify them with a profile: I vary the feature keeping the others fixed and
        check that the differences do not change sign.
        """),
)

EXERCISES = [E1, E2, E3]
