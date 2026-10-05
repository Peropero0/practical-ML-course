"""Module 8 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="duplicate_rows_and_optimistic_oob", kind="C", level=3, minutes=50, module=MODULE,
    title="Duplicate rows: the OOB error (and the cross-validation) say 100%",
    goal="""
        Build a noisy dataset in which every row appears **twice** (or almost: with a tiny measurement noise) and show that both the OOB accuracy and a random cross-validation become optimistic, because the copy of an example that is "outside the sample" is often inside. Fix it with
        validation **by group** (`GroupKFold`) and with the removal of the duplicates.
    """,
    dataset="""
        `make_classification(1000, 10 features, 2 informative, flip_y=0.4)` (very noisy: the true generalization is about 0.75). `X_d, y_d, group` is the same dataset with every row repeated twice (`group` = index of the original row); `X_q` is the version with the copy perturbed by a noise of 1e-3.
    """,
    purpose="""
        Recognize when an "internal" error (OOB or K-fold) is optimistic: it happens every time non-independent rows (duplicates, repeated measurements of the same subject, nearly identical images) end up in different folds. It is the group leakage of module 3, inside the forest.
    """,
    task=[
        "Train `RandomForestClassifier(200, oob_score=True, random_state=0, n_jobs=-1)` on the **clean** dataset (`X, y`) and save `clean_oob`; on the dataset with exact duplicates `X_d, y_d` save `dup_oob` and `near_oob` (with `X_q`).",
        "Cross-validation: `random_cv` (`StratifiedKFold(5, shuffle=True, random_state=0)` on the duplicates) and `group_cv` (`GroupKFold(5)` with `groups=group`, same model).",
        "Save `oob_inflation = dup_oob − clean_oob`, `cv_inflation = random_cv − group_cv`.",
        "Fix it: remove the exact duplicates (`np.unique` on the whole rows or `pd.DataFrame(...).drop_duplicates()`) and save `oob_after_dedup`. Why can't you remove the near-duplicates (`X_q`) with `drop_duplicates`? Save `n_rows_after_near_dedup` (rows left in `X_q` after `drop_duplicates`).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score

        X, y = make_classification(1000, n_features=10, n_informative=2, n_redundant=0, flip_y=0.4, random_state=3)
        X_d, y_d = np.r_[X, X], np.r_[y, y]
        group = np.r_[np.arange(1000), np.arange(1000)]
        X_q = np.r_[X, X + np.random.default_rng(0).normal(0, 1e-3, X.shape)]
        rf = lambda: RandomForestClassifier(200, oob_score=True, random_state=0, n_jobs=-1)
    """,
    starter=[
        C("""
        clean_oob = None
        dup_oob = None
        near_oob = None
        """),
        C("""
        random_cv = None
        group_cv = None
        oob_inflation = None
        cv_inflation = None
        """),
        C("""
        oob_after_dedup = None
        n_rows_after_near_dedup = None
        """),
    ],
    checks="""
        check(0.7 < clean_oob < 0.82, "on the clean dataset the OOB is realistic (0.70-0.82)", "RandomForestClassifier(oob_score=True) on X, y")
        check(dup_oob > 0.97 and near_oob > 0.95, "with exact duplicates the OOB goes over 0.97; with near-duplicates 0.95", "the copy of an 'outside' example is inside the bootstrap")
        check(random_cv > 0.9 and 0.7 < group_cv < 0.82, "random cross-validation on the duplicates goes over 0.9; by group it goes back to 0.70-0.82", "GroupKFold(5) with groups=group")
        check(oob_inflation > 0.2 and cv_inflation > 0.15 and close(oob_inflation, dup_oob - clean_oob), "inflation of over 0.2 for the OOB and over 0.15 for the cross-validation", "differences with the correct estimate")
        check(abs(oob_after_dedup - clean_oob) < 0.02 and n_rows_after_near_dedup == 2000, "after removing the duplicates the OOB goes back to the clean one; the near-duplicates are not removed (2000 rows stay)", "drop_duplicates compares identical values")
    """,
    hint="""
        `pd.DataFrame(np.c_[X_d, y_d]).drop_duplicates().to_numpy()` then separate the columns. `GroupKFold`: `cross_val_score(rf(), X_d, y_d, groups=group, cv=GroupKFold(5)).mean()`. `oob_score_` after `fit`.
    """,
    solution=[
        C("""
        clean_oob = rf().fit(X, y).oob_score_
        dup_oob = rf().fit(X_d, y_d).oob_score_
        near_oob = rf().fit(X_q, y_d).oob_score_
        print(f"OOB: clean {clean_oob:.3f} | exact duplicates {dup_oob:.3f} | near-duplicates {near_oob:.3f}")
        """),
        C("""
        random_cv = cross_val_score(rf(), X_d, y_d, cv=StratifiedKFold(5, shuffle=True, random_state=0)).mean()
        group_cv = cross_val_score(rf(), X_d, y_d, groups=group, cv=GroupKFold(5)).mean()
        oob_inflation = dup_oob - clean_oob
        cv_inflation = random_cv - group_cv
        print(f"cross-validation: random {random_cv:.3f} | by group {group_cv:.3f} | OOB inflation {oob_inflation:+.3f}, cross-validation {cv_inflation:+.3f}")
        """),
        C("""
        data = pd.DataFrame(np.c_[X_d, y_d]).drop_duplicates().to_numpy()
        oob_after_dedup = rf().fit(data[:, :-1], data[:, -1].astype(int)).oob_score_
        n_rows_after_near_dedup = len(pd.DataFrame(X_q).drop_duplicates())
        print(f"rows after removing the exact duplicates: {len(data)} | OOB: {oob_after_dedup:.3f}")
        print(f"near-duplicates: drop_duplicates leaves {n_rows_after_near_dedup} rows out of {len(X_q)} (none removed)")
        """),
    ],
    learned="""
        - **The OOB assumes independent examples.** If every row has a copy, for a tree that does not have a row in its own bootstrap the copy is there with probability 63%: the tree has already "seen" it and remembers it. Result: OOB accuracy over 0.99 on data whose true generalization is about 0.76.
        - **Random cross-validation is optimistic too** (0.94): the validation fold contains copies of training rows. The remedy is `GroupKFold` with a group identifier (customer, patient, source): honest estimate 0.75 (0.748). It is the same problem as M3 C2, inside a model that amplifies it (a forest memorizes).
        - **Near-duplicates are worse**: `drop_duplicates` does not see them (the values differ by 1e-3) and the OOB stays inflated. They are found by grouping on identifiers, with rounded hashes or with a nearest-neighbor search. In real data: repeated measurements of the same subject, images of consecutive frames, versions of a document.
        - A warning sign: **very high OOB or CV accuracy on a problem you expect to be noisy**, together with a huge gap with a test set collected at another time or from other subjects.
    """,
    pitfall="""
        Trusting the OOB (or a random K-fold) without knowing whether the rows are independent. Always ask: "are there rows that describe the same object?" If so, validate by group.
    """,
    interview=(
        "When does the out-of-bag error mislead?",
        """
        When the examples are not independent: duplicates, repeated measurements of the same subject, consecutive frames, temporal data. The copy of an "out-of-sample" example for a tree is often in its bootstrap, so the tree has already seen it and the OOB error is too low. In those cases I validate by group
        (GroupKFold on customer/patient) or by time, and I remove the exact and near duplicates.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="correlated_features_and_permutation_importance", kind="C", level=3, minutes=55, module=MODULE,
    title="Adding copies of a feature makes it look useless",
    goal="""
        Add `k` almost identical copies of the most important feature and observe that the permutation importance of **each** copy drops toward zero (the model can use the others) while the sum stays and the impurity importance is split. Then fix it with the **group permutation**:
        shuffling the correlated columns together.
    """,
    dataset="""
        `make_classification(2000, 8 features: 4 informative, class_sep=1, flip_y=0.05, shuffle=False)`, 60/40 split. The most important feature is column 0; the copies are `x0 + noise(0.05)`.
    """,
    purpose="""
        Understand a limit of the interpretation of tree models: with redundant features the individual importance is **not** the importance of the "information" they carry. An analyst who looks at the importances can conclude that a fundamental variable does not matter, just because copies of it exist.
    """,
    task=[
        "Train `RandomForestClassifier(200, random_state=0, n_jobs=-1)` without copies and compute `perm0` (permutation importance on the test set, `n_repeats=10`, `random_state=0`) of column 0.",
        "For `k` in `[1, 3, 5]` add `k` copies (`x0 + normal(0, 0.05)`, with `rng = np.random.default_rng(0)`, in both training and test), retrain and save in `table` (index `k`): `acc`, `orig_perm` (permutation of column 0), `mean_single_copy_perm` (mean of the permutations of the `k` copies), `orig_imp` (impurity importance of column 0) and `group_imp` (impurity sum of column 0 and the copies).",
        "**Group permutation:** write `group_perm(model, X, y, columns, n_repeats, seed)`: baseline accuracy minus the mean accuracy after permuting **with the same permutation** all the given columns (the same shuffling of rows). For `k = 5` save `group_perm_k5` (column 0 and the five copies together).",
        "Answer: `orig_perm_at_k5_is_zero` (True if `orig_perm` for `k = 5` is smaller than 0.02) and `group_recovers` (True if `group_perm_k5 > 0.8 * perm0`).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split

        X, y = make_classification(2000, n_features=8, n_informative=4, n_redundant=0, class_sep=1.0, flip_y=0.05, shuffle=False, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        rf = lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1)


        def with_copies(k, seed=0):
            rng = np.random.default_rng(seed)
            copies_tr = [X_tr[:, 0] + rng.normal(0, 0.05, len(X_tr)) for _ in range(k)]
            copies_te = [X_te[:, 0] + rng.normal(0, 0.05, len(X_te)) for _ in range(k)]
            return np.column_stack([X_tr] + copies_tr), np.column_stack([X_te] + copies_te)
    """,
    starter=[
        C("""
        perm0 = None
        """),
        C("""
        table = pd.DataFrame(index=[1, 3, 5], columns=["acc", "orig_perm", "mean_single_copy_perm", "orig_imp", "group_imp"], dtype=float)
        for k in table.index:
            A_tr, A_te = with_copies(k)
            ...
        table.round(3)
        """),
        C("""
        def group_perm(model, X, y, columns, n_repeats=10, seed=0):
            ...


        group_perm_k5 = None
        orig_perm_at_k5_is_zero = None
        group_recovers = None
        """),
    ],
    checks="""
        check(perm0 is not None and 0.15 < perm0 < 0.25, "without copies column 0 is worth about 0.2 of lost accuracy (0.15-0.25)", "permutation_importance(...).importances_mean[0]")
        check(table.notna().all().all() and table.loc[1, "orig_perm"] < 0.7 * perm0 and table.loc[3, "orig_perm"] < 0.1 * perm0 + 0.01, "with 1 copy the permutation of column 0 drops below 70%; with 3 below 15%", "the importance is split among the copies")
        check(table.loc[[3, 5], "orig_perm"].max() < 0.02 and table.loc[[3, 5], "mean_single_copy_perm"].max() < 0.02, "with 3 and 5 copies each column, alone, is worth less than 0.02: it looks useless", "the model uses the other copies")
        check(table["orig_imp"].is_monotonic_decreasing and table["group_imp"].is_monotonic_increasing, "the impurity of column 0 drops, that of the group rises: more copies = more total weight", "sum of the importances of the group")
        check(group_perm_k5 is not None and group_perm_k5 > 0.8 * perm0 and orig_perm_at_k5_is_zero is True and group_recovers is True, "permuting the group together recovers the importance (over 80% of the initial one)", "same row permutation for all the columns of the group")
    """,
    hint="""
        Single permutation: `permutation_importance(f, A_te, y_te, n_repeats=10, random_state=0).importances_mean`. Group: `base = model.score(X, y)`; for each repetition `p = rng.permutation(len(X)); Xp = X.copy(); Xp[:, columns] = X[p][:, columns]`; `loss = base - model.score(Xp, y)`; mean.
    """,
    solution=[
        C("""
        f0 = rf().fit(X_tr, y_tr)
        perm0 = permutation_importance(f0, X_te, y_te, n_repeats=10, random_state=0).importances_mean[0]
        print(f"permutation importance of column 0 without copies: {perm0:.3f}")
        """),
        C("""
        table = pd.DataFrame(index=[1, 3, 5], columns=["acc", "orig_perm", "mean_single_copy_perm", "orig_imp", "group_imp"], dtype=float)
        models = {}
        for k in table.index:
            A_tr, A_te = with_copies(k)
            f = rf().fit(A_tr, y_tr)
            models[k] = (f, A_te)
            perm = permutation_importance(f, A_te, y_te, n_repeats=10, random_state=0).importances_mean
            group = [0] + list(range(8, 8 + k))
            table.loc[k] = [f.score(A_te, y_te), perm[0], perm[8:8 + k].mean(), f.feature_importances_[0], f.feature_importances_[group].sum()]
        table.round(3)
        """),
        C("""
        def group_perm(model, X, y, columns, n_repeats=10, seed=0):
            rng = np.random.default_rng(seed)
            base = model.score(X, y)
            losses = []
            for _ in range(n_repeats):
                p = rng.permutation(len(X))
                Xp = X.copy()
                Xp[:, columns] = X[p][:, columns]                       # same permutation for all the columns of the group
                losses.append(base - model.score(Xp, y))
            return float(np.mean(losses))


        f5, A_te5 = models[5]
        group_perm_k5 = group_perm(f5, A_te5, y_te, [0] + list(range(8, 13)))
        orig_perm_at_k5_is_zero = bool(table.loc[5, "orig_perm"] < 0.02)
        group_recovers = bool(group_perm_k5 > 0.8 * perm0)
        print(f"group permutation (column 0 and 5 copies) = {group_perm_k5:.3f} (without copies: {perm0:.3f}) | column 0 alone with 5 copies: {table.loc[5, 'orig_perm']:.3f}")
        """),
    ],
    learned="""
        - **With one copy the permutation importance halves, with three or five it disappears:** 0.199 without copies → 0.103 with one copy → about 0 with three (−0.001) and with five (0.009). Shuffling a single column does not take information away from the model, which finds it in the copies. A report listing the individual importances would conclude that "x0 does not matter": it is the most important variable of the problem.
        - **The impurity importance is split** (0.21 → 0.05 for the original column) but the **sum of the group rises** (0.34 with five copies): the forest uses the copies instead of the original, precisely because they are interchangeable.
        - **The group permutation** (shuffling the correlated columns together, with the same row permutation) measures the importance of the **information** and not of the single column: it recovers (and exceeds) the original value: 0.24 against 0.2. It applies to groups you have identified by correlation or meaning (the measurements of the same sensor, the variables of a single record).
        - What to do in practice: before interpreting the importances **group the correlated features** (correlation matrix, hierarchical clustering) and compute the importance of the groups; or reduce the redundancy (PCA on a group, M6).
    """,
    pitfall="""
        Discarding the "unimportant" features after a permutation or after the impurity importance without checking the correlations. If you remove them one at a time the copies take their place, but if you remove the whole group you lose the information.
    """,
    interview=(
        "How do you interpret importances with highly correlated variables?",
        """
        With caution: the importance (by permutation or from impurity) is split among redundant variables and each looks unimportant. I group the correlated features (correlations, clustering) and compute the group permutation, shuffling all the columns of the group together, or I reduce the redundancy upstream.
        I do not remove features based on the individual importance without this check.
        """),
)

EXERCISES = [C1, C2]
