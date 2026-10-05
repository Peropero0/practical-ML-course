"""Module 8 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="how_many_trees_do_you_really_need", kind="D", level=2, minutes=50, module=MODULE,
    title="How many trees do you need? Reading OOB curves, test curves and disagreement between forests",
    goal="""
        Read how the OOB error, the test error and the **disagreement between two forests with different seeds** change as `n_estimators` grows, on two different problems: an "easy" one (the error settles early) and one with hard interactions (the error keeps dropping). Conclude when "a budget of 100 trees" is enough and when it is not.
    """,
    dataset="""
        `easy`: `make_classification(2000, 12 features, 5 informative, flip_y=0.08)`. `interactions`: 2000 points with 10 Gaussian features and `y = (x0·x1 > 0) XOR (x2 > 0.3)` with 5% of swapped labels. 60/40 split; `warm_start` to add trees.
    """,
    purpose="""
        Replace the rule "100-500 trees are fine" with reading the curves of the problem: how fast the error settles depends on how hard the signal is, and the **predictions** settle later than the accuracy.
    """,
    task=[
        "For each problem and for `n` in `[5, 20, 50, 100, 250, 500]` train with `warm_start=True` (increasing `n_estimators`, `oob_score=True`, `random_state=0`) and save `oob_err` and `test_err` (`pd.DataFrame` index `n`, columns `easy` and `interactions`, errors = 1 − accuracy).",
        "Save `disagreement` (same shape): for each `n`, the mean over the pairs of **6 independent forests** (`random_state=0..5`) of the fraction of test examples on which the predictions (`predict`) differ.",
        "Save `stable_n_error` (dictionary problem → the smallest `n` at which `oob_err` is within **0.01** of the value at 500) and `stable_n_predictions` (the smallest `n` at which `disagreement < 0.05`).",
        "Answer: `needs_more_trees` (name of the problem with the largest `stable_n_error`) and `predictions_still_unstable` (True if, in the problem with interactions, the disagreement at 500 trees exceeds 0.05 even though the OOB error is already stable).",
    ],
    setup="""
        from sklearn.datasets import make_classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        X1, y1 = make_classification(2000, n_features=12, n_informative=5, n_redundant=2, class_sep=0.8, flip_y=0.08, random_state=0)
        rng = np.random.default_rng(0)
        X2 = rng.normal(size=(2000, 10))
        y2 = ((X2[:, 0] * X2[:, 1] > 0) ^ (X2[:, 2] > 0.3)).astype(int)
        y2 = np.where(rng.random(2000) < 0.05, 1 - y2, y2)
        problems = {"easy": (X1, y1), "interactions": (X2, y2)}
        splits = {name: train_test_split(X, y, test_size=0.4, random_state=0) for name, (X, y) in problems.items()}
        ns = [5, 20, 50, 100, 250, 500]
    """,
    starter=[
        C("""
        oob_err = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        test_err = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        for name, (X_tr, X_te, y_tr, y_te) in splits.items():
            f = RandomForestClassifier(warm_start=True, oob_score=True, random_state=0, n_jobs=-1)
            for n in ns:
                ...
        pd.concat({"OOB": oob_err, "test": test_err}, axis=1).round(3)
        """),
        C("""
        disagreement = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        disagreement
        """),
        C("""
        stable_n_error = {}
        stable_n_predictions = {}
        needs_more_trees = None
        predictions_still_unstable = None
        """),
    ],
    checks="""
        check(oob_err.notna().all().all() and test_err.notna().all().all() and oob_err.loc[5, "easy"] > oob_err.loc[500, "easy"] + 0.03, "complete OOB and test curves; for the easy problem the OOB error drops by over 3 points from 5 to 500 trees", "1 - oob_score_ and 1 - score(X_te, y_te)")
        check(oob_err.loc[500, "interactions"] < oob_err.loc[100, "interactions"] - 0.05 and oob_err.loc[100, "easy"] - oob_err.loc[500, "easy"] < 0.02, "with the interactions the OOB error keeps dropping by over 5 points between 100 and 500 trees; in the easy problem by less than 2", "read the two columns")
        check(disagreement.notna().all().all() and disagreement["easy"].is_monotonic_decreasing and disagreement.loc[5, "easy"] > 0.1 and disagreement.loc[500, "easy"] < 0.05, "the disagreement between forests drops: over 0.10 with 5 trees, under 0.05 with 500 (easy problem)", "mean over the pairs of the 6 forests")
        check(stable_n_error["easy"] <= 100 and stable_n_error["interactions"] >= 250, "the easy problem settles within 100 trees; the interactions need at least 250", "oob_err within 0.01 of the value at 500")
        check(needs_more_trees == "interactions" and predictions_still_unstable is True and disagreement.loc[500, "interactions"] > 0.05 and stable_n_predictions["easy"] <= 100, "the interactions need more trees and at 500 the predictions still disagree on over 5%; the easy problem has stable predictions with 100 trees or fewer", "compare the two dictionaries and the table")
    """,
    hint="""
        Loop: `f.set_params(n_estimators=n).fit(X_tr, y_tr); oob_err.loc[n, name] = 1 - f.oob_score_; test_err.loc[n, name] = 1 - f.score(X_te, y_te)`. Disagreement: for each `n`, `P = [RandomForestClassifier(n, random_state=s, n_jobs=-1).fit(X_tr, y_tr).predict(X_te) for s in range(6)]` and `np.mean([(P[i] != P[j]).mean() for i in range(6) for j in range(i + 1, 6)])`.
    """,
    solution=[
        C("""
        oob_err = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        test_err = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        for name, (X_tr, X_te, y_tr, y_te) in splits.items():
            f = RandomForestClassifier(warm_start=True, oob_score=True, random_state=0, n_jobs=-1)
            for n in ns:
                f.set_params(n_estimators=n).fit(X_tr, y_tr)                     # adds trees to the ones already trained
                oob_err.loc[n, name] = 1 - f.oob_score_
                test_err.loc[n, name] = 1 - f.score(X_te, y_te)
        pd.concat({"OOB": oob_err, "test": test_err}, axis=1).round(3)
        """),
        C("""
        disagreement = pd.DataFrame(index=ns, columns=list(problems), dtype=float)
        for name, (X_tr, X_te, y_tr, y_te) in splits.items():
            for n in ns:
                P = [RandomForestClassifier(n, random_state=s, n_jobs=-1).fit(X_tr, y_tr).predict(X_te) for s in range(6)]
                disagreement.loc[n, name] = np.mean([(P[i] != P[j]).mean() for i in range(6) for j in range(i + 1, 6)])
        disagreement.round(3)
        """),
        C("""
        stable_n_error = {name: int(next(n for n in ns if abs(oob_err.loc[n, name] - oob_err.loc[500, name]) <= 0.01)) for name in problems}
        stable_n_predictions = {name: int(next((n for n in ns if disagreement.loc[n, name] < 0.05), 10 ** 9)) for name in problems}
        needs_more_trees = max(stable_n_error, key=stable_n_error.get)
        predictions_still_unstable = bool(disagreement.loc[500, "interactions"] > 0.05)
        print("trees to settle the OOB error within 0.01:", stable_n_error)
        print("trees for a disagreement between forests under 5% (10^9 = never):", stable_n_predictions)
        print("problem that needs more trees:", needs_more_trees, f"| disagreement at 500 trees in the problem with interactions: {disagreement.loc[500, 'interactions']:.3f}", "| predictions still unstable:", predictions_still_unstable)
        """),
    ],
    learned="""
        - **There is no magic number of trees.** In the easy problem the OOB error is almost flat from 100 trees on (0.128 at 100, 0.127 at 500); in the problem with interactions it keeps dropping up to 250 and beyond (from 0.45 to 0.26): the signal is hard to extract and each tree captures a fragment of it, so many trees are needed to accumulate it.
        - **The predictions can stay unstable even when the mean error has stopped.** In the problem with interactions the OOB error is stable from 250 trees, but **at 500 trees two forests with different seeds still disagree on 10% of the cases** (in the easy problem the disagreement drops below 5% already at 50 trees). If you need reproducible predictions (for example the ranking of customers to contact), you need more trees than are needed for the mean accuracy.
        - **Reading the OOB curve is a cheap check:** with `warm_start` you get all the points in a single incremental training. If the curve is still going down at the end, increase `n_estimators` (or revisit `max_features` and the minimum leaf size: the trees might be too weak).
        - A high `n_estimators` does not cause overfitting: at worst it stops gaining. That is why you choose "high enough" and pay in time, memory and latency.
    """,
    pitfall="""
        Fixing `n_estimators=100` everywhere because "it is the default". With easy problems it is plenty, with hard interaction problems it is too little. Look at the OOB curve of **your** problem.
    """,
    interview=(
        "How do you decide how many trees to use in a forest?",
        """
        I look at the OOB error curve as n_estimators grows (with warm_start it is a single incremental training) and pick a number after which it is flat. If I need stability of the predictions (rankings, reproducible decisions) I also check the disagreement between two forests with different seeds, which
        settles later than the mean error. The number is a budget: more trees do not overfit but cost time, memory and latency.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="four_importance_methods_compared", kind="D", level=3, minutes=55, module=MODULE,
    title="Four ways to measure importance: impurity, permutation, drop-column and group",
    goal="""
        On a designed dataset (two independent informative features, **a group of three correlated features** carrying the same information, a continuous noise column and a binary one) compare four importance measures of a random forest: impurity, permutation on the test set, **drop-column** (retrain without the feature and see
        how much the accuracy drops) and group permutation. Decide which method answers which question.
    """,
    dataset="""
        `generate()` (given): 3000 examples with `x0, x1` independent and informative, `g0, g1, g2` = three noisy versions of the same latent factor `z` (informative), `noise_c` (continuous) and `noise_b` (binary), 60/40 split. Forest: 200 trees, `random_state=0`.
    """,
    purpose="""
        Understand that "importance" is not one thing: each method answers a different question and has a blind spot. Know which one to use to decide whether to **remove** a feature, to **explain** the model, to **select** variables.
    """,
    task=[
        "Build `table` (index: the 7 features; columns `impurity`, `permutation`, `drop_column`) with: impurity importance (`feature_importances_`), permutation importance on the test set (`n_repeats=10`, `random_state=0`, mean, in accuracy units) and **drop-column** (test accuracy of the full model minus that of the model retrained **without that column**, same hyperparameters).",
        "Compute `group_permutation`: mean accuracy loss (`n_repeats=10`) permuting `g0, g1, g2` together with the same row permutation, and `group_drop`: accuracy lost retraining without all three.",
        "Answer: `impurity_rewards_noise` (True if `noise_c` has an impurity importance above 0.05), `single_permutation_underestimates_group` (True if the sum of the individual permutations of `g0, g1, g2` is less than half of `group_permutation`), `drop_column_underestimates_group` (True if the sum of the drop-columns of the three is less than half of `group_drop`).",
        "Save `drop_candidates`: the list of the names of the features with **drop-column ≤ 0.005** (the ones that, looking at them one at a time, seem removable) and `selection_error`: the list of those among `g0, g1, g2` that appear in `drop_candidates` (removing them all together you lose `group_drop` of accuracy).",
    ],
    setup="""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split


        def generate(n=3000, seed=0):
            rng = np.random.default_rng(seed)
            x0, x1, z = rng.normal(size=(3, n))
            g = np.column_stack([z + rng.normal(0, 0.15, n) for _ in range(3)])           # three noisy versions of the same factor
            y = (1.2 * x0 + 0.8 * x1 + 1.5 * z + rng.normal(0, 0.8, n) > 0).astype(int)
            X = np.column_stack([x0, x1, g, rng.normal(size=n), rng.integers(0, 2, n)])
            return X, y


        names = ["x0", "x1", "g0", "g1", "g2", "noise_c", "noise_b"]
        X, y = generate()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
        rf = lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1)
        f = rf().fit(X_tr, y_tr)
        full_acc = f.score(X_te, y_te)
        print("accuracy of the full model:", round(full_acc, 3))
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=names, columns=["impurity", "permutation", "drop_column"], dtype=float)
        table
        """),
        C("""
        group_permutation = None
        group_drop = None
        """),
        C("""
        impurity_rewards_noise = None
        single_permutation_underestimates_group = None
        drop_column_underestimates_group = None
        drop_candidates = None
        selection_error = None
        """),
    ],
    checks="""
        check(list(table.index) == names and table.notna().all().all() and close(table["impurity"].sum(), 1.0), "complete table; the impurity importances sum to 1", "seven rows and three columns")
        check(table.loc["noise_c", "impurity"] > 0.05 and table.loc["noise_c", "permutation"] < 0.01 and impurity_rewards_noise is True, "the continuous noise is worth over 5% by impurity and less than 0.01 by permutation", "impurity rewards features with many distinct values")
        check(table.loc[["g0", "g1", "g2"], "drop_column"].max() < 0.01 and table.loc[["g0", "g1", "g2"], "impurity"].min() > 0.1, "each copy has a drop-column below 0.01 (the others replace it) but an impurity importance over 0.1", "the copies are interchangeable")
        check(group_permutation > 0.06 and group_drop > 0.05 and single_permutation_underestimates_group is True and drop_column_underestimates_group is True, "the group together is worth over 0.06 (permutation) and 0.05 (drop); the sum of the single importances is worth less than half of it", "permute or remove the three columns together")
        check("noise_c" in drop_candidates and "noise_b" in drop_candidates and set(selection_error) == {"g0", "g1", "g2"}, "the drop-column alone flags as removable also all three copies of the informative group: selection error", "drop-column <= 0.005")
    """,
    hint="""
        Drop-column: `A = np.delete(X_tr, j, axis=1); B = np.delete(X_te, j, axis=1); full_acc - rf().fit(A, y_tr).score(B, y_te)`. Group: `np.delete(X_tr, [2, 3, 4], axis=1)`. Group permutation: same row permutation for the three columns (see C2).
        Permutation method: `permutation_importance(f, X_te, y_te, n_repeats=10, random_state=0).importances_mean`.
    """,
    solution=[
        C("""
        result = permutation_importance(f, X_te, y_te, n_repeats=10, random_state=0)
        table = pd.DataFrame(index=names, columns=["impurity", "permutation", "drop_column"], dtype=float)
        table["impurity"] = f.feature_importances_
        table["permutation"] = result.importances_mean
        for j, name in enumerate(names):
            without = rf().fit(np.delete(X_tr, j, axis=1), y_tr).score(np.delete(X_te, j, axis=1), y_te)          # retrain without the column
            table.loc[name, "drop_column"] = full_acc - without
        table.round(3)
        """),
        C("""
        group = [2, 3, 4]
        rng = np.random.default_rng(0)
        losses = []
        for _ in range(10):
            p = rng.permutation(len(X_te))
            Xp = X_te.copy()
            Xp[:, group] = X_te[p][:, group]                                          # the three columns together, same permutation
            losses.append(full_acc - f.score(Xp, y_te))
        group_permutation = float(np.mean(losses))
        group_drop = float(full_acc - rf().fit(np.delete(X_tr, group, axis=1), y_tr).score(np.delete(X_te, group, axis=1), y_te))
        print(f"group g0-g1-g2: permutation {group_permutation:.3f} | drop-column {group_drop:.3f}")
        """),
        C("""
        impurity_rewards_noise = bool(table.loc["noise_c", "impurity"] > 0.05)
        g = ["g0", "g1", "g2"]
        single_permutation_underestimates_group = bool(table.loc[g, "permutation"].sum() < 0.5 * group_permutation)
        drop_column_underestimates_group = bool(table.loc[g, "drop_column"].sum() < 0.5 * group_drop)
        drop_candidates = list(table.index[table["drop_column"] <= 0.005])
        selection_error = [n for n in g if n in drop_candidates]
        print("removable according to the drop-column, one at a time:", drop_candidates)
        print("... of which belonging to the informative group:", selection_error)
        """),
    ],
    learned="""
        - **The impurity rewards the continuous noise** (`noise_c` is worth 5.8% of importance: many thresholds to try) while by permutation and drop-column it is zero: they are methods that measure how much the **model gets worse** on new data by removing or shuffling that column.
        - **With three copies of the same factor, each one looks useless** with the drop-column (−0.008, −0.013, 0.001: removing one leaves the other two) and is worth little with the single permutation (0.06, 0.02, 0.01, in total less than half of the 0.25 of the group), yet the group is worth a lot. The three columns together carry the factor `z`, a strong predictor of the problem: removing them all would cost about 18 points of accuracy (0.878 → 0.70).
        - **Typical selection error:** a rule "remove the features with drop-column ~0" applied to a group of correlated features removes them all (one at a time each is redundant) and the information is lost. The right rule is to remove **one at a time, retraining**, or to reason by groups.
        - Summary of the questions: **impurity** = "what does the tree use on the training set?" (fast, biased); **permutation** = "what does the model lean on to predict?" (underestimates groups); **drop-column** = "can I remove this column?" (expensive: one retraining per column; underestimates groups); **group** = "how much is this information worth?".
    """,
    pitfall="""
        Calling a number "importance" without saying which question it answers. The colleague who uses `feature_importances_` to "see what is needed" and the one who uses drop-column to "see what to remove" can reach opposite conclusions, and both are right for their own question.
    """,
    interview=(
        "How do you choose which features to remove from a model?",
        """
        Not from the importance alone: impurity is biased toward features with many values, permutation and drop-column underestimate groups of correlated features. I group the redundant variables, remove one group (or one feature) at a time retraining and checking the performance on validation data,
        and keep the smallest set that does not get worse. The final cut also depends on the cost of collecting and maintaining each feature.
        """),
)

EXERCISES = [D1, D2]
