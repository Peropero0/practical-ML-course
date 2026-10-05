"""Module 3 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_03_pipeline_and_validation"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_cv_results", kind="D", level=2, minutes=40, module=MODULE,
    title="Reading `cv_results_`: who really wins, and which parameter matters",
    goal="""
        Analyze the results table of a `GridSearchCV` on a random forest: separate the truly better combinations from the
        equivalent ones, apply the standard error rule to choose the simplest model and measure overfitting and the importance
        of each hyperparameter.
    """,
    dataset="""
        `ds.loans(with_missing=False)` (four numeric columns and the target `defaulted`). The setup cell runs a `GridSearchCV`
        on a `RandomForestClassifier(n_estimators=60)` with `max_depth` in `[2, 4, 6, 10, None]` and `min_samples_leaf` in `[1, 5, 20]`
        (15 combinations, 5 folds, `return_train_score=True`) and produces `r`, the DataFrame of `cv_results_`.
    """,
    purpose="""
        Stop looking only at `best_params_`. A results table contains much more: how clear the win is, how much of the
        difference is within the noise, where the model memorizes and which hyperparameter really moves the performance.
    """,
    task=[
        "Sort `r` by `rank_test_score`. Save `best` (the first row), `best_mean` and `best_std` (`mean_test_score` and `std_test_score`).",
        "Apply the **standard error rule**: save in `candidates_1se` (DataFrame) the rows with `mean_test_score >= best_mean − best_std`, and in "
        "`n_candidates_1se` their number. They are the combinations statistically indistinguishable from the best.",
        "Among the `candidates_1se` choose the **simplest**: minimum `max_depth` (with `None` considered the maximum); on a tie, larger `min_samples_leaf`. "
        "Save the row in `choice_1se`.",
        "Compute `gap = mean_train_score − mean_test_score` for every row and save `best_gap` (of the best one) and `max_gap` (the largest in the table).",
        "With `groupby` on the values of each hyperparameter (mean of `mean_test_score`), save `depth_range` and `leaf_range` (maximum minus minimum "
        "of the mean per value) and `most_influential_parameter` (`\"max_depth\"` or `\"min_samples_leaf\"`).",
    ],
    setup="""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import GridSearchCV, StratifiedKFold

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]]
        y = df["defaulted"]
        grid = {"max_depth": [2, 4, 6, 10, None], "min_samples_leaf": [1, 5, 20]}
        search = GridSearchCV(RandomForestClassifier(n_estimators=60, random_state=0), grid,
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc", return_train_score=True).fit(X, y)
        r = pd.DataFrame(search.cv_results_)
        r["max_depth"] = r["param_max_depth"].map(lambda v: 99 if v is None else v).astype(int)       # None = no limit
        r["min_samples_leaf"] = r["param_min_samples_leaf"].astype(int)
        r[["max_depth", "min_samples_leaf", "mean_test_score", "std_test_score", "mean_train_score", "rank_test_score"]].sort_values("rank_test_score").head(5).round(3)
    """,
    starter=[
        C("""
        r = r.sort_values("rank_test_score").reset_index(drop=True)
        best = ...
        best_mean = ...
        best_std = ...
        """),
        C("""
        candidates_1se = ...
        n_candidates_1se = ...
        choice_1se = ...
        """),
        C("""
        r["gap"] = ...
        best_gap = ...
        max_gap = ...
        """),
        C("""
        depth_range = ...
        leaf_range = ...
        most_influential_parameter = ...
        """),
    ],
    checks="""
        check(best["rank_test_score"] == 1 and close(best_mean, r["mean_test_score"].max()) and best_std > 0,
              "the best is the first row and has a positive standard deviation", "best = r.iloc[0]")
        expected = r[r["mean_test_score"] >= best_mean - best_std]
        check(n_candidates_1se == len(expected) and n_candidates_1se >= 5,
              "many combinations are statistically indistinguishable from the best (at least 5)", "candidates_1se = rows with mean_test_score >= best_mean - best_std")
        order = candidates_1se.assign(_d=candidates_1se["max_depth"]).sort_values(["_d", "min_samples_leaf"], ascending=[True, False])
        check(choice_1se["max_depth"] == order.iloc[0]["max_depth"] and choice_1se["min_samples_leaf"] == order.iloc[0]["min_samples_leaf"],
              "choice_1se: the simplest combination among the equivalent ones", "minimum max_depth (None = 99), on a tie maximum min_samples_leaf")
        check(choice_1se["max_depth"] <= best["max_depth"] and choice_1se["mean_test_score"] >= best_mean - best_std,
              "the choice is as simple as the best or more so, without losing appreciably", "the 1-SE choice cannot be more complex than the best")
        check(close(best_gap, best["mean_train_score"] - best["mean_test_score"]) and max_gap >= best_gap and max_gap > 0.2,
              "the maximum gap (overfitting) exceeds 0.2", "gap = mean_train_score - mean_test_score")
        check(most_influential_parameter in ("max_depth", "min_samples_leaf") and
              most_influential_parameter == ("max_depth" if depth_range > leaf_range else "min_samples_leaf"),
              "most_influential_parameter consistent with the ranges", "the parameter with the larger range of the means")
    """,
    hint="""
        For the choice: `candidates_1se.sort_values(["max_depth", "min_samples_leaf"], ascending=[True, False]).iloc[0]`. Range:
        `m = r.groupby("max_depth")["mean_test_score"].mean(); m.max() - m.min()`.
    """,
    solution=[
        C("""
        r = r.sort_values("rank_test_score").reset_index(drop=True)
        best = r.iloc[0]
        best_mean, best_std = best["mean_test_score"], best["std_test_score"]
        print(f"best: depth {best['max_depth']}, leaf {best['min_samples_leaf']} -> AUC {best_mean:.4f} +/- {best_std:.4f}")
        """),
        C("""
        candidates_1se = r[r["mean_test_score"] >= best_mean - best_std]
        n_candidates_1se = len(candidates_1se)
        choice_1se = candidates_1se.sort_values(["max_depth", "min_samples_leaf"], ascending=[True, False]).iloc[0]
        print(f"{n_candidates_1se} combinations out of {len(r)} are within one standard error of the best")
        print(f"1-SE choice: depth {choice_1se['max_depth']}, leaf {choice_1se['min_samples_leaf']} -> AUC {choice_1se['mean_test_score']:.4f}")
        """),
        C("""
        r["gap"] = r["mean_train_score"] - r["mean_test_score"]
        best_gap = r.iloc[0]["gap"]
        max_gap = r["gap"].max()
        print(f"train-validation gap: best {best_gap:.3f} | maximum {max_gap:.3f}")
        """),
        C("""
        m_depth = r.groupby("max_depth")["mean_test_score"].mean()
        m_leaf = r.groupby("min_samples_leaf")["mean_test_score"].mean()
        depth_range = m_depth.max() - m_depth.min()
        leaf_range = m_leaf.max() - m_leaf.min()
        most_influential_parameter = "max_depth" if depth_range > leaf_range else "min_samples_leaf"
        print("mean AUC per max_depth:", m_depth.round(3).to_dict())
        print("mean AUC per min_samples_leaf:", m_leaf.round(3).to_dict())
        print("most influential:", most_influential_parameter)
        """),
    ],
    learned="""
        - The standard deviation across folds is large compared to the differences between combinations: many (here most) are
          **statistically indistinguishable** from the best. The "winner" is little more than a random choice in the middle of a group of equals.
        - The **standard error rule** picks among the equals the simplest model (shallower, more regularized): at equal
          estimated performance it usually generalizes better, and costs less to serve.
        - The **gap** between training and validation shows where the model memorizes: the combination with no depth limit and leaves of a
          single example has a gap of 0.25 (training close to 1, validation much lower), against 0.07 for the best. A huge gap is not a bug, it is the
          signature of overfitting.
        - Aggregating by value of each hyperparameter shows which one matters: here the maximum depth moves the mean by about 0.014 of AUC and
          `min_samples_leaf` by about 0.010. They are small differences, of the same order as the standard deviation across folds: among these
          hyperparameters none changes the performance dramatically.
    """,
    pitfall="""
        Taking `best_params_` as a discovery ("the optimal depth is 6"). With the variability of the folds the "optimal" depth changes if
        you change seed or sample: the correct message is "among these combinations there is no demonstrable difference".
    """,
    interview=(
        "Your GridSearchCV says the best has AUC 0.812 with max_depth=8, but the second has 0.811 with max_depth=4. Which one do you send to production?",
        """
        I look at the standard deviation across folds: if it is 0.03 the difference of 0.001 is noise, and I apply the standard error rule choosing
        the simplest model among the indistinguishable ones: the one with max_depth=4 (less overfitting, faster, more stable). If I had
        reasons to prefer the other (for example a different test set that confirms it) I would write them down, but "first in the ranking" is not a reason.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="validation_and_learning_curves", kind="D", level=2, minutes=40, module=MODULE,
    title="Validation and learning curves: diagnosing a model",
    goal="""
        Compute a validation curve (depth of a tree) and two learning curves (logistic regression and random forest), and use them
        to say whether a model suffers from underfitting or overfitting and whether more data would be needed.
    """,
    dataset="""
        `ds.loans(with_missing=False)`: the four numeric columns and the target `defaulted`, 1500 rows. Stratified 5-fold cross-validation and
        AUC scoring. Models: `DecisionTreeClassifier` (for the validation curve), `LogisticRegression` and `RandomForestClassifier(60)`.
    """,
    purpose="""
        Learn to read the two plots that answer the two most useful questions of a project: "is the model too simple or too
        complex?" (validation curve) and "would adding data help?" (learning curve). The answers are computed, not guessed.
    """,
    task=[
        "With `validation_curve` compute the training and validation AUC of the tree for `max_depth` in `[1, 2, 3, 5, 8, 12, 20]` and save "
        "`train_v`, `val_v` (means over the folds, array of 7 values), `best_depth` (the value with the maximum validation) and `gap_v = train_v − val_v`.",
        "Assign to `diagnosis` a dictionary with three keys (`\"depth_1\"`, `\"depth_20\"`, `\"depth_best\"`) and values `\"underfitting\"`, `\"overfitting\"` or "
        "`\"adequate\"` applying the rules: **overfitting** if the gap exceeds 0.15; **underfitting** if the validation is below 0.72; otherwise **adequate**.",
        "With `learning_curve` (`train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]`) compute for the logistic and for the forest the validation means at each "
        "size: `val_lr` and `val_rf` (arrays of 5 values).",
        "Save `lr_gain` and `rf_gain`: the validation AUC with 100% of the data minus the one with 60% (how much you would still gain by adding data). "
        "Assign to `more_data_needed` a dictionary `{\"lr\": bool, \"rf\": bool}`: `True` if the gain exceeds 0.01.",
        "Save `sufficient_fraction`, a dictionary `{\"lr\": ..., \"rf\": ...}` with the **minimum** fraction of data (among `sizes`) for which the validation "
        "AUC is within 0.005 of the one obtained with all the data.",
    ],
    setup="""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, learning_curve, validation_curve
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        df = ds.loans(with_missing=False)
        X = df[["age", "income", "job_tenure", "amount"]].to_numpy()
        y = df["defaulted"].to_numpy()
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        depths = [1, 2, 3, 5, 8, 12, 20]
    """,
    starter=[
        C("""
        tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth", param_range=depths,
                                  cv=cv, scoring="roc_auc")
        train_v, val_v = ..., ...
        best_depth = ...
        gap_v = ...
        pd.DataFrame({"max_depth": depths, "train": train_v, "validation": val_v, "gap": gap_v}).round(3)
        """),
        C("""
        diagnosis = {"depth_1": ..., "depth_20": ..., "depth_best": ...}
        """),
        C("""
        sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        _, _, va_lr = learning_curve(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, train_sizes=sizes, cv=cv, scoring="roc_auc")
        _, _, va_rf = learning_curve(RandomForestClassifier(60, min_samples_leaf=5, random_state=0), X, y, train_sizes=sizes, cv=cv, scoring="roc_auc")
        val_lr, val_rf = ..., ...
        lr_gain = ...
        rf_gain = ...
        more_data_needed = {"lr": ..., "rf": ...}
        sufficient_fraction = {"lr": ..., "rf": ...}
        """),
    ],
    checks="""
        check(len(train_v) == 7 and len(val_v) == 7 and best_depth == depths[int(np.argmax(val_v))], "validation curve computed, best_depth consistent", "best_depth = depths[np.argmax(val_v)]")
        check(2 <= best_depth <= 8, "the best depth is intermediate (between 2 and 8): neither too little nor too much", "look at the validation: it goes up and then down")
        check(gap_v[-1] > 0.15 and gap_v[0] < 0.05 and close(gap_v, train_v - val_v), "the gap grows with the depth (tiny at 1, huge at 20)", "gap_v = train_v - val_v")
        check(diagnosis == {"depth_1": "underfitting", "depth_20": "overfitting", "depth_best": "adequate"}, "correct diagnosis for the three trees",
              "depth_1: low validation (underfitting); depth_20: huge gap (overfitting); the best: adequate")
        check(len(val_lr) == 5 and len(val_rf) == 5 and close(lr_gain, val_lr[-1] - val_lr[2]) and close(rf_gain, val_rf[-1] - val_rf[2]),
              "gains computed (100% minus 60% of the data)", "gain = val[-1] - val[2]")
        check(more_data_needed == {"lr": bool(lr_gain > 0.01), "rf": bool(rf_gain > 0.01)} and not more_data_needed["lr"] and not more_data_needed["rf"],
              "with this data neither model gains more than 0.01 by adding data beyond 60%: the limit is not the quantity", "compare the two gains with 0.01")
        check(sufficient_fraction["lr"] < sufficient_fraction["rf"] and val_lr[-1] > val_rf[-1],
              "the logistic needs less data and its final validation is better than the forest's", "the forest needs more data and stays below the logistic")
    """,
    hint="""
        `tr.mean(axis=1)` and `va.mean(axis=1)`. For the learning curve: `va_lr.mean(axis=1)`. `more_data_needed = {"lr": bool(lr_gain > 0.01), "rf": bool(rf_gain > 0.01)}`.
        A rule for the diagnosis: `gap = train - val`; if `gap > 0.15` overfitting, elif `val < 0.72` underfitting, otherwise adequate.
    """,
    solution=[
        C("""
        tr, va = validation_curve(DecisionTreeClassifier(random_state=0), X, y, param_name="max_depth", param_range=depths,
                                  cv=cv, scoring="roc_auc")
        train_v, val_v = tr.mean(axis=1), va.mean(axis=1)
        best_depth = depths[int(np.argmax(val_v))]
        gap_v = train_v - val_v
        display(pd.DataFrame({"max_depth": depths, "train": train_v, "validation": val_v, "gap": gap_v}).round(3))
        print("best depth:", best_depth)
        """),
        C("""
        def classify(i):
            if gap_v[i] > 0.15:
                return "overfitting"
            if val_v[i] < 0.72:
                return "underfitting"
            return "adequate"


        diagnosis = {"depth_1": classify(0), "depth_20": classify(len(depths) - 1), "depth_best": classify(depths.index(best_depth))}
        print(diagnosis)
        """),
        C("""
        sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
        _, _, va_lr = learning_curve(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X, y, train_sizes=sizes, cv=cv, scoring="roc_auc")
        _, _, va_rf = learning_curve(RandomForestClassifier(60, min_samples_leaf=5, random_state=0), X, y, train_sizes=sizes, cv=cv, scoring="roc_auc")
        val_lr, val_rf = va_lr.mean(axis=1), va_rf.mean(axis=1)
        lr_gain = val_lr[-1] - val_lr[2]
        rf_gain = val_rf[-1] - val_rf[2]
        more_data_needed = {"lr": bool(lr_gain > 0.01), "rf": bool(rf_gain > 0.01)}
        print("validation AUC as the data grows - logistic:", val_lr.round(3), "| forest:", val_rf.round(3))
        print(f"gain going from 60% to 100%: logistic {lr_gain:+.3f} | forest {rf_gain:+.3f} -> {more_data_needed}")
        sufficient_fraction = {name: next(f for f, v in zip(sizes, val) if v >= val[-1] - 0.005)
                               for name, val in [("lr", val_lr), ("rf", val_rf)]}
        print("minimum sufficient fraction of data:", sufficient_fraction)
        """),
    ],
    learned="""
        - **Validation curve**: with depth 1 the tree is too simple (training and validation both low and close: **underfitting**);
          beyond depth 5-8 the training goes toward 1 while the validation drops (**overfitting**: huge gap). The best point is in the middle.
        - **Learning curve**: the logistic is already saturated with 20% of the data (0.78 with 240 rows and with 1200): it has a limited capacity, its
          limit is the model and the features, not the quantity of data. The forest needs more data (it reaches 0.767 only from 40%) but stays
          below the logistic at every size: here more data is not the right lever, you need better features or a different model.
        - The two curves answer different questions: validation tells **which complexity**, learning tells **whether to invest in data**. They are
          the cheapest way to decide where to spend the next day of work.
    """,
    pitfall="""
        Interpreting a learning curve with a single seed and no standard deviation. A gain of 0.005 is within the fold noise: decisions
        (collecting more data, changing model) must be made on differences that exceed the variability, and with repeated folds when needed.
    """,
    interview=(
        "Your learning curve shows a large and stable gap between training and validation, even with all the data. What do you do?",
        """
        A large and stable gap is overfitting that additional data won't solve by themselves: first I reduce the complexity of the model
        (regularization, less depth, fewer features, early stopping), then I reevaluate. Only if the gap shrinks as the data grows (the
        validation curve still rises) does it make sense to collect more. If training and validation are both low and close, instead, the problem is the
        model being too simple or the features being barely informative.
        """),
)

EXERCISES = [D1, D2]
