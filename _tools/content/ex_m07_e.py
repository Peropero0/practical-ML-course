"""Module 7 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="importances_with_the_wrong_column_names", kind="E", level=2, minutes=40, module=MODULE, start="must_fail",
    title="Feature importances with the wrong column names",
    goal="""
        Find out why a report says the largest importance belongs to the column `pclass` when the model mostly uses **sex**: the colleague assigned the importances of the pipeline to the names of the original columns, but the `ColumnTransformer` has **reordered the columns** (categoricals first, then numerics).
        Fix it with `get_feature_names_out()`.
    """,
    dataset="""
        `ds.titanic()`: 9 columns (`pclass, sex, age, siblings_spouses, parents_children, fare, port, deck, title`), stratified 75/25 split. Pipeline: `ColumnTransformer` that puts the categoricals **first** (`sex, port, deck, title`, with `OrdinalEncoder`) and the numerics **after** (`pclass, age, ...`), followed by `DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)`.
    """,
    purpose="""
        Recognize a frequent bug in reports: when the pipeline reorders, adds (one-hot) or removes columns, position `i` of the importances **no longer** corresponds to column `i` of `X`. If the number of columns stays the same, no error flags the problem (and the report is published wrong).
    """,
    task=[
        "Run the colleague's cell: `top_bug` (name of the most important feature according to the report) and `imp_bug`.",
        "Save `pipeline_names` (list `get_feature_names_out()` of the preprocessing part) and `different_positions` (how many positions `i` have `X_tr.columns[i] != pipeline_names[i]` without the prefix `cat__`/`num__`).",
        "Fix it: `imp_ok` (`pd.Series` with the importances and `pipeline_names` as index, with the prefixes removed) and `top_ok`.",
        "Check with the permutation (`permutation_importance` on the pipeline, `n_repeats=10`, `random_state=0`, `scoring=\"roc_auc\"`) that `top_ok` really is the most important column: save `top_permutation` (name).",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.tree import DecisionTreeClassifier

        df = ds.titanic()
        X, y = df.drop(columns="survived"), df["survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        CAT = ["sex", "port", "deck", "title"]
        NUM = ["pclass", "age", "siblings_spouses", "parents_children", "fare"]
        prep = ColumnTransformer([("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder()), CAT),
                                  ("num", SimpleImputer(strategy="median"), NUM)])
        pipe = make_pipeline(prep, DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)).fit(X_tr, y_tr)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        imp_bug = pd.Series(pipe[-1].feature_importances_, index=X_tr.columns)          # "the importances, with the column names"
        top_bug = imp_bug.idxmax()
        print(f"most important feature according to the report: {top_bug} ({imp_bug.max():.2f})")
        imp_bug.round(3).sort_values(ascending=False).head(4)
        """),
        M("### Your diagnosis and the fix"),
        C("""
        pipeline_names = None
        different_positions = None
        """),
        C("""
        imp_ok = None
        top_ok = None
        top_permutation = None
        """),
    ],
    checks="""
        check(top_bug == "pclass", "the colleague's report says the most important feature is `pclass`", "don't modify the colleague's code")
        check(pipeline_names is not None and len(pipeline_names) == 9 and pipeline_names[0].startswith("cat__sex") and different_positions >= 6, "the pipeline has 9 columns in a different order (cat before num): at least 6 different positions", "pipe[:-1].get_feature_names_out()")
        check(imp_ok is not None and top_ok == "sex" and imp_ok.max() > 0.4 and not any(str(n).startswith(("cat__", "num__")) for n in imp_ok.index), "the most important feature is `sex` (over 0.4); the names have no prefix", "remove 'cat__' and 'num__' from the names")
        check(close(imp_ok.sum(), 1.0) and set(imp_ok.index) == set(X_tr.columns), "same columns as X, importances that sum to 1", "reindex with the correct names")
        check(top_permutation == top_ok, "the permutation confirms: the most important column is `sex`", "permutation_importance(pipe, X_te, y_te, ...) has the same names as X_te.columns")
    """,
    hint="""
        `pipeline_names = list(pipe[:-1].get_feature_names_out())`; without the prefix: `[n.split("__", 1)[1] for n in pipeline_names]`. Permutation: `pd.Series(permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=0, scoring="roc_auc").importances_mean, index=X_te.columns).idxmax()`
        (here the order is that of `X_te`, because the permutation acts on the input columns of the pipeline).
    """,
    solution=[
        C("""
        imp_bug = pd.Series(pipe[-1].feature_importances_, index=X_tr.columns)          # "the importances, with the column names"
        top_bug = imp_bug.idxmax()
        print(f"most important feature according to the report: {top_bug} ({imp_bug.max():.2f})")
        imp_bug.round(3).sort_values(ascending=False).head(4)
        """),
        M("""
        **Where the bug is.** `feature_importances_` has one entry for each column **coming out of the preprocessing**, in the order in which the `ColumnTransformer` produces them: first the 4 categoricals (`sex`, `port`, `deck`, `title`), then the 5 numerics. The colleague labels them with `X_tr.columns`, where the order is different
        (`pclass`, `sex`, `age`, ...): the very high importance of **sex** (position 0 in the output) is attributed to `pclass` (position 0 in the input). The lengths match (9 and 9), so there is no error.
        """),
        C("""
        pipeline_names = list(pipe[:-1].get_feature_names_out())
        clean = [n.split("__", 1)[1] for n in pipeline_names]
        different_positions = sum(a != b for a, b in zip(X_tr.columns, clean))
        print("input columns :", list(X_tr.columns))
        print("output columns:", clean)
        print("different positions:", different_positions, "out of 9")
        """),
        C("""
        imp_ok = pd.Series(pipe[-1].feature_importances_, index=clean)
        top_ok = imp_ok.idxmax()
        perm = permutation_importance(pipe, X_te, y_te, n_repeats=10, random_state=0, scoring="roc_auc")
        top_permutation = pd.Series(perm.importances_mean, index=X_te.columns).idxmax()
        print(f"correct importances, top three: {imp_ok.nlargest(3).round(3).to_dict()} | most important: {top_ok} | by permutation: {top_permutation}")
        """),
    ],
    learned="""
        - **The importances follow the columns coming out of the preprocessing**, not those going in. `ColumnTransformer` reorders (the blocks in the order in which you declare them), the one-hot multiplies the columns, imputers with indicators add some, selectors remove some. The only correct way to get the names is `get_feature_names_out()` of the pipeline (without the last step).
        - The bug is invisible when **the number of columns stays the same**: with the one-hot you would get a length error and fix it; with the ordinal encoding (9 columns in and out) the report comes out wrong and nobody notices. A report with `pclass` at the top is plausible ("the rich survive"), and the wrong story is easy to tell.
        - The check is always the same: **compare with a second method** (the permutation importance, which works on the input columns and has the same names as `X`) or with your knowledge of the problem (`sex` and `title` are obvious predictors).
        - Every time you stick names on a numeric array coming from a pipeline, ask who decided the order.
    """,
    pitfall="""
        Using `X.columns` as the index of `feature_importances_`, `coef_` or of the `shap_values` arrays of a model inside a pipeline. If the preprocessing changes the number or the order of the columns, the label is wrong without any warning.
    """,
    interview=(
        "How do you assign names to the importances of a model inside a pipeline?",
        """
        With `get_feature_names_out()` of the preprocessing part (`pipeline[:-1]`), because the final model sees columns that are reordered, expanded or removed. Then I check with a second method, like the permutation on the input columns, that the conclusions agree, and I add an automatic test that compares the length and the names.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="the_customer_id_that_predicts_the_outcome", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="A tree with AUC 0.99 on loans: it was the identifier",
    goal="""
        Find out why a tree on the loans reaches AUC 0.99 in cross-validation: the dataset includes `customer_id`, assigned in order of **file closing** (defaulted loans have higher ids), so a threshold on the id separates almost perfectly. Measure the AUC on **new customers** (ids even higher) and
        remove the column.
    """,
    dataset="""
        `dataset()` (given): 1500 training customers from `ds.loans(with_missing=False)` with `ratio` and a column `customer_id` assigned in order of outcome (first the regular ones, then the defaulters); `new` is a group of 300 customers with **consecutive ids above the maximum** of the training set (as would happen in production). Tree: `DecisionTreeClassifier(min_samples_leaf=30, random_state=0)`.
    """,
    purpose="""
        Recognize identifier leakage: any column that encodes **when or how** the data was collected (progressive ids, timestamps, file names, insertion order) can correlate with the target in a non-reproducible way. A tree, which looks for thresholds on any column, finds it right away.
    """,
    task=[
        "Run the colleague's cell: `cv_auc_bug` (cross-validation AUC with `customer_id`) and `id_importance` (share of `feature_importances_` of the tree trained on all the data).",
        "Compute `id_outcome_corr` (Spearman correlation between `customer_id` and `defaulted`) and `new_auc_bug`: AUC on `X_new`, `y_new` with the model trained with the id on the full training set.",
        "Fix it: remove `customer_id` and save `cv_auc_ok` and `new_auc_ok`.",
        "Save `n_distinct_new_values` (number of distinct values of `predict_proba` of the model with the id on the new customers): what does the model see?",
    ],
    setup="""
        from scipy.stats import spearmanr
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.tree import DecisionTreeClassifier


        def dataset():
            df = ds.loans(n=1800, with_missing=False)
            df["ratio"] = df["amount"] / df["income"]
            df = df.sample(frac=1, random_state=0).reset_index(drop=True)
            train, new = df.iloc[:1500].copy(), df.iloc[1500:].copy()
            train = train.sort_values("defaulted", kind="stable").reset_index(drop=True)      # ids in order of outcome: first the regular ones, then the defaulters
            train["customer_id"] = np.arange(1500)
            new["customer_id"] = np.arange(1500, 1800)                                         # the new customers have ids above the maximum of the training set
            return train, new


        train, new = dataset()
        FEATURES = ["ratio", "age", "job_tenure", "income", "amount"]
        X_tr, y_tr = train[FEATURES + ["customer_id"]], train["defaulted"]
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        print("defaulters in the training set:", round(y_tr.mean(), 3), "| ids of the new customers:", int(new["customer_id"].min()), "-", int(new["customer_id"].max()))
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        tree = DecisionTreeClassifier(min_samples_leaf=30, random_state=0)
        cv_auc_bug = cross_val_score(tree, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        id_importance = pd.Series(tree.fit(X_tr, y_tr).feature_importances_, index=X_tr.columns)["customer_id"]
        print(f"cross-validation AUC: {cv_auc_bug:.3f} | importance of customer_id: {id_importance:.2f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        id_outcome_corr = None
        new_auc_bug = None
        n_distinct_new_values = None
        """),
        C("""
        cv_auc_ok = None
        new_auc_ok = None
        """),
    ],
    checks="""
        check(cv_auc_bug > 0.95 and id_importance > 0.8, "the colleague: AUC over 0.95 and customer_id is worth over 80% of the importance", "don't modify the colleague's code")
        check(id_outcome_corr is not None and id_outcome_corr > 0.6, "the id is strongly correlated with the outcome (Spearman over 0.6)", "spearmanr(train['customer_id'], train['defaulted'])")
        check(new_auc_bug is not None and 0.45 < new_auc_bug < 0.55 and n_distinct_new_values <= 2, "on the new customers the AUC is 0.5: the model with the id gives everyone the same score (at most two distinct values)", "all the new ids are beyond the last threshold")
        check(cv_auc_ok is not None and 0.65 < cv_auc_ok < 0.85 and new_auc_ok > 0.62, "without the id: cross-validation AUC 0.65-0.85, on the new customers over 0.62", "FEATURES only (without customer_id)")
    """,
    hint="""
        New customers: `X_new = new[FEATURES + ["customer_id"]]`, `y_new = new["defaulted"]`; `roc_auc_score(y_new, model.predict_proba(X_new)[:, 1])` with `model = DecisionTreeClassifier(...).fit(X_tr, y_tr)`. Without the id: `X_tr[FEATURES]` and `X_new[FEATURES]`.
    """,
    solution=[
        C("""
        tree = DecisionTreeClassifier(min_samples_leaf=30, random_state=0)
        cv_auc_bug = cross_val_score(tree, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        id_importance = pd.Series(tree.fit(X_tr, y_tr).feature_importances_, index=X_tr.columns)["customer_id"]
        print(f"cross-validation AUC: {cv_auc_bug:.3f} | importance of customer_id: {id_importance:.2f}")
        """),
        M("""
        **Where the bug is.** `customer_id` was assigned **after** the outcome (in order of closing): the regular customers have low ids and the defaulters high ids. A single threshold on the id separates the classes almost perfectly, and a tree finds it at the root. In cross-validation it works (the validation ids are in the same range as the training ones),
        but for a new customer the id is simply **higher than all the others**: it falls in the last leaf, the one of the defaulters (or of the regular ones), and all the new customers get the same score.
        """),
        C("""
        X_new, y_new = new[FEATURES + ["customer_id"]], new["defaulted"]
        id_outcome_corr = float(spearmanr(train["customer_id"], train["defaulted"])[0])
        with_id = DecisionTreeClassifier(min_samples_leaf=30, random_state=0).fit(X_tr, y_tr)
        p_new = with_id.predict_proba(X_new)[:, 1]
        new_auc_bug = roc_auc_score(y_new, p_new)
        n_distinct_new_values = int(np.unique(p_new).size)
        print(f"id-outcome correlation: {id_outcome_corr:.2f} | AUC on the new customers: {new_auc_bug:.3f} | distinct values of predict_proba on the new ones: {n_distinct_new_values}")
        """),
        C("""
        without_id = DecisionTreeClassifier(min_samples_leaf=30, random_state=0)
        cv_auc_ok = cross_val_score(without_id, X_tr[FEATURES], y_tr, cv=cv, scoring="roc_auc").mean()
        new_auc_ok = roc_auc_score(y_new, without_id.fit(X_tr[FEATURES], y_tr).predict_proba(X_new[FEATURES])[:, 1])
        print(f"without customer_id: cross-validation AUC {cv_auc_ok:.3f} | on the new customers {new_auc_ok:.3f}")
        """),
    ],
    learned="""
        - **An AUC of 0.99 on a credit problem is a warning sign**, not a success: the realistic value is about 0.73-0.78 (M4). When a model is "too good", look first at the **importances**: `customer_id` was worth 100%.
        - **An identifier that encodes the order of collection is not a feature.** Here the id is assigned after the outcome; in other cases it is a timestamp, a batch number, the file name. Cross-validation does not see it (the validation folds are in the same id range as the training set) and in production the id is out of range.
        - **The new customers all receive the same score** (the AUC is 0.5, `predict_proba` has at most two values): the tree has thresholds on the id only inside the range seen, and an id higher than all the others falls in the last leaf. A useless model with a validation that said 0.99.
        - Without the id the tree has AUC 0.73 (in cross-validation and on the new customers), consistent with what other models get. Removing suspicious columns (ids, dates, codes) is a standard step of a serious pipeline: list them explicitly in a `ColumnTransformer` with `remainder="drop"`.
    """,
    pitfall="""
        Leaving the "technical" columns (ids, row indices, insertion timestamps) in the dataset because "the model will work out by itself whether they are needed". A tree uses them if they correlate with the target in the sample, even for a reason that does not hold in production.
    """,
    interview=(
        "Your model has a very high AUC in validation. What do you check before trusting it?",
        """
        The feature importances (a dominant column, especially if it is an id or a timestamp), the structure of the validation (same period/same customers in training and test), consistency with a simple baseline, and I run a test with new data obtained the way it would be in production. I remove identifiers and columns that
        encode the collection process, and repeat the evaluation with a temporal or group split.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="ccp_alpha_depends_on_the_target_scale", kind="E", level=3, minutes=40, module=MODULE, start="must_fail",
    title="The same pruning prunes nothing: `ccp_alpha` depends on the units of the target",
    goal="""
        Find out why the same `ccp_alpha=5e-4` that gave 182 leaves on prices in hundreds of thousands of dollars leaves **2453 leaves** when the prices are in dollars: cost-complexity pruning uses the impurity (the squared error) in units of the target, so `alpha` has to be rescaled with the variance of `y`.
    """,
    dataset="""
        `ds.california()`: `y` in hundreds of thousands of dollars (`y_tr`, `y_te`) and the same in dollars (`y_tr_dollars = y_tr * 100000`, `y_te_dollars`). Tree: `DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=...)`, 75/25 split.
    """,
    purpose="""
        Recognize a parameter that depends on the **units** of the problem: `ccp_alpha` (regression) and `min_impurity_decrease` are expressed in impurity units, that is in the square of the target. A value found "on scale" is not transferable: change the units and the regularization disappears (or becomes excessive).
    """,
    task=[
        "Run the colleague's cell: `ref_leaves` (pruning with `ccp_alpha=5e-4` on `y` in hundreds of thousands) and `bug_leaves` (same `ccp_alpha` on `y` in dollars), with the respective `ref_rmse` and `bug_rmse` (in **hundreds of thousands of dollars**, to compare them).",
        "Save `scale_factor = 100000 ** 2` and `alpha_dollars = 5e-4 * scale_factor`: the equivalent `alpha` for the target in dollars.",
        "Save `ok_leaves` and `ok_rmse` training the tree on the target in dollars with `alpha_dollars` (RMSE reported in hundreds of thousands). They must match `ref_leaves` and `ref_rmse`.",
        "Write `relative_alpha(alpha, y)` that returns `alpha / np.var(y)`: a **dimensionless** parameter. Check that `relative_alpha(5e-4, y_tr)` and `relative_alpha(alpha_dollars, y_tr_dollars)` match and save the value in `rel_alpha`.",
    ],
    setup="""
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        y_tr_dollars, y_te_dollars = y_tr * 100000, y_te * 100000


        def tree(alpha, y_train):
            return DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=alpha).fit(X_tr, y_train)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        t_ref = tree(5e-4, y_tr)                                         # target in hundreds of thousands of dollars
        t_bug = tree(5e-4, y_tr_dollars)                                  # same alpha, target in dollars
        ref_leaves, bug_leaves = t_ref.get_n_leaves(), t_bug.get_n_leaves()
        ref_rmse = mean_squared_error(y_te, t_ref.predict(X_te), squared=False)
        bug_rmse = mean_squared_error(y_te_dollars, t_bug.predict(X_te), squared=False) / 100000
        print(f"leaves: {ref_leaves} (target in hundreds of thousands) against {bug_leaves} (in dollars) | RMSE: {ref_rmse:.3f} against {bug_rmse:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        scale_factor = None
        alpha_dollars = None
        """),
        C("""
        ok_leaves = None
        ok_rmse = None
        """),
        C("""
        def relative_alpha(alpha, y):
            ...


        rel_alpha = None
        """),
    ],
    checks="""
        check(ref_leaves < 250 and bug_leaves > 2000 and bug_rmse > ref_rmse + 0.005, "the colleague: 182 leaves against over 2000 (no pruning), with a worse RMSE", "don't modify the colleague's code")
        check(scale_factor == 1e10 and close(alpha_dollars, 5e6), "the scale factor is 1e10 (the square of 100000); equivalent alpha 5e6", "impurity = squared error: it scales like the square of the target")
        check(ok_leaves == ref_leaves and abs(ok_rmse - ref_rmse) < 1e-6, "with the rescaled alpha the number of leaves and the error match those on the original scale", "tree(alpha_dollars, y_tr_dollars)")
        check(close(relative_alpha(5e-4, y_tr), rel_alpha) and close(relative_alpha(alpha_dollars, y_tr_dollars), rel_alpha), "relative alpha (alpha / variance of the target) equal in the two units", "alpha / np.var(y)")
    """,
    hint="""
        `alpha_dollars = 5e-4 * 100000 ** 2`; `tree(alpha_dollars, y_tr_dollars)`; RMSE: `mean_squared_error(y_te_dollars, m.predict(X_te), squared=False) / 100000`. `relative_alpha`: `return alpha / np.var(y)`.
    """,
    solution=[
        C("""
        t_ref = tree(5e-4, y_tr)                                         # target in hundreds of thousands of dollars
        t_bug = tree(5e-4, y_tr_dollars)                                  # same alpha, target in dollars
        ref_leaves, bug_leaves = t_ref.get_n_leaves(), t_bug.get_n_leaves()
        ref_rmse = mean_squared_error(y_te, t_ref.predict(X_te), squared=False)
        bug_rmse = mean_squared_error(y_te_dollars, t_bug.predict(X_te), squared=False) / 100000
        print(f"leaves: {ref_leaves} (target in hundreds of thousands) against {bug_leaves} (in dollars) | RMSE: {ref_rmse:.3f} against {bug_rmse:.3f}")
        """),
        M("""
        **Where the bug is.** Cost-complexity pruning removes a branch if the impurity reduction it brings is smaller than `alpha` per leaf. The impurity of a regression tree is the squared error **in units of the target squared**: with `y` in dollars it is 10¹⁰ times larger. The same `alpha = 5e-4` is therefore worth
        10⁻¹⁴ of the variance of the target: no branch is cut and the tree stays the complete one.
        """),
        C("""
        scale_factor = 100000 ** 2
        alpha_dollars = 5e-4 * scale_factor
        t_ok = tree(alpha_dollars, y_tr_dollars)
        ok_leaves = t_ok.get_n_leaves()
        ok_rmse = mean_squared_error(y_te_dollars, t_ok.predict(X_te), squared=False) / 100000
        print(f"equivalent alpha for the target in dollars: {alpha_dollars:.1e} -> {ok_leaves} leaves, RMSE {ok_rmse:.3f}")
        """),
        C("""
        def relative_alpha(alpha, y):
            return alpha / np.var(y)


        rel_alpha = relative_alpha(5e-4, y_tr)
        print(f"relative alpha (fraction of the variance of the target): {rel_alpha:.2e} = {relative_alpha(alpha_dollars, y_tr_dollars):.2e} in dollars")
        """),
    ],
    learned="""
        - **`ccp_alpha` (regression) and `min_impurity_decrease` are in units of the target squared.** Changing units (dollars, thousands, logarithm) changes the pruning drastically and silently: with the same number, 182 leaves or 2453. The code raises no error and the RMSE gets worse only a little (0.653 against 0.640), so nobody notices.
        - **It also holds if you transform the target:** moving to `log(y)` (for example for a price model) makes any `alpha` found before useless. If you need a transferable parameter, express it as a **fraction of the variance** (`alpha / var(y)`) and convert it back when you change scale.
        - For classification the impurity (Gini, entropy) is already dimensionless (between 0 and 1), so `ccp_alpha` is comparable between problems; it is a case where the intuition "it holds for everyone" is right, but not for regression.
        - Rule: **never copy a hyperparameter** from another problem or another unit without asking yourself in which units it is. If a hyperparameter has no effect (or a huge one), check something measurable (here: the number of leaves) and not only the final metric.
    """,
    pitfall="""
        Copying a `ccp_alpha` from a colleague's notebook, from a tutorial or from a previous project. The number has no meaning outside the units of the target on which it was chosen.
    """,
    interview=(
        "Why can a hyperparameter found on one dataset fail on another, even a similar one?",
        """
        Because many depend on the scale of the data or on the sample size: the ccp_alpha of a regression tree is in units of the target squared, the regularization of a logistic regression depends on the scale of the features and on n, the learning rate on the number of trees. I always search the hyperparameters on the current problem,
        and when I transfer a value I re-express it in dimensionless form (for example as a fraction of the variance of the target).
        """),
)

EXERCISES = [E1, E2, E3]
