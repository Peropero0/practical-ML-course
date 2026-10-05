"""Module 7 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="titanic_tree_tuning_and_rules", kind="B", level=2, minutes=50, module=MODULE,
    title="A tree on the Titanic: overfitting, tuning and readable rules",
    goal="""
        Build a pipeline with a classification tree on the Titanic data (imputation, encoding of the categoricals with one-hot or ordinal), measure how much a tree **without limits** overfits, choose `max_depth`, `min_samples_leaf` and `ccp_alpha` with `GridSearchCV`
        and read the rules of the final tree.
    """,
    dataset="""
        `ds.titanic()` (1309 passengers: `pclass, sex, age, siblings_spouses, parents_children, fare, port, deck, title`; target `survived`), stratified 75/25 split. `age` and `deck` have missing values. Validation: `StratifiedKFold(5, shuffle=True, random_state=0)`, AUC.
    """,
    purpose="""
        See in a real case that an unregularized tree is a bad model (it memorizes), that a few parameters turn it into a good readable model, and that the choice of the encoding of the categoricals matters little. Also: a tree is the only model of the course whose rules you can print.
    """,
    task=[
        "Write `build(encoding, **tree_params)`: `Pipeline` of a `ColumnTransformer` (numeric `pclass, age, siblings_spouses, parents_children, fare` with `SimpleImputer(strategy=\"median\")`; categorical `sex, port, deck, title` with `SimpleImputer(strategy=\"most_frequent\")` "
        "and `OneHotEncoder(handle_unknown=\"ignore\")` if `encoding == \"ohe\"`, or `OrdinalEncoder(handle_unknown=\"use_encoded_value\", unknown_value=-1)` if `\"ord\"`; `sparse_threshold=0`) and `DecisionTreeClassifier(random_state=0, **tree_params)`.",
        "Compute `cv_auc` (DataFrame, index: `no limits`, `max_depth=3`, `max_depth=5`, `min_samples_leaf=10`, `min_samples_leaf=20`, `ccp_alpha=0.005`; columns `ohe` and `ord`) with the mean cross-validation AUC.",
        "`GridSearchCV` (same `cv`, `scoring=\"roc_auc\"`) on the `ohe` pipeline with `max_depth` in `[2, 3, 4, 5, 6, 8, None]`, `min_samples_leaf` in `[1, 5, 10, 20, 40]` and `ccp_alpha` in `[0, 0.002, 0.005, 0.01]`: save `best` (dictionary of the parameters without the prefix), `test_auc` (test AUC) and `n_leaves`.",
        "Read the rules: save `root` (name of the feature at the root, from `get_feature_names_out()` of the preprocessing part) and `rules` (the `export_text` string with the feature names) of the final tree; print the first lines.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
        from sklearn.tree import DecisionTreeClassifier, export_text

        df = ds.titanic()
        X, y = df.drop(columns="survived"), df["survived"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        NUM = ["pclass", "age", "siblings_spouses", "parents_children", "fare"]
        CAT = ["sex", "port", "deck", "title"]
    """,
    starter=[
        C("""
        def build(encoding, **tree_params):
            ...
        """),
        C("""
        cv_auc = pd.DataFrame(index=["no limits", "max_depth=3", "max_depth=5", "min_samples_leaf=10", "min_samples_leaf=20", "ccp_alpha=0.005"], columns=["ohe", "ord"], dtype=float)
        cv_auc
        """),
        C("""
        best = None
        test_auc = None
        n_leaves = None
        """),
        C("""
        root = None
        rules = None
        """),
    ],
    checks="""
        check(cv_auc.notna().all().all() and cv_auc.loc["no limits"].max() < 0.76, "the tree without limits has AUC below 0.76 (overfitting)", "DecisionTreeClassifier(random_state=0) with no parameters")
        check(cv_auc.loc["max_depth=3"].min() > 0.83 and cv_auc.drop("no limits").min().min() > 0.8, "with any regularization the AUC goes above 0.8 (over 0.83 with max_depth=3)", "compare the rows")
        check(abs(cv_auc.loc["max_depth=3", "ohe"] - cv_auc.loc["max_depth=3", "ord"]) < 0.02, "the encoding (ohe or ordinal) changes little: less than 0.02 with max_depth=3", "compare the ohe and ord columns")
        check(set(best) == {"max_depth", "min_samples_leaf", "ccp_alpha"} and best["max_depth"] in (3, 4, 5) and test_auc > 0.83, "the search picks a depth between 3 and 5; test AUC above 0.83", "keys without the prefix 'decisiontreeclassifier__'")
        check(3 <= n_leaves <= 40, "the final tree is small: from 3 to 40 leaves", "get_n_leaves() of the last step")
        check(isinstance(root, str) and ("sex" in root or "title" in root) and "class:" in rules and rules.count("\\n") >= 6, "the root is `sex` or `title`; the rules are a text with the classes", "export_text(tree, feature_names=names)")
    """,
    hint="""
        Names: `names = list(model[:-1].get_feature_names_out())`; `export_text(model[-1], feature_names=names)`; root: `names[model[-1].tree_.feature[0]]`. Parameters after the search: `{k.split("__")[1]: v for k, v in search.best_params_.items()}`.
    """,
    solution=[
        C("""
        def build(encoding, **tree_params):
            enc = OneHotEncoder(handle_unknown="ignore") if encoding == "ohe" else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), NUM),
                                      ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), enc), CAT)], sparse_threshold=0)
            return make_pipeline(prep, DecisionTreeClassifier(random_state=0, **tree_params))
        """),
        C("""
        configs = {"no limits": {}, "max_depth=3": {"max_depth": 3}, "max_depth=5": {"max_depth": 5},
                   "min_samples_leaf=10": {"min_samples_leaf": 10}, "min_samples_leaf=20": {"min_samples_leaf": 20}, "ccp_alpha=0.005": {"ccp_alpha": 0.005}}
        cv_auc = pd.DataFrame(index=list(configs), columns=["ohe", "ord"], dtype=float)
        for name, params in configs.items():
            for encoding in ["ohe", "ord"]:
                cv_auc.loc[name, encoding] = cross_val_score(build(encoding, **params), X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        cv_auc.round(3)
        """),
        C("""
        grid = {"decisiontreeclassifier__max_depth": [2, 3, 4, 5, 6, 8, None], "decisiontreeclassifier__min_samples_leaf": [1, 5, 10, 20, 40],
                "decisiontreeclassifier__ccp_alpha": [0, 0.002, 0.005, 0.01]}
        search = GridSearchCV(build("ohe"), grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        best = {k.split("__")[1]: v for k, v in search.best_params_.items()}
        test_auc = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])
        n_leaves = search.best_estimator_[-1].get_n_leaves()
        print(f"best parameters: {best} | CV AUC {search.best_score_:.4f} | test AUC {test_auc:.4f} | leaves {n_leaves}")
        """),
        C("""
        model = search.best_estimator_
        names = list(model[:-1].get_feature_names_out())
        root = names[model[-1].tree_.feature[0]]
        rules = export_text(model[-1], feature_names=names)
        print("feature at the root:", root)
        print("\\n".join(rules.splitlines()[:14]))
        """),
    ],
    learned="""
        - **A tree without limits memorizes:** cross-validation AUC 0.72 against 0.85 with `max_depth=3`. It is not a flaw of the algorithm, it is its nature: without a stopping criterion it builds a leaf for each group of examples. The regularization parameters (`max_depth`, `min_samples_leaf`, `ccp_alpha`)
          are not optional.
        - **The best tree is small:** the search picks depth 3, with no other constraints, with test AUC 0.86 and **8 leaves**. With very few parameters a tree gets performance close to much more complex models, and it is readable.
        - **The encoding of the categoricals (one-hot or ordinal) matters little** with `max_depth=3`: the tree chooses thresholds and the one-hot makes them explicit. With many levels the one-hot fragments the data and the ordinal encoding (with a sensible order) is more efficient.
        - **The rules are the product**: the root is the most informative question (`sex`/`title`: "women and children first"), and in three or four levels you read the rules that explain almost the whole prediction. It is the reason a single tree still makes sense when you need to explain.
    """,
    pitfall="""
        Reporting the rules of a tree as "the true structure of the phenomenon". With slightly different data the root and the lower levels would change (C2). The rules should be read as a **description** of this model, not as a law.
    """,
    interview=(
        "A decision tree is interpretable: how far do you trust its rules?",
        """
        The rules describe precisely how that model decides, but a single tree is unstable: small changes in the data can change the root and the thresholds. That is why I regularize it (depth or minimum leaves chosen in cross-validation), check the stability with a bootstrap and use the rules as
        an explanation of the model, not as a discovery about the world. If accuracy is needed I use a forest or a boosting and explain with permutation importances and SHAP.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="regression_tree_on_california", kind="B", level=2, minutes=50, module=MODULE,
    title="Regression trees on house prices",
    goal="""
        Compare a linear regression with a regression tree on the California prices, look at how the validation RMSE changes as `max_depth` and `min_samples_leaf` vary, choose both with a search, and check how much the geographic position is worth (a tree on latitude and
        longitude alone).
    """,
    dataset="""
        `ds.california()` (20,640 districts, 8 features; target: median price in hundreds of thousands of dollars), 75/25 split with `random_state=0`; RMSE with `squared=False`. Validation: `KFold(5, shuffle=True, random_state=0)` on the training set.
    """,
    purpose="""
        See that an **unregularized** tree does not beat linear regression, that with the right minimum leaf size it beats it by a lot, and compare with the geographic KNN of module 5 (0.53): the proximity structure mattered a lot here too.
    """,
    task=[
        "Save `rmse` (dictionary) on the test set of `linear` (`LinearRegression`) and `base tree` (`DecisionTreeRegressor(random_state=0)` with no parameters).",
        "With `KFold(5, shuffle=True, random_state=0)` on the training set compute `depth_curve` (`pd.Series`, mean RMSE for `max_depth` in `[3, 5, 8, 12, 50]`: 50 is equivalent to no limit) and `leaf_curve` (mean RMSE for `min_samples_leaf` in `[1, 5, 20, 50, 100]`, no depth limit). Save `best_depth` and `best_leaf`.",
        "`GridSearchCV` (same `cv`, `scoring=\"neg_root_mean_squared_error\"`) on `max_depth` in `[6, 8, 10, 12, None]` and `min_samples_leaf` in `[5, 10, 20, 40]`: save `params` (dictionary), `tree_rmse` (test RMSE) and `n_leaves`.",
        "Train a tree with only the pair `Latitude`, `Longitude` and `min_samples_leaf=20`: save `geo_rmse`. Answer with `geo_is_enough = True` if `geo_rmse` is within 0.05 of `tree_rmse`.",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        cv = KFold(5, shuffle=True, random_state=0)
        error = lambda model, columns=None: mean_squared_error(y_te, model.fit(X_tr if columns is None else X_tr[columns], y_tr).predict(X_te if columns is None else X_te[columns]), squared=False)
    """,
    starter=[
        C("""
        rmse = {}
        """),
        C("""
        depth_curve = pd.Series(dtype=float)
        leaf_curve = pd.Series(dtype=float)
        best_depth = None
        best_leaf = None
        """),
        C("""
        params = None
        tree_rmse = None
        n_leaves = None
        """),
        C("""
        geo_rmse = None
        geo_is_enough = None
        """),
    ],
    checks="""
        check(set(rmse) == {"linear", "base tree"} and abs(rmse["base tree"] - rmse["linear"]) < 0.03, "the unregularized tree (0.74) is on par with linear regression (0.735)", "error(model) with squared=False")
        check(list(depth_curve.index) == [3, 5, 8, 12, 50] and best_depth == 8 and depth_curve[8] < depth_curve[50] - 0.05, "the best depth is 8; with no limit (50) the RMSE is over 0.05 higher", "idxmin of the curve")
        check(list(leaf_curve.index) == [1, 5, 20, 50, 100] and best_leaf in (20, 50) and leaf_curve[1] > leaf_curve[best_leaf] + 0.1, "best min_samples_leaf 20 or 50; with 1 the RMSE is over 0.1 higher", "idxmin of the curve")
        check(params["min_samples_leaf"] in (10, 20, 40) and tree_rmse < 0.65 and tree_rmse < rmse["linear"] - 0.08, "the optimized tree beats the linear one by over 0.08 (RMSE below 0.65)", "GridSearchCV on max_depth and min_samples_leaf")
        check(n_leaves >= 200, "the best tree has hundreds of leaves: unlike the Titanic many rules are needed here", "get_n_leaves() of the final model")
        check(geo_rmse < 0.66 and geo_is_enough is True, "with the geographic position alone the RMSE is within 0.05 of the full tree", "DecisionTreeRegressor on the two coordinates only")
    """,
    hint="""
        `cross_val_score(DecisionTreeRegressor(max_depth=d, random_state=0), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")`, then `-mean`. Use a dictionary: `pd.Series({d: cv_rmse(max_depth=d) for d in [3, 5, 8, 12, 50]})`.
    """,
    solution=[
        C("""
        rmse = {"linear": error(LinearRegression()), "base tree": error(DecisionTreeRegressor(random_state=0))}
        print({k: round(v, 3) for k, v in rmse.items()})
        """),
        C("""
        def cv_rmse(**kw):
            return -cross_val_score(DecisionTreeRegressor(random_state=0, **kw), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()


        depth_curve = pd.Series({d: cv_rmse(max_depth=d) for d in [3, 5, 8, 12, 50]})
        leaf_curve = pd.Series({l: cv_rmse(min_samples_leaf=l) for l in [1, 5, 20, 50, 100]})
        best_depth, best_leaf = depth_curve.idxmin(), leaf_curve.idxmin()
        print("cross-validation RMSE by max_depth:", depth_curve.round(3).to_dict())
        print("cross-validation RMSE by min_samples_leaf:", leaf_curve.round(3).to_dict())
        """),
        C("""
        search = GridSearchCV(DecisionTreeRegressor(random_state=0), {"max_depth": [6, 8, 10, 12, None], "min_samples_leaf": [5, 10, 20, 40]}, cv=cv,
                              scoring="neg_root_mean_squared_error").fit(X_tr, y_tr)
        params = search.best_params_
        tree_rmse = mean_squared_error(y_te, search.predict(X_te), squared=False)
        n_leaves = search.best_estimator_.get_n_leaves()
        print(f"best parameters: {params} | test RMSE {tree_rmse:.3f} (linear {rmse['linear']:.3f}) | leaves {n_leaves}")
        """),
        C("""
        geo_rmse = error(DecisionTreeRegressor(min_samples_leaf=20, random_state=0), ["Latitude", "Longitude"])
        geo_is_enough = bool(geo_rmse <= tree_rmse + 0.05)
        print(f"tree on the coordinates only: RMSE {geo_rmse:.3f} | within 0.05 of the full tree: {geo_is_enough}")
        """),
    ],
    learned="""
        - **An unregularized tree is worth as much as a straight line** (RMSE 0.74 against 0.735): it memorizes the 15,000 examples and generalizes like a linear model. With `min_samples_leaf=20` it drops to 0.62 (−16%): the minimum leaf size is the most effective regularization, because it ties the stability of the leaf means to the amount of data.
        - **Hundreds of leaves** are needed (605): the phenomenon is complex (interactions between place, income and age) and a tree with 20 leaves would not be enough. On the Titanic about twenty rules were enough; here the readable rules are too many.
        - **The geographic position alone is enough:** with latitude and longitude the RMSE is 0.61, as with all the features (0.62): the price depends mostly on where you are. Remember the KNN of module 5? With 0.53 it is even better: the step predictions of a tree are less suited to a smooth surface in space.
        - The choices of `max_depth` and `min_samples_leaf` partly replace each other: with large leaves the depth is limited in practice. Search the two together, or fix one and search the other.
    """,
    pitfall="""
        Choosing `max_depth` without looking at `min_samples_leaf`: with a large `min_samples_leaf` the depth is not needed, and with limited depth the leaves stay large even without the rule. The two parameters overlap and have to be evaluated together.
    """,
    interview=(
        "How do you regularize a regression tree?",
        """
        With the minimum leaf size (`min_samples_leaf`), which is the most direct control of the variance, with the maximum depth and with cost-complexity pruning (`ccp_alpha`), choosing in cross-validation. An unregularized tree memorizes the training set: here it was worth as much as a straight line,
        with the right minimum leaf size it beat the line by over 15%.
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="cost_complexity_pruning_in_cv", kind="B", level=3, minutes=55, module=MODULE,
    title="Cost-complexity pruning chosen in cross-validation",
    goal="""
        Use `cost_complexity_pruning_path` and `ccp_alpha` to prune a regression tree on the house prices, choose `alpha` in cross-validation (with the pruning **inside** each fold) applying both the minimum rule and the "one standard deviation" rule (the smallest tree within
        one standard error of the best), and compare the number of leaves and the error.
    """,
    dataset="""
        `ds.california()`, 75/25 split with `random_state=0`; base tree `DecisionTreeRegressor(min_samples_leaf=5, random_state=0)`; validation `KFold(5, shuffle=True, random_state=0)`.
    """,
    purpose="""
        Understand how `ccp_alpha` turns the complexity/error trade-off into a single continuous parameter, and learn the parsimony rule (1-SE) to choose the simplest model that cannot be told apart from the best.
    """,
    task=[
        "Train the base tree on the training set and save `n_alphas` (number of values in `cost_complexity_pruning_path(...).ccp_alphas`) and `initial_leaves` (`get_n_leaves()` of the unpruned tree).",
        "For `alphas = np.geomspace(1e-5, 1e-2, 30)` compute in cross-validation (pruning inside each fold: pass `ccp_alpha` to the estimator) `cv_rmse` (mean RMSE, `pd.Series` indexed by `alphas`) and `cv_se` (standard error = standard deviation of the fold RMSEs / √5), and `leaves` (number of leaves of the tree pruned with that `alpha` trained on the whole training set).",
        "Save `alpha_min` (the `alpha` with minimum RMSE) and `alpha_1se`: the **largest** `alpha` with `cv_rmse <= cv_rmse.min() + cv_se[alpha_min]`.",
        "On the test set save `rmse_min` and `rmse_1se` (trees trained with those `alpha` on the full training set) and `leaves_min`, `leaves_1se`.",
    ],
    setup="""
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import KFold, cross_val_score, train_test_split
        from sklearn.tree import DecisionTreeRegressor

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        cv = KFold(5, shuffle=True, random_state=0)
        base = DecisionTreeRegressor(min_samples_leaf=5, random_state=0)
        alphas = np.geomspace(1e-5, 1e-2, 30)
    """,
    starter=[
        C("""
        n_alphas = None
        initial_leaves = None
        """),
        C("""
        cv_rmse = pd.Series(dtype=float)
        cv_se = pd.Series(dtype=float)
        leaves = pd.Series(dtype=float)
        """),
        C("""
        alpha_min = None
        alpha_1se = None
        """),
        C("""
        rmse_min = None
        rmse_1se = None
        leaves_min = None
        leaves_1se = None
        """),
    ],
    checks="""
        check(n_alphas is not None and n_alphas > 1500 and initial_leaves > 2000, "the base tree has over 2000 leaves and over 1500 candidate alpha values", "cost_complexity_pruning_path and get_n_leaves")
        check(len(cv_rmse) == 30 and len(cv_se) == 30 and len(leaves) == 30 and leaves.is_monotonic_decreasing and leaves.iloc[0] > 1000 and leaves.iloc[-1] < 30, "30 alpha values; the leaves decrease (from over 1000 to under 30)", "more alpha, more pruning")
        check(1e-4 <= alpha_min <= 1e-3 and alpha_1se >= alpha_min and cv_rmse[alpha_1se] <= cv_rmse.min() + cv_se[alpha_min] + 1e-12, "best alpha between 1e-4 and 1e-3; 1-SE alpha >= alpha_min and within one standard error", "the largest alpha within cv_rmse.min() + cv_se[alpha_min]")
        check(leaves_1se <= leaves_min and leaves_min < initial_leaves / 5, "the 1-SE tree has at most the leaves of the minimum one; both well below the initial leaves", "get_n_leaves()")
        check(rmse_1se < rmse_min + 0.03 and rmse_min < 0.66, "the 1-SE tree loses less than 0.03 of RMSE compared with the minimum (RMSE below 0.66)", "compare rmse_min and rmse_1se")
    """,
    hint="""
        `scores = -cross_val_score(DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=a), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")`; `cv_rmse[a] = scores.mean()`; `cv_se[a] = scores.std(ddof=1) / np.sqrt(5)`.
        `alpha_1se = max(a for a in alphas if cv_rmse[a] <= cv_rmse.min() + cv_se[alpha_min])`.
    """,
    solution=[
        C("""
        tree = base.fit(X_tr, y_tr)
        n_alphas = len(tree.cost_complexity_pruning_path(X_tr, y_tr).ccp_alphas)
        initial_leaves = tree.get_n_leaves()
        print(f"candidate alpha values: {n_alphas} | leaves of the unpruned tree: {initial_leaves}")
        """),
        C("""
        cv_rmse, cv_se, leaves = {}, {}, {}
        for a in alphas:
            scores = -cross_val_score(DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=a), X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
            cv_rmse[a], cv_se[a] = scores.mean(), scores.std(ddof=1) / np.sqrt(5)
            leaves[a] = DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=a).fit(X_tr, y_tr).get_n_leaves()
        cv_rmse, cv_se, leaves = pd.Series(cv_rmse), pd.Series(cv_se), pd.Series(leaves)
        print(pd.DataFrame({"CV RMSE": cv_rmse, "standard error": cv_se, "leaves": leaves}).round(4))
        """),
        C("""
        alpha_min = cv_rmse.idxmin()
        alpha_1se = max(a for a in alphas if cv_rmse[a] <= cv_rmse.min() + cv_se[alpha_min])
        print(f"alpha of the minimum: {alpha_min:.2e} | alpha with the 1-SE rule: {alpha_1se:.2e}")
        """),
        C("""
        def try_alpha(a):
            m = DecisionTreeRegressor(min_samples_leaf=5, random_state=0, ccp_alpha=a).fit(X_tr, y_tr)
            return mean_squared_error(y_te, m.predict(X_te), squared=False), m.get_n_leaves()


        (rmse_min, leaves_min), (rmse_1se, leaves_1se) = try_alpha(alpha_min), try_alpha(alpha_1se)
        print(f"minimum: test RMSE {rmse_min:.3f}, {leaves_min} leaves | 1-SE: test RMSE {rmse_1se:.3f}, {leaves_1se} leaves")
        """),
    ],
    learned="""
        - **`ccp_alpha` is a single continuous parameter for the complexity:** the full tree has over 2000 leaves and about 2000 candidate `alpha` values (each value corresponds to a subtree); as `alpha` grows the leaves drop from thousands to a handful. Pruning is a "softer" way of controlling the size compared with `max_depth`.
        - **The pruning goes inside the cross-validation** (`ccp_alpha` is a parameter of the estimator, so each fold prunes independently): choosing `alpha` on the training set and then validating on the same folds is selection optimism.
        - **The 1-SE rule picks the simplest model that cannot be told apart from the best:** the standard error (about 0.006) lets you move from the `alpha` of the minimum (4.5e-4, 199 leaves, test RMSE 0.639) to a slightly larger one (5.7e-4, 163 leaves, 0.645): 18% fewer leaves for 6 thousandths of RMSE. The rule does not always change the choice: with a sharp minimum the difference is small, with flat curves it can be very large. Fewer leaves = more readable, faster and more stable.
        - **`ccp_alpha` depends on the scale of the target** (it is expressed in impurity units): the best value here is not transferable to a problem with the price in dollars instead of hundreds of thousands (exercise E3).
    """,
    pitfall="""
        Choosing `ccp_alpha` by looking at the test set or at the same folds on which the base tree was chosen. Pruning is a hyperparameter like the others and requires a validation that has not already seen it.
    """,
    interview=(
        "What is cost-complexity pruning?",
        """
        I build the full tree and then minimize error + alpha·(number of leaves): for each alpha there is an optimal subtree and the subtrees are nested, so you get a sequence of smaller and smaller trees. I choose alpha in cross-validation, usually with the standard error rule
        to prefer the simplest tree. Unlike stopping the growth early, it looks at the overall effect of a branch.
        """),
)

EXERCISES = [B1, B2, B3]
