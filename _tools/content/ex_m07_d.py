"""Module 7 - type D exercises: interpret."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_07_decision_trees"

# ---------------------------------------------------------------------------------------- D1

D1 = dict(
    id="D1", slug="reading_the_structure_of_a_tree", kind="D", level=2, minutes=50, module=MODULE,
    title="Reading a tree from the inside: nodes, paths, probabilities and rules",
    goal="""
        Open a `DecisionTreeClassifier` trained on the Titanic and read its structure (`tree_`): number of nodes and leaves, root, purity of the leaves; rebuild `predict_proba` **by hand** by walking through the tree; find the leaf with the highest survival probability and write its rule in plain English.
    """,
    dataset="""
        `ds.titanic()` prepared in `X_tr, X_te, y_tr, y_te` with numeric columns: `pclass, age, fare` (missing values imputed with the median), `siblings_spouses`, `parents_children`, and the dummies `sex_male`, `title_Mr`, `title_Miss`, `title_Mrs`. Tree: `DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0)`; `names` is the list of the column names.
    """,
    purpose="""
        Learn to read what scikit-learn hides: `tree_` is a handful of arrays (children, features, thresholds, values) and the whole model is a table. Reading it teaches you what a tree really is and how to explain a prediction ("this passenger ended up in that leaf because...").
    """,
    task=[
        "Save `n_nodes`, `n_leaves`, `depth` (of the tree), `root` (name of the feature at the root), `root_threshold` and `root_samples` (number of examples at the root).",
        "Write `proba_by_hand(tree, x)`: start from node 0 and go down following `children_left`/`children_right` (`x[feature] <= threshold` goes left) until a leaf (`children_left == -1`); return the class fractions of the leaf (`tree_.value` holds the **normalized** fractions in scikit-learn >= 1.4 and the counts in "
        "earlier versions: always normalize by dividing by the sum). Check that it matches `predict_proba` on the **whole** test set: save `matches` (True/False).",
        "For each leaf compute `samples` (number of examples) and `purity` (fraction of the most frequent class). Save `largest_leaf` (node index of the leaf with the most samples) and its `largest_purity`, and `n_pure_leaves` (leaves with purity 1).",
        "**Rule:** among the leaves with at least 20 samples find the one with the highest survival probability and save `rule` as a list of strings (one per condition, for example `\"sex_male <= 0.5\"`) obtained by climbing from the node to the root. Also save `rule_proba` (fraction of survivors in the leaf).",
    ],
    setup="""
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier

        df = ds.titanic()
        y = df["survived"].to_numpy()
        X = df[["pclass", "age", "fare", "siblings_spouses", "parents_children"]].copy()
        X = X.fillna(X.median())                                                   # age and fare have a few missing values
        X["sex_male"] = (df["sex"] == "male").astype(int)
        for t in ["Mr", "Miss", "Mrs"]:
            X[f"title_{t}"] = (df["title"] == t).astype(int)
        names = list(X.columns)
        X_tr, X_te, y_tr, y_te = train_test_split(X.to_numpy(), y, test_size=0.3, stratify=y, random_state=0)
        tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=0).fit(X_tr, y_tr)
        T = tree.tree_
        print("columns:", names)
    """,
    starter=[
        C("""
        n_nodes = None
        n_leaves = None
        depth = None
        root = None
        root_threshold = None
        root_samples = None
        """),
        C("""
        def proba_by_hand(tree, x):
            ...
        """),
        C("""
        matches = None
        """),
        C("""
        largest_leaf = None
        largest_purity = None
        n_pure_leaves = None
        """),
        C("""
        rule = None
        rule_proba = None
        """),
    ],
    checks="""
        check(n_nodes == T.node_count and n_leaves == tree.get_n_leaves() and depth == tree.get_depth() and n_nodes == 2 * n_leaves - 1, "nodes, leaves and depth read from tree_ (a binary tree has 2 * leaves - 1 nodes)", "T.node_count, get_n_leaves(), get_depth()")
        check(root == names[T.feature[0]] and close(root_threshold, T.threshold[0]) and root_samples == len(y_tr), "root, threshold and samples at the root (the whole training set)", "T.feature[0], T.threshold[0], T.n_node_samples[0]")
        check(matches is True, "proba_by_hand matches predict_proba on the whole test set", "go down with children_left/right; normalize tree_.value")
        leaves = np.flatnonzero(T.children_left == -1)
        samples = T.n_node_samples[leaves]
        check(largest_leaf == leaves[samples.argmax()] and samples.max() >= 100 and largest_purity > 0.7, "the largest leaf has over 100 examples (and purity above 0.7)", "n_node_samples of the leaves")
        purities = np.array([T.value[n][0].max() / T.value[n][0].sum() for n in leaves])
        check(n_pure_leaves == int((np.isclose(purities, 1.0)).sum()) and n_pure_leaves >= 1, "correct number of pure leaves (at least one)", "purity = max(value) / sum(value)")
        check(isinstance(rule, list) and len(rule) >= 3 and all(isinstance(r, str) and ("<=" in r or ">" in r) for r in rule) and rule_proba > 0.85, "rule with at least 3 conditions and survival probability over 0.85", "climb with a child -> parent dictionary")
        check(any("sex_male" in r or "title_Mr" in r for r in rule), "the rule involves sex (or title)", "the conditions of the safest leaf")
    """,
    hint="""
        `T.children_left[n] == -1` identifies a leaf. Fraction: `v = T.value[n][0]; v / v.sum()`. Parent: build `parent = {}` with `for n in range(T.node_count): if T.children_left[n] != -1: parent[T.children_left[n]] = (n, "<="); parent[T.children_right[n]] = (n, ">")`.
        Rule: climb from the leaf to the root, for each step `f"{names[T.feature[p]]} {sign} {T.threshold[p]:.1f}"`, and reverse the order.
    """,
    solution=[
        C("""
        n_nodes, n_leaves, depth = T.node_count, tree.get_n_leaves(), tree.get_depth()
        root, root_threshold, root_samples = names[T.feature[0]], float(T.threshold[0]), int(T.n_node_samples[0])
        print(f"{n_nodes} nodes, {n_leaves} leaves, depth {depth} | root: {root} <= {root_threshold:.2f} with {root_samples} examples")
        """),
        C("""
        def proba_by_hand(tree, x):
            T = tree.tree_
            n = 0
            while T.children_left[n] != -1:                                       # until you are in a leaf
                n = T.children_left[n] if x[T.feature[n]] <= T.threshold[n] else T.children_right[n]
            v = T.value[n][0]
            return v / v.sum()                                                     # normalize: scikit-learn versions differ (counts or fractions)
        """),
        C("""
        by_hand = np.array([proba_by_hand(tree, x) for x in X_te])
        matches = bool(np.allclose(by_hand, tree.predict_proba(X_te)))
        print("proba by hand = predict_proba on all the", len(X_te), "test examples:", matches)
        """),
        C("""
        leaves = np.flatnonzero(T.children_left == -1)
        samples = T.n_node_samples[leaves]
        purities = np.array([T.value[n][0].max() / T.value[n][0].sum() for n in leaves])
        largest_leaf = int(leaves[samples.argmax()])
        largest_purity = float(purities[samples.argmax()])
        n_pure_leaves = int(np.isclose(purities, 1.0).sum())
        print(f"largest leaf: node {largest_leaf} with {samples.max()} examples and purity {largest_purity:.2f} | pure leaves: {n_pure_leaves} out of {len(leaves)}")
        """),
        C("""
        parent = {}
        for n in range(T.node_count):
            if T.children_left[n] != -1:
                parent[T.children_left[n]] = (n, "<=")
                parent[T.children_right[n]] = (n, ">")
        large = [n for n in leaves if T.n_node_samples[n] >= 20]
        best_leaf = max(large, key=lambda n: T.value[n][0][1] / T.value[n][0].sum())              # probability of class 1 (survived)
        rule, n = [], best_leaf
        while n in parent:
            p, sign = parent[n]
            rule.append(f"{names[T.feature[p]]} {sign} {T.threshold[p]:.1f}")
            n = p
        rule = rule[::-1]
        rule_proba = float(T.value[best_leaf][0][1] / T.value[best_leaf][0].sum())
        print(f"the safest leaf (>= 20 examples, node {best_leaf}, {T.n_node_samples[best_leaf]} examples): {rule_proba:.0%} survive if")
        for r in rule:
            print("   ", r)
        """),
    ],
    learned="""
        - **A tree is a table**: `tree_` holds, for each node, the feature and the threshold, the two children and the counts. `predict_proba` is a descent from one node to the next followed by a normalization; the rebuild by hand matches on every example. There is nothing more to it.
        - **Binary trees have `2 × leaves − 1` nodes**: the leaves tell the complexity of the model and the depth the maximum length of an explanation.
        - **The purity of a leaf is its uncertainty:** with `min_samples_leaf=10` many leaves are not pure and the probability is a fraction over at least ten examples. With leaves of 1-2 examples the same "probability" would be 0 or 1 and meaningless.
        - **Every prediction has a written rule**: the safest leaf among those with at least 20 examples is a combination of a few conditions (women and children in the first classes): climbing the tree you produce the explanation "this person survives with a probability of 97% because...". It is the advantage that no other model in the course has.
    """,
    pitfall="""
        Reading `tree_.value` without checking the version: in scikit-learn < 1.4 it holds the **counts**, from 1.4 the fractions. Code that uses the counts as fractions (or the other way round) breaks silently when the library is updated: always normalize.
    """,
    interview=(
        "How do you explain the prediction of a tree to a customer?",
        """
        I show the path from the example to the leaf (`decision_path`) as a list of conditions ("male, third class, age above 9"), and the fraction of similar training examples that are in the leaf, with their number: it is an exact local explanation, not an approximation.
        With deep trees or forests the explanation becomes more complex and I use permutation importances or SHAP.
        """),
)

# ---------------------------------------------------------------------------------------- D2

D2 = dict(
    id="D2", slug="impurity_importance_vs_permutation", kind="D", level=3, minutes=55, module=MODULE,
    title="Impurity importances reward noise: comparison with permutation",
    goal="""
        Add to the Titanic two columns **with no information** (a continuous random number and a random ticket code with 100,000 values) and compare the impurity importances (`feature_importances_`, computed on the training set) with those by **permutation** on the test set: the first gives the two noise columns a large share, the second
        brings them to zero.
    """,
    dataset="""
        `ds.titanic()` with two added columns: `random_number` (uniform) and `ticket_code` (random integers). Pipeline: median imputation for the numeric ones, mode + `OrdinalEncoder` for the categorical ones, `DecisionTreeClassifier(min_samples_leaf=5, random_state=0)`; stratified 70/30 split. `pipe` is already trained; `names` is the list of the column names in the order of the tree.
    """,
    purpose="""
        Know when **not** to trust `feature_importances_`. It is the best known bias of trees: a feature with many possible thresholds has more chances of finding a lucky one on the training set and the model uses it; the permutation importance on new data is not fooled.
    """,
    task=[
        "Save `imp` (`pd.Series` with `feature_importances_`, index `names`) and `perm` (`pd.Series` with the mean of `permutation_importance(pipe, X_te, y_te, n_repeats=20, random_state=0, scoring=\"roc_auc\").importances_mean`, index: the columns of `X_te` **reordered as `names`**).",
        "Save `top5_impurity` and `top5_permutation`: the lists of the names of the 5 features with the largest importance according to each method.",
        "Save `noise_share_impurity` (sum of the impurity importances of `random_number` and `ticket_code`) and `noise_share_permutation` (sum of their permutation values, **clipped at zero**, divided by the sum of all the positive values).",
        "Also check `test_auc`: the test AUC of the pipeline. Is the tree, with the two useless columns, worse or better than a tree without them (`auc_without`)? Save `auc_without` (same tree without the two columns).",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import OrdinalEncoder
        from sklearn.tree import DecisionTreeClassifier

        df = ds.titanic()
        X, y = df.drop(columns="survived").copy(), df["survived"]
        r = np.random.default_rng(0)
        X["random_number"] = r.uniform(size=len(X))
        X["ticket_code"] = r.integers(0, 100000, len(X))
        NUM = ["pclass", "age", "siblings_spouses", "parents_children", "fare", "random_number", "ticket_code"]
        CAT = ["sex", "port", "deck", "title"]
        names = NUM + CAT
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)


        def build(num, cat):
            prep = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OrdinalEncoder()), cat)])
            return make_pipeline(prep, DecisionTreeClassifier(min_samples_leaf=5, random_state=0))


        pipe = build(NUM, CAT).fit(X_tr, y_tr)
    """,
    starter=[
        C("""
        imp = None
        perm = None
        """),
        C("""
        top5_impurity = None
        top5_permutation = None
        """),
        C("""
        noise_share_impurity = None
        noise_share_permutation = None
        """),
        C("""
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        auc_without = None
        """),
    ],
    checks="""
        check(list(imp.index) == names and close(imp.sum(), 1.0) and list(perm.index) == names, "the two series have the same names in the same order; the impurity importances sum to 1", "use X_te[names] or reindex(names) for the permutation")
        check("sex" in top5_impurity and "sex" in top5_permutation and top5_impurity[0] == "sex" and top5_permutation[0] == "sex", "`sex` is at the top of both", "the most important feature")
        check("random_number" in top5_impurity and "random_number" not in top5_permutation, "`random_number` is in the top 5 by impurity but not by permutation", "compare the two lists")
        check(noise_share_impurity > 0.15 and noise_share_permutation < 0.06, "the noise is worth over 15% of the impurity importance and less than 6% of the permutation one", "sum of the two useless features")
        check(auc_without is not None and auc_without > test_auc - 0.03, "without the useless columns the AUC does not get worse (and often improves)", "build(NUM without noise, CAT)")
    """,
    hint="""
        `perm_res = permutation_importance(pipe, X_te, y_te, n_repeats=20, random_state=0, scoring="roc_auc")`; its `importances_mean` are in the order of `X_te.columns`: `pd.Series(perm_res.importances_mean, index=X_te.columns).reindex(names)`. The impurity importances: `pd.Series(pipe[-1].feature_importances_, index=names)`
        (the `ColumnTransformer` puts the numeric ones first and then the categorical ones, as in `names`).
    """,
    solution=[
        C("""
        imp = pd.Series(pipe[-1].feature_importances_, index=names)
        result = permutation_importance(pipe, X_te, y_te, n_repeats=20, random_state=0, scoring="roc_auc")
        perm = pd.Series(result.importances_mean, index=X_te.columns).reindex(names)
        pd.DataFrame({"impurity (training)": imp, "permutation (test)": perm}).round(3)
        """),
        C("""
        top5_impurity = list(imp.nlargest(5).index)
        top5_permutation = list(perm.nlargest(5).index)
        print("top 5 by impurity:", top5_impurity, "\\ntop 5 by permutation:", top5_permutation)
        """),
        C("""
        noise = ["random_number", "ticket_code"]
        noise_share_impurity = float(imp[noise].sum())
        positive = perm.clip(lower=0)
        noise_share_permutation = float(positive[noise].sum() / positive.sum())
        print(f"share of the noise: impurity {noise_share_impurity:.1%} | permutation {noise_share_permutation:.1%}")
        """),
        C("""
        test_auc = roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])
        without = build([c for c in NUM if c not in noise], CAT).fit(X_tr, y_tr)
        auc_without = roc_auc_score(y_te, without.predict_proba(X_te)[:, 1])
        print(f"test AUC with the two useless columns: {test_auc:.3f} | without: {auc_without:.3f}")
        """),
    ],
    learned="""
        - **The impurity importance gives 18% to the noise:** `random_number` (10.5%) is more important than `siblings_spouses` and on par with `age`, and `ticket_code` (7.7%) is sixth. Neither column contains information: the tree uses them to **memorize** the training cases. It is a known flaw of trees: features with
          many distinct values offer more thresholds to try and more chances of finding a lucky one.
        - **The permutation on the test set puts them back where they belong** (contribution 0 or slightly negative: shuffling a useless column does not make anything worse, or improves by chance). It measures how much the AUC **on new data** depends on that column: it is the operational definition of "useful".
        - The model with the two useless columns is a little **worse** than the one without (the noise increases the variance): another reason not to trust `feature_importances_` as a feature selection tool.
        - Rule: use the impurity importance for a quick first look, the permutation importance (on data not used for training) for the conclusions, and remember that with correlated features even the permutation splits the importance among the copies (it underestimates all of them).
    """,
    pitfall="""
        Selecting features by keeping those with an impurity importance above a threshold. Random columns with many values exceed the threshold, and informative ones with few distinct values (booleans) can stay below it.
    """,
    interview=(
        "Why can the feature_importances_ of a random forest be misleading?",
        """
        Because it is computed on the training set and with the impurity: features with many distinct values (continuous, identifiers) have more candidate thresholds and more chances of reducing the impurity by chance, so they receive importance even if they are noise. I use the permutation importance on a validation set, which measures
        how much the performance gets worse by shuffling the column, or SHAP; and I keep in mind that correlated features share the importance.
        """),
)

EXERCISES = [D1, D2]
