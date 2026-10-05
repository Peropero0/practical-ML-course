"""Module 5 - type E exercises: find the bug."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- E1

E1 = dict(
    id="E1", slug="the_nearest_neighbor_is_you", kind="E", level=3, minutes=45, module=MODULE, start="must_fail",
    title="The neighbors feature that contains the label of the point itself",
    goal="""
        Find out why a random forest to which "the label of the nearest neighbor" was added as a feature relies blindly on that column in training and does worse in test: for the **training** points the nearest neighbor is the point itself.
        Fix it by excluding itself and measure how much the defect weighs.
    """,
    dataset="""
        `make_moons(1500, noise=0.4)` split 50/50: `X_tr, X_te, y_tr, y_te`. `nn = NearestNeighbors(n_neighbors=1).fit(X_tr)` searches the neighbors in the training set.
    """,
    purpose="""
        Recognize a typical bug of "feature engineering with neighbors" (KNN as a feature for stacking, local target means, spatial target encoding): any feature computed **from the target of the neighbors** must exclude the point itself,
        otherwise it contains the answer.
    """,
    task=[
        "Run the colleague's cell: `bug_test_acc` is lower than that of a random forest without the new feature.",
        "Save `self_share` (fraction of training points whose 'nearest neighbor' returned by `nn.kneighbors(X_tr)` is itself) and `train_feature_acc` / `test_feature_acc` (accuracy of the feature alone used as a prediction, on the training and on the test set).",
        "Write `neighbor_feature(nn, Xq, y_ref, exclude_self)` that returns the label of the nearest neighbor, skipping the point itself when `exclude_self` is true (you need to query `n_neighbors=2`, that is a new `NearestNeighbors`).",
        "Rebuild the training feature with `exclude_self=True` and the test one with `False` (the test set is not in the training set): save `ok_test_acc`, and the importances of the new column `bug_imp` and `ok_imp`. Also save `X_only_acc` (random forest on the two coordinates only).",
    ],
    setup="""
        from sklearn.datasets import make_moons
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import NearestNeighbors

        X, y = make_moons(1500, noise=0.4, random_state=0)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        nn = NearestNeighbors(n_neighbors=1).fit(X_tr)
        rf = lambda: RandomForestClassifier(200, random_state=0, n_jobs=-1)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        f_tr = y_tr[nn.kneighbors(X_tr)[1][:, 0]]            # label of the nearest neighbor, for every training point
        f_te = y_tr[nn.kneighbors(X_te)[1][:, 0]]            # and for every test point
        A_tr, A_te = np.c_[X_tr, f_tr], np.c_[X_te, f_te]
        bug_model = rf().fit(A_tr, y_tr)
        bug_test_acc = bug_model.score(A_te, y_te)
        bug_imp = bug_model.feature_importances_[-1]
        print(f"test accuracy: {bug_test_acc:.3f} | importance of the new column: {bug_imp:.2f}")
        """),
        M("### Your diagnosis"),
        C("""
        self_share = None
        train_feature_acc = None
        test_feature_acc = None
        """),
        C("""
        def neighbor_feature(nn, Xq, y_ref, exclude_self):
            ...
        """),
        C("""
        ok_test_acc = None
        ok_imp = None
        X_only_acc = None
        """),
    ],
    checks="""
        check(bug_test_acc < 0.82, "the colleague's model is below 0.82 on the test set", "don't modify the colleague's code")
        check(self_share == 1.0 and train_feature_acc == 1.0, "for all the training points the 'nearest neighbor' is itself: the feature is the label", "compare the returned indices with np.arange(len(X_tr))")
        check(test_feature_acc < 0.85, "on the test set the same feature is much less accurate (below 0.85)", "the feature only holds for the label of the neighbor in the training set")
        check(np.array_equal(neighbor_feature(nn, X_tr, y_tr, True), y_tr[NearestNeighbors(n_neighbors=2).fit(X_tr).kneighbors(X_tr)[1][:, 1]]) and np.array_equal(neighbor_feature(nn, X_te, y_tr, False), f_te), "neighbor_feature skips itself only when needed", "for the training set query two neighbors and take the second")
        check(ok_test_acc > bug_test_acc + 0.02, "with the correct feature the test accuracy goes up by over 0.02", "exclude_self=True for the training set, False for the test set")
        check(bug_imp > 2.5 * ok_imp, "the colleague's model gives the new column over 2.5 times the importance of the correct one", "feature_importances_[-1]")
        check(abs(ok_test_acc - X_only_acc) < 0.03, "the neighbors feature, done right, adds nothing to the coordinates (gap below 0.03)", "the forest already had the two coordinates")
    """,
    hint="""
        `nn2 = NearestNeighbors(n_neighbors=2).fit(X_tr); idx = nn2.kneighbors(X_tr)[1][:, 1]` (column 1: the first neighbor other than itself, since itself is at distance 0).
        For the test set there is nothing to skip: no test point is in the training set.
    """,
    solution=[
        C("""
        f_tr = y_tr[nn.kneighbors(X_tr)[1][:, 0]]            # label of the nearest neighbor, for every training point
        f_te = y_tr[nn.kneighbors(X_te)[1][:, 0]]            # and for every test point
        A_tr, A_te = np.c_[X_tr, f_tr], np.c_[X_te, f_te]
        bug_model = rf().fit(A_tr, y_tr)
        bug_test_acc = bug_model.score(A_te, y_te)
        bug_imp = bug_model.feature_importances_[-1]
        print(f"test accuracy: {bug_test_acc:.3f} | importance of the new column: {bug_imp:.2f}")
        """),
        M("""
        **Where the bug is.** `nn.kneighbors(X_tr)` searches the neighbors of every training point **inside the training set**: the nearest neighbor of a point is the point itself, at distance 0. The column `f_tr` is therefore identical to `y_tr`. The forest learns
        "copy the new column" (importance 0.67). On the test set the same column is the label of a **different** point and is right only 80% of the time: the model relies on information that it will not have in production.
        """),
        C("""
        train_indices = nn.kneighbors(X_tr)[1][:, 0]
        self_share = float((train_indices == np.arange(len(X_tr))).mean())
        train_feature_acc = float((f_tr == y_tr).mean())
        test_feature_acc = float((f_te == y_te).mean())
        print(f"points whose neighbor is itself: {self_share:.0%} | accuracy of the feature: training {train_feature_acc:.2f}, test {test_feature_acc:.2f}")
        """),
        C("""
        def neighbor_feature(nn, Xq, y_ref, exclude_self):
            if exclude_self:
                indices = NearestNeighbors(n_neighbors=2).fit(Xq).kneighbors(Xq)[1][:, 1]      # Xq is the training set: column 0 = itself, column 1 = first real neighbor
            else:
                indices = nn.kneighbors(Xq)[1][:, 0]
            return y_ref[indices]
        """),
        C("""
        A_tr_ok = np.c_[X_tr, neighbor_feature(nn, X_tr, y_tr, True)]
        A_te_ok = np.c_[X_te, neighbor_feature(nn, X_te, y_tr, False)]
        ok_model = rf().fit(A_tr_ok, y_tr)
        ok_test_acc = ok_model.score(A_te_ok, y_te)
        ok_imp = ok_model.feature_importances_[-1]
        X_only_acc = rf().fit(X_tr, y_tr).score(X_te, y_te)
        print(f"test accuracy: bug {bug_test_acc:.3f} -> fixed {ok_test_acc:.3f} (coordinates only: {X_only_acc:.3f}) | importance of the column: {bug_imp:.2f} -> {ok_imp:.2f}")
        """),
    ],
    learned="""
        - **For the training points the nearest neighbor is the point itself.** Every feature built from the target of the neighbors (label of the neighbor, local mean of the target, fraction of positives) computed **on the same training set** contains the answer. The downstream
          model exploits it 100% and in production, where the point is not in the training set, the feature is much less informative.
        - The effect is measurable: the colleague's feature is worth 1.00 in training and 0.8 in test, the forest assigns it 67% of the importance (against 19% in the correct model) and the test accuracy drops by four points.
        - The fix is to exclude the point itself (`n_neighbors + 1` and skip the first column) or, better, to build the feature **out of sample** with cross-validation (`cross_val_predict` of a KNN): it is the rule of stacking and of target encoding.
        - With the feature done right nothing is gained: the forest already sees the coordinates and the neighbors feature adds no information. Before building a complicated feature, ask yourself whether the downstream model already has the same information.
    """,
    pitfall="""
        Building "proximity features" with `kneighbors(X_train)` and using them in training. Every time you evaluate a KNN or a derivative of it on the training set, the nearest neighbor is you.
    """,
    interview=(
        "How do you do stacking (or target encoding) without leakage?",
        """
        The features built from the target must be computed **out of sample**: for every row of the training set I use a model or statistic trained on the other rows (K-fold: `cross_val_predict`, target encoding with folds). For the test set I use the model trained on the whole training set.
        That way the distribution of the feature in training and in test coincides and the next model learns to trust it by the right amount.
        """),
)

# ---------------------------------------------------------------------------------------- E2

E2 = dict(
    id="E2", slug="columns_in_a_different_order_in_production", kind="E", level=2, minutes=40, module=MODULE, start="must_fail",
    title="The same columns, in another order: the model does not notice",
    goal="""
        See what happens when a model trained on numpy arrays receives in production a DataFrame with the same columns in a different order: **no error** and almost random predictions. Build a pipeline that selects the columns by name and
        a schema checking function.
    """,
    dataset="""
        `load_wine(as_frame=True)`: 13 numeric columns, 3 classes, stratified 70/30 split: `X_tr, X_te, y_tr, y_te` (DataFrames). `X_prod` is the test set with the columns shuffled, as it would arrive from another system.
    """,
    purpose="""
        Recognize the "position instead of name" bug: with `.to_numpy()` or `.values` the information on the names disappears and the model reads the `k`-th column whatever it is. It is an integration problem, not a model problem, and it is very common.
    """,
    task=[
        "Run the colleague's cell: `test_acc` is high, `bug_prod_acc` (same examples, shuffled columns) collapses, and no code reported an error.",
        "Save `changed_positions`: how many of the 13 columns of `X_prod` are in a different position from `X_tr`.",
        "Write `check_schema(X, expected)`: it raises `ValueError` if an expected column is missing or there is an extra one; otherwise it returns `X[list(expected)]` (columns reordered as in training).",
        "Rebuild the model with a `ColumnTransformer` that **selects the columns by name** (`StandardScaler` on all the columns, given by name) and train it on the DataFrames: save `ok_model`, `ok_test_acc` and `ok_prod_acc` (on the DataFrame with the shuffled columns, without reordering them by hand).",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.datasets import load_wine
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        wine = load_wine(as_frame=True)
        X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.3, stratify=wine.target, random_state=0)
        columns = list(X_tr.columns)
        prod_order = list(np.random.default_rng(1).permutation(columns))
        X_prod = X_te[prod_order]                                    # same columns and same examples, different order
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        bug_model = make_pipeline(StandardScaler(), KNeighborsClassifier(7)).fit(X_tr.to_numpy(), y_tr)
        test_acc = bug_model.score(X_te.to_numpy(), y_te)
        bug_prod_acc = bug_model.score(X_prod.to_numpy(), y_te)
        print(f"test accuracy: {test_acc:.3f} | in production (columns in a different order): {bug_prod_acc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        changed_positions = None
        """),
        C("""
        def check_schema(X, expected):
            ...
        """),
        C("""
        ok_model = None
        ok_test_acc = None
        ok_prod_acc = None
        """),
    ],
    checks="""
        check(test_acc > 0.9 and bug_prod_acc < 0.7, "the colleague's model: over 0.9 on the test set, below 0.7 with the shuffled columns, with no errors", "don't modify the colleague's code")
        check(changed_positions == sum(a != b for a, b in zip(columns, prod_order)) and changed_positions >= 10, "at least 10 columns out of 13 are in a different position", "compare the two lists position by position")
        check(list(check_schema(X_prod, columns).columns) == columns and (check_schema(X_prod, columns).to_numpy() == X_te.to_numpy()).all(), "check_schema reorders the columns as in training", "X[list(expected)]")
        for bad, name in [(X_prod.drop(columns=columns[0]), "missing column"), (X_prod.assign(extra=0.0), "extra column")]:
            try:
                check_schema(bad, columns)
                raised = False
            except ValueError:
                raised = True
            check(raised, f"check_schema raises ValueError with a {name}", "compare set(X.columns) with set(expected)")
        check(ok_model is not None and ok_prod_acc > 0.9 and abs(ok_prod_acc - ok_test_acc) < 1e-12, "with the selection by name the shuffled columns give the same result as the test set", "ColumnTransformer([('num', StandardScaler(), columns)])")
    """,
    hint="""
        `ok_model = make_pipeline(ColumnTransformer([("num", StandardScaler(), columns)]), KNeighborsClassifier(7)).fit(X_tr, y_tr)`; `ok_model.score(X_prod, y_te)`. The `ColumnTransformer` selects by **name** and always produces the same output order.
        `check_schema`: `if set(X.columns) != set(expected): raise ValueError(...)`.
    """,
    solution=[
        C("""
        bug_model = make_pipeline(StandardScaler(), KNeighborsClassifier(7)).fit(X_tr.to_numpy(), y_tr)
        test_acc = bug_model.score(X_te.to_numpy(), y_te)
        bug_prod_acc = bug_model.score(X_prod.to_numpy(), y_te)
        print(f"test accuracy: {test_acc:.3f} | in production (columns in a different order): {bug_prod_acc:.3f}")
        """),
        M("""
        **Where the bug is.** With `.to_numpy()` the model lost the names: at prediction time it reads the column in position 0 as if it were `alcohol`, the one in position 1 as `malic_acid`, and so on. If the columns change order, every feature ends up in the
        wrong position and the scaler and the KNN work on numbers of another magnitude. No check intervenes because the shapes coincide.
        """),
        C("""
        changed_positions = sum(a != b for a, b in zip(columns, prod_order))
        print("columns in a different position:", changed_positions, "out of", len(columns))
        """),
        C("""
        def check_schema(X, expected):
            missing, extra = set(expected) - set(X.columns), set(X.columns) - set(expected)
            if missing or extra:
                raise ValueError(f"different schema: missing {sorted(missing)}, extra {sorted(extra)}")
            return X[list(expected)]                                           # same order as in training
        """),
        C("""
        ok_model = make_pipeline(ColumnTransformer([("num", StandardScaler(), columns)]), KNeighborsClassifier(7)).fit(X_tr, y_tr)
        ok_test_acc = ok_model.score(X_te, y_te)
        ok_prod_acc = ok_model.score(X_prod, y_te)
        print(f"with selection by name: test {ok_test_acc:.3f} | production {ok_prod_acc:.3f}")
        """),
    ],
    learned="""
        - **A model trained on numpy has no names**: it reads the columns by position. If in production the order changes (another query, another system, a column added in the middle) the predictions get ruined with no error at all: here from over 0.9 to about 0.4.
        - Structural remedy: **train on DataFrames and select the columns by name** in the `ColumnTransformer`. The order of the incoming data no longer matters (recent scikit-learn versions also check `feature_names_in_` and will raise an error, but your
          pipeline must be correct even when it doesn't: with 1.1 the check is only a warning).
        - Second level of defense: an explicit **schema check** at the entrance (expected columns, types, plausible ranges) that fails loudly. In production a clear error is worth far more than a wrong prediction.
        - KNN is particularly vulnerable because the distances mix all the columns, but the same holds for a linear regression (coefficients attached to the wrong columns).
    """,
    pitfall="""
        Converting the DataFrames to arrays (`.values`, `.to_numpy()`) "to be sure scikit-learn works" at the start of the pipeline. It is exactly the step that erases the protection of the names.
    """,
    interview=(
        "How do you avoid schema errors when a model goes into production?",
        """
        I train on DataFrames with selection of the columns by name inside the pipeline, save the list of expected columns with the model and verify the incoming schema (names, types, ranges). I add automated tests with example data in a different order and with missing
        columns, and monitor the distribution of the features in production against that of the training set.
        """),
)

# ---------------------------------------------------------------------------------------- E3

E3 = dict(
    id="E3", slug="normalizer_instead_of_standardscaler", kind="E", level=2, minutes=35, module=MODULE, start="must_fail",
    title="Normalizer is not a scaler: it normalizes the rows, not the columns",
    goal="""
        Find out why a KNN on the wines with `Normalizer` "to bring everything onto the same scale" gets 0.80 instead of 0.96: `Normalizer` divides **every row** by its length, erases the magnitude of the samples and does not touch the relative scales of the columns.
        Replace it with `StandardScaler` and understand when `Normalizer` is instead the right choice.
    """,
    dataset="""
        `load_wine`: 178 wines, 13 measurements (`proline` has a scale of hundreds, `hue` of 1), 3 classes. Validation: `StratifiedKFold(5, shuffle=True, random_state=0)`.
    """,
    purpose="""
        Tell apart two transformations that the name makes you confuse: **scaler** (per column, learned from the training set) and `Normalizer` (per row, stateless). And recognize that "no error and accuracy 0.80" is not a result to accept without a baseline.
    """,
    task=[
        "Run the colleague's cell: `bug_acc` (KNN with `Normalizer`). Also compute `no_transform_acc` (KNN without transformations) and `std_acc` (with `StandardScaler`): did the colleague improve anything compared to doing nothing?",
        "Check what `Normalizer` does: `norms_after` (Euclidean norm of every row after `Normalizer().fit_transform(X)`) and `unit_norms` (True if they are all 1). Save `deviations_after` (`pd.Series`: standard deviation of each column after `Normalizer`), `dominant_column_after` (name of the column with the largest standard deviation) and `scale_ratio` (maximum / minimum standard deviation): "
        "were the columns brought to the same scale?",
        "Show that the length disappears: `proportional_distance` = Euclidean distance between `Normalizer` applied to `X.iloc[0]` and to `2 * X.iloc[0]` (two wines identical except for a factor 2).",
        "Answer with a letter: `what_normalizer_does = 'c'` (a: standardizes the columns; b: brings every column into [0, 1]; c: divides every row by its norm).",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import Normalizer, StandardScaler

        X, y = load_wine(return_X_y=True, as_frame=True)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        M("### The colleague's code"),
        C("""
        bug = make_pipeline(Normalizer(), KNeighborsClassifier(7))       # "that way all the columns are on the same scale"
        bug_acc = cross_val_score(bug, X, y, cv=cv).mean()
        print(f"accuracy with Normalizer: {bug_acc:.3f}")
        """),
        M("### Your diagnosis and the fix"),
        C("""
        no_transform_acc = None
        std_acc = None
        """),
        C("""
        norms_after = None
        unit_norms = None
        deviations_after = None
        dominant_column_after = None
        scale_ratio = None
        proportional_distance = None
        what_normalizer_does = None
        """),
    ],
    checks="""
        check(bug_acc < 0.85, "the colleague gets less than 0.85", "don't modify the colleague's code")
        check(std_acc is not None and std_acc > 0.94 and std_acc > bug_acc + 0.1, "with StandardScaler the accuracy goes up by over 0.1 (above 0.94)", "make_pipeline(StandardScaler(), KNeighborsClassifier(7))")
        check(no_transform_acc is not None and bug_acc > no_transform_acc, "Normalizer improves on doing nothing, but only a little compared to the scaler: it is an accidental improvement", "no_transform_acc: just KNeighborsClassifier(7)")
        check(unit_norms is True and close(norms_after, 1.0), "after Normalizer every row has norm 1", "np.linalg.norm(..., axis=1)")
        check(dominant_column_after == deviations_after.idxmax() == "magnesium" and scale_ratio > 50, "after Normalizer another column dominates (`magnesium`) and the scales still differ by over 50 times", "std per column of the result")
        check(proportional_distance is not None and proportional_distance < 1e-12, "two wines that differ by a factor 2 have distance 0 after Normalizer", "Normalizer erases the length of the row")
        check(what_normalizer_does == "c", "correct answer: it divides every row by its norm", "c")
    """,
    hint="""
        `Xn = Normalizer().fit_transform(X)`; `np.linalg.norm(Xn, axis=1)`; `pd.DataFrame(Xn, columns=X.columns).std()`. Distance: `Normalizer().transform([X.iloc[0], 2 * X.iloc[0]])` then `np.linalg.norm(a - b)`.
    """,
    solution=[
        C("""
        bug = make_pipeline(Normalizer(), KNeighborsClassifier(7))       # "that way all the columns are on the same scale"
        bug_acc = cross_val_score(bug, X, y, cv=cv).mean()
        print(f"accuracy with Normalizer: {bug_acc:.3f}")
        """),
        M("""
        **Where the bug is.** `Normalizer` is a **per-row** transformation: it divides every wine by its own length, so that the vector of the 13 measurements has norm 1. It does not remove the mean of the columns nor bring them onto the same scale. Since `proline`
        (hundreds) dominates the length of every row, after the transformation `proline` is almost always 1 (its variability disappears) and what comes to dominate the distance is another column, `magnesium`, with a standard deviation more than a hundred times that of `nonflavanoid_phenols`.
        """),
        C("""
        no_transform_acc = cross_val_score(KNeighborsClassifier(7), X, y, cv=cv).mean()
        std_acc = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(7)), X, y, cv=cv).mean()
        print(f"no transformation: {no_transform_acc:.3f} | Normalizer: {bug_acc:.3f} | StandardScaler: {std_acc:.3f}")
        """),
        C("""
        Xn = Normalizer().fit_transform(X)
        norms_after = np.linalg.norm(Xn, axis=1)
        unit_norms = bool(np.allclose(norms_after, 1.0))
        deviations_after = pd.Series(Xn.std(axis=0), index=X.columns)
        dominant_column_after = deviations_after.idxmax()
        scale_ratio = float(deviations_after.max() / deviations_after.min())
        a, b = Normalizer().transform(np.array([X.iloc[0], 2 * X.iloc[0]]))
        proportional_distance = float(np.linalg.norm(a - b))
        what_normalizer_does = "c"
        print("unit norms:", unit_norms, "| dominant column after Normalizer:", dominant_column_after, f"(scales: ratio {scale_ratio:.0f}x)", "| distance between x and 2x after Normalizer:", proportional_distance)
        print(deviations_after.round(4).sort_values(ascending=False).head(4).to_string())
        """),
    ],
    learned="""
        - `Normalizer` **is not a scaler**: it divides every row by its norm. It is useful when the **direction** of the vector matters and not the length (word frequencies in documents of different length, histograms, cosine similarity): two documents
          equal except for a factor 2 (one long, one short) become identical. In the wines the length is information: a wine with more of everything is different from one with less of everything.
        - **The columns do not end up on the same scale**: after `Normalizer` the variability of `proline` disappears (it is almost always 1) and the dominant column becomes `magnesium`, with a standard deviation more than a hundred times that of the least variable column. The result (0.80) is far below that of the scaler (0.96) and only a little better than doing nothing (0.69).
        - The name is misleading: `StandardScaler` and `MinMaxScaler` transform the **columns** with training statistics; `Normalizer` transforms the **rows** without learning anything (`fit` does nothing) and the pipeline treats it like any other step.
        - The warning sign was "0.80 on a dataset where a scaled KNN gets 0.96": you discover it only by comparing with **a correct baseline**. When a preprocessing "improves" a result, always ask yourself compared to what.
    """,
    pitfall="""
        Choosing a transformer by its name ("normalize": so it brings everything to the same scale) without reading what it does. In scikit-learn "normalize" always means "bring the norm of the row to 1", which is different from standardizing.
    """,
    interview=(
        "What is the difference between StandardScaler, MinMaxScaler and Normalizer?",
        """
        The first two work per **column** with statistics learned from the training set: StandardScaler removes the mean and divides by the standard deviation, MinMaxScaler brings every column into an interval (usually 0-1). Normalizer works per **row** and divides every example by its
        norm: it is suited when the direction matters (text, cosine) and does not change the relative scales of the columns. I choose based on what must be comparable: the columns with each other or the examples with each other.
        """),
)

EXERCISES = [E1, E2, E3]
