"""Module 5 - type C exercises: break the model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- C1

C1 = dict(
    id="C1", slug="curse_of_dimensionality", kind="C", level=2, minutes=45, module=MODULE,
    title="Breaking the KNN with noise columns",
    goal="""
        Add to a two-dimensional problem a growing number of pure noise columns and measure how a KNN collapses while a logistic and a random forest hold up. Measure the **concentration of the distances** (the nearest neighbor
        becomes almost as far as the farthest) and repair with feature selection inside the pipeline.
    """,
    dataset="""
        `generate(d)` (given): 3000 points of `make_moons(noise=0.25)` with `d` columns of Gaussian noise added, 50/50 split in `X_tr, X_te, y_tr, y_te`. The two informative columns are always the first two.
    """,
    purpose="""
        See with numbers why KNN is the model most exposed to the curse of dimensionality: every useless column adds a random component to the distance, and with enough columns the "true" neighbors are no longer the nearest ones.
        The same does not happen to models that select the features (trees) or ignore them (regularized logistic).
    """,
    task=[
        "For `d` in `[0, 2, 5, 10, 20, 50, 100, 200]`: generate the data and compute on the test set the accuracy of `KNeighborsClassifier(5)`, of `LogisticRegression(max_iter=1000)` and of `RandomForestClassifier(100, random_state=0, n_jobs=-1)`, "
        "and `concentration` (given: `concentration(X_te)`, ratio between the mean distance to the nearest neighbor and the mean distance to the farthest on a sample). Save everything in `table` (index `d`, columns `knn`, `lr`, `rf`, `concentration`).",
        "Save `collapse_d`: the first `d` at which the KNN accuracy drops **below 0.65**.",
        "Save `loss`: dictionary `{'knn': ..., 'lr': ..., 'rf': ...}` with `accuracy(d=0) − accuracy(d=200)` for each model.",
        "**Repair:** for `d = 100` train a `Pipeline` `SelectKBest(f_classif, k=2)` + `KNeighborsClassifier(5)` and save `selection_acc` (test accuracy) and `chosen_columns` (indices of the selected columns: `get_support().nonzero()[0]`).",
    ],
    setup="""
        from sklearn.datasets import make_moons
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import SelectKBest, f_classif
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline


        def generate(d, seed=0):
            X, y = make_moons(3000, noise=0.25, random_state=0)
            noise = np.random.default_rng(seed).normal(size=(len(X), d))
            return train_test_split(np.c_[X, noise], y, test_size=0.5, random_state=0)


        def concentration(X, n=300):
            D = np.sqrt(((X[:n, None, :] - X[None, n:2 * n, :]) ** 2).sum(-1))       # distances between two different groups of points
            return D.min(1).mean() / D.max(1).mean()
    """,
    starter=[
        C("""
        rows = {}
        for d in [0, 2, 5, 10, 20, 50, 100, 200]:
            X_tr, X_te, y_tr, y_te = generate(d)
            rows[d] = ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        collapse_d = None
        loss = None
        """),
        C("""
        X_tr, X_te, y_tr, y_te = generate(100)
        selection_acc = None
        chosen_columns = None
        """),
    ],
    checks="""
        check(list(table.columns) == ["knn", "lr", "rf", "concentration"] and list(table.index) == [0, 2, 5, 10, 20, 50, 100, 200], "table with the four columns for the eight values of d", "index d, columns knn, lr, rf, concentration")
        check(table.loc[0, "knn"] > 0.9 and table.loc[200, "knn"] < 0.62, "the KNN goes from over 0.9 to less than 0.62 (almost chance)", "test accuracy at d = 0 and d = 200")
        check(table.loc[200, "lr"] > 0.8 and table.loc[200, "rf"] > 0.85, "the logistic and the random forest stay above 0.8 and 0.85", "they don't use the distances in the full space")
        check(table["concentration"].is_monotonic_increasing and table.loc[0, "concentration"] < 0.05 and table.loc[200, "concentration"] > 0.7, "the nearest/farthest ratio goes from below 0.05 to above 0.7", "with noise the nearest neighbor is as far as the farthest")
        check(collapse_d in (50, 100), "the KNN drops below 0.65 with 50-100 noise columns", "first d with knn < 0.65")
        check(loss["knn"] > 0.3 and loss["lr"] < 0.06 and loss["rf"] < 0.06, "loss: KNN over 0.3, the others below 0.06", "accuracy(0) - accuracy(200)")
        check(selection_acc > 0.92 and sorted(chosen_columns.tolist()) == [0, 1], "with the selection the accuracy goes back to 0.92 or more and the chosen columns are the first two", "SelectKBest inside the pipeline")
    """,
    hint="""
        Row: `{"knn": KNeighborsClassifier(5).fit(X_tr, y_tr).score(X_te, y_te), "lr": ..., "rf": ..., "concentration": concentration(X_te)}`. Selection: `sel = make_pipeline(SelectKBest(f_classif, k=2), KNeighborsClassifier(5)).fit(X_tr, y_tr)`; `sel[0].get_support().nonzero()[0]`.
    """,
    solution=[
        C("""
        rows = {}
        for d in [0, 2, 5, 10, 20, 50, 100, 200]:
            X_tr, X_te, y_tr, y_te = generate(d)
            rows[d] = {"knn": KNeighborsClassifier(5).fit(X_tr, y_tr).score(X_te, y_te),
                       "lr": LogisticRegression(max_iter=1000).fit(X_tr, y_tr).score(X_te, y_te),
                       "rf": RandomForestClassifier(100, random_state=0, n_jobs=-1).fit(X_tr, y_tr).score(X_te, y_te),
                       "concentration": concentration(X_te)}
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        collapse_d = int(table.index[(table["knn"] < 0.65).to_numpy().argmax()])
        loss = {m: table.loc[0, m] - table.loc[200, m] for m in ["knn", "lr", "rf"]}
        print("the KNN drops below 0.65 with d =", collapse_d, "| losses:", {m: round(v, 3) for m, v in loss.items()})
        """),
        C("""
        X_tr, X_te, y_tr, y_te = generate(100)
        sel = make_pipeline(SelectKBest(f_classif, k=2), KNeighborsClassifier(5)).fit(X_tr, y_tr)
        selection_acc = sel.score(X_te, y_te)
        chosen_columns = sel[0].get_support().nonzero()[0]
        print(f"with d = 100: KNN {table.loc[100, 'knn']:.3f} -> with selection {selection_acc:.3f} | chosen columns: {chosen_columns}")
        """),
    ],
    learned="""
        - **Every useless column costs the KNN**: the accuracy drops from 0.93 (without noise) to 0.79 with 10 useless columns, 0.64 with 50 and 0.57 with 200 (chance is 0.5). The logistic does not change (0.83-0.86) and the random forest
          drops a little (0.93 → 0.89): a tree can simply not use the useless columns.
        - The explanation is geometric: the squared distance is the sum of the contributions of all the coordinates, and every noise column adds a **random** amount of the same order of magnitude. The ratio between the nearest neighbor and the farthest
          goes from 0.03 to 0.78: with enough dimensions "everyone is equally far" and the neighbor is chosen by the noise.
        - The repair is **selecting the features** (here `SelectKBest` with the F test, inside the pipeline to avoid leakage): with 100 noise columns the accuracy goes back to 0.93, equal to the case without noise. In general you need
          feature selection, dimension reduction (PCA, module 6) or a model that selects them by itself.
        - Rule of thumb: before a KNN ask yourself **how many of your columns really carry information about the distance** (not about the target). Many columns correlated with the target but little with each other are a good sign; many detail columns are not.
    """,
    pitfall="""
        Adding all the available features to a KNN "because more information can't hurt". With distance-based models it hurts, silently and in proportion to the number of useless columns.
    """,
    interview=(
        "What is the curse of dimensionality and how does it hit KNN?",
        """
        As the number of dimensions grows the space "empties out": the points are almost all at the same distance from each other, the ratio between the nearest neighbor and the farthest tends to 1 and the neighbors stop being meaningful. In KNN it is a direct problem because
        the prediction depends only on the distance. It is mitigated with feature selection or reduction, with learned metrics or with models that select them by themselves (trees, regularized linear models), or with a lot more data.
        """),
)

# ---------------------------------------------------------------------------------------- C2

C2 = dict(
    id="C2", slug="noisy_labels_and_choice_of_k", kind="C", level=3, minutes=50, module=MODULE,
    title="Wrong labels: where k = 1 collapses and large k holds up",
    goal="""
        Corrupt a growing share of the training labels (0%, 10%, 20%, 30%, 40%) and measure, for each level, the test accuracy as a function of `k`: the KNN with `k = 1` **memorizes** the errors, while the best `k` grows with the noise.
        Check that cross-validation on the noisy training set still finds a `k` that works.
    """,
    dataset="""
        `make_moons(2000, noise=0.15)` split 50/50: `X_tr, X_te, y_tr, y_te` (the test set has **clean labels**). `noisy(p)` (given) returns the training labels with a fraction `p` swapped at random.
    """,
    purpose="""
        Understand KNN as a **local average model**: with `k = 1` it copies every error and its accuracy collapses with the noise; with large `k` it votes among many neighbors and the errors compensate. Noise in the labels is the situation in which the
        bias-variance tradeoff is most evident.
    """,
    task=[
        "For `p` in `[0, 0.1, 0.2, 0.3, 0.4]` and `k` in `Ks = [1, 3, 5, 9, 15, 25, 41, 61, 91, 121]`: train `KNeighborsClassifier(k)` on the training set with labels `noisy(p)` and measure the accuracy on the **clean** test set. Save `acc` (DataFrame rows `p`, columns `k`).",
        "Save `optimal_k` (`pd.Series` indexed by `p`: the `k` with maximum test accuracy), `optimal_acc` (the corresponding accuracy) and `k1_acc` (accuracy with `k = 1`).",
        "Save `k1_train_acc` (`pd.Series`): training accuracy of `k = 1` **measured on the noisy labels** (the ones it was trained with). What is it? What sense does it make as a measure?",
        "**Without looking at the test set:** for each `p` choose `k` with `GridSearchCV` (5 folds) on the noisy training set: save `cv_k` (`pd.Series`) and `acc_with_cv_k` (test accuracy with that `k`).",
    ],
    setup="""
        from sklearn.datasets import make_moons
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.neighbors import KNeighborsClassifier

        X, y = make_moons(2000, noise=0.15, random_state=1)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, random_state=0)
        Ks = [1, 3, 5, 9, 15, 25, 41, 61, 91, 121]
        ps = [0, 0.1, 0.2, 0.3, 0.4]


        def noisy(p):
            \"\"\"Training labels with a fraction p swapped at random (the test set stays clean).\"\"\"
            swap = np.random.default_rng(0).random(len(y_tr)) < p
            return np.where(swap, 1 - y_tr, y_tr)
    """,
    starter=[
        C("""
        acc = pd.DataFrame(index=ps, columns=Ks, dtype=float)
        for p in ps:
            for k in Ks:
                ...
        acc.round(3)
        """),
        C("""
        optimal_k = None
        optimal_acc = None
        k1_acc = None
        k1_train_acc = None
        """),
        C("""
        cv_k = None
        acc_with_cv_k = None
        """),
    ],
    checks="""
        check(acc.shape == (5, 10) and acc.notna().all().all(), "complete 5 x 10 table", "one accuracy for every pair (p, k)")
        check(optimal_k.is_monotonic_increasing and optimal_k[0] <= 9 and optimal_k[0.4] >= 61, "the optimal k grows with the noise: from 9 or less (p = 0) to 61 or more (p = 0.4)", "argmax per row of acc")
        check(optimal_acc.min() > 0.97, "with the right k the accuracy stays above 0.97 even with 40% wrong labels", "optimal_acc = maximum per row")
        check(k1_acc[0.4] < 0.7 and k1_acc[0] > 0.97 and k1_acc[0.4] < k1_acc[0.2] < k1_acc[0.1] < k1_acc[0], "with k = 1 the accuracy keeps falling, down to below 0.7", "the k = 1 column of acc")
        check((k1_train_acc == 1.0).all(), "with k = 1 the training accuracy (on noisy labels) is always 1.0: it memorizes the errors too", "the nearest neighbor of a training point is the point itself")
        check(cv_k[0.3] >= 25 and cv_k[0.4] >= 25 and (acc_with_cv_k > 0.97).all(), "cross-validation on the noisy training set finds large k (25 or more with 30-40% noise) and the test accuracy stays above 0.97", "GridSearchCV(KNeighborsClassifier(), {'n_neighbors': Ks}, cv=5)")
    """,
    hint="""
        `KNeighborsClassifier(k).fit(X_tr, noisy(p)).score(X_te, y_te)`. `optimal_k = acc.idxmax(axis=1)`; `optimal_acc = acc.max(axis=1)`; for the training set: `m.score(X_tr, noisy(p))`.
        `GridSearchCV(KNeighborsClassifier(), {"n_neighbors": Ks}, cv=5).fit(X_tr, noisy(p))` and then `best_params_["n_neighbors"]`.
    """,
    solution=[
        C("""
        acc = pd.DataFrame(index=ps, columns=Ks, dtype=float)
        for p in ps:
            y_noisy = noisy(p)
            for k in Ks:
                acc.loc[p, k] = KNeighborsClassifier(k).fit(X_tr, y_noisy).score(X_te, y_te)          # noisy training labels, clean test set
        acc.round(3)
        """),
        C("""
        optimal_k = acc.idxmax(axis=1)
        optimal_acc = acc.max(axis=1)
        k1_acc = acc[1]
        k1_train_acc = pd.Series({p: KNeighborsClassifier(1).fit(X_tr, noisy(p)).score(X_tr, noisy(p)) for p in ps})
        print(pd.DataFrame({"optimal k": optimal_k, "optimal acc": optimal_acc, "acc with k=1": k1_acc, "train with k=1": k1_train_acc}).round(3))
        """),
        C("""
        cv_k, acc_with_cv_k = {}, {}
        for p in ps:
            search = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": Ks}, cv=5).fit(X_tr, noisy(p))
            cv_k[p] = search.best_params_["n_neighbors"]
            acc_with_cv_k[p] = search.score(X_te, y_te)
        cv_k, acc_with_cv_k = pd.Series(cv_k), pd.Series(acc_with_cv_k)
        print(pd.DataFrame({"k chosen in cv": cv_k, "test accuracy": acc_with_cv_k}).round(3))
        """),
    ],
    learned="""
        - **With `k = 1` the training accuracy is always 1.0**, and with 40% wrong labels it means the model has memorized all the errors. Measured on the clean test set the same choice is worth 0.99 without noise and 0.63 with 40%.
          A training accuracy of 1.0 says nothing.
        - The **optimal `k` grows with the noise** (from 5 to 15, 41 and 121): every time the labels are less reliable it is better to vote among more neighbors, so the errors compensate. It is the bias-variance tradeoff in pure form: high noise = high variance
          = you need a smoother model.
        - With the right `k` the test accuracy stays at 0.99 **even with 40% wrong labels**: KNN (with enough data and a problem with only two dimensions) is very robust to symmetric label noise, as long as `k` is chosen well.
        - Cross-validation on the **noisy** training set still finds large `k`: the validation folds have the same noise, so the models that memorize the noise get penalized. You don't need clean labels to choose `k`, but you do need
          many examples: with little data and a lot of noise the choice becomes unstable.
    """,
    pitfall="""
        Using `k = 1` or `k = 3` as a "default" without validating. With noisy labels (frequent in real data: human annotations, legacy systems) they are the worst choices. And never trust the training accuracy of a KNN.
    """,
    interview=(
        "You suspect that 20% of the labels are wrong. What do you change in a KNN?",
        """
        I increase `k` (and choose it in cross-validation, which sees the same noise) to compensate for the errors by voting among more neighbors; I use 1/distance weights with caution because a wrong neighbor very close weighs too much. If I can, I clean up: methods like
        `k=1` with a filter (edited nearest neighbours) or the search for labels inconsistent with the neighbors. And I always measure on a test set with verified labels, otherwise I measure the noise too.
        """),
)

EXERCISES = [C1, C2]
