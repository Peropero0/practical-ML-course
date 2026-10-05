"""Module 8 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_08_random_forest"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="the_ladder_from_tree_to_extra_trees", kind="F", level=2, minutes=50, module=MODULE,
    title="From a tree to Extra Trees: how much each ingredient gains",
    goal="""
        Compare four models **built one on top of the other** on three datasets (wines, tumors, digits): a single tree, bagging of trees (variance reduced by the average), the random forest (bagging + random features) and Extra Trees (random thresholds on top of random features). Measure accuracy in repeated cross-validation and time,
        and quantify the gain of each ingredient.
    """,
    dataset="""
        `load_wine`, `load_breast_cancer`, `load_digits`; validation `RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)`; 100 trees, `random_state=0`.
    """,
    purpose="""
        See the story of the module in numbers: each ingredient (averaging, decorrelation, random thresholds) adds a gain. And remember that small differences (tumors: 0.5 points between forest and Extra Trees) are noise, while the jump from the single tree is large.
    """,
    task=[
        "For each dataset and model (`tree` = `DecisionTreeClassifier`, `bagging` = `BaggingClassifier(DecisionTreeClassifier(), 100)`, `forest` = `RandomForestClassifier(100)`, `extra` = `ExtraTreesClassifier(100)`) compute with `cross_validate` `acc` (mean) and `t` (mean total fit time per fold, seconds). All with `random_state=0`, `n_jobs=-1` where it exists. Save `table` (index `(dataset, model)`).",
        "Save `gains` (`pd.DataFrame`, index: the datasets; columns `averaging` = bagging − tree, `random_features` = forest − bagging, `random_thresholds` = extra − forest).",
        "Save `order_holds` (True if, for **all** the datasets, `tree < bagging <= forest + 0.005` and `extra >= forest − 0.005`) and `dominant_gain` (name of the column of `gains` with the largest sum).",
    ],
    setup="""
        from sklearn.datasets import load_breast_cancer, load_digits, load_wine
        from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
        from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
        from sklearn.tree import DecisionTreeClassifier

        datasets = {"wine": load_wine(return_X_y=True), "tumors": load_breast_cancer(return_X_y=True), "digits": load_digits(return_X_y=True)}
        models = {
            "tree": lambda: DecisionTreeClassifier(random_state=0),
            "bagging": lambda: BaggingClassifier(DecisionTreeClassifier(), n_estimators=100, random_state=0, n_jobs=-1),
            "forest": lambda: RandomForestClassifier(100, random_state=0, n_jobs=-1),
            "extra": lambda: ExtraTreesClassifier(100, random_state=0, n_jobs=-1),
        }
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    """,
    starter=[
        C("""
        rows = {}
        for name, (X, y) in datasets.items():
            for model_name, create in models.items():
                ...
        table = pd.DataFrame(rows).T
        table.round(3)
        """),
        C("""
        gains = None
        order_holds = None
        dominant_gain = None
        """),
    ],
    checks="""
        check(table.shape == (12, 2) and list(table.columns) == ["acc", "t"], "12 rows (3 datasets x 4 models) and columns acc, t", "one row per (dataset, model) pair")
        check(list(gains.columns) == ["averaging", "random_features", "random_thresholds"] and list(gains.index) == list(datasets), "gains per dataset and per ingredient", "columns: averaging, random_features, random_thresholds")
        check((gains["averaging"] > 0.02).all() and dominant_gain == "averaging", "averaging the trees always gains over 2 points and it is the dominant gain", "bagging - tree")
        check(gains["random_features"].min() > -0.002 and gains["random_features"].max() < 0.03 and (gains["random_features"] >= 0).sum() >= 2, "random features gain little (0.5-2.5 points) and do not make things worse", "forest - bagging")
        check(order_holds is True and table.loc[("digits", "extra"), "acc"] > table.loc[("digits", "tree"), "acc"] + 0.1, "the order tree < bagging <= forest <= extra holds on all the datasets; on the digits the single tree is 10 points below Extra Trees", "compare the rows")
        check(table.loc[("wine", "tree"), "t"] < table.loc[("wine", "bagging"), "t"] / 3, "the single tree is much faster than the bagging (over three times)", "the cost grows with the number of trees")
    """,
    hint="""
        `r = cross_validate(create(), X, y, cv=cv)`; `acc = r["test_score"].mean()`; `t = r["fit_time"].mean()`. Gains: `table["acc"].unstack().loc[list(datasets)]` has the datasets as rows (in the original order) and the models as columns: `u["bagging"] - u["tree"]`, etc.
    """,
    solution=[
        C("""
        rows = {}
        for name, (X, y) in datasets.items():
            for model_name, create in models.items():
                r = cross_validate(create(), X, y, cv=cv)
                rows[(name, model_name)] = {"acc": r["test_score"].mean(), "t": r["fit_time"].mean()}
        table = pd.DataFrame(rows).T
        table["acc"].unstack()[["tree", "bagging", "forest", "extra"]].round(3)
        """),
        C("""
        u = table["acc"].unstack().loc[list(datasets)]                       # rows in the order of the datasets
        gains = pd.DataFrame({"averaging": u["bagging"] - u["tree"], "random_features": u["forest"] - u["bagging"], "random_thresholds": u["extra"] - u["forest"]})
        order_holds = bool(((u["tree"] < u["bagging"]) & (u["bagging"] <= u["forest"] + 0.005) & (u["extra"] >= u["forest"] - 0.005)).all())
        dominant_gain = gains.sum().idxmax()
        print(gains.round(3))
        print("order holds:", order_holds, "| dominant gain:", dominant_gain)
        """),
    ],
    learned="""
        - **Averaging is the huge gain:** from the single tree to the bagging you gain 3-9 points (wine +4, tumors +3, digits +9.5): each tree is very variable and the average removes almost all the noise. It is 70-90% of the total gain.
        - **Random features add little but cost nothing:** forest − bagging is +1.7 (wine), +0.6 (tumors), +2.4 (digits): more decorrelated trees, a slightly better average. With data that have few informative features the gain is larger.
        - **Extra Trees** (random thresholds on top of random features) often does a little better still (+0.5-0.8): more bias per tree, less variance for the average, and it is also faster to train (no search for the optimal threshold). On the tumors the difference with the forest is half a point: in the noise.
        - **The time** is proportional to the number of trees: a hundred trees cost a hundred times one tree (on the bagging with a single thread even more). The gain of 3-10 points is often worth the cost, but know that you are paying it.
    """,
    pitfall="""
        Reporting "the random forest is the best" from a table where the difference with Extra Trees or with bagging is half a point. With repeated validation and differences of 0.5-1 point we are in the noise: the solid story is the jump from the single tree.
    """,
    interview=(
        "What does the random forest add to bagging, and Extra Trees to the random forest?",
        """
        Bagging reduces the variance by averaging trees on different bootstraps; the random forest adds the random subsampling of the features at each split to decorrelate the trees; Extra Trees also picks the threshold at random (it does not use the bootstrap by default): it increases the bias of each tree and reduces the variance of the average further,
        and it is faster. In practice the big jump is from a single tree to an ensemble; the differences between the variants are a point or less and have to be checked on your own problem.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="when_the_forest_loses", kind="F", level=3, minutes=60, module=MODULE,
    title="When the forest loses: linear relationship, trend and cost",
    goal="""
        Identify the three situations in which a random forest is not the best choice, comparing it with linear/logistic regression: (1) a **linear** relationship with many features (the linear model wins and costs a hundredth), (2) **extrapolation** of a trend (the forest is flat), (3) **cost** of training, prediction and memory
        (the forest is hundreds of times heavier). And the situation in which it wins: nonlinear interactions.
    """,
    dataset="""
        Three problems (`problems()`, given): `linear` (3000 examples, 60 features of which 15 linear), `xor` (2000 examples, `y = (x0 > 0) XOR (x1 > 0)` with 8 noise features), `trend` (series with linear trend and seasonality, training on the first 300 months, test on the next 100). Models: logistic (or linear), random forest of 200 trees.
    """,
    purpose="""
        Develop the criterion for choosing: the forest is an excellent default but not a universal one. Measure for each problem accuracy (or RMSE), `fit` time, `predict` time and model size.
    """,
    task=[
        "For `linear` and `xor` (classification, 60/40 split with `random_state=0`) train `make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))` and `RandomForestClassifier(200, random_state=0, n_jobs=-1)`: save `cls_table` (index `(problem, model)`, columns `acc`, `t_fit`, `t_predict`, `mb`) with `mb = len(pickle.dumps(model)) / 1e6`.",
        "For `trend` (regression) train `LinearRegression` and `RandomForestRegressor(200, random_state=0, n_jobs=-1)` on the first 300 months with features `t`, `sin`, `cos`; save `trend_rmse` (dictionary model → RMSE on the next 100 months).",
        "Save `linear_wins_on` (sorted list of the problems where the linear/logistic model is better or on par, that is within 0.01 of accuracy or with a lower RMSE), `forest_wins_on` (the problems where the forest beats the logistic by over 0.1), `fit_ratio` (forest `t_fit` / logistic `t_fit` on `linear`) and `mb_ratio`.",
    ],
    setup="""
        import pickle
        import time
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.linear_model import LinearRegression, LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler


        def problems():
            rng = np.random.default_rng(0)
            Xl = rng.normal(size=(3000, 60))
            w = np.r_[rng.normal(size=15), np.zeros(45)]
            yl = rng.binomial(1, 1 / (1 + np.exp(-2.5 * Xl @ w / np.linalg.norm(w))))
            Xx = rng.normal(size=(2000, 10))
            yx = ((Xx[:, 0] > 0) ^ (Xx[:, 1] > 0)).astype(int)
            yx = np.where(rng.random(2000) < 0.05, 1 - yx, yx)
            t = np.arange(400)
            y_t = 5 + 0.05 * t + 2 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, 400)
            F = np.c_[t, np.sin(2 * np.pi * t / 12), np.cos(2 * np.pi * t / 12)]
            return {"linear": (Xl, yl), "xor": (Xx, yx)}, (F, y_t)


        classification, (F_trend, y_trend) = problems()
    """,
    starter=[
        C("""
        rows = {}
        for name, (X, y) in classification.items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
            models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "forest": RandomForestClassifier(200, random_state=0, n_jobs=-1)}
            for model_name, m in models.items():
                ...
        cls_table = pd.DataFrame(rows).T
        cls_table.round(3)
        """),
        C("""
        trend_rmse = {}
        """),
        C("""
        linear_wins_on = None
        forest_wins_on = None
        fit_ratio = None
        mb_ratio = None
        """),
    ],
    checks="""
        check(cls_table.shape == (4, 4) and list(cls_table.columns) == ["acc", "t_fit", "t_predict", "mb"], "4 rows (2 problems x 2 models) and columns acc, t_fit, t_predict, mb", "one row per (problem, model) pair")
        check(set(trend_rmse) == {"linear", "forest"} and trend_rmse["linear"] < 1 and trend_rmse["forest"] > 2.5, "trend: the linear one is off by less than 1, the forest by more than 2.5", "test on months 300-399")
        check(linear_wins_on == ["linear", "trend"] and forest_wins_on == ["xor"], "the linear model wins (or ties) on 'linear' and 'trend'; the forest beats the logistic by over 0.1 on 'xor'", "compare the accuracies and the RMSEs")
        check(cls_table.loc[("xor", "logistic"), "acc"] < 0.6 and cls_table.loc[("xor", "forest"), "acc"] > 0.9, "on the XOR the logistic is at chance (below 0.6), the forest over 0.9", "the XOR is not linear")
        check(fit_ratio > 20 and mb_ratio > 100, "the forest costs over 20 times in training and over 100 times in memory compared with the logistic", "t_fit and mb")
    """,
    hint="""
        `t0 = time.time(); m.fit(X_tr, y_tr); t_fit = time.time() - t0`; `t0 = time.time(); acc = m.score(X_te, y_te); t_predict = time.time() - t0`; `rows[(name, model_name)] = {...}`. Trend: `tr = F_trend[:, 0] < 300`; `LinearRegression().fit(F_trend[tr], y_trend[tr])`; RMSE on `~tr`.
    """,
    solution=[
        C("""
        rows = {}
        for name, (X, y) in classification.items():
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)
            models = {"logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), "forest": RandomForestClassifier(200, random_state=0, n_jobs=-1)}
            for model_name, m in models.items():
                t0 = time.time(); m.fit(X_tr, y_tr); t_fit = time.time() - t0
                t0 = time.time(); acc = m.score(X_te, y_te); t_predict = time.time() - t0
                rows[(name, model_name)] = {"acc": acc, "t_fit": t_fit, "t_predict": t_predict, "mb": len(pickle.dumps(m)) / 1e6}
        cls_table = pd.DataFrame(rows).T
        cls_table.round(3)
        """),
        C("""
        tr = F_trend[:, 0] < 300
        trend_rmse = {}
        for name, m in {"linear": LinearRegression(), "forest": RandomForestRegressor(200, random_state=0, n_jobs=-1)}.items():
            m.fit(F_trend[tr], y_trend[tr])
            trend_rmse[name] = float(np.sqrt(np.mean((m.predict(F_trend[~tr]) - y_trend[~tr]) ** 2)))
        print({k: round(v, 2) for k, v in trend_rmse.items()})
        """),
        C("""
        acc = cls_table["acc"].unstack()
        linear_wins_on = sorted([p for p in acc.index if acc.loc[p, "logistic"] >= acc.loc[p, "forest"] - 0.01] + (["trend"] if trend_rmse["linear"] < trend_rmse["forest"] else []))
        forest_wins_on = sorted(p for p in acc.index if acc.loc[p, "forest"] > acc.loc[p, "logistic"] + 0.1)
        fit_ratio = cls_table.loc[("linear", "forest"), "t_fit"] / cls_table.loc[("linear", "logistic"), "t_fit"]
        mb_ratio = cls_table.loc[("linear", "forest"), "mb"] / cls_table.loc[("linear", "logistic"), "mb"]
        print("the linear model wins (or ties) on:", linear_wins_on, "| the forest wins on:", forest_wins_on)
        print(f"cost on 'linear': training {fit_ratio:.0f} times, memory {mb_ratio:.0f} times")
        """),
    ],
    learned="""
        - **Linear relationship with many features:** the logistic (0.80) beats the forest (0.76) and costs a hundredth of the time and memory. When the true link is a linear combination of many variables, a linear model estimates it with few parameters, while trees have to approximate it with many thresholds.
        - **Interactions (XOR):** it is the ground of the forest: 0.93 against 0.52 for the logistic. No linear combination separates the four quadrants; trees find the two thresholds and their interaction without you specifying it.
        - **Extrapolation:** on a trend the forest is off by more than seven times as much as the line (3.1 against 0.43): beyond the range seen in training it predicts a constant value. You need a model with structure (linear, ARIMA) or a hybrid (M7 C1).
        - **Cost:** the forest takes about 0.7 seconds of training and 5-7 MB (200 trees on 1800 rows: they grow with the data, as in the 244 MB of B2), the logistic 14 milliseconds and 3 KB. To serve the model on a device, with low latency or with many models in parallel, the cost matters as much as the accuracy.
        - These are not arguments against the forest (it is the best first attempt on tabular data): they are the **scenarios in which to also try a linear model** and compare with the same validation.
    """,
    pitfall="""
        Using a forest by default without ever trying a linear baseline. On linear relationships and on trends the forest can lose, and the baseline tells you anyway how much the added complexity is worth.
    """,
    interview=(
        "In which cases would you not use a random forest?",
        """
        When you need to extrapolate (trends, nonstationary series), when the relationship is linear with many features (a regularized linear model is better and stable), when you need a small, fast model (devices, latency) or simple explanations, and with very sparse high-dimensional data (text). I always compare it with a linear baseline
        and with a boosting, using the same validation.
        """),
)

EXERCISES = [F1, F2]
