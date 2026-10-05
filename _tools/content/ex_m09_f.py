"""Module 9 - type F exercises: compare."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_09_xgboost"

# ---------------------------------------------------------------------------------------- F1

F1 = dict(
    id="F1", slug="five_tree_implementations_compared", kind="F", level=2, minutes=50, module=MODULE,
    title="Five implementations compared: speed and quality as the data grows",
    goal="""
        Compare `GradientBoostingClassifier` (classic scikit-learn), `HistGradientBoostingClassifier`, `XGBClassifier` with `tree_method="hist"` and `"exact"` and `RandomForestClassifier` on three training set sizes (2000, 8000, 32000 examples): how much the time grows, what the AUC is worth and what matters more (algorithm or amount of data).
    """,
    dataset="""
        `make_classification(40000, 20 features, 8 informative, flip_y=0.05)`: the first `n` examples are the training set, the last 8000 the test set. All the models have 100 trees of depth 4 (the forest has free depth).
    """,
    purpose="""
        Choose an implementation with judgment: the classic scikit-learn boosting is linear in `n × features × thresholds` per tree and does not scale; the histogram versions (Hist, XGBoost `hist`) quantize the features into 256 bins and grow almost linearly; the quality changes little, the time enormously.
    """,
    task=[
        "Build `table` (`pd.DataFrame` with a multi-index `(method, n)`, columns `seconds` (`fit` time with `time.perf_counter`) and `auc` on the test set) for the five methods and the three `n`.",
        "Save `time_growth` (`pd.Series`, per method: `seconds(n=32000) / seconds(n=2000)`) and `gb_speedup` (time of `GradientBoostingClassifier` divided by the time of `XGBClassifier hist`, at `n = 32000`).",
        "Save `auc_spread_32000` (maximum AUC minus minimum among the five methods at `n = 32000`) and `data_gain` (`pd.Series`, per method: `auc(32000) − auc(2000)`).",
        "Save `scalability_conclusion` (`\"classic gb does not scale\"` or `\"all scale equally\"`) and `matters_more` (`\"algorithm\"` or `\"data\"`: what moves the AUC more, comparing `auc_spread_32000` with the mean of `data_gain`).",
    ],
    setup="""
        import time

        import xgboost as xgb
        from sklearn.datasets import make_classification
        from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        Xb, yb = make_classification(40000, n_features=20, n_informative=8, class_sep=0.7, flip_y=0.05, random_state=1)
        X_te, y_te = Xb[-8000:], yb[-8000:]
        sizes = [2000, 8000, 32000]
        models = {
            "GradientBoosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=0),
            "HistGradientBoosting": lambda: HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=False, random_state=0),
            "XGBoost hist": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=4, tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=0),
            "XGBoost exact": lambda: xgb.XGBClassifier(n_estimators=100, max_depth=4, tree_method="exact", eval_metric="logloss", n_jobs=-1, random_state=0),
            "RandomForest": lambda: RandomForestClassifier(100, n_jobs=-1, random_state=0),
        }
    """,
    starter=[
        C("""
        index = pd.MultiIndex.from_product([list(models), sizes], names=["method", "n"])
        table = pd.DataFrame(index=index, columns=["seconds", "auc"], dtype=float)
        for name, create in models.items():
            for n in sizes:
                ...
        table.round(3)
        """),
        C("""
        time_growth = None
        gb_speedup = None
        """),
        C("""
        auc_spread_32000 = None
        data_gain = None
        scalability_conclusion = None
        matters_more = None
        """),
    ],
    checks="""
        check(table.notna().all().all() and table.shape == (15, 2) and (table["auc"] > 0.88).all(), "fifteen complete measurements, AUC always over 0.88", "time with time.perf_counter around fit")
        check(time_growth is not None and time_growth["GradientBoosting"] > 10 and time_growth["XGBoost hist"] < 6 and time_growth["HistGradientBoosting"] < 6, "the classic boosting grows by over 10 times from 2000 to 32000 examples, the histogram versions by less than 6", "seconds(32000) / seconds(2000)")
        check(gb_speedup is not None and gb_speedup > 15, "at 32000 examples the classic boosting is over 15 times slower than XGBoost hist", "GB time / XGBoost hist time")
        check(auc_spread_32000 is not None and auc_spread_32000 < 0.02 and (data_gain > 0.02).all(), "at 32000 examples the five methods differ by less than 2 points of AUC, while going from 2000 to 32000 examples gains over 2 points for each", "maximum AUC - minimum AUC; auc(32000) - auc(2000)")
        check(scalability_conclusion == "classic gb does not scale" and matters_more == "data", "the classic boosting does not scale and the factor that matters most is the amount of data", "compare 0.011 with 0.03")
    """,
    hint="""
        `t0 = time.perf_counter(); m = create().fit(X_tr, y_tr); seconds = time.perf_counter() - t0`, with `X_tr, y_tr = Xb[:n], yb[:n]`. Then `table.loc[(name, n)] = [seconds, roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])]`. The series per method: `table["seconds"].unstack()`.
    """,
    solution=[
        C("""
        index = pd.MultiIndex.from_product([list(models), sizes], names=["method", "n"])
        table = pd.DataFrame(index=index, columns=["seconds", "auc"], dtype=float)
        for name, create in models.items():
            for n in sizes:
                X_tr, y_tr = Xb[:n], yb[:n]
                t0 = time.perf_counter()
                m = create().fit(X_tr, y_tr)
                seconds = time.perf_counter() - t0
                table.loc[(name, n)] = [seconds, roc_auc_score(y_te, m.predict_proba(X_te)[:, 1])]
        table.round(3)
        """),
        C("""
        seconds = table["seconds"].unstack().loc[list(models)]
        auc = table["auc"].unstack().loc[list(models)]
        time_growth = seconds[32000] / seconds[2000]
        gb_speedup = float(seconds.loc["GradientBoosting", 32000] / seconds.loc["XGBoost hist", 32000])
        print(seconds.round(2).to_string())
        print("\\ntime growth from 2000 to 32000 examples:", time_growth.round(1).to_dict())
        print(f"at 32000 examples GradientBoosting is {gb_speedup:.0f} times slower than XGBoost hist")
        """),
        C("""
        auc_spread_32000 = float(auc[32000].max() - auc[32000].min())
        data_gain = auc[32000] - auc[2000]
        scalability_conclusion = "classic gb does not scale"
        matters_more = "data" if data_gain.mean() > auc_spread_32000 else "algorithm"
        print(auc.round(3).to_string())
        print(f"\\ngap between methods at 32000 examples: {auc_spread_32000:.3f} | mean gain going from 2000 to 32000 examples: {data_gain.mean():.3f} -> matters more: {matters_more}")
        """),
    ],
    learned="""
        - **The classic scikit-learn boosting does not scale:** 1.6 s at 2000 examples, 30 s at 32000 (19 times, for 16 times the data) against 0.3-0.4 s for XGBoost `hist` and about 1 s for `HistGradientBoosting`. At 32000 examples it is 70 times slower than XGBoost. The difference is the threshold search algorithm: exact (it sorts the features at every tree) against histogram (256 bins built once).
        - **`XGBoost exact` also scales badly:** from 0.3 s to 2.8 s (about 8 times). The histogram method is the default recommended by XGBoost 2 onwards, and for large data it is the only way.
        - **The quality is almost the same:** at 32000 examples the five methods are within 1 point of AUC (0.941-0.952), while going from 2000 to 32000 examples is worth 3-4 points for all of them. With enough data the choice of the algorithm moves less than the amount of data or the feature engineering.
        - **Practical choice:** `HistGradientBoosting` if you want to stay in scikit-learn (no dependencies, it handles NaN and categoricals); `XGBoost` (or LightGBM) if you need refined early stopping, monotonic constraints, GPU, `pred_contribs`; `GradientBoostingClassifier` only for teaching experiments or very small datasets.
    """,
    pitfall="""
        Comparing algorithms on a small dataset and concluding "they are all the same in speed": the differences of scale emerge only as `n` grows. Always measure at several sizes.
    """,
    interview=(
        "Why is histogram boosting much faster than the classic one?",
        """
        It quantizes each feature into at most 256 bins once and then, for each tree, searches the threshold only among the bins by building histograms of gradients: the cost per tree is linear in n and independent of the number of distinct values. The exact method has to sort or scan all the values at every tree. The quality changes very little because 256
        candidate thresholds are almost always enough.
        """),
)

# ---------------------------------------------------------------------------------------- F2

F2 = dict(
    id="F2", slug="when_boosting_does_not_win", kind="F", level=3, minutes=60, module=MODULE,
    title="When boosting does not win: extrapolation, noisy linear signal, interactions",
    goal="""
        Compare Ridge, RandomForest and XGBoost on **five regimes** built to show the strengths and weaknesses of trees: a trend to extrapolate, a linear signal with a lot of noise (with `n = 150` and `n = 5000`), a signal made of interactions (with `n = 600` and `n = 5000`), and the hybrid remedy "Ridge + XGBoost on the residuals".
    """,
    dataset="""
        Five synthetic data generators in `regimes` (given): `trend` (`y = 2t + sin(3t)` with increasing `t`, the test set is the last 30% of the time, so outside the training range), `noisy linear` (30 features of which 6 with nonzero coefficients, noise `σ = 2`) and `interactions` (`y = ±x₁` with sign decided by `x₀`, plus `1.5·[x₂x₃ > 0]`).
    """,
    purpose="""
        Know in advance where a tree model (forest or boosting) is a bad choice: **outside the training range it does not extrapolate** (it predicts a constant), with a noisy linear signal and few data it **wastes** its flexibility, while with nonlinear interactions it wins by a lot. And learn the remedy: the linear model captures the global structure, the trees the rest.
    """,
    task=[
        "Build `table` (test RMSE; index: the five regimes, columns `ridge`, `rf`, `xgb`) with `RidgeCV(alphas=np.logspace(-2, 3, 20))`, `RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1)` and `XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0, n_jobs=-1)`. Save `winner` (`pd.Series`: method with the lowest RMSE per regime) and `xgb_ridge_ratio` (`xgb / ridge` per regime).",
        "**Why the trend?** Redo the `trend` regime with a **random** 70/30 split (interpolation): save `random_trend_table` (`pd.Series` with the three RMSEs) and `random_trend_ratio` (`xgb / ridge`). The problem is the extrapolation, not the function.",
        "**Hybrid remedy:** `hybrid = ridge + XGBoost on the residuals` (train Ridge, then `XGBRegressor(...)` on the residuals `y − ridge.predict(X)`; prediction = sum). Save `hybrid_rmse` (`pd.Series`, per regime).",
        "Save `rules` (dictionary with three entries: `trend`, `noisy linear`, `interactions` → a short sentence: which model you choose and why).",
    ],
    setup="""
        import xgboost as xgb
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import RidgeCV
        from sklearn.metrics import mean_squared_error


        def trend(seed=0):
            rng = np.random.default_rng(seed)
            n = 1200
            t = np.linspace(0, 10, n)
            X = np.c_[t, np.sin(3 * t)]
            y = 2 * t + np.sin(3 * t) + rng.normal(0, 0.5, n)
            c = int(0.7 * n)
            return X[:c], y[:c], X[c:], y[c:]


        def noisy_linear(n, seed):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(n + 2000, 30))
            beta = np.zeros(30)
            beta[:6] = [1, -1, 0.8, -0.8, 0.5, 0.5]
            y = X @ beta + rng.normal(0, 2, n + 2000)
            return X[:n], y[:n], X[n:], y[n:]


        def interactions(n, seed):
            rng = np.random.default_rng(seed)
            X = rng.normal(size=(n + 2000, 8))
            y = np.where(X[:, 0] > 0, X[:, 1], -X[:, 1]) * 2 + (X[:, 2] * X[:, 3] > 0) * 1.5 + rng.normal(0, 0.5, n + 2000)
            return X[:n], y[:n], X[n:], y[n:]


        regimes = {
            "trend (extrapolation)": trend(),
            "noisy linear, n=150": noisy_linear(150, 1),
            "noisy linear, n=5000": noisy_linear(5000, 2),
            "interactions, n=600": interactions(600, 3),
            "interactions, n=5000": interactions(5000, 4),
        }
        models = {
            "ridge": lambda: RidgeCV(alphas=np.logspace(-2, 3, 20)),
            "rf": lambda: RandomForestRegressor(300, min_samples_leaf=3, random_state=0, n_jobs=-1),
            "xgb": lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=0, n_jobs=-1),
        }
        rmse = lambda y, p: mean_squared_error(y, p, squared=False)
    """,
    starter=[
        C("""
        table = pd.DataFrame(index=list(regimes), columns=list(models), dtype=float)
        for regime, (X_tr, y_tr, X_te, y_te) in regimes.items():
            ...
        table.round(3)
        """),
        C("""
        winner = None
        xgb_ridge_ratio = None
        """),
        C("""
        random_trend_table = None
        random_trend_ratio = None
        """),
        C("""
        hybrid_rmse = pd.Series(dtype=float)
        rules = {}
        """),
    ],
    checks="""
        check(table.notna().all().all() and table.shape == (5, 3), "complete 5 x 3 table", "RMSE per regime and method")
        check(winner is not None and winner["trend (extrapolation)"] == "ridge" and winner["noisy linear, n=150"] == "ridge" and winner["noisy linear, n=5000"] == "ridge", "Ridge wins on the trend and on the noisy linear signal (even with 5000 examples)", "idxmin per row")
        check(winner is not None and winner["interactions, n=600"] in ("rf", "xgb") and winner["interactions, n=5000"] in ("rf", "xgb") and table.loc["interactions, n=5000", "ridge"] > 2.5 * table.loc["interactions, n=5000", "xgb"], "with interactions the trees win, and Ridge has an RMSE over 2.5 times larger at n=5000", "ridge does not see the interactions")
        check(xgb_ridge_ratio is not None and xgb_ridge_ratio["trend (extrapolation)"] > 4 and 1.0 < xgb_ridge_ratio["noisy linear, n=5000"] < 1.15 and 1.02 < xgb_ridge_ratio["noisy linear, n=150"], "on the trend XGBoost has an RMSE over 4 times larger; on the noisy linear one it is always worse (slightly at n=5000, less than 15%)", "xgb / ridge")
        check(random_trend_table is not None and random_trend_ratio < 1.2 and random_trend_table.min() < 0.6, "with a random split the three methods are close (xgb / ridge below 1.2): the disaster was the extrapolation", "random 70/30 split of the trend regime")
        check(len(hybrid_rmse) == 5 and all(hybrid_rmse[r] <= 1.1 * table.loc[r].min() for r in table.index), "the hybrid is within 10% of the best in all the regimes", "ridge + xgb on the residuals")
        check(hybrid_rmse["trend (extrapolation)"] < 0.7 and hybrid_rmse["interactions, n=5000"] < 0.85, "the hybrid solves the trend (below 0.7) and the interactions at n=5000 (below 0.85)", "the trees correct only the residuals")
        check(set(rules) == {"trend", "noisy linear", "interactions"} and all(isinstance(v, str) and len(v) > 15 for v in rules.values()), "three rules, one per regime", "one sentence per regime")
    """,
    hint="""
        `models[name]().fit(X_tr, y_tr).predict(X_te)` and `rmse(y_te, p)`. Random split for the trend: `X, y = np.r_[X_tr, X_te], np.r_[y_tr, y_te]` then `train_test_split(X, y, test_size=0.3, random_state=0)`. Hybrid: `r = RidgeCV(...).fit(X_tr, y_tr)`, `g = xgb.XGBRegressor(...).fit(X_tr, y_tr - r.predict(X_tr))`, prediction `r.predict(X_te) + g.predict(X_te)`.
    """,
    solution=[
        C("""
        table = pd.DataFrame(index=list(regimes), columns=list(models), dtype=float)
        for regime, (X_tr, y_tr, X_te, y_te) in regimes.items():
            for name, create in models.items():
                table.loc[regime, name] = rmse(y_te, create().fit(X_tr, y_tr).predict(X_te))
        winner = table.idxmin(axis=1)
        xgb_ridge_ratio = table["xgb"] / table["ridge"]
        print(table.round(3).to_string())
        print("\\nwinner:", winner.to_dict())
        print("xgb / ridge:", xgb_ridge_ratio.round(2).to_dict())
        """),
        C("""
        from sklearn.model_selection import train_test_split

        X_tr, y_tr, X_te, y_te = regimes["trend (extrapolation)"]
        X_t, y_t = np.r_[X_tr, X_te], np.r_[y_tr, y_te]
        A, B, ya, yb = train_test_split(X_t, y_t, test_size=0.3, random_state=0)
        random_trend_table = pd.Series({name: rmse(yb, create().fit(A, ya).predict(B)) for name, create in models.items()})
        random_trend_ratio = float(random_trend_table["xgb"] / random_trend_table["ridge"])
        print("trend with a random split:", random_trend_table.round(3).to_dict(), "| xgb / ridge", round(random_trend_ratio, 2))
        """),
        C("""
        def hybrid(X_tr, y_tr, X_te):
            linear = models["ridge"]().fit(X_tr, y_tr)
            rest = models["xgb"]().fit(X_tr, y_tr - linear.predict(X_tr))
            return linear.predict(X_te) + rest.predict(X_te)


        hybrid_rmse = pd.Series({regime: rmse(y_te, hybrid(X_tr, y_tr, X_te)) for regime, (X_tr, y_tr, X_te, y_te) in regimes.items()})
        print(pd.DataFrame({"best single": table.min(axis=1), "hybrid": hybrid_rmse}).round(3).to_string())
        """),
        C("""
        rules = {
            "trend": "Linear model (or hybrid): trees do not extrapolate outside the training range and predict a constant.",
            "noisy linear": "Ridge/regularized: with a linear signal and a lot of noise, trees spend useless flexibility and overfit.",
            "interactions": "Boosting or forest: they capture the nonlinear interactions that a linear model cannot represent.",
        }
        rules
        """),
    ],
    learned="""
        - **Trees do not extrapolate:** on the trend outside the training range XGBoost has an RMSE of 2.98 against 0.48 for Ridge (6 times), and the forest 3.30. The model predicts the mean of the outermost leaves, that is a constant. With a random split (interpolation) the difference disappears (0.54 against 0.51): the problem is not the function but being out of range. On data with a time trend, prices or growth, a tree alone is not enough.
        - **With a noisy linear signal Ridge always wins,** even at n=5000 (1.98 against 2.04, 3%): with 150 examples the gap rises to 10% (2.24 against 2.47). Trees fragment the signal into thresholds and pay for its noise. Before installing XGBoost, a regularized linear baseline is a 10-second test.
        - **With interactions the picture flips:** Ridge has RMSE 2.15 at n=5000 against 0.71 for XGBoost (3 times worse) because it cannot represent `±x₁` as a function of the sign of `x₀`. The advantage grows with `n`: at n=600 XGBoost is 1.16 and Ridge 2.13 (1.8 times); at n=5000 the ratio reaches 3.
        - **The hybrid is safe:** a linear model captures the global structure (and the extrapolation), boosting on the residuals the nonlinear part. In all five regimes it is within 6% of the best single model (trend: 0.51 against 0.48; linear: 2.28 against 2.24 and 2.01 against 1.98) and with the interactions it is even better (1.08 against 1.16; 0.65 against 0.71). It is the configuration to try when you do not know which regime your problem belongs to.
    """,
    pitfall="""
        Applying boosting "because it wins on Kaggle" to a problem with a time trend or one that has to be extrapolated: outside the range it sees a constant. And comparing only with other trees: the regularized linear baseline always goes in the table.
    """,
    interview=(
        "When would you choose a linear model instead of boosting?",
        """
        With a mostly linear and noisy signal, few data, the need to extrapolate outside the training range (trends, growth), or requirements of interpretability and stability. Boosting wins with interactions and nonlinearity and with enough data. In any case I always put a regularized linear baseline in the table and also evaluate the hybrid
        (linear plus boosting on the residuals).
        """),
)

EXERCISES = [F1, F2]
