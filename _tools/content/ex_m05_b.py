"""Module 5 - type B exercises: implement with scikit-learn."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, M  # noqa: E402

MODULE = "module_05_knn"

# ---------------------------------------------------------------------------------------- B1

B1 = dict(
    id="B1", slug="knn_on_mixed_data", kind="B", level=2, minutes=50, module=MODULE,
    title="KNN on mixed data: which columns to put in the distance",
    goal="""
        Build a KNN on mixed data (scaled numerics + one-hot categoricals) with a `ColumnTransformer`, measure in cross-validation what happens when adding an **informative** category (`contract_type`) and a **noisy** one
        (`city`), choose the hyperparameters with `GridSearchCV` and compare the result with a logistic regression.
    """,
    dataset="""
        `ds.loans(with_missing=False)` with `ratio = amount/income` (1500 applications, 19% defaulters): stratified 70/30 split. Numerics: `ratio, age, job_tenure, income, amount`. Categoricals: `contract_type` (4 levels, informative)
        and `city` (12 levels, unrelated to the outcome).
    """,
    purpose="""
        Learn to **decide what goes into the distance**: every added column weighs in the measure of "nearness", so a useless feature is a cost, not a neutral. You will also see an honest result: on this problem KNN does not beat a linear model.
    """,
    task=[
        "Write `build(numerics, categoricals, k=25)`: `Pipeline` of a `ColumnTransformer` (`StandardScaler` on the numerics, `OneHotEncoder(handle_unknown=\"ignore\")` on the categoricals; `sparse_threshold=0`, and if a list is empty the block is not there) "
        "and `KNeighborsClassifier(k, weights=\"distance\")`.",
        "With `StratifiedKFold(5, shuffle=True, random_state=0)` on the training set compute the mean AUC of four variants and save it in `cv_auc` (`pd.Series`, index: `numerics`, `+ contract`, `+ city`, `+ both`).",
        "Save `harmful_variable` (`\"contract\"` or `\"city\"`: the one that, added to the numerics, **lowers** the AUC the most) and `city_loss = cv_auc['numerics'] - cv_auc['+ city']`.",
        "`GridSearchCV` (same `cv`, `scoring=\"roc_auc\"`) on the `numerics` variant with `n_neighbors` in `[5, 15, 25, 45, 75, 125]`, `weights` in `[\"uniform\", \"distance\"]` and `p` in `[1, 2]`: save `best_params` (dictionary of the names without prefix) and `knn_test_auc` (test AUC of the best model). "
        "Then compare with `lr_test_auc`: a `LogisticRegression(max_iter=1000)` with the same preprocessing of the numerics and `contract_type`.",
    ],
    setup="""
        from sklearn.compose import ColumnTransformer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        df = ds.loans(with_missing=False)
        df["ratio"] = df["amount"] / df["income"]
        NUM = ["ratio", "age", "job_tenure", "income", "amount"]
        X, y = df[NUM + ["contract_type", "city"]], df["defaulted"]
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
    """,
    starter=[
        C("""
        def build(numerics, categoricals, k=25):
            ...
        """),
        C("""
        cv_auc = pd.Series(dtype=float)
        harmful_variable = None
        city_loss = None
        cv_auc
        """),
        C("""
        best_params = None
        knn_test_auc = None
        lr_test_auc = None
        """),
    ],
    checks="""
        check(isinstance(build(NUM, ["city"]), Pipeline) and build(NUM, []).fit(X_tr, y_tr).predict(X_te).shape == (len(X_te),), "build returns a Pipeline that also works without categoricals", "the 'cat' block must not be created if the list is empty")
        check(list(cv_auc.index) == ["numerics", "+ contract", "+ city", "+ both"] and 0.72 < cv_auc["numerics"] < 0.82, "four variants; AUC of the numerics between 0.72 and 0.82", "index: numerics, + contract, + city, + both")
        check(harmful_variable == "city" and close(city_loss, cv_auc["numerics"] - cv_auc["+ city"]) and city_loss > 0.005, "the city (noise) lowers the AUC by over 0.005", "compare + contract and + city")
        check(abs(cv_auc["+ contract"] - cv_auc["numerics"]) < 0.02 and cv_auc["+ both"] < cv_auc["+ contract"], "the contract does not help the KNN (difference below 0.02) and adding the city too makes it worse", "read the series cv_auc")
        check(set(best_params) == {"n_neighbors", "weights", "p"} and best_params["n_neighbors"] >= 25, "the search prefers a large k (25 or more): the signal is weak and noisy", "keys without the prefix 'kneighborsclassifier__'")
        check(knn_test_auc > 0.7 and lr_test_auc > knn_test_auc - 0.005, "the logistic is not beaten by the optimized KNN (or nearly tied)", "compare knn_test_auc and lr_test_auc on the test set")
    """,
    hint="""
        `blocks = []; if numerics: blocks.append(("num", StandardScaler(), numerics)); if categoricals: blocks.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricals))`.
        Search: `GridSearchCV(build(NUM, []), {"kneighborsclassifier__n_neighbors": [...], "kneighborsclassifier__weights": [...], "kneighborsclassifier__p": [1, 2]}, ...)`; then `roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])`.
        The columns not needed by a variant stay in `X_tr`: the `ColumnTransformer` with `remainder="drop"` discards them.
    """,
    solution=[
        C("""
        def build(numerics, categoricals, k=25):
            blocks = []
            if numerics:
                blocks.append(("num", StandardScaler(), numerics))
            if categoricals:
                blocks.append(("cat", OneHotEncoder(handle_unknown="ignore"), categoricals))
            return make_pipeline(ColumnTransformer(blocks, sparse_threshold=0), KNeighborsClassifier(k, weights="distance"))
        """),
        C("""
        variants = {"numerics": [], "+ contract": ["contract_type"], "+ city": ["city"], "+ both": ["contract_type", "city"]}
        cv_auc = pd.Series({name: cross_val_score(build(NUM, cat), X_tr, y_tr, cv=cv, scoring="roc_auc").mean() for name, cat in variants.items()})
        delta = {"contract": cv_auc["+ contract"] - cv_auc["numerics"], "city": cv_auc["+ city"] - cv_auc["numerics"]}
        harmful_variable = min(delta, key=delta.get)
        city_loss = cv_auc["numerics"] - cv_auc["+ city"]
        print(cv_auc.round(4).to_string())
        print(f"change of AUC by adding contract_type: {delta['contract']:+.4f} | by adding city: {delta['city']:+.4f} -> harmful: {harmful_variable}")
        """),
        C("""
        grid = {"kneighborsclassifier__n_neighbors": [5, 15, 25, 45, 75, 125], "kneighborsclassifier__weights": ["uniform", "distance"], "kneighborsclassifier__p": [1, 2]}
        search = GridSearchCV(build(NUM, []), grid, cv=cv, scoring="roc_auc").fit(X_tr, y_tr)
        best_params = {k.replace("kneighborsclassifier__", ""): v for k, v in search.best_params_.items()}
        knn_test_auc = roc_auc_score(y_te, search.predict_proba(X_te)[:, 1])
        lr = make_pipeline(ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), ["contract_type"])], sparse_threshold=0), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
        lr_test_auc = roc_auc_score(y_te, lr.predict_proba(X_te)[:, 1])
        print(f"best parameters: {best_params} (cv AUC {search.best_score_:.4f})")
        print(f"test AUC: optimized KNN {knn_test_auc:.4f} | logistic {lr_test_auc:.4f}")
        """),
    ],
    learned="""
        - **Noisy features cost**: adding `city` (12 dummies unrelated to the outcome) lowers the AUC of the KNN, while in a logistic the same variable is simply ignored (coefficients close to 0). In KNN every column
          enters the distance with the same weight, the information is not enough to compensate.
        - Even an **informative** category like `contract_type` does not help the KNN here: the distance between two 0/1 dummies is 0 or 1.4, an enormous difference compared to the standardized numerics, and the signal of `unemployed` (7% of the cases) gets lost in the noise of
          all the other coordinates. A logistic uses the same signal well (M4 B1).
        - The optimized KNN chooses a large `k`: when the signal is weak and noisy you need many neighbors to average (with small `k` the estimate is too variable). Note that the best (`k = 125`) is **the edge of the grid**: it is the signal to extend it and retry with even larger `k`.
        - Honest comparison: the optimized KNN (0.776) and the logistic (0.776) tie. A simpler model, faster at prediction and more interpretable wins at equal AUC: it is the default choice until KNN proves otherwise.
    """,
    pitfall="""
        Treating dummies as if they were "free" in the distance. A category with 12 levels adds 12 0/1 columns that dominate the standardized numerics when the levels differ: a KNN on data with many categories requires
        explicit choices (weight of the block, Gower distance) or a different representation (target encoding, embedding).
    """,
    interview=(
        "How do you treat categorical variables in a distance-based model?",
        """
        One-hot with `handle_unknown` is the starting point, but with many categories or few informative ones it adds noise to the distance: I check with an ablation in cross-validation, consider weighting the block (multiplying the dummies by a factor),
        reducing the levels or using a distance designed for mixed data (Gower). If the categories matter a lot, a tree model handles them better and does not require these choices.
        """),
)

# ---------------------------------------------------------------------------------------- B2

B2 = dict(
    id="B2", slug="knn_regressor_and_geography", kind="B", level=2, minutes=55, module=MODULE,
    title="The price depends on where you are: KNN regressor on latitude and longitude",
    goal="""
        Compare a linear regression with several KNN regressors on house prices in California, show that the KNN with **only latitude and longitude** beats the linear model with all the features, and
        that a distance with **per-feature weights** (small weight on income) beats both even more. The weight is chosen on a validation set, not on the test set.
    """,
    dataset="""
        `ds.california()` (20,640 districts, 8 features: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`; target: median price in hundreds of thousands of dollars): 75/25 split with `random_state=0`.
        Error: RMSE (`mean_squared_error(..., squared=False)`).
    """,
    purpose="""
        Understand that KNN is a model with a **proximity structure** that must be chosen: for house prices geographic proximity is almost everything. And see the effect of choosing the weights of the features in the distance, which is
        the simplest way to "teach" KNN which columns matter.
    """,
    task=[
        "Compute the test RMSE of five models and save it in `rmse` (dictionary): `linear` (`LinearRegression`, 8 features), `raw knn` (`KNeighborsRegressor(15)`, 8 features **without** scaler), `scaled knn` (with `StandardScaler`, 8 features), `geo knn` "
        "(`KNeighborsRegressor(15)` on `Latitude` and `Longitude` only), `geo + income knn` (the same two plus `MedInc`, scaled).",
        "`GridSearchCV` (5 folds, `scoring=\"neg_root_mean_squared_error\"`) on `Latitude` and `Longitude` with `n_neighbors` in `[3, 5, 10, 15, 25, 40, 60]` and `weights` in `[\"uniform\", \"distance\"]`: save `geo_params` and `best_geo_rmse` (test RMSE).",
        "**Per-feature weights.** Standardize `Latitude`, `Longitude`, `MedInc` (scaler learned on the training set) and multiply the income column by a weight `w`. Choose `w` among `[0, 0.05, 0.1, 0.2, 0.35, 0.5, 1]` with a validation set "
        "(20% of the training set, `train_test_split(random_state=1)`), with `KNeighborsRegressor(10, weights=\"distance\")`: save `rmse_per_w` (`pd.Series` on the validation set) and `best_w`.",
        "With that `w`, retrain on the **whole** training set and save `weighted_rmse` (test RMSE).",
    ],
    setup="""
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error
        from sklearn.model_selection import GridSearchCV, train_test_split
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = ds.california()
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
        GEO = ["Latitude", "Longitude"]


        def rmse_of(model, columns):
            return mean_squared_error(y_te, model.fit(X_tr[columns], y_tr).predict(X_te[columns]), squared=False)
    """,
    starter=[
        C("""
        rmse = {}
        rmse
        """),
        C("""
        geo_params = None
        best_geo_rmse = None
        """),
        C("""
        cols = GEO + ["MedInc"]
        rmse_per_w = pd.Series(dtype=float)
        best_w = None
        weighted_rmse = None
        """),
    ],
    checks="""
        check(set(rmse) == {"linear", "raw knn", "scaled knn", "geo knn", "geo + income knn"}, "five models compared", "keys: linear, raw knn, scaled knn, geo knn, geo + income knn")
        check(rmse["raw knn"] > rmse["linear"] and rmse["scaled knn"] < rmse["linear"], "without scaling the KNN is worse than the linear one; with scaling better", "Population (scale 1000) dominates the raw distance")
        check(rmse["geo knn"] < rmse["scaled knn"] - 0.05 and rmse["geo knn"] < rmse["linear"] - 0.15, "the KNN with just two coordinates beats the one with 8 features and the linear one (by over 0.15)", "fewer features but the right ones")
        check(rmse["geo + income knn"] >= rmse["geo knn"] - 0.02, "adding the income with the same weight as the coordinates does not improve (or barely)", "the standardized income weighs as much as latitude and longitude")
        check(geo_params["n_neighbors"] <= 15 and best_geo_rmse < 0.55, "the search on the coordinates alone chooses a small k (15 or less); RMSE below 0.55", "GridSearchCV on Latitude and Longitude")
        check(list(rmse_per_w.index) == [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1] and best_w == rmse_per_w.idxmin() and 0 < best_w <= 0.2, "the best weight of the income is small but nonzero (between 0.05 and 0.2)", "try the seven weights on the validation set")
        check(weighted_rmse < rmse["geo knn"] - 0.03 and weighted_rmse < best_geo_rmse, "with the weights, the test RMSE drops by over 0.03 compared to the coordinates alone and also beats the search on k", "retrain with best_w on the whole training set")
    """,
    hint="""
        Weights: `sc = StandardScaler().fit(X_fit[cols]); A = sc.transform(X_fit[cols]); A[:, 2] *= w`. Apply the same `w` to the validation set. After the choice: retrain on the whole `X_tr` with a scaler learned on all of `X_tr`.
    """,
    solution=[
        C("""
        rmse = {
            "linear": rmse_of(LinearRegression(), list(X.columns)),
            "raw knn": rmse_of(KNeighborsRegressor(15), list(X.columns)),
            "scaled knn": rmse_of(make_pipeline(StandardScaler(), KNeighborsRegressor(15)), list(X.columns)),
            "geo knn": rmse_of(KNeighborsRegressor(15), GEO),
            "geo + income knn": rmse_of(make_pipeline(StandardScaler(), KNeighborsRegressor(15)), GEO + ["MedInc"]),
        }
        print(pd.Series(rmse).round(3).to_string())
        """),
        C("""
        search = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": [3, 5, 10, 15, 25, 40, 60], "weights": ["uniform", "distance"]}, cv=5, scoring="neg_root_mean_squared_error").fit(X_tr[GEO], y_tr)
        geo_params = search.best_params_
        best_geo_rmse = mean_squared_error(y_te, search.predict(X_te[GEO]), squared=False)
        print(f"best parameters on the coordinates: {geo_params} | test RMSE {best_geo_rmse:.4f}")
        """),
        C("""
        cols = GEO + ["MedInc"]
        X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=1)


        def rmse_with_weight(w, X_a, y_a, X_b, y_b):
            sc = StandardScaler().fit(X_a[cols])
            A, B = sc.transform(X_a[cols]), sc.transform(X_b[cols])
            A[:, 2] *= w                                            # reduces the weight of the income in the distance
            B[:, 2] *= w
            return mean_squared_error(y_b, KNeighborsRegressor(10, weights="distance").fit(A, y_a).predict(B), squared=False)


        weights = [0, 0.05, 0.1, 0.2, 0.35, 0.5, 1]
        rmse_per_w = pd.Series({w: rmse_with_weight(w, X_fit, y_fit, X_val, y_val) for w in weights})
        best_w = rmse_per_w.idxmin()
        weighted_rmse = rmse_with_weight(best_w, X_tr, y_tr, X_te, y_te)
        print(rmse_per_w.round(4).to_string())
        print(f"best weight of the income: {best_w} | test RMSE with the weights: {weighted_rmse:.4f} (coordinates only: {best_geo_rmse:.4f}, linear: {rmse['linear']:.4f})")
        """),
    ],
    learned="""
        - **Without scaling the KNN is worse than the linear one** (`Population` has a scale of thousands and alone decides the distances); with scaling it improves but stays below the KNN on the two coordinates alone: the other six features add noise to the
          distance (neighborhoods near in feature space are not near on the map).
        - A KNN on latitude and longitude is a **geographic proximity model**: it beats the linear regression with all the features (0.53 against 0.74) because the price of houses depends on where you are in a nonlinear way (coast, metropolitan areas) and
          a linear model cannot represent it without engineered features.
        - **The weights in the distance are hyperparameters.** Adding the income with weight 1 makes it worse, with a weight of about 0.05-0.1 it improves a lot (test RMSE 0.46 with `w = 0.05`): the information on income is useful, but only to break ties between geographic neighbors. It is the first step toward
          "learning the metric" (metric learning, NCA).
        - The choice of `w` must be made on a validation set or in cross-validation, never on the test set: here you did it with 20% of the training set and the test set measured once.
    """,
    pitfall="""
        Concluding "KNN is better than linear models" from this exercise. It holds here because the structure is spatial and the coordinates carry enormous weight; on other problems (many features, additive effects) the opposite holds (see C1 and F2).
    """,
    interview=(
        "How do you decide which features to use in a KNN and with what weight?",
        """
        I start from a few features that have a proximity meaning for the problem (coordinates, time, product categories), scale them and try a weight per feature (or group of features) as a hyperparameter, evaluating on a validation set or in cross-validation.
        If the candidate features are many, I use a selection or a reduction (PCA, metric learning) before the KNN, or I switch to a model that learns by itself which features matter (trees, regularized linear models).
        """),
)

# ---------------------------------------------------------------------------------------- B3

B3 = dict(
    id="B3", slug="knn_imputer_with_known_truth", kind="B", level=3, minutes=50, module=MODULE,
    title="KNNImputer: measuring imputation when you know the true values",
    goal="""
        Evaluate imputations (mean, median, `KNNImputer` with various `k` and weights) with a clean method: **erase known values** and measure the reconstruction error. Verify that without scaling the `KNNImputer` works worse and that a better imputation does not
        automatically translate into a better classification.
    """,
    dataset="""
        `load_wine` standardized (`Z`, truth) with 15% of the cells made missing at random (`Zm`, mask `mask`). `X_raw` is the same matrix in the original units, with the same missing cells (`Xm`).
    """,
    purpose="""
        Know **how much** an imputer is worth, not just that it "works". The known-truth trick (masking values you know) is the standard for comparing imputations, and you will need it in the following modules.
    """,
    task=[
        "Write `imputation_rmse(imputer)`: apply `fit_transform` to `Zm` and return the RMSE **only on the masked cells** with respect to `Z`.",
        "Save `base_rmse` with `mean` and `median` (`SimpleImputer`), and `knn_rmse` (`pd.Series`) for `KNNImputer(n_neighbors=k)` with `k` in `[1, 2, 3, 5, 10, 15, 30, 50]`. Save `best_k`. Add `knn5_distance` (k=5, `weights=\"distance\"`).",
        "**Without scaling:** apply `KNNImputer(5)` to `Xm` (original units), convert the result to standardized units (`(R - X.mean(0)) / X.std(0)` with the statistics of the complete data) and save `rmse_without_scaling` on the masked cells.",
        "**Downstream effect.** With a `Pipeline` `StandardScaler` → imputer → `LogisticRegression(max_iter=1000)` compute in cross-validation (`StratifiedKFold(5, shuffle=True, random_state=0)`) the accuracy on `Xm` for `mean` and for `KNNImputer(5)`: save `mean_acc`, `knn_acc`.",
    ],
    setup="""
        from sklearn.datasets import load_wine
        from sklearn.impute import KNNImputer, SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = load_wine(return_X_y=True)
        mu, sd = X.mean(0), X.std(0)
        Z = (X - mu) / sd                                            # standardized truth
        mask = np.random.default_rng(0).random(Z.shape) < 0.15       # 15% of the cells missing at random
        Zm = np.where(mask, np.nan, Z)
        Xm = np.where(mask, np.nan, X)
        print(f"{mask.mean():.1%} of cells masked, {mask.sum()} values to reconstruct")
    """,
    starter=[
        C("""
        def imputation_rmse(imputer):
            ...
        """),
        C("""
        base_rmse = {"mean": None, "median": None}
        knn_rmse = pd.Series(dtype=float)
        best_k = None
        knn5_distance = None
        """),
        C("""
        rmse_without_scaling = None
        mean_acc = None
        knn_acc = None
        """),
    ],
    checks="""
        check(close(imputation_rmse(SimpleImputer()), base_rmse["mean"]) and 0.95 < base_rmse["mean"] < 1.15, "RMSE of the mean imputation: about 1 (the standard deviation of the data)", "on the masked cells only")
        check(list(knn_rmse.index) == [1, 2, 3, 5, 10, 15, 30, 50] and best_k == knn_rmse.idxmin(), "RMSE curve as a function of k and best k", "one imputer for every value of k")
        check(knn_rmse.min() < 0.85 * base_rmse["mean"] and knn_rmse[1] > knn_rmse.min() + 0.03, "the KNNImputer reduces the error by over 15% compared to the mean; with k = 1 it is much noisier", "the wine columns are correlated")
        check(3 <= best_k <= 30 and close(knn5_distance, imputation_rmse(KNNImputer(n_neighbors=5, weights="distance"))), "best k between 3 and 30; result with weights computed", "weights='distance'")
        check(rmse_without_scaling > knn_rmse[5] + 0.03, "without scaling the KNNImputer reconstructs worse (over 0.03 more RMSE)", "proline dominates the distance")
        check(abs(knn_acc - mean_acc) < 0.05 and knn_acc > 0.9, "the classification changes little (less than 5 points): the task is easy and the logistic is robust to missing values", "compare knn_acc and mean_acc")
    """,
    hint="""
        `R = imputer.fit_transform(Zm); np.sqrt(((R[mask] - Z[mask]) ** 2).mean())`. Without scaling: `R = KNNImputer(5).fit_transform(Xm); Rz = (R - mu) / sd`. Pipeline: `make_pipeline(StandardScaler(), SimpleImputer(), LogisticRegression(max_iter=1000))`
        (`StandardScaler` ignores the NaN when it computes mean and standard deviation).
    """,
    solution=[
        C("""
        def imputation_rmse(imputer):
            R = imputer.fit_transform(Zm)
            return float(np.sqrt(((R[mask] - Z[mask]) ** 2).mean()))          # only the cells we know and have hidden
        """),
        C("""
        base_rmse = {"mean": imputation_rmse(SimpleImputer()), "median": imputation_rmse(SimpleImputer(strategy="median"))}
        knn_rmse = pd.Series({k: imputation_rmse(KNNImputer(n_neighbors=k)) for k in [1, 2, 3, 5, 10, 15, 30, 50]})
        best_k = int(knn_rmse.idxmin())
        knn5_distance = imputation_rmse(KNNImputer(n_neighbors=5, weights="distance"))
        print(base_rmse, "\\n", knn_rmse.round(3).to_string(), f"\\nbest k: {best_k} | k=5 with 1/distance weights: {knn5_distance:.3f}")
        """),
        C("""
        R = KNNImputer(n_neighbors=5).fit_transform(Xm)
        Rz = (R - mu) / sd                                                   # I bring the result back to standardized units to compare it with Z
        rmse_without_scaling = float(np.sqrt(((Rz[mask] - Z[mask]) ** 2).mean()))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        mean_acc = cross_val_score(make_pipeline(StandardScaler(), SimpleImputer(), LogisticRegression(max_iter=1000)), Xm, y, cv=cv).mean()
        knn_acc = cross_val_score(make_pipeline(StandardScaler(), KNNImputer(n_neighbors=5), LogisticRegression(max_iter=1000)), Xm, y, cv=cv).mean()
        print(f"RMSE without scaling: {rmse_without_scaling:.3f} (with scaling, k=5: {knn_rmse[5]:.3f})")
        print(f"cross-validated accuracy: mean {mean_acc:.3f} | KNNImputer {knn_acc:.3f}")
        """),
    ],
    learned="""
        - **The known-truth method:** hide cells you know, impute, compare. It is the only way to measure an imputer without relying on an indirect metric. With the RMSE on the masked cells the mean is worth about 1 (the standard deviation: it uses no
          information from the other columns), the KNNImputer drops to about 0.77: **a quarter less error** because the chemical measurements of the wines are correlated with each other.
        - `k` has a U-shaped curve: with `k = 1` the imputation copies a single neighbor (noisy), with a huge `k` it tends to the mean. The maximum is flat between 5 and 30.
        - **Scaling is mandatory** here too (the imputer uses distances): without scaling `proline` dominates and the error goes back almost to the level of the mean.
        - Improving the imputation does not necessarily improve the downstream model: with an easy problem (wines, logistic) the classification changes by a few points. Whether the imputation matters is measured on the task, not just on the reconstruction.
    """,
    pitfall="""
        Comparing imputers only on the final accuracy of an easy classifier: differences of one or two points are noise. Also measure the reconstruction error with hidden values, and remember that the imputer must be learned inside the pipeline on the training set alone.
    """,
    interview=(
        "How do you evaluate an imputation method?",
        """
        With the known-truth trick: I hide a share of values I know (with the missingness mechanism I consider realistic), impute and measure the error on the hidden cells, besides the effect on the final task in cross-validation with the imputer inside the pipeline.
        I remember that the mechanism matters: if the data are missing in a non-random way, the results on random missingness are optimistic.
        """),
)

EXERCISES = [B1, B2, B3]
