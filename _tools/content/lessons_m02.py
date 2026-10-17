"""Module 2 lessons: preprocessing."""
import sys
import textwrap
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nb import C, HEADER, Cell, M, build_lesson, details  # noqa: E402

MODULE = "module_02_preprocessing"
TITLE = "Preprocessing: scaling, encoding categoricals, missing data"
LESSONS = [
    ("01_scaling_and_transformations.ipynb", "2.1", "Scaling and transformations",
     "StandardScaler, MinMax, Robust, log and PowerTransformer: when they are needed and what they do."),
    ("02_encoding_categoricals.ipynb", "2.2", "Encoding categoricals",
     "One-hot, ordinal, frequency and target encoding; rare and never-seen categories."),
    ("03_missing_data.ipynb", "2.3", "Missing data",
     "MCAR, MAR, MNAR; simple, KNN and iterative imputation; the missingness indicator."),
]
PATH = [
    ("Lesson 2.1", ["A1", "B3", "C1", "F1"]),
    ("Lesson 2.2", ["A2", "B1", "D1", "C2", "E1", "E2", "A3", "E3", "F2"]),
    ("Lesson 2.3", ["B2", "C3", "D2"]),
]


def _setup(extra: str = "") -> Cell:
    """Setup cell of a lesson: the common imports plus the optional `extra` code."""
    return C(HEADER + ("\n" + textwrap.dedent(extra).strip("\n") if extra else ""))


def lesson_2_1() -> List[Cell]:
    """Cells of lesson 2.1."""
    return [
        M("""
        # Lesson 2.1 · Scaling and transformations

        A machine learning algorithm does not know what euros or years are: it only sees numbers. Scaling brings the features onto
        comparable scales; transformations change the shape of their distribution. This lesson shows what
        each tool does on loan application data, and when it is not needed at all.

        **What you will learn**

        - Which models need scaling and which don't, and why.
        - What `fit` computes and what `transform` applies, in each scaler.
        - How Standard, MinMax and Robust behave in the presence of outliers.
        - How to reduce skewness with log, `PowerTransformer` and `QuantileTransformer`.
        """),
        _setup("""
        from scipy import stats
        from sklearn.preprocessing import (MinMaxScaler, PowerTransformer, QuantileTransformer,
                                           RobustScaler, StandardScaler)

        df = ds.loans(with_missing=False)
        df[["age", "income", "job_tenure", "amount"]].describe().round(1)
        """),
        M("""
        ## Who needs scaling

        | Model family | Why | Scaling |
        |---|---|---|
        | Distance-based (KNN, k-means, kernel SVM) | the Euclidean distance sums the squares of the differences: the feature with the largest unit dominates | **needed** |
        | Gradient-based and penalized (regularized regression, neural networks) | the penalty and the speed of convergence depend on the unit | **needed** |
        | PCA | it looks for the directions of maximum variance: the feature with the numerically largest variance wins | **needed** |
        | Trees, forests, boosting | they compare one feature at a time with a threshold: the order of the values does not change | useless |
        | Unregularized linear and logistic regression | rescaling a feature rescales the coefficient in the opposite direction | useless for predictions (useful for interpreting) |
        """),
        M("""
        ## Three scalers, three ideas

        ```
        StandardScaler   z = (x − mean) / standard deviation      mean 0, standard deviation 1
        MinMaxScaler     z = (x − min) / (max − min)               everything in [0, 1]
        RobustScaler     z = (x − median) / (q75 − q25)            ignores the extremes
        ```

        Mean, standard deviation, minimum and maximum are **sensitive to outliers**; median and interquartile range are not.
        """),
        C("""
        income = df[["income"]]
        print("income: median", int(income.median().iloc[0]), "| maximum", int(income.max().iloc[0]))
        tab = {}
        for name, scaler in [("standard", StandardScaler()), ("minmax", MinMaxScaler()), ("robust", RobustScaler())]:
            z = scaler.fit_transform(income).ravel()
            tab[name] = {"typical value (median)": np.median(z), "outlier (maximum)": z.max(), "std deviation": z.std(),
                         "share of data in [-1, 1]": np.mean(np.abs(z) <= 1)}
        pd.DataFrame(tab).round(2)
        """),
        M("""
        With `MinMaxScaler` the typical income ends up near 0.2 and all the others are squeezed into the same narrow interval
        because a single huge value fixed the maximum. `RobustScaler` centers on the median and uses the interquartile range:
        the typical values have a comparable spread, and the outlier stays an outlier (large value) without ruining the others.
        """),
        M("""
        ## `fit`, `transform`, `fit_transform`

        Every scaler **learns** some numbers with `fit` (mean and standard deviation, or minimum and maximum...) and **applies** them with
        `transform`. The learned attributes have a trailing underscore.
        """),
        C("""
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(df[["age", "income"]], test_size=0.25, random_state=0)
        sc = StandardScaler().fit(tr)
        print("learned on the TRAINING set: mean", sc.mean_.round(1), "| scale", sc.scale_.round(1))
        print("test with transform         :", sc.transform(te).mean(axis=0).round(3), " <- it is not zero, and that is right")
        print("test with fit_transform     :", StandardScaler().fit_transform(te).mean(axis=0).round(3), " <- recomputed on the test: WRONG")
        """),
        M("""
        > **Common mistake.** `fit` only on the training set, `transform` on everything else. Refitting the scaler on the test set (or
        > fitting it on the whole dataset before the split) is a leakage mistake that exercises E1 and E3 of module 1 and this
        > module make you hunt for. Inside a `Pipeline` (module 3) it is impossible to do it.
        """),
        M("""
        ## Changing the shape: log, Yeo-Johnson, quantiles

        Scaling does not change the **shape** of a distribution. If a feature is strongly skewed (incomes, prices, counts)
        and the model is linear or distance-based, it is worth transforming it:

        - `np.log1p(x)`: simple, requires x ≥ 0 (or x > -1);
        - `PowerTransformer(method="yeo-johnson")`: it chooses by itself the power that makes the distribution more symmetric;
        - `QuantileTransformer(output_distribution="normal")`: it maps the quantiles onto a Gaussian. It is monotonic but it **squashes**
          the distances: two different extreme values end up almost equal.
        """),
        C("""
        x = df[["income"]]
        transformed = {
            "raw": x.to_numpy(),
            "log1p": np.log1p(x).to_numpy(),
            "yeo-johnson": PowerTransformer().fit_transform(x),
            "quantiles -> normal": QuantileTransformer(n_quantiles=200, output_distribution="normal").fit_transform(x),
        }
        pd.DataFrame({k: {"skewness": stats.skew(v.ravel()), "kurtosis": stats.kurtosis(v.ravel())}
                      for k, v in transformed.items()}).round(2)
        """),
        M("## Check your understanding\n\n" + details(
            "I have to train a Random Forest: do I standardize the features?",
            "No. Every split compares a feature with a threshold and the order of the values does not change with a monotonic transformation: "
            "the chosen splits are identical, and scaling does not change the predictions. It costs time and complicates the pipeline.") + "\n\n" + details(
            "I have a huge outlier in income. MinMaxScaler or RobustScaler?",
            "RobustScaler: the maximum is exactly the value that ruins the MinMaxScaler, because it fixes the denominator. With the median and "
            "the interquartile range the typical values keep a sensible scale. If the outlier is a measurement error, it is better to "
            "correct it or remove it beforehand.")),
    ]


def lesson_2_2() -> List[Cell]:
    """Cells of lesson 2.2."""
    return [
        M("""
        # Lesson 2.2 · Encoding categoricals

        Almost all the scikit-learn models accept only numbers. Turning a text column into numbers is a modeling choice,
        not a formality: the encoding decides which information (and which false information) the model can use.

        **What you will learn**

        - To tell apart nominal, ordinal and high-cardinality variables, and choose the right encoding for each.
        - To use `OneHotEncoder` and `OrdinalEncoder` with never-seen categories and rare categories.
        - To understand why target encoding is powerful and dangerous.
        """),
        _setup("""
        from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

        df = ds.loans(with_missing=False)
        df[["city", "occupation", "education", "contract_type"]].nunique()
        """),
        M("""
        ## Three types of categorical variable

        | Type | Example | Typical encoding |
        |---|---|---|
        | **Nominal**, few categories | `contract_type` (4 levels) | one-hot |
        | **Ordinal** | `education` (middle school < ... < phd) | ordinal with the order made explicit |
        | **High cardinality** | `occupation` (40 levels) | frequency, target encoding, grouping of the rare ones; one-hot only with suitable models and data |

        A typical mistake is encoding a nominal variable with integers (Rome = 0, Milan = 1, Naples = 2): the linear model
        reads an order and a distance that do not exist.
        """),
        M("## One-hot: one column per category"),
        C("""
        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        M_ohe = ohe.fit_transform(df[["contract_type"]])
        print("shape:", M_ohe.shape)
        print("columns:", list(ohe.get_feature_names_out()))
        pd.DataFrame(M_ohe[:3], columns=ohe.get_feature_names_out(), index=df.index[:3])
        """),
        M("""
        `handle_unknown="ignore"` makes a category never seen in training produce a row of zeros instead of an error: it is
        almost always what you need in production. `drop="first"` removes one column per category, avoiding redundancy (the
        sum of the columns is always 1, that is equal to the intercept): it is needed only with unregularized linear models.

        > **Version note.** With `drop` other than `None` scikit-learn 1.1 requires `handle_unknown="error"`. In scikit-learn 1.2 the
        > parameter `sparse=False` is renamed `sparse_output=False`, and in 1.4 the old name is removed: the notebooks of this course use `sparse=False`,
        > so they need scikit-learn 1.1 to 1.3.
        """),
        M("## Rare categories: `min_frequency`"),
        C("""
        ohe_rare = OneHotEncoder(sparse=False, handle_unknown="infrequent_if_exist", min_frequency=40)
        M_rare = ohe_rare.fit_transform(df[["occupation"]])
        print("occupation: 40 levels ->", M_rare.shape[1], "columns (the categories with fewer than 40 rows end up together)")
        print(list(ohe_rare.get_feature_names_out())[-3:])
        """),
        M("## Ordinal: the order must be explicit"),
        C("""
        order = ds.EDUCATION_ORDER
        oe = OrdinalEncoder(categories=[order], handle_unknown="use_encoded_value", unknown_value=-1)
        pd.DataFrame({"education": df.education[:5], "code": oe.fit_transform(df[["education"]])[:5].ravel()})
        """),
        M("""
        Without `categories=[order]` the `OrdinalEncoder` sorts the categories alphabetically ("bachelor" < "high school" < "master"
        < ...): an order that has nothing to do with the meaning.

        ## Frequency and target encoding

        With 40 occupations a one-hot adds 40 nearly empty columns. Two single-column alternatives:

        - **Frequency encoding**: every category becomes its frequency. Simple and with no leakage, but it loses the information about the target.
        - **Target encoding**: every category becomes the mean (with *smoothing*) of the target in the category. Very informative and very
          dangerous: if the mean of a row is computed **including the row itself**, the encoding contains the target. It is
          computed out-of-fold, as you will see in exercises A3 and E3.
        """),
        C("""
        freq = df["occupation"].map(df["occupation"].value_counts(normalize=True))
        te_naive = df["occupation"].map(df.groupby("occupation")["defaulted"].mean())     # it includes the row itself!
        print("occupations with fewer than 10 rows:", int((df["occupation"].value_counts() < 10).sum()))
        print("correlation of the 'naive' target encoding with the target:", round(np.corrcoef(te_naive, df["defaulted"])[0, 1], 3))
        """),
        M("> **Common mistake.** Encoding a nominal variable with integers (1, 2, 3...) and giving it to a linear model or to a KNN: the model reads an order and a distance that do not exist. Integers are fine only for ordinals with the order declared, or for trees (exercise C2). And building the dummies with `pd.get_dummies` separately on training and test produces different columns (exercise E1)."),
        M("## Check your understanding\n\n" + details(
            "Why don't I encode `city` with numbers from 0 to 11?",
            "Because a linear model (or KNN) would treat city 11 as 'eleven times city 1' and city 5 as 'halfway between 0 and "
            "10': an invented order and distance. The correct encodings for a nominal are one-hot, frequency or target encoding. "
            "A tree can get by with numbers (with enough splits it isolates every category), but it is an exception.") + "\n\n" + details(
            "What happens to `OneHotEncoder(handle_unknown='error')` in production if a new city arrives?",
            "It raises an error and the service stops. With `handle_unknown='ignore'` the row of zeros is a prudent choice: the model "
            "sees 'none of the known cities'. In both cases it is worth monitoring the appearance of new categories: it is a drift signal.")),
    ]


def lesson_2_3() -> List[Cell]:
    """Cells of lesson 2.3."""
    return [
        M("""
        # Lesson 2.3 · Missing data

        A missing value is not an error to eliminate: it is information about how the data were collected, and sometimes it is
        the most important information. This lesson distinguishes the missingness mechanisms, shows the scikit-learn imputers and
        explains why the choice of imputation changes the distribution of the data.

        **What you will learn**

        - The three mechanisms: MCAR, MAR and MNAR, and what can be inferred from the data.
        - `SimpleImputer`, `KNNImputer`, `IterativeImputer` and the missingness indicator.
        - Why imputation with the mean shortens the variance and weakens correlations.
        """),
        _setup("""
        from sklearn.experimental import enable_iterative_imputer          # noqa: F401  (needed to import it)
        from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

        df = ds.loans()
        df.isna().mean().round(3)
        """),
        M("""
        ## Three mechanisms

        | Mechanism | Meaning | Example |
        |---|---|---|
        | **MCAR** (Missing Completely At Random) | the missingness depends on nothing | a sensor that loses a reading at random |
        | **MAR** (Missing At Random) | it depends on other **observed** variables | someone who is unemployed does not report their job tenure |
        | **MNAR** (Missing Not At Random) | it depends on the **missing** value itself | someone who earns a lot does not declare their income |

        From a sample you can **detect** dependencies on the observed variables (MAR), but you **cannot prove MCAR nor
        distinguish MNAR**: the missing value, by definition, we do not see.
        """),
        C("""
        missing = df["income"].isna()
        print("default rate with income present :", round(df.loc[~missing, "defaulted"].mean(), 3))
        print("default rate with income missing :", round(df.loc[missing, "defaulted"].mean(), 3))
        """),
        M("""
        The fact that income is missing is **predictive**: whoever does not declare it defaults much more often. Imputing the value and
        forgetting that it was missing throws this information away. That is why `SimpleImputer(add_indicator=True)` adds a
        binary "was missing" column for every column with missing values.

        ## The imputers

        - `SimpleImputer(strategy="mean" | "median" | "most_frequent" | "constant")`: one value per column, fast and robust.
        - `KNNImputer(n_neighbors=5)`: the mean of the values of the most similar neighbors; it requires scaled features.
        - `IterativeImputer`: it models each feature as a function of the others and iterates; it captures the correlations, but it is slower and less stable.
        """),
        C("""
        num = df[["age", "income", "job_tenure", "amount"]]
        for name, imp in [("mean", SimpleImputer(strategy="mean")), ("median", SimpleImputer(strategy="median")),
                          ("with indicator", SimpleImputer(strategy="median", add_indicator=True)),
                          ("KNN", KNNImputer(n_neighbors=5)), ("iterative", IterativeImputer(random_state=0))]:
            Z = imp.fit_transform(num)
            print(f"{name:15s} shape {Z.shape} | remaining nan {int(np.isnan(Z).sum())}")
        """),
        M("""
        ## What imputation with the mean does

        Replacing the missing values with a constant **shortens the variance** and **weakens the correlations** with the other variables:
        all the imputed values are equal, so they do not vary together with anything. If a fraction p of the values is
        imputed with the mean, the variance drops by about a factor (1 − p).
        """),
        C("""
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        for p in [0.1, 0.3, 0.5]:
            x_miss = np.where(rng.random(5000) < p, np.nan, x)
            x_imp = SimpleImputer(strategy="mean").fit_transform(x_miss.reshape(-1, 1)).ravel()
            print(f"imputed share {p:.0%}: variance {x_imp.var():.2f} (true 1.00, expected {1 - p:.2f})")
        """),
        M("> **Common mistake.** Imputing with the mean (or dropping the incomplete rows) without asking **why** the value is missing: if the missingness is informative, imputation erases it, and the mean reduces variance and correlations (exercises C3 and D2). You impute inside the pipeline, adding the missingness indicator when it matters."),
        M("## Check your understanding\n\n" + details(
            "I drop all the rows with at least one missing value. Is that fine?",
            "Only with MCAR and with very few rows lost. In the loans dataset about 29% of the rows have at least one missing value, "
            "and the dropped rows are not a random sample: whoever does not declare income defaults much more often. The model "
            "would see a population different from the one it will be used on.") + "\n\n" + details(
            "When do I fit the imputer?",
            "Only on the training set, like any other transformer: the mean (or the model of the neighbors) is a statistic learned "
            "from the data. Fitting it on the whole dataset lets information from the test set into the training.")),
    ]


def build() -> None:
    """Write the lesson notebooks of this module."""
    for name, f in [("01_scaling_and_transformations.ipynb", lesson_2_1),
                    ("02_encoding_categoricals.ipynb", lesson_2_2),
                    ("03_missing_data.ipynb", lesson_2_3)]:
        build_lesson(MODULE, name, f())


if __name__ == "__main__":
    build()
