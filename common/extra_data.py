"""Datasets for modules 2 onwards (used through `from common import datasets as ds`, like all the others)."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

EDUCATION_ORDER = ["middle school", "high school", "bachelor", "master", "phd"]


def loans(n: int = 1500, seed: int = 4, with_missing: bool = True) -> pd.DataFrame:
    """Personal loan applications: mixed data, as dirty as in real life.

    Columns: age, income, job_tenure, amount (numeric); city (nominal, 12 levels), occupation
    (nominal with high cardinality, 40 levels), education (ordinal, see EDUCATION_ORDER), contract_type
    (nominal, 4 levels); defaulted (0/1 target, about 20%).

    With `with_missing=True` it inserts missing values with different mechanisms (hidden on purpose):
    some random, others linked to other columns or to the target.

    Args:
        n: number of applications.
        seed: seed of the random generator.
        with_missing: if False the dataset is complete.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(42, 12, n), 21, 75).round()
    edu_idx = rng.choice(5, n, p=[0.18, 0.38, 0.28, 0.12, 0.04])
    contract = rng.choice(["permanent", "fixed_term", "self_employed", "unemployed"], n, p=[0.55, 0.2, 0.18, 0.07])
    tenure = np.clip(age - 21 - rng.exponential(4, n), 0, None).round(1)
    tenure = np.where(contract == "unemployed", 0.0, tenure)
    log_inc = 9.9 + 0.12 * edu_idx + 0.008 * tenure + rng.normal(0, 0.4, n) - 0.9 * (contract == "unemployed")
    income = np.exp(log_inc).round(-1)
    city = rng.choice([f"city_{i:02d}" for i in range(12)], n,
                      p=np.array([12, 10, 9, 8, 7, 6, 5, 4, 3, 2, 2, 1]) / 69)
    weights = 1 / np.arange(1, 41) ** 0.8
    occ_idx = rng.choice(40, n, p=weights / weights.sum())
    occupation_effect = np.random.default_rng(99).normal(0, 0.8, 40)                 # hidden risk of each occupation
    amount = np.exp(rng.normal(9.2, 0.6, n)).round(-1)
    logit = (-3.0 + 2.2 * amount / income + occupation_effect[occ_idx] - 0.15 * edu_idx - 0.01 * (age - 40)
             + np.select([contract == "self_employed", contract == "fixed_term", contract == "unemployed"],
                         [0.4, 0.3, 1.4], 0.0))
    defaulted = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({
        "age": age, "income": income, "job_tenure": tenure, "amount": amount,
        "city": city, "occupation": [f"occ_{i:02d}" for i in occ_idx],
        "education": np.array(EDUCATION_ORDER, dtype=object)[edu_idx], "contract_type": contract,
        "defaulted": defaulted,
    })
    if with_missing:
        r = np.random.default_rng(seed + 1000)
        df.loc[r.random(n) < 0.06, "age"] = np.nan                                                # random
        df.loc[r.random(n) < np.where(defaulted == 1, 0.30, 0.06), "income"] = np.nan             # depends on the target
        df.loc[r.random(n) < np.where(contract == "unemployed", 0.45, 0.04), "job_tenure"] = np.nan  # depends on the contract
        df.loc[r.random(n) < 0.08, "education"] = np.nan                                          # random
        df.loc[r.random(n) < 0.03, "occupation"] = np.nan                                         # random
    return df


def titanic() -> pd.DataFrame:
    """Titanic (1309 passengers) with the leaking columns removed (lifeboat, body).

    Columns: pclass, sex, age, siblings_spouses, parents_children, fare, port, deck (from the cabin, often
    missing), title (from the name); survived (0/1 target). On first use it is downloaded from OpenML and cached in
    `data/`; without network a synthetic version with the same columns is returned.
    """
    try:
        from sklearn.datasets import fetch_openml

        d = fetch_openml("titanic", version=1, as_frame=True, data_home=str(DATA_DIR)).frame
    except Exception as exc:
        print(f"[datasets] Titanic download failed ({exc.__class__.__name__}): using a synthetic version.")
        return _titanic_synthetic()
    title = d["name"].str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Other")
    return pd.DataFrame({
        "pclass": d["pclass"].astype(int), "sex": d["sex"].astype(str), "age": d["age"],
        "siblings_spouses": d["sibsp"].astype(int), "parents_children": d["parch"].astype(int), "fare": d["fare"],
        "port": d["embarked"].astype(object), "deck": d["cabin"].str[0], "title": title,
        "survived": d["survived"].astype(int),
    })


def _titanic_synthetic(n: int = 1300, seed: int = 0) -> pd.DataFrame:
    """Synthetic Titanic with the same columns, used when the real dataset cannot be downloaded."""
    rng = np.random.default_rng(seed)
    pclass = rng.choice([1, 2, 3], n, p=[0.25, 0.21, 0.54])
    sex = rng.choice(["male", "female"], n, p=[0.64, 0.36])
    age = np.where(rng.random(n) < 0.2, np.nan, np.clip(rng.normal(30, 13, n), 1, 80).round())
    logit = -0.6 + 2.4 * (sex == "female") - 0.72 * (pclass - 2)
    return pd.DataFrame({
        "pclass": pclass, "sex": sex, "age": age, "siblings_spouses": rng.poisson(0.5, n),
        "parents_children": rng.poisson(0.4, n), "fare": np.exp(rng.normal(3.0, 1.0, n)) / pclass,
        "port": rng.choice(["S", "C", "Q"], n, p=[0.7, 0.2, 0.1]),
        "deck": np.where(rng.random(n) < 0.77, None, rng.choice(list("ABCDEF"), n)),
        "title": np.where(sex == "female", "Miss", "Mr"), "survived": rng.binomial(1, 1 / (1 + np.exp(-logit))),
    })


def patient_measurements(n_patients: int = 60, n_measures: int = 10, seed: int = 8) -> pd.DataFrame:
    """Repeated measurements on different patients: 10 rows for each patient.

    Columns: patient (identifier), m1, m2, m3 (measures: a stable trait of the patient plus a little noise),
    outcome (it depends mostly on a hidden individual characteristic, not predictable for a new patient).

    Args:
        n_patients: number of patients.
        n_measures: rows per patient.
        seed: seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    n = n_patients * n_measures
    pat = np.repeat(np.arange(n_patients), n_measures)
    base = rng.normal(0, 3, n_patients)                       # hidden individual effect
    traits = rng.normal(0, 1, (n_patients, 3))                # stable characteristics: the patient's "fingerprint"
    X = traits[pat] + 0.15 * rng.normal(size=(n, 3))
    outcome = base[pat] + 0.5 * X[:, 0] + rng.normal(0, 0.5, n)
    return pd.DataFrame({"patient": pat, "m1": X[:, 0], "m2": X[:, 1], "m3": X[:, 2], "outcome": outcome})


def time_series(n: int = 600, seed: int = 2) -> pd.DataFrame:
    """Monthly series with a random-walk trend plus an annual seasonality.

    Columns: t (time index), s (sin), c (cos) of the annual season, y (to predict). The rows are in chronological order.

    Args:
        n: number of months.
        seed: seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = np.cumsum(rng.normal(0, 0.35, n))
    y = trend + 3 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 0.5, n)
    return pd.DataFrame({"t": t, "s": np.sin(2 * np.pi * t / 12), "c": np.cos(2 * np.pi * t / 12), "y": y})
