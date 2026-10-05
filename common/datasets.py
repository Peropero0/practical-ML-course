"""Datasets used in the exercises.

Three families:
  * datasets bundled with scikit-learn or downloaded once and cached in `data/`;
  * SYNTHETIC datasets generated with a fixed seed, where the generating process is known
    (so you can check whether the model found the truth again);
  * small hand-written datasets (Anscombe).

All the functions are deterministic: same code, same data.
"""
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CALIFORNIA_SYNTHETIC = False



def california() -> Tuple[pd.DataFrame, pd.Series]:
    """California housing (20640 districts, 8 features).

    Returns (X, y): X is a DataFrame, y is the median house value in hundreds
    of thousands of dollars. The target is capped at 5.0 (the most expensive houses are all "5").
    If the download fails it falls back to a synthetic version with the same structure and sets
    `CALIFORNIA_SYNTHETIC` to True.

    Returns:
        (X, y): the 8 features and the target `price`.
    """
    global CALIFORNIA_SYNTHETIC
    try:
        from sklearn.datasets import fetch_california_housing

        d = fetch_california_housing(data_home=str(DATA_DIR), as_frame=True)
        CALIFORNIA_SYNTHETIC = False
        return d.data, d.target.rename("price")
    except Exception as exc:  # no network or site unreachable
        print(f"[datasets] California download failed ({exc.__class__.__name__}): "
              "using a synthetic version. The numeric values in the exercises will change.")
        CALIFORNIA_SYNTHETIC = True
        return _california_synthetic()


def _california_synthetic(n: int = 8000, seed: int = 0) -> Tuple[pd.DataFrame, pd.Series]:
    """Synthetic stand-in for California housing: same columns and similar structure, `n` rows, fixed `seed`."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "MedInc": rng.lognormal(1.2, 0.5, n).clip(0.5, 15),
        "HouseAge": rng.integers(1, 53, n).astype(float),
        "AveRooms": rng.normal(5.4, 1.5, n).clip(1.5, 20),
        "AveBedrms": rng.normal(1.1, 0.3, n).clip(0.5, 5),
        "Population": rng.lognormal(7.0, 0.7, n).clip(3, 35000),
        "AveOccup": rng.lognormal(1.0, 0.3, n).clip(1, 20),
        "Latitude": rng.uniform(32.5, 42, n),
        "Longitude": rng.uniform(-124.3, -114.3, n),
    })
    y = (0.42 * X.MedInc + 0.008 * X.HouseAge - 0.15 * X.AveOccup
         - 0.10 * (X.Latitude - 36) + rng.normal(0, 0.6, n) + 0.3)
    return X, y.clip(0.15, 5.0).rename("price")


def diabetes() -> Tuple[pd.DataFrame, pd.Series]:
    """scikit-learn Diabetes (442 patients, 10 already standardized features).

    Returns:
        (X, y): the 10 features and `progression`, a measure of disease progression after one year.
    """
    from sklearn.datasets import load_diabetes

    d = load_diabetes(as_frame=True)
    return d.data, d.target.rename("progression")



def anscombe() -> pd.DataFrame:
    """Anscombe's quartet: 4 datasets of 11 points with the same summary statistics.

    Returns:
        A DataFrame with columns dataset ("I".."IV"), x, y.
    """
    x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
    y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
    y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
    rows = []
    for name, x, y in [("I", x123, y1), ("II", x123, y2), ("III", x123, y3), ("IV", x4, y4)]:
        rows += [{"dataset": name, "x": xi, "y": yi} for xi, yi in zip(x, y)]
    return pd.DataFrame(rows)



def salaries(n: int = 600, seed: int = 7) -> pd.DataFrame:
    """Gross annual salaries (euros) of employees.

    Generating process (hidden on purpose): log(salary) is linear in the features.

    Args:
        n: number of employees.
        seed: seed of the random generator (same seed, same data).
    """
    rng = np.random.default_rng(seed)
    years = rng.uniform(0, 30, n).round(1)
    hours = rng.integers(20, 51, n)
    city = rng.binomial(1, 0.35, n)
    degree = rng.binomial(1, 0.45, n)
    log_s = 9.6 + 0.03 * years + 0.008 * hours + 0.15 * city + 0.10 * degree + rng.normal(0, 0.12, n)
    return pd.DataFrame({
        "years_experience": years,
        "weekly_hours": hours,
        "big_city": city,
        "degree": degree,
        "salary": np.exp(log_s).round(0),
    })


def house_prices(n: int = 400, seed: int = 3) -> pd.DataFrame:
    """House prices in euros, with a right-skewed distribution.

    Args:
        n: number of houses.
        seed: seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(35, 220, n)
    rooms = np.clip((area / 40 + rng.normal(0, 0.6, n)).round(), 1, 7)
    dist = rng.exponential(6, n).clip(0.2, 30)
    log_p = 10.6 + 0.011 * area + 0.04 * rooms - 0.03 * dist + rng.normal(0, 0.18, n)
    return pd.DataFrame({
        "area": area.round(1),
        "rooms": rooms.astype(int),
        "distance_center_km": dist.round(2),
        "price": np.exp(log_p).round(0),
    })


def energy_consumption(n: int = 160, seed: int = 11) -> pd.DataFrame:
    """Daily energy consumption (kWh) as a function of the outside temperature (degrees C).

    The relationship is U-shaped: more is used in the cold (heating) and in the heat (air conditioning).

    Args:
        n: number of days.
        seed: seed of the random generator.

    Returns:
        A DataFrame with columns temperature and consumption.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(-5, 35, n)
    kwh = 50 + 0.16 * (t - 17) ** 2 + rng.normal(0, 6, n)
    return pd.DataFrame({"temperature": t.round(2), "consumption": kwh.round(2)})


def pure_noise(n: int = 100, p: int = 1000, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """X and y are INDEPENDENT Gaussians: there is no signal to find.

    Args:
        n: number of rows.
        p: number of features.
        seed: seed of the random generator.

    Returns:
        (X, y) with shapes (n, p) and (n,).
    """
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), rng.normal(size=n)


def trending_series(n: int = 400, seed: int = 5) -> pd.DataFrame:
    """Time series with two features that drift over time (increasing trend).

    Columns: t, x1, x2, y. The rows are in chronological order.

    Args:
        n: number of time steps.
        seed: seed of the random generator.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x1 = 10 + 0.05 * t + rng.normal(0, 1.0, n)
    x2 = 5 + 0.02 * t + rng.normal(0, 1.0, n)
    y = 3 + 1.5 * x1 - 2.0 * x2 + rng.normal(0, 1.0, n)
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "y": y})


def residual_mysteries(n: int = 200, seed: int = 21) -> Dict[str, pd.DataFrame]:
    """Four one-variable (x, y) datasets: one respects the assumptions, three violate one of them.

    Returns a dict {"A": DataFrame, "B": ..., "C": ..., "D": ...}. The rows are in
    order of acquisition. Which violation it is, you have to discover from the residuals.

    Args:
        n: rows per dataset.
        seed: seed of the random generator.
    """
    rng = np.random.default_rng(seed)

    xa = rng.uniform(1, 10, n)                                   # sigma grows with x
    ya = 2 + 1.5 * xa + rng.normal(0, 1, n) * (0.15 + 0.45 * xa)

    xb = rng.uniform(0, 10, n)                                   # correct model
    yb = 1 + 1.2 * xb + rng.normal(0, 1.5, n)

    xc = rng.uniform(-3, 3, n)                                   # curved relationship
    yc = 1 + 0.5 * xc + 0.9 * xc ** 2 + rng.normal(0, 0.8, n)

    xd = rng.uniform(0, 10, n)                                   # errors linked over time
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = 0.92 * e[i - 1] + rng.normal(0, 0.8)
    yd = 1 + 1.2 * xd + e

    return {k: pd.DataFrame({"x": x, "y": y})
            for k, (x, y) in {"A": (xa, ya), "B": (xb, yb), "C": (xc, yc), "D": (xd, yd)}.items()}


from .extra_data import *  # noqa: F401,F403,E402  (modules 2 onwards)
from .extra_data import _titanic_synthetic  # noqa: F401,E402
