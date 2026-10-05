"""Recurring plots, so they don't have to be rewritten in every notebook."""
from typing import Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy import stats


def residual_panel(y_true: Union[Sequence[float], np.ndarray], y_pred: Union[Sequence[float], np.ndarray], title: str = "") -> Figure:
    """Three diagnostic plots of the residuals: residuals vs predicted, Q-Q plot, histogram.

    Args:
        y_true: observed values.
        y_pred: values predicted by the model.
        title: optional title for the whole figure.

    Returns:
        The matplotlib figure with the three panels.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].scatter(y_pred, residuals, s=12, alpha=0.6)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set(xlabel="predicted values", ylabel="residual", title="Residuals vs predicted")
    stats.probplot(residuals, plot=axes[1])
    axes[1].set_title("Q-Q plot of the residuals")
    axes[2].hist(residuals, bins=25, color="#1F6E5C", alpha=0.8)
    axes[2].set(xlabel="residual", title="Distribution of the residuals")
    if title:
        fig.suptitle(title, y=1.03)
    fig.tight_layout()
    return fig
