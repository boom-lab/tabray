"""Tests for visualization helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from data_sparsity.utils.viz import plot_grid_case


def test_plot_grid_case_supports_one_dimensional_variables(monkeypatch):
    """plot_grid_case should render a 1D variable on the full 2D grid."""
    ds = xr.Dataset(
        data_vars={
            "var0": ("x1", np.array([1.0, np.nan, 3.0, 4.0])),
        },
        coords={
            "x0": ("x0", np.array([0.1, 0.4, 0.7])),
            "x1": ("x1", np.array([0.2, 0.5, 0.8, 0.9])),
        },
    )
    df = pd.DataFrame(
        {
            "x0": [0.4, 0.4, 0.4, 0.4],
            "x1": [0.2, 0.8, 0.9, 0.5],
            "var0": [1.0, 3.0, 4.0, np.nan],
        }
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_grid_case(ds, df, "var0", 0, "one-dimensional variable")
    fig = plt.gcf()
    ax = fig.axes[0]
    assert len(ax.get_yticks()) > 0
    assert ax.collections[-1].get_offsets().shape[0] == 3
    plt.close("all")


def test_plot_grid_case_supports_two_dimensional_variables(monkeypatch):
    """plot_grid_case should handle a variable with two dimensions."""
    ds = xr.Dataset(
        data_vars={
            "var0": (
                ("x0", "x1"),
                np.array(
                    [
                        [1.0, np.nan, 2.0, np.nan],
                        [np.nan, 4.0, np.nan, 5.0],
                        [6.0, np.nan, np.nan, 8.0],
                    ]
                ),
            ),
        },
        coords={
            "x0": ("x0", np.array([0.1, 0.4, 0.7])),
            "x1": ("x1", np.array([0.2, 0.5, 0.8, 0.9])),
        },
    )
    df = pd.DataFrame(
        {
            "x0": [0.1, 0.4, 0.7],
            "x1": [0.2, 0.5, 0.8],
            "var0": [1.0, 4.0, 6.0],
        }
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    plot_grid_case(ds, df, "var0", 0, "two-dimensional variable")
    plt.close("all")
