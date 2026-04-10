"""
Visualization utilities for backtest results, feature importance, and model comparison.

All plots are designed to work in Jupyter notebooks with matplotlib/plotly.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)

# Color palette
COLORS = {
    "strategy": "#2196F3",
    "benchmark": "#FF9800",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "primary": "#3F51B5",
    "secondary": "#E91E63",
}


def plot_equity_curve(
    result: pd.DataFrame,
    title: str = "Equity Curve: Strategy vs Benchmark",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the equity curve of the strategy vs buy-and-hold benchmark.

    Args:
        result: Backtest result DataFrame from BacktestEngine.run().
        title: Chart title.
        figsize: Figure size.
        save_path: If given, save the figure to this path.

    Returns:
        matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    initial = result["portfolio_value"].iloc[0]
    strategy_equity = result["portfolio_value"] / initial
    benchmark_equity = result["price"] / result["price"].iloc[0]

    ax.plot(strategy_equity.index, strategy_equity.values,
            label="Strategy", color=COLORS["strategy"], linewidth=1.5)
    ax.plot(benchmark_equity.index, benchmark_equity.values,
            label="Benchmark (Buy & Hold)", color=COLORS["benchmark"], linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Normalized Value")
    ax.set_xlabel("Date")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved equity curve to {save_path}")

    return fig


def plot_drawdown(
    result: pd.DataFrame,
    title: str = "Drawdown Analysis",
    figsize: tuple = (14, 4),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot the drawdown series for strategy and benchmark.
    """
    pv = result["portfolio_value"]
    strategy_dd = (pv - pv.cummax()) / pv.cummax()
    bench_dd = (result["price"] - result["price"].cummax()) / result["price"].cummax()

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(strategy_dd.index, strategy_dd.values, 0,
                    color=COLORS["strategy"], alpha=0.4, label="Strategy Drawdown")
    ax.fill_between(bench_dd.index, bench_dd.values, 0,
                    color=COLORS["benchmark"], alpha=0.3, label="Benchmark Drawdown")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_monthly_returns(
    result: pd.DataFrame,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Heatmap of monthly returns.
    """
    ret = result["strategy_return"]

    # Build monthly return table
    monthly = ret.resample("M").sum()
    monthly_df = pd.DataFrame({
        "year": monthly.index.year,
        "month": monthly.index.month,
        "return": monthly.values,
    })
    pivot = monthly_df.pivot(index="year", columns="month", values="return")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Return")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_feature_importance(
    importances: pd.Series,
    top_n: int = 20,
    title: str = "Feature Importance",
    figsize: tuple = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.
    """
    sorted_imp = importances.sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(sorted_imp.index, sorted_imp.values, color=COLORS["primary"], alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart comparing models across metrics.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "f1", "auc"]

    available = [m for m in metrics if m in comparison_df.columns]
    if not available:
        logger.warning("No matching metrics found for comparison plot.")
        return plt.figure()

    models = comparison_df.index.tolist()
    n_metrics = len(available)
    n_models = len(models)
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(n_models)

    for i, metric in enumerate(available):
        offset = (i - n_metrics / 2 + 0.5) * width
        vals = comparison_df[metric].values
        ax.bar(x + offset, vals, width, label=metric.capitalize(), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_signals(
    prices: pd.Series,
    signals: pd.Series,
    title: str = "Trading Signals",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot price with buy/sell signal markers overlaid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(prices.index, prices.values, color="gray", linewidth=1, alpha=0.7, label="Price")

    buy_mask = signals == 1
    sell_mask = signals == 0

    ax.scatter(prices.index[buy_mask], prices.values[buy_mask],
               marker="^", color=COLORS["positive"], s=30, label="Buy Signal", zorder=5)
    ax.scatter(prices.index[sell_mask], prices.values[sell_mask],
               marker="v", color=COLORS["negative"], s=30, label="Sell Signal", zorder=5)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Price")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
