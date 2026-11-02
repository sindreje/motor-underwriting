"""Grafer til appen. Hver funksjon tar inn tabeller og returnerer en figur."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
POINT = "#b8b6ae"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SOFT,
    "axes.titlecolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 14,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

CATEGORY_ORDER = {
    "driver_age": ["18-25", "26-30", "31-40", "41-50", "51-60", "61-70", "71+"],
    "vehicle_age": ["0", "1-3", "4-7", "8-12", "13+"],
    "vehicle_power": ["4-5", "6-7", "8-9", "10+"],
    "area": list("ABCDEF"),
}


def number(value: float, decimals: int = 0) -> str:
    """Norsk tallformat: 12 345,6"""
    text = f"{value:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


def percent(value: float, decimals: int = 1) -> str:
    return f"{number(value * 100, decimals)} %"


def signed_percent(value: float, decimals: int = 1) -> str:
    return f"{'+' if value >= 0 else '−'}{number(abs(value) * 100, decimals)} %"


def _ordered(table: pd.DataFrame, feature: str, sort_column: str) -> pd.DataFrame:
    """Faste bånd beholder rekkefølgen sin. Andre grupper sorteres på verdi."""
    if feature in CATEGORY_ORDER:
        rank = {c: i for i, c in enumerate(CATEGORY_ORDER[feature])}
        return table.assign(_rank=table["category"].astype(str).map(rank)).sort_values("_rank", ascending=False)
    return table.sort_values(sort_column)


def segment_chart(table: pd.DataFrame, feature: str, column: str, title: str, average: float, is_percent: bool):
    data = _ordered(table.assign(category=table["category"].astype(str)), feature, column)
    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.32 * len(data) + 1.2)))
    ax.barh(data["category"], data[column], color=BLUE, height=0.68)
    ax.axvline(average, color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    label = percent(average) if is_percent else number(average)
    ax.annotate(f"Porteføljen: {label}", (average, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9, color=INK_SOFT)
    ax.set_title(title, pad=26)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: percent(v, 0) if is_percent else number(v)))
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    fig.tight_layout()
    return fig


def relativity_chart(table: pd.DataFrame, feature: str, title: str):
    data = _ordered(table.assign(category=table["category"].astype(str)), feature, "relativity")
    fig, ax = plt.subplots(figsize=(7.5, max(2.6, 0.32 * len(data) + 1.2)))
    positions = np.arange(len(data))
    ax.hlines(positions, 1, data["relativity"], color=BLUE, linewidth=2)
    ax.scatter(data["relativity"], positions, color=BLUE, s=42, zorder=3)
    ax.axvline(1, color=AXIS, linewidth=1.2)
    ax.set_yticks(positions, data["category"])
    ax.set_xscale("log")
    low, high = min(data["relativity"].min(), 1), max(data["relativity"].max(), 1)
    ax.set_xlim(low * 0.92, high * 1.08)
    ticks = [t for t in (0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.25, 1.5, 2, 3, 4) if low * 0.92 <= t <= high * 1.08]
    ax.set_xticks(ticks, [number(t, 2) for t in ticks])
    ax.minorticks_off()
    ax.set_title(title)
    ax.set_xlabel("Relativitet mot referansegruppen (1,00)")
    ax.grid(axis="x")
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    return fig


def funnel_chart(fleets: pd.DataFrame, selected: pd.Series | None = None):
    """Viser A/E mot størrelse. Små flåter spriker mest, selv uten reell forskjell."""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.scatter(fleets["expected_claims"], fleets["actual_vs_expected"], s=9, color=POINT, alpha=0.55, linewidths=0)
    x = np.geomspace(max(fleets["expected_claims"].min(), 0.1), fleets["expected_claims"].max() * 1.15, 200)
    ax.plot(x, 1 + 2 / np.sqrt(x), color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)), label="Ventet tilfeldig variasjon (±2 SD)")
    ax.plot(x, np.maximum(1 - 2 / np.sqrt(x), 0), color=INK_SOFT, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.axhline(1, color=AXIS, linewidth=1.2)
    if selected is not None:
        ax.scatter(selected["expected_claims"], selected["actual_vs_expected"], s=90, color=ORANGE,
                   edgecolors=SURFACE, linewidths=2, zorder=4, label=selected["fleet_name"])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: number(v, 0) if v >= 1 else number(v, 1)))
    ax.set_ylim(0, min(fleets["actual_vs_expected"].quantile(0.995) * 1.1, 6))
    ax.set_xlabel("Forventet antall skader (logaritmisk skala)")
    ax.set_ylabel("Faktisk / forventet (A/E)")
    ax.set_title("Hvilke flåter skiller seg ut?")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def price_chart(current: float, technical: float, proposed: float):
    labels = ["Anbefalt premie", "Teknisk pris", "Dagens premie (simulert)"]
    values = [proposed, technical, current]
    fig, ax = plt.subplots(figsize=(7.5, 2.4))
    bars = ax.barh(labels, values, color=[ORANGE, BLUE, POINT], height=0.62)
    for bar, value in zip(bars, values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"  {number(value)} €", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, max(values) * 1.2)
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    fig.tight_layout()
    return fig


def credibility_chart(k: float, expected_claims: float, z: float):
    fig, ax = plt.subplots(figsize=(7.5, 3.1))
    top = max(expected_claims * 1.6, k * 2)
    x = np.linspace(0, top, 200)
    ax.plot(x, x / (x + k), color=BLUE, linewidth=2)
    ax.scatter([expected_claims], [z], s=90, color=ORANGE, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"Denne flåten: Z = {percent(z, 0)}", (expected_claims, z), xytext=(8, 10),
                textcoords="offset points", fontsize=10, color=INK)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, top)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent(v, 0)))
    ax.set_xlabel("Forventet antall skader")
    ax.set_ylabel("Vekt på flåtens historikk")
    ax.set_title("Større flåter får mer vekt på egen historikk")
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def lift_chart(lift: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.plot(lift["group"], lift["predicted_frequency"], color=BLUE, linewidth=2, marker="o", markersize=6, label="Predikert")
    ax.plot(lift["group"], lift["actual_frequency"], color=ORANGE, linewidth=2, marker="o", markersize=6, label="Faktisk")
    ax.set_xticks(lift["group"])
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: percent(v, 0)))
    ax.set_xlabel("Risikogruppe (1 = lavest predikert risiko)")
    ax.set_ylabel("Skadefrekvens")
    ax.set_title("Treffer modellen på data den ikke har sett?")
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    fig.tight_layout()
    return fig


def action_chart(counts: pd.Series):
    data = counts.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    bars = ax.barh(data.index, data.values, color=BLUE, height=0.66)
    for bar, value in zip(bars, data.values):
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"  {number(value)}", va="center", color=INK)
    ax.set_xlim(0, max(data.max() * 1.15, 1))
    ax.xaxis.set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Anbefaling per flåte")
    fig.tight_layout()
    return fig
