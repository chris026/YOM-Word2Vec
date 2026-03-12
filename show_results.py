from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_FILES = [
    Path("tests/MBA.csv"),
    Path("tests/W2V.csv"),
]
METRICS = ["HitRate", "Recall", "MRR", "Precision", "Positives"]


def month_label(months: int) -> str:
    return f"{months} Month" if months == 1 else f"{months} Months"


def assign_model_index(df: pd.DataFrame) -> pd.Series:
    """Assign model index based on repeated K cycles (e.g. 5,10,20 | 5,10,20 ...)."""
    if df.empty:
        return pd.Series(dtype="int64")

    first_k = df.iloc[0]["K"]
    model_index: list[int] = []
    current_model = 1

    for i, k_value in enumerate(df["K"].tolist()):
        if i > 0 and k_value == first_k:
            current_model += 1
        model_index.append(current_model)

    return pd.Series(model_index, index=df.index, dtype="int64")


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, sep=";", decimal=",")

    required_cols = {"K"}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(
            f"{csv_path} fehlt erforderliche Spalte(n): {sorted(missing_required)}"
        )

    df["Model"] = assign_model_index(df)
    df["K"] = pd.to_numeric(df["K"], errors="coerce")
    return df


def plot_dataset(df: pd.DataFrame, title: str, output_path: Path | None = None) -> None:
    available_metrics = [m for m in METRICS if m in df.columns]
    if not available_metrics:
        raise ValueError("Keine darstellbaren Metrik-Spalten gefunden.")

    include_k5_impact = (
        "Recall" in df.columns and "HitRate" in df.columns and not df[df["K"] == 5].empty
    )
    total_plots = len(available_metrics) + (1 if include_k5_impact else 0)

    ncols = 3
    nrows = math.ceil(total_plots / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes_flat[idx]
        for model_id, model_df in df.groupby("Model", sort=True):
            model_df = model_df.sort_values("K")
            ax.plot(
                model_df["K"],
                model_df[metric],
                marker="o",
                linewidth=2,
                label=month_label(int(model_id)),
            )

        ax.set_title(metric)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(title="Training data")

    if include_k5_impact:
        impact_df = df[df["K"] == 5].copy().sort_values("Model")
        impact_ax = axes_flat[len(available_metrics)]
        for metric in ["Recall"]:
            impact_ax.plot(
                impact_df["Model"],
                impact_df[metric],
                marker="o",
                linewidth=2,
                label=metric,
            )
        impact_ax.set_title("K=5: Impact of More Training Data")
        impact_ax.set_xlabel("Training data (months)")
        #impact_ax.set_ylabel("Percent")
        impact_ax.set_xticks(impact_df["Model"].astype(int).tolist())
        #impact_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        impact_ax.grid(True, alpha=0.3)
        impact_ax.legend(title="Metric")

    for idx in range(total_plots, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {output_path}")


def plot_two_month_comparison(
    datasets: list[tuple[Path, pd.DataFrame]], output_path: Path | None = None
) -> None:
    model_id = 2
    selected: list[tuple[str, pd.DataFrame]] = []

    for csv_path, df in datasets:
        model_df = df[df["Model"] == model_id].copy()
        if not model_df.empty:
            selected.append((csv_path.stem, model_df.sort_values("K")))

    if len(selected) < 2:
        print(
            "Vergleichsplot uebersprungen: Mindestens zwei Dateien mit 2-Monats-Modell noetig."
        )
        return

    metrics = [m for m in METRICS if all(m in model_df.columns for _, model_df in selected)]
    if not metrics:
        print("Vergleichsplot uebersprungen: Keine gemeinsamen Metrik-Spalten gefunden.")
        return

    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4.5 * nrows))
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for idx, metric in enumerate(metrics):
        ax = axes_flat[idx]
        for label, model_df in selected:
            ax.plot(
                model_df["K"],
                model_df[metric],
                marker="o",
                linewidth=2,
                label=label,
            )
        ax.set_title(metric)
        ax.set_xlabel("K")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(title="Modell")

    for idx in range(len(metrics), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Comparison: 2-month model", fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Gespeichert: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Liest Ergebnis-CSV(s) ein und visualisiert Metriken pro Modell."
    )
    parser.add_argument(
        "csv_files",
        nargs="*",
        type=Path,
        default=DEFAULT_FILES,
        help="Pfad(e) zu CSV-Dateien (default: tests/erg_diana.csv tests/erg_christian.csv)",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optionales Verzeichnis zum Speichern der Plots als PNG.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Plots nicht interaktiv anzeigen (nur speichern oder stille Ausfuehrung).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show or args.save_dir is not None:
        plt.switch_backend("Agg")

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    datasets: list[tuple[Path, pd.DataFrame]] = []
    for csv_path in args.csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"Datei nicht gefunden: {csv_path}")

        df = load_results(csv_path)
        datasets.append((csv_path, df))
        output_path = (
            args.save_dir / f"{csv_path.stem}_plot.png" if args.save_dir else None
        )
        plot_dataset(df=df, title=f"Results: {csv_path.name}", output_path=output_path)

    comparison_output_path = (
        args.save_dir / "comparison_2_month_model.png" if args.save_dir else None
    )
    plot_two_month_comparison(datasets=datasets, output_path=comparison_output_path)

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
