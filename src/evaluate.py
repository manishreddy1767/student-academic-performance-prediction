"""Evaluation metrics and report figure generation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    """Calculate the four requested regression metrics."""
    mse = mean_squared_error(actual, predicted)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "mse": float(mse),
        "rmse": float(mse**0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def save_eda_figures(frame: pd.DataFrame, output_dir: Path) -> None:
    """Save EDA charts based only on the supplied dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 5))
    sns.histplot(frame["G3"], bins=range(0, 22), discrete=True)
    plt.title("Distribution of final mathematics grade (G3)")
    plt.xlabel("Final grade (0-20)")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=150)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(frame.select_dtypes("number").corr(), cmap="vlag", center=0)
    plt.title("Correlation matrix of numeric variables")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.boxplot(data=frame, x="studytime", y="G3", ax=axes[0])
    sns.scatterplot(data=frame, x="absences", y="G3", ax=axes[1])
    sns.boxplot(data=frame, x="failures", y="G3", ax=axes[2])
    axes[0].set_title("Study time and G3")
    axes[1].set_title("Absences and G3")
    axes[2].set_title("Failures and G3")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_relationships.png", dpi=150)
    plt.close()


def save_evaluation_figures(
    actual, predictions: dict[str, object], output_dir: Path
) -> None:
    """Save actual/predicted, residual, and model comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    first_name, first_pred = next(iter(predictions.items()))

    plt.figure(figsize=(7, 5))
    for name, predicted in predictions.items():
        plt.scatter(actual, predicted, alpha=0.55, label=name)
    plt.plot([0, 20], [0, 20], "k--", label="Perfect prediction")
    plt.xlabel("Actual G3")
    plt.ylabel("Predicted G3")
    plt.title("Actual versus predicted final grades")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()

    residuals = pd.Series(actual) - pd.Series(first_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(residuals, kde=True, ax=axes[0])
    axes[0].set_title(f"Residual distribution ({first_name})")
    axes[0].set_xlabel("Actual - predicted")
    axes[1].scatter(first_pred, residuals, alpha=0.6)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title(f"Residuals versus predictions ({first_name})")
    axes[1].set_xlabel("Predicted G3")
    axes[1].set_ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_dir / "residuals.png", dpi=150)
    plt.close()
