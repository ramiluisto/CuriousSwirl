"""
Plotting functions for semantic similarity analysis.

All functions log what they plot for testability.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

PAIR_TYPE_COLORS = {
    "synonyms": "#4C72B0",
    "antonyms": "#DD8452",
    "random_noun_pairs": "#55A868",          # legacy
    "random_pairs": "#C44E52",               # legacy
    "shuffled_synonym_words": "#55A868",     # new
    "shuffled_antonym_words": "#C44E52",     # new
    "antonym_words": "#DD8452",              # base word analysis
    "synonym_words": "#4C72B0",              # base word analysis
}

MODEL_COLORS = {
    "bert_base_cased": "#4C72B0",
    "modernbert_base": "#DD8452",
    "word2vec": "#55A868",
    "glove": "#C44E52",
}


def plot_metric_distributions(
    metrics_by_type: Dict[str, np.ndarray],
    metric_name: str,
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Violin/box plot of a metric across pair types.

    Args:
        metrics_by_type: {pair_type: array of metric values}.
        metric_name: Name of the metric (for axis label).
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = []
    labels = []
    for pair_type, values in sorted(metrics_by_type.items()):
        data.append(values)
        labels.append(pair_type)
        logger.info("Plotted %d points for %s %s", len(values), pair_type, metric_name)

    parts = ax.violinplot(data, showmeans=True, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Distribution of {metric_name}")
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved distribution plot to %s", save_path)

    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Plot a confusion matrix heatmap.

    Args:
        cm: Confusion matrix array (n_classes x n_classes).
        class_names: Labels for each class.
        title: Plot title.
        save_path: If provided, save figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    fig.tight_layout()

    total = cm.sum()
    logger.info("Plotted confusion matrix: %d total predictions, %d classes", total, len(class_names))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved confusion matrix to %s", save_path)

    return fig


def plot_pvalue_heatmap(
    pvalues: Dict[str, Dict[str, float]],
    title: str = "p-value Heatmap",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> plt.Figure:
    """Plot a heatmap of p-values from pairwise comparisons.

    Args:
        pvalues: {comparison_name: {metric_name: p_value}}.
        title: Plot title.
        save_path: If provided, save figure.
    """
    comparisons = sorted(pvalues.keys())
    if not comparisons:
        logger.warning("No comparisons to plot")
        fig, ax = plt.subplots()
        return fig

    metrics = sorted(next(iter(pvalues.values())).keys())
    matrix = np.zeros((len(comparisons), len(metrics)))

    for i, comp in enumerate(comparisons):
        for j, metric in enumerate(metrics):
            matrix[i, j] = pvalues[comp].get(metric, float("nan"))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, annot=True, fmt=".2e", cmap="RdYlGn_r",
        xticklabels=metrics, yticklabels=comparisons,
        vmin=0, vmax=0.05, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()

    logger.info("Plotted p-value heatmap: %d comparisons x %d metrics", len(comparisons), len(metrics))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved p-value heatmap to %s", save_path)

    return fig


def plot_classification_comparison(
    results: List[Dict],
    title: str = "Classification Results",
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Bar chart comparing classification accuracies.

    Args:
        results: List of dicts with 'classifier', 'accuracy', and optionally 'task'.
        title: Plot title.
        save_path: If provided, save figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [r.get("task", r.get("classifier", "?")) for r in results]
    accuracies = [r.get("accuracy", 0.0) for r in results]
    classifiers = [r.get("classifier", "unknown") for r in results]

    colors = {"logistic": "#4C72B0", "xgboost": "#55A868", "shallow_nn": "#C44E52"}
    bar_colors = [colors.get(c, "#999999") for c in classifiers]

    bars = ax.bar(range(len(labels)), accuracies, color=bar_colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Chance (binary)")

    fig.tight_layout()

    logger.info("Plotted %d classification results", len(results))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved classification comparison to %s", save_path)

    return fig


def plot_metric_histograms(
    metrics_by_type: Dict[str, np.ndarray],
    metric_name: str,
    title: str = "",
    bins: int = 50,
    density: bool = False,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Overlaid histograms of one metric across selected pair types.

    Args:
        metrics_by_type: {pair_type: array of metric values}.
        metric_name: Name of the metric (for axis label).
        title: Plot title.
        bins: Number of histogram bins.
        density: If True, normalize histograms to show frequency (density).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for pair_type, values in sorted(metrics_by_type.items()):
        color = PAIR_TYPE_COLORS.get(pair_type, "#999999")
        ax.hist(values, bins=bins, alpha=0.5, label=pair_type, color=color, density=density)
        logger.info("Histogram: %d points for %s %s", len(values), pair_type, metric_name)

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Frequency" if density else "Count")
    ax.set_title(title or f"Distribution of {metric_name}")
    ax.legend()
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved histogram plot to %s", save_path)

    return fig


def plot_cross_model_boxplots(
    data_by_model: Dict[str, np.ndarray],
    metric_name: str,
    pair_type: str,
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Side-by-side box plots comparing one metric+pair_type across multiple models.

    Args:
        data_by_model: {model_name: array of metric values}.
        metric_name: Name of the metric (for axis label).
        pair_type: Pair type being compared.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_names = sorted(data_by_model.keys())
    data = [data_by_model[m] for m in model_names]
    colors = [MODEL_COLORS.get(m, "#999999") for m in model_names]

    bp = ax.boxplot(data, patch_artist=True, tick_labels=model_names)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} for {pair_type} across models")
    fig.tight_layout()

    logger.info("Cross-model boxplot: %d models for %s / %s", len(model_names), metric_name, pair_type)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved cross-model boxplot to %s", save_path)

    return fig


def plot_effect_size_heatmap(
    effect_sizes: Dict[str, Dict[str, float]],
    title: str = "Cohen's d Effect Sizes",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> plt.Figure:
    """Heatmap of Cohen's d values. Rows=comparisons, cols=metrics.

    Args:
        effect_sizes: {comparison_name: {metric_name: cohens_d}}.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    comparisons = sorted(effect_sizes.keys())
    if not comparisons:
        logger.warning("No effect sizes to plot")
        fig, ax = plt.subplots()
        return fig

    metrics = sorted(next(iter(effect_sizes.values())).keys())
    matrix = np.zeros((len(comparisons), len(metrics)))

    for i, comp in enumerate(comparisons):
        for j, metric in enumerate(metrics):
            matrix[i, j] = effect_sizes[comp].get(metric, float("nan"))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        xticklabels=metrics, yticklabels=comparisons,
        center=0, vmin=-vmax, vmax=vmax, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()

    logger.info("Effect size heatmap: %d comparisons x %d metrics", len(comparisons), len(metrics))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved effect size heatmap to %s", save_path)

    return fig


def plot_cross_model_effect_sizes(
    data_by_model: Dict[str, Dict[str, float]],
    comparison: str,
    title: str = "",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Cohen's d heatmap for a single comparison across multiple models.
    Rows=models, cols=metrics.

    Args:
        data_by_model: {model_slug: {metric_name: cohens_d}}.
        comparison: The comparison label.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    models = sorted(data_by_model.keys())
    if not models:
        logger.warning("No models to plot")
        fig, ax = plt.subplots()
        return fig

    metrics = sorted(next(iter(data_by_model.values())).keys())
    matrix = np.zeros((len(models), len(metrics)))

    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            matrix[i, j] = data_by_model[model].get(metric, float("nan"))

    fig, ax = plt.subplots(figsize=figsize)
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    sns.heatmap(
        matrix, annot=True, fmt=".2f", cmap="RdBu_r",
        xticklabels=metrics, yticklabels=models,
        center=0, vmin=-vmax, vmax=vmax, ax=ax,
    )
    ax.set_title(title or f"Cohen's d: {comparison}")
    fig.tight_layout()

    logger.info("Cross-model effect sizes: %d models x %d metrics for %s", len(models), len(metrics), comparison)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved cross-model effect size heatmap to %s", save_path)

    return fig


def plot_classification_grouped_bars(
    results_df: "pd.DataFrame",
    value_col: str = "accuracy",
    group_col: str = "classifier",
    x_col: str = "model",
    title: str = "",
    chance_level: Optional[float] = 0.5,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Grouped bar chart of classification results.

    Args:
        results_df: DataFrame with columns for model, classifier, and value.
        value_col: Column to plot on y-axis (e.g. 'accuracy').
        group_col: Column for grouping bars (e.g. 'classifier').
        x_col: Column for x-axis positions (e.g. 'model').
        title: Plot title.
        chance_level: If not None, draw a horizontal line at this level.
        save_path: If provided, save figure to this path.
    """
    import pandas as pd

    fig, ax = plt.subplots(figsize=figsize)

    x_labels = sorted(results_df[x_col].unique())
    groups = sorted(results_df[group_col].unique())
    n_groups = len(groups)
    bar_width = 0.8 / max(n_groups, 1)
    classifier_colors = {"logistic": "#4C72B0", "xgboost": "#55A868", "shallow_nn": "#C44E52"}

    for idx, group in enumerate(groups):
        group_data = results_df[results_df[group_col] == group]
        values = []
        for x_label in x_labels:
            match = group_data[group_data[x_col] == x_label]
            values.append(match[value_col].values[0] if len(match) > 0 else 0.0)

        positions = np.arange(len(x_labels)) + idx * bar_width
        color = classifier_colors.get(group, f"C{idx}")
        ax.bar(positions, values, bar_width, label=group, color=color, alpha=0.85)

    ax.set_xticks(np.arange(len(x_labels)) + bar_width * (n_groups - 1) / 2)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylabel(value_col.replace("_", " ").title())
    ax.set_ylim(0, 1.0)
    ax.set_title(title or f"Classification: {value_col}")
    ax.legend()

    if chance_level is not None:
        ax.axhline(y=chance_level, color="gray", linestyle="--", alpha=0.5, label="Chance")

    fig.tight_layout()

    logger.info(
        "Grouped bar chart: %d x-values, %d groups, col=%s",
        len(x_labels), n_groups, value_col,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved grouped bar chart to %s", save_path)

    return fig


def plot_projection_scatter(
    coords: np.ndarray,
    labels: List[str],
    title: str = "2D Projection",
    pair_types_to_show: Optional[List[str]] = None,
    alpha: float = 0.6,
    point_size: float = 15,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 8),
    dpi: int = 300,
) -> plt.Figure:
    """2D scatter plot of projection coordinates colored by pair type.

    Args:
        coords: Array of shape (n_samples, 2).
        labels: Pair type label for each sample.
        title: Plot title.
        pair_types_to_show: If provided, only show these pair types.
        alpha: Point transparency.
        point_size: Marker size.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = sorted(set(labels))
    if pair_types_to_show is not None:
        unique_labels = [l for l in unique_labels if l in pair_types_to_show]

    for label in unique_labels:
        mask = np.array([l == label for l in labels])
        color = PAIR_TYPE_COLORS.get(label, "#999999")
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=label, alpha=alpha, s=point_size, edgecolors="none",
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend(markerscale=3, fontsize=12)
    fig.tight_layout()

    n_shown = sum(1 for l in labels if pair_types_to_show is None or l in pair_types_to_show)
    logger.info("Projection scatter: %d points, %d pair types", n_shown, len(unique_labels))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved projection scatter to %s", save_path)

    return fig


def plot_projection_grid(
    grid_results,
    labels,
    row_values,
    col_values,
    row_label,
    col_label,
    method,
    model_name="",
    input_type="",
    point_size=1.0,
    alpha=0.4,
    dpi=300,
) -> plt.Figure:
    """Render a 4x4 grid of projection scatters varying two hyperparameters.

    Args:
        grid_results: List of (row_val, col_val, coords) tuples.
        labels: np.ndarray of pair type strings for each point.
        row_values: List of row hyperparameter values.
        col_values: List of column hyperparameter values.
        row_label: Display name for the row parameter.
        col_label: Display name for the column parameter.
        method: Projection method name (e.g. "t-SNE", "UMAP").
        model_name: Model display name for suptitle.
        input_type: Input type display name for suptitle.
        point_size: Marker size.
        alpha: Point transparency.
        dpi: Figure DPI.

    Returns:
        matplotlib Figure with 4x4 subplots (A4 landscape).
    """
    n_rows = len(row_values)
    n_cols = len(col_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.69, 8.27))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    # Index results by (row_val, col_val)
    result_map = {(rv, cv): coords for rv, cv, coords in grid_results}

    unique_labels = sorted(set(labels))
    labels_arr = np.array(labels)

    for ri, rv in enumerate(row_values):
        for ci, cv in enumerate(col_values):
            ax = axes[ri, ci]
            coords = result_map.get((rv, cv))
            if coords is None:
                ax.set_visible(False)
                continue
            for label in unique_labels:
                mask = labels_arr == label
                color = PAIR_TYPE_COLORS.get(label, "#999999")
                ax.scatter(
                    coords[mask, 0], coords[mask, 1],
                    c=color, s=point_size, alpha=alpha,
                    edgecolors="none", rasterized=True,
                )
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"{col_label}={cv}", fontsize=7)
            if ci == 0:
                ax.set_ylabel(f"{row_label}={rv}", fontsize=7)

    # Legend at bottom
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=PAIR_TYPE_COLORS.get(l, "#999999"),
               markersize=8, label=l)
        for l in unique_labels
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(unique_labels),
        fontsize=9, frameon=False,
    )

    suptitle = f"{method} Hyperparameter Grid"
    if model_name:
        suptitle += f" -- {model_name}"
    if input_type:
        suptitle += f" ({input_type})"
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    logger.info(
        "Projection grid: %dx%d, method=%s, %d points",
        n_rows, n_cols, method, len(labels),
    )
    return fig


def plot_scree(
    explained_variance_ratio: np.ndarray,
    cumulative: bool = True,
    title: str = "PCA Scree Plot",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Bar chart of PCA explained variance with optional cumulative line.

    Args:
        explained_variance_ratio: Array of variance ratios per component.
        cumulative: If True, overlay cumulative variance line.
        title: Plot title.
        save_path: If provided, save figure to this path.
    """
    n = len(explained_variance_ratio)
    components = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(components, explained_variance_ratio * 100, color="#4C72B0", alpha=0.7,
           label="Individual")

    if cumulative:
        cum_var = np.cumsum(explained_variance_ratio) * 100
        ax.plot(components, cum_var, "o-", color="#DD8452", markersize=4,
                label="Cumulative")

        # Threshold markers
        for threshold, style in [(90, "--"), (95, ":")]:
            ax.axhline(y=threshold, color="gray", linestyle=style, alpha=0.5)
            ax.text(n * 0.95, threshold + 0.5, f"{threshold}%", ha="right",
                    va="bottom", fontsize=9, color="gray")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0.5, n + 0.5)
    fig.tight_layout()

    logger.info(
        "Scree plot: %d components, top-2=%.1f%%, top-10=%.1f%%",
        n,
        explained_variance_ratio[:2].sum() * 100,
        explained_variance_ratio[:min(10, n)].sum() * 100,
    )

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved scree plot to %s", save_path)

    return fig


def plot_scree_multi(
    evr_by_group: Dict[str, np.ndarray],
    cumulative: bool = True,
    title: str = "PCA Scree Plot",
    max_components: Optional[int] = None,
    colors: Optional[Dict[str, str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Overlaid scree curves for multiple groups (e.g. pair types).

    Args:
        evr_by_group: Mapping of group label to explained variance ratio array.
        cumulative: If True, plot cumulative variance; otherwise individual.
        title: Plot title.
        max_components: If set, truncate curves to this many components.
        colors: Optional mapping of group label to color hex string.
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = colors or {}
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (label, evr) in enumerate(evr_by_group.items()):
        if max_components is not None:
            evr = evr[:max_components]
        n = len(evr)
        components = np.arange(1, n + 1)

        color = colors.get(label, color_cycle[idx % len(color_cycle)])
        if cumulative:
            values = np.cumsum(evr) * 100
        else:
            values = evr * 100
        ax.plot(components, values, "o-", color=color, markersize=4, label=label)

    if cumulative:
        for threshold, style in [(90, "--"), (95, ":")]:
            ax.axhline(y=threshold, color="gray", linestyle=style, alpha=0.5)
            ax.text(0.98, threshold + 0.5, f"{threshold}%", ha="right",
                    va="bottom", fontsize=9, color="gray",
                    transform=ax.get_yaxis_transform())

    ax.set_xlabel("Principal Component")
    ylabel = "Cumulative Variance (%)" if cumulative else "Explained Variance (%)"
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    n_groups = len(evr_by_group)
    logger.info("Scree multi: %d groups, cumulative=%s", n_groups, cumulative)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved scree multi plot to %s", save_path)

    return fig


# =============================================================================
# Confounder Analysis Plots
# =============================================================================


def plot_jaccard_heatmap(
    jaccard_matrix: Dict[str, Dict[str, float]],
    title: str = "Vocabulary Overlap (Jaccard)",
    save_path: Optional[Path] = None,
    figsize: tuple = (8, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Heatmap of Jaccard similarity between pair type vocabularies."""
    import pandas as pd
    types = sorted(jaccard_matrix.keys())
    data = [[jaccard_matrix[ta][tb] for tb in types] for ta in types]
    df = pd.DataFrame(data, index=types, columns=types)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=1,
                square=True, ax=ax)
    ax.set_title(title)
    fig.tight_layout()

    logger.info("Jaccard heatmap: %d types", len(types))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved Jaccard heatmap to %s", save_path)

    return fig


def plot_pos_stacked_bars(
    pos_df: "pd.DataFrame",
    title: str = "POS Distribution per Pair Type",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Stacked bar chart of POS proportions per pair type."""
    pivot = pos_df.pivot_table(
        index="pair_type", columns="pos", values="fraction", fill_value=0,
    )

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", stacked=True, ax=ax, colormap="Set2")
    ax.set_ylabel("Fraction")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.legend(title="POS", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    logger.info("POS stacked bars: %d types, %d POS categories",
                len(pivot), len(pivot.columns))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved POS stacked bars to %s", save_path)

    return fig


def plot_degree_distributions(
    graph_stats: Dict[str, Dict],
    pairs_by_type: Dict[str, List],
    title: str = "Word Degree Distribution",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
    dpi: int = 300,
) -> plt.Figure:
    """Overlaid degree histograms for each pair type's co-occurrence graph."""
    from collections import defaultdict as dd

    fig, ax = plt.subplots(figsize=figsize)

    for ptype, pairs in pairs_by_type.items():
        adj: Dict[str, set] = {}
        for w1, w2 in pairs:
            adj.setdefault(w1, set()).add(w2)
            adj.setdefault(w2, set()).add(w1)
        degrees = [len(v) for v in adj.values()]
        if degrees:
            color = PAIR_TYPE_COLORS.get(ptype, None)
            ax.hist(degrees, bins=range(1, max(degrees) + 2), alpha=0.5,
                    label=ptype, color=color, edgecolor="white")

    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    logger.info("Degree distribution plot: %d types", len(pairs_by_type))

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved degree distribution to %s", save_path)

    return fig


def plot_morphology_pie(
    morph_counts: Dict[str, int],
    title: str = "Antonym Morphology Breakdown",
    save_path: Optional[Path] = None,
    figsize: tuple = (7, 7),
    dpi: int = 300,
) -> plt.Figure:
    """Pie chart of prefix vs semantic antonym breakdown."""
    fig, ax = plt.subplots(figsize=figsize)

    labels = list(morph_counts.keys())
    sizes = list(morph_counts.values())
    colors = ["#4C72B0", "#DD8452"] + sns.color_palette("Set2", max(0, len(labels) - 2))

    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)],
           startangle=90)
    ax.set_title(title)

    logger.info("Morphology pie: %s", morph_counts)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info("Saved morphology pie to %s", save_path)

    return fig
