################################################################
#
# Plots using plotly
#
################################################################

import math
from typing import Optional

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_model_performance(
    models: list[str],
    tasks: list[str],
    performance_data: dict[str, list[float]],
    agg_scores: Optional[dict[str, float]],
    mode: str = "overlay",  # "overlay" or "grid"
    title: str = "Model Performance",
    save_path: Optional[str] = None,
    save_format: str = "html",  # "html", "png", "pdf", "svg"
):
    """
    Create radar charts for model performance across tasks.

    Args:
        models: List of model names
        tasks: List of task names
        performance_data: Dict mapping model names to performance scores (0-100)
        mode: "overlay" for single chart, "grid" for subplot grid
        title: Main title for the plot(s)
        save_path: Optional path to save the plot
        save_format: Format for saving ("html", "png", "pdf", "svg")
        show_plot: Whether to display the plot
        renderer: How to display ("browser", "png", "svg", "notebook")

    Returns:
        plotly.graph_objects.Figure
    """

    # Validate input
    if not all(model in performance_data for model in models):
        missing = [m for m in models if m not in performance_data]
        raise ValueError(f"Missing performance data for models: {missing}")

    if mode == "overlay":
        fig = _create_overlay_plot(models, tasks, performance_data, title, agg_scores)
    elif mode == "grid":
        fig = _create_grid_plot(models, tasks, performance_data, title, save_path, agg_scores)
    else:
        raise ValueError("Mode must be 'overlay' or 'grid'")

    # Save the plot if requested
    if save_path:
        if save_format.lower() == "html":
            fig.write_html(save_path)
        elif save_format.lower() in ["png", "jpg", "jpeg", "svg", "pdf"]:
            fig.write_image(save_path)
        else:
            raise ValueError("Unsupported save format. Use 'html', 'png', 'pdf', or 'svg'")

    return fig

def _create_overlay_plot(models, tasks, performance_data, title, agg_scores):
    """Create a single radar chart with all models overlaid."""
    fig = go.Figure()

    # Color palette for different models
    colors = px.colors.qualitative.Set3

    for i, model in enumerate(models):
        scores = performance_data[model]

        # Add the first point at the end to close the radar chart
        scores_closed = scores + [scores[0]]
        tasks_closed = tasks + [tasks[0]]

        color = colors[i % len(colors)]

        if agg_scores:
            model = f"[{agg_scores[model]:.2f}] {model}"

        fig.add_trace(go.Scatterpolar(
            r=scores_closed,
            theta=tasks_closed,
            fill='toself',
            name=model,
            line={"color": color, "width": 2},
            fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
            hovertemplate=f'<b>{model}</b><br>Task: %{{theta}}<br>Score: %{{r:.1f}}%<extra></extra>'
        ))

    fig.update_layout(
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "ticksuffix": '%',
                "gridcolor": 'lightgray',
                "angle": 90,
            },
            "angularaxis": {
                "tickfont": {"size": 12},
                "gridcolor": 'lightgray',
                "rotation": 5,
            },
            "domain":{"x":[0, 0.6], "y":[0.1, 0.9]}
        },
        title={"text": title, "x": 0.5, "font": {"size": 16}},
        showlegend=True,
        legend={
            "orientation": "v",
            "yanchor": "top",
            "y": 0.9,
            "xanchor": "left",
            "x": 0.65
        },
        width=1400,
        height=700,
        margin={"l": 50, "r": 200, "t": 80, "b": 50}
    )

    return fig

def _create_grid_plot(models, tasks, performance_data, title, save_path, agg_scores):
    """Create a grid of radar charts, one for each model."""
    n_models = len(models)

    # Calculate grid dimensions
    n_cols = math.ceil(math.sqrt(n_models))
    n_rows = math.ceil(n_models / n_cols)

    # Create subplots with polar projections
    subplot_titles = [f"{model}" for model in models]

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{'type': 'polar'}] * n_cols for _ in range(n_rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )

    # Color palette
    colors = px.colors.qualitative.Set3

    for i, model in enumerate(models):
        row = i // n_cols + 1
        col = i % n_cols + 1

        scores = performance_data[model]

        # Ensure we have scores for all tasks
        if len(scores) != len(tasks):
            raise ValueError(f"Model {model} has {len(scores)} scores but {len(tasks)} tasks")

        # Close the radar chart
        scores_closed = scores + [scores[0]]
        tasks_closed = tasks + [tasks[0]]

        color = colors[i % len(colors)]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=tasks_closed,
                fill='toself',
                name=model,
                showlegend=False,
                line={"color": color, "width": 2},
                fillcolor=color.replace('rgb', 'rgba').replace(')', ', 0.2)'),
                hovertemplate=f'<b>{model}</b><br>Task: %{{theta}}<br>Score: %{{r:.1f}}%<extra></extra>'
            ),
            row=row, col=col
        )

        fig.update_layout(
            font={"size": 10},  # Overall font size
            # Or more specifically for subplot titles:
            annotations=[{"font": {"size": 10}} for annotation in fig.layout.annotations], # type: ignore
        )


        # Update individual polar chart settings
        fig.update_polars(
            radialaxis={
                "visible": True,
                "range": [0, 100],
                "ticksuffix": '%',
                "gridcolor": 'lightgray',
                "tickfont": {"size": 9},
                "angle": 90,
            },
            angularaxis={
                "tickfont": {"size": 8},
                "gridcolor": 'lightgray',
                "rotation": -5,
            },
            row=row, col=col
        )

    # Calculate figure dimensions based on grid size
    width = max(1200, 500 * n_cols)
    height = max(1000, 500 * n_rows)

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 18}},
        width=width,
        height=height,
        margin={"t": 80, "b": 40, "l": 40, "r": 40}
    )

    return fig
