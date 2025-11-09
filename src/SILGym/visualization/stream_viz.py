"""
Skill stream visualization utilities.

This module provides visualization functions for skill training streams,
separated from configuration logic to make matplotlib an optional dependency.
"""

from typing import Optional, Tuple
from SILGym.utils.logger import get_logger

logger = get_logger(__name__)


def visualize_skill_stream(
    config,
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None,
    table: bool = False,
) -> None:
    """
    Visualize a skill training stream as a timeline.

    This function creates a timeline visualization showing which model components
    (policy, interface, decoder) are trained in each phase of the skill stream.

    Args:
        config: SkillStreamConfig instance to visualize.
        figsize: Figure size as (width, height) tuple.
        save_path: Path to save the figure. If None, figure is displayed.
        table: If True, include a table showing evaluation tasks.

    Returns:
        None. Figure is either saved to disk or displayed.

    Raises:
        ImportError: If matplotlib is not available.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.path as mpath
        import matplotlib.lines as mlines
    except ImportError:
        logger.warning(
            "Matplotlib not available. Cannot visualize stream. "
            "Install with: pip install matplotlib"
        )
        return

    # Create subplots
    if table:
        fig, (ax_timeline, ax_table) = plt.subplots(
            2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_timeline = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.6))
        ax_table = None

    # Draw timeline
    _draw_timeline(ax_timeline, config, mlines, mpath, mpatches)

    # Draw table if requested
    if table and ax_table:
        _draw_table(ax_table, config)

    plt.tight_layout()

    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Figure saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()


def _draw_timeline(ax, config, mlines, mpath=None, mpatches=None) -> None:
    """
    Draw the timeline visualization on the given axes.

    Args:
        ax: Matplotlib axes object.
        config: SkillStreamConfig instance.
        mlines: matplotlib.lines module.
        mpath: matplotlib.path module (unused but kept for compatibility).
        mpatches: matplotlib.patches module (unused but kept for compatibility).
    """
    # Configuration
    target_positions = {"policy": 2, "interface": 1, "decoder": 0}
    color_map = {"policy": "limegreen", "interface": "dodgerblue", "decoder": "gold"}
    edge_color_map = {
        "policy": "green",
        "interface": "blue",
        "decoder": "darkgoldenrod",
    }

    num_phases = len(config.datastream)
    if num_phases == 0:
        return

    # Background
    for i in range(num_phases):
        color = "#f5f5f5" if i % 2 == 0 else "#e0e0e0"
        ax.axvspan(i, i + 1, color=color, alpha=0.5, ec="none")

    # Track last positions
    last_pos = {target: None for target in target_positions}

    # Plot phases
    for i, phase in enumerate(config.datastream):
        x = i + 0.5

        # Plot targets
        for target in target_positions:
            if target in phase.train_targets:
                y = target_positions[target]

                # Connect to previous
                if last_pos[target]:
                    last_x, last_y, last_i = last_pos[target]
                    style = "solid" if i == last_i + 1 else "--"
                    ax.plot(
                        [last_x, x],
                        [last_y, y],
                        "gray",
                        linewidth=3,
                        linestyle=style,
                        zorder=1,
                    )

                # Plot marker
                ax.plot(
                    x,
                    y,
                    "o",
                    markersize=14,
                    color=color_map[target],
                    markeredgecolor=edge_color_map[target],
                    markeredgewidth=3,
                    zorder=2,
                )

                last_pos[target] = (x, y, i)

        # Phase name
        ax.text(
            x,
            -0.7,
            phase.phase_name,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Configure axes
    ax.set_xlim(0, num_phases)
    ax.set_ylim(-1.5, 2.5)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Decoder", "Interface", "Policy"])
    ax.set_xticks([i + 0.5 for i in range(num_phases)])
    ax.set_xticklabels([str(i) for i in range(num_phases)])
    ax.set_xlabel("Phase Index")
    ax.set_title(f"Skill Stream: {config.scenario_id}", fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    handles = [
        mlines.Line2D(
            [],
            [],
            color=color_map[t],
            marker="o",
            markersize=14,
            label=t.capitalize(),
            markeredgecolor=edge_color_map[t],
            markeredgewidth=3,
            linestyle="None",
        )
        for t in ["policy", "interface", "decoder"]
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=False,
    )


def _draw_table(ax, config) -> None:
    """
    Draw the evaluation tasks table on the given axes.

    Args:
        ax: Matplotlib axes object.
        config: SkillStreamConfig instance.
    """
    ax.axis("off")

    # Build table data
    table_data = []
    for phase in config.datastream:
        if phase.eval_tasks:
            names = [str(t.get("data_name", t)) for t in phase.eval_tasks]
            count = len(names)
            task_str = ", ".join(names[:10]) + (" ..." if count > 10 else "")
        else:
            count = 0
            task_str = "N/A"
        table_data.append([phase.phase_name, str(count), task_str])

    # Create table
    if any(row[2] != "N/A" for row in table_data):
        table = ax.table(
            cellText=table_data,
            colLabels=["Phase", "Count", "Evaluation Tasks"],
            loc="center",
            cellLoc="center",
        )

        # Style table
        for (row, col), cell in table.get_celld().items():
            cell.set_height(0.1)
            cell.set_linewidth(0.1)

            # Column widths
            widths = [0.15, 0.1, 0.75]
            cell.set_width(widths[col] if col < len(widths) else 0.25)

            # Colors
            if row == 0:
                cell.set_facecolor("#40466e")
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("#f5f5f5" if row % 2 == 0 else "#ffffff")
