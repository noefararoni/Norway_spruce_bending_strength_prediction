"""Save evaluation figures without opening an interactive window."""


def plot_predictions(
    table,
    path,
    *,
    title="Linear Regression - Original Dataset",
    prediction_label="Cross-validation predictions",
    physics_style=False,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .config import TARGET
    from .evaluation import metrics

    actual, predicted = table[TARGET], table["predicted_value"]
    low, high = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
    padding = (high - low or 1.0) * 0.08
    low, high = low - padding, high + padding
    scores = metrics(actual, predicted)
    r_squared = f"{scores['r2']:.3f}" if scores["r2"] is not None else "N/A"
    fig, ax = plt.subplots(figsize=(9, 6) if physics_style else (7, 7))
    ax.scatter(actual, predicted, s=55, alpha=0.75, color="tab:blue", label=prediction_label)
    ax.plot([low, high], [low, high], "--", color="tab:blue", label="Perfect prediction")
    ax.set(
        xlabel="Measured Bending Strength [MPa]",
        ylabel="Predicted Bending Strength [MPa]",
        title=title,
        xlim=(low, high),
        ylim=(low, high),
    )
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    if not physics_style:
        ax.set_aspect("equal", adjustable="box")
    metric_text = f"MSE = {scores['mse']:.2f} MPa²\nRMSE = {scores['rmse']:.2f} MPa\n"
    if physics_style and "fold" in table:
        fold_mse = (
            table.assign(_squared=(actual - predicted) ** 2).groupby("fold")["_squared"].mean()
        )
        average_rmse = float((fold_mse**0.5).mean())
        metric_text += f"Average fold RMSE = {average_rmse:.2f} MPa\n"
    else:
        metric_text += f"MAE = {scores['mae']:.2f} MPa\n"
    ax.text(
        0.97,
        0.04,
        metric_text + r"$R^2$" + f" = {r_squared}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "black", "alpha": 0.9},
    )
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
