import matplotlib.pyplot as plt
import matplotlib as mpl
from config import load_config
from pathlib import Path


CONFIG_PATH = Path(__file__).parent.parent / "configs" / "config.yaml"
cfg = load_config(CONFIG_PATH)


def setup_plotting():

    plt.style.use(cfg.plots.style)
    mpl.rcParams.update({
        "figure.figsize": cfg.plots.fig_size,
        "figure.dpi": cfg.plots.dpi,
        'font.size': cfg.plots.font_size,
        'axes.grid': cfg.plots.grid,
        'grid.alpha': cfg.plots.alpha,
        "axes.spines.top": cfg.plots.spines_top,
        "axes.spines.right": cfg.plots.spines_right,
    })

def visualise_mistakes(model, y_true, y_pred, residuals):
    fig, axes = plt.subplots(1, 3)

    axes[0].scatter(y_true, y_pred)
    axes[0].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()], 'r--')
    axes[0].set_title(f'{model.__class__.__name__} — Predicted vs Actual')

    axes[1].scatter(y_pred, residuals)
    axes[1].axhline(0, color='r', linestyle='--')
    axes[1].set_title('Residuals vs Predicted')

    axes[2].hist(residuals, bins=50)
    axes[2].set_title('Распределение остатков')

    plt.tight_layout()
    plt.show()

def visualise_for_target(df, y_train):
    fig, ax = plt.subplots()
    ax.scatter(df, y_train)
    ax.set_title(df.name)
    plt.show()