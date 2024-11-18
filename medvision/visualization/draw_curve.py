from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import sklearn
from matplotlib.ticker import FixedFormatter


def draw_froc_curve(average_fps, sensitivity, save_path=None, bLogPlot=True, **kwargs):
    """Plot the FROC curve.

    Args:
        average_fps (list): A list containing the average number of false
        positives per image for different thresholds.
        sensitivity (list):  A list containing overall sensitivity for
        different thresholds.
        save_path (str): path to save the froc curve drawing.
    """
    average_fps = np.append(average_fps, 64)
    sensitivity = np.append(sensitivity, sensitivity[-1])
    plt.figure()
    plt.plot(average_fps, sensitivity, **kwargs)
    plt.xlabel("Average number of false positives per scan")
    plt.ylabel("Sensitivity")
    plt.title("FROC Curve")
    plt.xlim(0.125, 64)
    plt.ylim([0, 1])
    plt.legend(loc="lower right")

    ax = plt.gca()
    xaxis = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    if bLogPlot:
        plt.xscale("log", basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter(xaxis))

    # set your ticks manually
    ax.xaxis.set_ticks(xaxis)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, which="both")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def draw_roc_curve(fpr: Sequence[float], tpr: Sequence[float], save_path: Optional[str] = None):
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (auc = %0.4f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xticks()
    plt.yticks()
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(b=True, which="both")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def draw_pr_curve(
    precision: Sequence[float], recall: Sequence[float], save_path: Optional[str] = None
):
    pr_auc = sklearn.metrics.auc(precision, recall)
    plt.figure()
    plt.plot(
        recall,
        precision,
        color="darkorange",
        lw=2,
        label="PR curve (auc = %0.4f)" % pr_auc,
    )
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xticks()
    plt.yticks()
    plt.title("PR Curve")
    plt.legend(loc="lower right")
    plt.grid(b=True, which="both")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
