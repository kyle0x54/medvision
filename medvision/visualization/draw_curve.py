import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import numpy as np
import sklearn


def draw_froc(average_fps,
              sensitivity,
              save_path=None,
              bLogPlot=True,
              **kwargs):
    """ Plot the FROC curve.

    Args:
        average_fps (list): A list containing the average number of false
        positives per image for different thresholds.
        sensitivity (list):  A list containing overall sensitivity for
        different thresholds.
        save_path (str): path to save the froc curve drawing.
    """
    average_fps = np.append(average_fps, 64)
    sensitivity = np.append(sensitivity, sensitivity[-1])
    plt.plot(average_fps, sensitivity, **kwargs)
    plt.xlabel('Average number of false positives per scan')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.xlim(0.125, 64)
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

    ax = plt.gca()
    xaxis = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
    if bLogPlot:
        plt.xscale('log', basex=2)
        ax.xaxis.set_major_formatter(FixedFormatter(xaxis))

    # set your ticks manually
    ax.xaxis.set_ticks(xaxis)
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True, which='both')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def save_roc_curve(roc_curve, save_path=None):
    fpr, tpr, _ = roc_curve
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (auc = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xticks()
    plt.yticks()
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def save_pr_curve(pr_curve, save_path=None):
    precision, recall, _ = pr_curve
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xticks()
    plt.yticks()
    plt.title('PR Curve')
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
