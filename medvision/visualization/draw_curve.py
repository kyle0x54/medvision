import matplotlib.pyplot as plt
import sklearn


def draw_froc(average_fps, sensitivity, save_path=None, **kwargs):
    """ Plot the FROC curve.

    Args:
        average_fps (list): A list containing the average number of false
        positives per image for different thresholds.
        sensitivity (list):  A list containing overall sensitivity for
        different thresholds.
        save_path (str): path to save the froc curve drawing.
    """
    plt.figure()
    plt.xlabel('False Positives')
    plt.ylabel('Sensitivity')
    plt.title('FROC Curve')
    plt.ylim([0, 1])
    plt.plot(average_fps, sensitivity, **kwargs)
    plt.grid()
    if save_path is not None:
        plt.savefig(save_path)
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
    plt.show()
