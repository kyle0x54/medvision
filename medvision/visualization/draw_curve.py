import matplotlib.pyplot as plt


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
