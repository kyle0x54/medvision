import math
import matplotlib.pyplot as plt


def _imshow_tight(im, title, cmap):
    plt.imshow(im, cmap=cmap)
    plt.axis('off')
    plt.xlim([0, im.shape[1]])
    plt.ylim([im.shape[0], 0])
    plt.title(title)


def imshow(ims, fig_name=None, titles=None, cmap=None, num_cols=None):
    """ Show an image or multiple images in a single canvas.

    Args:
        ims (ndarray of list of ndarrays): images to be shown.
        fig_name (str): name of the plot.
        cmap (str): the same as matplotlib 'cmap'.
        num_cols (int): image number per column for multiple images display.
            If not given, this parameter is automatically determined.
    """
    if isinstance(ims, (list, tuple)):
        num_ims = len(ims)
    else:
        num_ims = 1
        ims = [ims]

    if titles is None:
        titles = [''] * len(ims)
    else:
        assert len(titles) == len(ims)

    if num_cols is None:
        num_cols = int(math.ceil(math.sqrt(num_ims)))
    assert num_cols <= len(ims)
    num_rows = int((num_ims + num_cols - 1) // num_cols)

    plt.figure(fig_name)

    for i, im in enumerate(ims):
        plt.subplot(num_rows, num_cols, i + 1)
        _imshow_tight(im, titles[i], cmap)

    plt.show()


if __name__ == '__main__':
    import cv2
    im = cv2.imread('../../tests/data/pngs/Blue-Ogi.png', cv2.IMREAD_GRAYSCALE)
    imshow(im, fig_name='show single image', cmap='gray')
    im = cv2.imread('../../tests/data/pngs/Blue-Ogi.png', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imshow([im] * 5, fig_name='show multiple images',
           titles=[str(i) for i in range(5)])
