import collections
import math
import matplotlib.pyplot as plt


def _imshow_tight(img, title, cmap):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.title(title)


def imshow(imgs, num_cols=None, fig_name=None, titles='', cmap=None):
    """ Show an image or multiple images in a single canvas.

    Args:
        imgs (ndarray or tuple/list[ndarray]): images to be shown.
        num_cols (int): image number per column for multiple images display.
            If not given, this parameter is automatically determined.
        fig_name (str): name of the plot.
        titles (str or tuple[str]): sub-plot titles.
        cmap (str): the same as matplotlib 'cmap'.
    """
    if not isinstance(imgs, collections.Sequence):
        imgs = [imgs]
    num_imgs = len(imgs)

    if isinstance(titles, str):
        titles = [titles] * num_imgs
    assert len(titles) == num_imgs

    if num_cols is None:
        num_cols = int(math.ceil(math.sqrt(num_imgs)))
    assert num_cols <= len(imgs)
    num_rows = int((num_imgs + num_cols - 1) // num_cols)

    plt.figure(fig_name)

    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        _imshow_tight(img, titles[i], cmap)

    plt.show()


if __name__ == '__main__':
    import cv2
    im = cv2.imread('../../tests/data/pngs/Blue-Ogi.png', cv2.IMREAD_GRAYSCALE)
    imshow(im, fig_name='show single image', titles='name', cmap='gray')
    im = cv2.imread('../../tests/data/pngs/Blue-Ogi.png', cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imshow([im] * 5, fig_name='show multiple images',
           titles=[str(i) for i in range(5)])
