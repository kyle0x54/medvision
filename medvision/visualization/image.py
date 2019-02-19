import math
import matplotlib.pyplot as plt
import medvision as mv


def _imshow_tight(img, title, cmap):
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.title(title)


def imshow(imgs, fig_name=None, titles=None, cmap=None, num_cols=None):
    """ Show an image or multiple images in a single canvas.

    Args:
        imgs (ndarray or tuple/list[ndarray]): images to be shown.
        fig_name (str): name of the plot.
        titles (tuple/list[str]): sub-plot titles.
        cmap (str): the same as matplotlib 'cmap'.
        num_cols (int): image number per column for multiple images display.
            If not given, this parameter is automatically determined.
    """
    if mv.isarrayinstance(imgs):
        num_imgs = len(imgs)
    else:
        num_imgs = 1
        imgs = [imgs]

    if titles is None:
        titles = [''] * len(imgs)
    elif isinstance(titles, str):
        titles = [titles] * len(imgs)
    else:  # mv.isarrayinstance(titles)
        assert len(titles) == len(imgs)

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
