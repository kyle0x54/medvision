import collections
import math
from enum import Enum, unique

import cv2
import matplotlib.pyplot as plt
import numpy as np

import medvision as mv


@unique
class Color(Enum):
    """An enum that defines common colors."""

    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


class Switcher:
    def __init__(self, ax, imgs):
        self.ax = ax

        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.index = 0

        self.im = ax.imshow(imgs[self.index])
        self.update()

    def on_press(self, event):
        self.index = (self.index + 1) % self.num_imgs
        self.update()

    def update(self):
        self.im.set_data(self.imgs[self.index])
        self.im.axes.figure.canvas.draw_idle()
        try:
            self.im.axes.figure.canvas.flush_events()
        except NotImplementedError:
            pass


def _imshow_tight(img, title):
    cmap = "gray" if img.ndim == 2 else None
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.title(title)


def _imshow_switcher(imgs, title=""):
    """imshow with channel changer.

    An enhanced version of 'imshow'. Images (stored in 'imgs') can be
    switched by single clicking.

    Args:
        imgs (list[ndarray]): images to be displayed.
        title (str): title of the plot.
    """
    fig, ax = plt.subplots(num=title)
    fig.tight_layout()
    ax.set_title(title)
    ax.axis("off")
    ref_img = imgs[0]
    ax.set_xlim([0, ref_img.shape[1]])
    ax.set_ylim([ref_img.shape[0], 0])
    switcher = Switcher(ax, imgs)
    fig.canvas.mpl_connect("button_press_event", switcher.on_press)
    plt.show()


def imshow_dynamic(imgs, title=""):
    """ imshow with channel changer.

    An enhanced version of 'imshow'. Images (stored in 'imgs') can be
    switched by single clicking.

    Args:
        imgs (list[ndarray]): images to be displayed.
        title (str): title of the plot.
    """
    fig, ax = plt.subplots(num=title)
    fig.tight_layout()
    ax.set_title(title)
    ax.axis('off')
    ref_img = imgs[0]
    ax.set_xlim([0, ref_img.shape[1]])
    ax.set_ylim([ref_img.shape[0], 0])
    switcher = Switcher(ax, imgs)
    fig.canvas.mpl_connect("button_press_event", switcher.on_press)
    plt.show()


def imshow(
    imgs,
    num_cols=None,
    fig_name=None,
    titles="",
    show=True,
    save_path=None,
):
    """Show an image or multiple images in a single canvas.

    Args:
        imgs (ndarray or tuple/list[ndarray]): images to be shown.
        num_cols (int): image number per column for multiple images display.
            If not given, this parameter is automatically determined.
        fig_name (str): name of the plot.
        titles (str or list[str]): sub-plot titles.
        show (bool): True: show the image; False: save the image.
        save_path (str, optional): path to save the image.
    """
    if not isinstance(imgs, collections.abc.Sequence):
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
    plt.tight_layout()

    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i + 1)
        _imshow_tight(img, titles[i])

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def imshow_bboxes(
    img,
    bboxes,
    score_thr=0,
    colors=Color.green,
    top_k=-1,
    thickness=1,
    font_scale=0.5,
    font_thickness=1,
    font_color=Color.white,
    title="",
    show=True,
    save_path=None,
):
    """Draw bounding boxes on an image.

    To display detection result or compare detection results by different
    algorithms.

    Args:
        img (str or ndarray): image (or file path) to be displayed.
        bboxes (list or ndarray): a list of ndarray of shape (k, 4) or (n, 5).
        score_thr (float): minimum score of bboxes to be shown.
        colors (Color or list[Color]): color or list of colors.
        top_k (int): plot the first k bboxes only if set positive.
            Otherwise, plot all the bboxes.
        thickness (int): line thickness.
        font_scale (float): font scales of texts.
        font_thickness (int): font thickness.
        font_color (Color):  color of font.
        title (str): title of the plot.
        show (bool): True: show the image; False: save the image.
        save_path (str, optional): path to save the image.
    """
    if isinstance(img, str):
        img = mv.imread(img)
    else:
        img = img if img.ndim == 3 else mv.gray2rgb(img)

    if isinstance(bboxes, np.ndarray):
        bboxes = [bboxes]

    if isinstance(colors, Color):
        colors = [colors] * len(bboxes)

    assert len(bboxes) == len(colors)

    plot_prob = True if bboxes[0].shape[1] == 5 else False

    if score_thr > 0:
        for i in range(len(bboxes)):
            assert bboxes[i].shape[1] == 5
            scores = bboxes[i][:, -1]
            indices = scores > score_thr
            bboxes[i] = bboxes[i][indices, :]

    img_with_result = img.copy()
    for i, _bboxes in enumerate(bboxes):
        _bboxes_int = _bboxes.astype(np.int32)
        if top_k <= 0:
            _top_k = _bboxes.shape[0]
        else:
            _top_k = min(top_k, _bboxes.shape[0])
        for j in range(_top_k):
            left_top = (_bboxes_int[j, 0], _bboxes_int[j, 1])
            right_bottom = (_bboxes_int[j, 2], _bboxes_int[j, 3])
            cv2.rectangle(img_with_result, left_top, right_bottom, colors[i].value, thickness)
            if plot_prob:
                label_text = "%.2f" % _bboxes[j, -1]
                ((text_width, text_height), _) = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                cv2.rectangle(
                    img_with_result,
                    (left_top[0], left_top[1] - int(1.3 * text_height)),
                    (left_top[0] + text_width, left_top[1]),
                    colors[i].value,
                    -1,
                )
                cv2.putText(
                    img_with_result,
                    label_text,
                    (left_top[0], left_top[1] - int(0.3 * text_height)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_color.value,
                    thickness=font_thickness,
                    lineType=cv2.LINE_AA,
                )

    if show:
        _imshow_switcher([img_with_result, img], title)

    if save_path is not None:
        mv.imwrite(save_path, img_with_result)


if __name__ == "__main__":
    im = cv2.imread("../../tests/data/pngs/Blue-Ogi.png", cv2.IMREAD_GRAYSCALE)
    imshow(im, fig_name="show single image", titles="name")

    h, w = im.shape[:2]
    bbox = np.array([w // 3, h // 3, w * 2 // 3, h * 2 // 3, 0.5]).reshape(-1, 5)
    imshow_bboxes(im, bbox, score_thr=0.2, title="draw bounding boxes")

    im = cv2.imread("../../tests/data/pngs/Blue-Ogi.png", cv2.IMREAD_COLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    imshow([im] * 5, fig_name="show multiple images", titles=[str(i) for i in range(5)])
