from collections import OrderedDict

import cv2
import math
import numpy as np
import SimpleITK as sitk

import medvision as mv
from medvision.annotation.rws import load_rws_contour
from medvision.annotation.rws2mask import rws2multimasks, rws2mask


def get_lut_value(data, ww, wc):
    if math.isclose(ww, 1.0):
        max_val, min_val = data.max(), data.min()
        ww = max_val - min_val
        wc = (max_val + min_val) / 2.0

    alpha = 255.0 / (ww - 1.0)
    beta = ((0.5 - wc) / (ww - 1.0) + 0.5) * 255.0
    minClipVal = round(wc - 0.5 - (ww - 1) / 2)
    maxClipVal = round(wc - 0.5 + (ww - 1) / 2)
    # minClipVal = max(minClipVal, data.min())
    # maxClipVal = min(maxClipVal, data.max())
    lutValue = np.clip(data, minClipVal, maxClipVal)
    # note that image might be converted to other types with np.clip().
    # (for instance, minClipVal, maxClipVal are of int32 type, but image is of uint16 type)
    # so we force casting image type to float32 to avoid potential opencv error
    lutValue = cv2.convertScaleAbs(lutValue.astype(np.float32), alpha=alpha, beta=beta)

    return lutValue


def dcmread(dcm_path: str):
    img_itk = sitk.ReadImage(dcm_path)
    image_array = sitk.GetArrayFromImage(img_itk)
    image_array = np.squeeze(image_array)

    dicom_info = {
        "wc": int(img_itk.GetMetaData("0028|1050")),
        "ww": int(img_itk.GetMetaData("0028|1051")),
    }

    return image_array, dicom_info


def read_dcm_as_8uc3(dcm_path: str) -> np.ndarray:
    img, dicom_info = dcmread(dcm_path)
    img_uint8 = get_lut_value(img, dicom_info["ww"], dicom_info["wc"])
    return cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)


def split_colormap(
    colormap: OrderedDict[str, tuple[int, int, int]]
) -> tuple[OrderedDict[str, int], OrderedDict[int, tuple[int, int, int]]]:
    class2label = OrderedDict()
    label2color = OrderedDict()
    for i, (key, value) in enumerate(colormap.items(), 1):
        class2label[key] = i
        label2color[i] = value
    return class2label, label2color


def gen_lut(label2color: OrderedDict[int, tuple[int, int, int]]):
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for label, color in label2color.items():
        lut[label][0] = np.array(color)
    return lut


def mask_alpha_blend(
    image: np.ndarray,
    mask: np.ndarray,
    label2color: OrderedDict[int, tuple[int, int, int]],
    alpha: float
) -> None:
    """ N.B. This is a inplace operation to image."""
    # Generate look up table to map mask intensity to predefined colors
    lut = gen_lut(label2color)
    maskRgb = cv2.applyColorMap(mask, lut)

    # Alpha blend
    image[mask > 0] = cv2.addWeighted(image, 1 - alpha, maskRgb, alpha, 0)[mask > 0]


def draw_contours(
    image: np.ndarray,
    mask: np.ndarray,
    label2color: OrderedDict[int, tuple[int, int, int]],
):
    """ N.B. This is a inplace operation to image."""
    for label in label2color.keys():
        bmask = (mask == label).astype(np.uint8)
        contours, _ = cv2.findContours(bmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, label2color[label], 3)


def plot_rws_multimask(
    dcm_path: str,
    rws_path: str,
    colormap: dict[str, tuple[int, int, int]],
    mode: str = 'contour',  # 'mask'
    alpha: float = 0.2, # N.B. only used in mask mode
):
    # Convert rws to mask
    rws_contour = load_rws_contour(rws_path)
    class2label, label2color = split_colormap(colormap)
    masks = rws2multimasks(rws_contour, class2label)

    # Load dicom image and convert to 3-channel uint8 type
    img_8uc3 = read_dcm_as_8uc3(dcm_path)

    # Draw contours
    img_with_overlay = img_8uc3.copy()
    if mode == 'contour':
        for label, mask in zip(label2color.keys(), masks):
            draw_contours(img_with_overlay, mask, {label: label2color[label]})
    elif mode == 'mask':
        for label, mask in zip(label2color.keys(), masks):
            mask_alpha_blend(img_with_overlay, mask, label2color, alpha)
    else:
        raise ValueError(f"Wrong mode [{mode}], only support 'contour' and 'mask' ")

    mv.imshow_dynamic([img_with_overlay, img_8uc3], dcm_path)


def plot_rws(
    dcm_path: str,
    rws_path: str,
    colormap: dict[str, tuple[int, int, int]],
    mode: str = 'contour',  # 'mask'
    alpha: float = 0.2, # N.B. only used in mask mode
):
    # Convert rws to mask
    rws_contour = load_rws_contour(rws_path)
    class2label, label2color = split_colormap(colormap)
    mask, _ = rws2mask(rws_contour, class2label)
    mask = mask.astype(np.uint8)

    # Load dicom image and convert to 3-channel uint8 type
    img_8uc3 = read_dcm_as_8uc3(dcm_path)

    # Draw contours
    img_with_overlay = img_8uc3.copy()
    if mode == 'contour':
        draw_contours(img_with_overlay, mask, label2color)
    elif mode == 'mask':
        mask_alpha_blend(img_with_overlay, mask, label2color, alpha)
    else:
        raise ValueError(f"Wrong mode [{mode}], only support 'contour' and 'mask' ")

    mv.imshow_dynamic([img_with_overlay, img_8uc3], dcm_path)


def main():
    dcm_path = '/home/liupengfei/tmp/1.3.12.2.1107.5.12.7.3294.30000014031300443610900000053/R_MLO.dcm'
    rws_path = '/home/liupengfei/tmp/1.3.12.2.1107.5.12.7.3294.30000014031300443610900000053/R_MLO.json'
    colormap = OrderedDict([("breast", (255, 0, 0)), ("2", (0, 255, 0))])
    # plot_rws_multimask(dcm_path, rws_path, colormap, mode='mask', alpha=0.4)
    plot_rws(dcm_path, rws_path, colormap, mode='mask', alpha=0.4)


if __name__ == '__main__':
    main()
