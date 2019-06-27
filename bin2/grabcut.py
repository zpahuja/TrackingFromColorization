import numpy as np
import cv2


def grabcut(image, mask, erosion_kernel_size=30, dilation_kernel_size=50):
    """
    mask and image size must be equal. not tested.
    """
    if np.all(mask == 0):
        return mask

    erosion_kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    eroded = cv2.erode(mask, erosion_kernel, iterations=1)

    dilation_kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

    mask[dilated == 1] = 3
    mask[eroded == 1] = 1

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, _, _ = cv2.grabCut(
        image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    # _color = image * mask[:,:,np.newaxis]
    return mask


def _grabcut(image, mask):
    """
    mask and image size must be equal. not tested.
    """
    erosion_kernel = np.ones((15, 15), np.uint8)
    eroded = cv2.erode(mask, erosion_kernel, iterations=1)

    dilation_kernel = np.ones((40, 40), np.uint8)
    dilated = cv2.dilate(mask, dilation_kernel, iterations=1)

    mask[dilated == 1] = 3
    mask[eroded == 1] = 1

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask, _, _ = cv2.grabCut(
        image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return mask, eroded, dilated
