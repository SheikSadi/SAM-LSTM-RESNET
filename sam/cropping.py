import os
import numpy as np
import cv2
import math
from sam.config import *


def maxSubarrayFL(array, width, threshold):
    """
    https://ieeexplore.ieee.org/document/7780430
    """
    sub_idx = None
    sum_max = -1
    array_cum = np.cumsum(np.pad(array, (1, 0), "constant"))
    if threshold > 0 and width > 0:
        for idx in range(len(array_cum) - width):
            sum_0 = array_cum[idx + width] - array_cum[idx]
            if sum_0 >= threshold and sum_0 > sum_max:
                sub_idx = idx
                sum_max = sum_0
    return sub_idx, sum_max


def fixedAspRatioRectangle(saliency_mapped_ndarray, a_r, attention):
    """
    https://ieeexplore.ieee.org/document/7780430
    """
    G, r, tau = saliency_mapped_ndarray, a_r, attention

    G_pos_c = np.cumsum(G, axis=0)
    G_pos = np.cumsum(G_pos_c, axis=1)

    m, n = G.shape
    i, j, w, h = 0, 0, n, m
    Smin = -1
    i1, i2 = 0, 0
    T = tau * G_pos[m - 1, n - 1]

    while i2 < m and i1 - 1 < m:
        h0 = i2 - i1 + 1
        w0 = int(math.ceil(h0 * r))
        if w0 > n:
            i1 += 1
        else:
            if i1 - 1 < 0:
                array = G_pos_c[i2, :]
            else:
                array = G_pos_c[i2, :] - G_pos_c[i1 - 1, :]
            j1, S0 = maxSubarrayFL(array, w0, T)

            if j1:
                if w0 * h0 < w * h or (w0 * h0 == w * h and S0 > Smin):
                    i, j, w, h = i1, j1, w0, h0
                    Smin = S0
                i1 += 1
            else:
                i2 += 1
    return i, j, w, h


def crop(
    original_image_path,
    saliency_map_path,
    cropped_image_path,
    boxed_image_path,
    a_r,
    attention,
):
    salient_ndimage = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)
    i, j, w, h = fixedAspRatioRectangle(salient_ndimage, a_r, attention)
    original_ndimage = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    cropped_ndimage = original_ndimage[i : i + h, j : j + w, :]
    crop_success = cv2.imwrite(cropped_image_path, cropped_ndimage.astype(int))
    # Draw a diagonal blue line with thickness of 5 px
    blue = (255, 0, 0)
    thickness = 5
    cv2.rectangle(original_ndimage, (j, i), (j + w, i + h), blue, thickness)
    box_success = cv2.imwrite(boxed_image_path, original_ndimage.astype(int))
    
    return crop_success and box_success


def batch_crop_images(
    originals_folder,
    maps_folder,
    crops_folder,
    boxes_folder,
    aspect_ratio,
    retained_attention,
):
    current_dir = os.getcwd()
    originals_path = os.path.join(current_dir, originals_folder)
    maps_path = os.path.join(current_dir, maps_folder)
    crops_path = os.path.join(current_dir, crops_folder)
    boxes_path = os.path.join(current_dir, boxes_folder)

    for path in [crops_path, boxes_path]:
        if not os.path.exists(path):
            os.mkdir(path)

    if not os.path.exists(crops_path):
        os.mkdir(crops_path)

    for fname in os.listdir(maps_path):
        if fname in os.listdir(originals_path) and fname not in os.listdir(crops_path):
            original_file = os.path.join(originals_path, fname)
            mapping_file = os.path.join(maps_path, fname)
            crop_file = os.path.join(crops_path, fname)
            box_file = os.path.join(boxes_path, fname)

            crop_success = crop(
                original_file,
                mapping_file,
                crop_file,
                box_file,
                aspect_ratio,
                retained_attention,
            )
            if crop_success:
                print("Cropped and boxed %s successfully" % fname)
            else:
                print("Cropping and boxing %s failed" % fname)
