import os
import numpy as np
import cv2
import math
from sam.config import *


def find_max_subarray(array: np.ndarray, window_w: int, threshold: float) -> tuple:
    assert len(array.shape) == 1

    best_sum = -1
    start_idx = None

    array_cum = np.pad(np.cumsum(array), pad_width=[(1, 0)])

    max_start_idx = array.shape[0] - window_w

    for idx in range(max_start_idx + 1):
        cumsum_upto_windowend = array_cum[idx + window_w]
        cumsum_before_windowstart = array_cum[idx]
        subarray_sum = cumsum_upto_windowend - cumsum_before_windowstart
        if subarray_sum > threshold and subarray_sum > best_sum:
            best_sum = subarray_sum
            start_idx = idx

    return start_idx, best_sum


def default_maximizer(box_area, img_area, attention_kept, total_attention):
    area_factor = box_area / img_area
    attention_factor = attention_kept / total_attention
    # We want to maximize this value
    factor = attention_factor / area_factor
    return int(factor * 1000)


def find_best_rectangle(array2d, asp_ratio, keep_attention, maximizer):
    """
    to_maximize can be user-defined function
    and it will take in 4 input arguments -
    box_area, img_area, attention_kept, total_attention
    """
    array2d_hcum = np.pad(np.cumsum(array2d, axis=0), pad_width=[(1, 0), (0, 0)])
    img_h, img_w = array2d.shape
    img_area = img_h * img_w

    total_attention = np.sum(array2d)
    threshold_attention = keep_attention * total_attention

    # initialize
    y_start = 0
    min_height = 1
    y_finish = y_start + min_height
    best_factor = -1

    while True:
        window_h = y_finish - y_start
        window_w = math.ceil(asp_ratio * window_h)

        if not (
            y_finish <= img_h
            and window_h >= min_height
            and window_h <= img_h
            and window_w <= img_w
        ):
            break

        subarray2d = array2d_hcum[y_finish] - array2d_hcum[y_start]
        x_start, attention_kept = find_max_subarray(
            subarray2d, window_w, threshold_attention
        )
        if attention_kept > 0:
            box_area = window_w * window_h
            factor = maximizer(box_area, img_area, attention_kept, total_attention)
            if factor < 0:
                raise KeyboardInterrupt("WTF!")
            elif factor > best_factor:
                best_factor = factor
                x, y, w, h = x_start, y_start, window_w, window_h
                best_attention = attention_kept
            y_start += 1
        else:
            y_finish += 1

    print(
        f"Attention kept: {round(best_attention/total_attention*100,2)}% "
        f"at an area: {round(w*h/img_area*100,2)}%"
    )

    return x, y, w, h


def crop(
    original_image_path,
    saliency_map_path,
    cropped_image_path,
    boxed_image_path,
    a_r,
    attention,
    maximizer,
):
    salient_ndimage = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)

    x, y, w, h = find_best_rectangle(salient_ndimage, a_r, attention, maximizer)

    original_ndimage = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    cropped_ndimage = original_ndimage[y : y + h, x : x + w, :]
    crop_success = cv2.imwrite(cropped_image_path, cropped_ndimage.astype(int))
    # Draw a diagonal blue line with thickness of 5 px
    blue = (255, 0, 0)
    thickness = 5
    cv2.rectangle(original_ndimage, (x, y), (x + w, y + h), blue, thickness)
    box_success = cv2.imwrite(boxed_image_path, original_ndimage.astype(int))

    return crop_success and box_success


def batch_crop_images(
    originals_folder,
    maps_folder,
    crops_folder,
    boxes_folder,
    aspect_ratio,
    retained_attention,
    maximizer=default_maximizer,
):
    for _dir in [crops_folder, boxes_folder]:
        if not os.path.exists(_dir):
            os.mkdir(_dir)

    for fname in os.listdir(maps_folder):
        if fname in os.listdir(originals_folder):
            original_file = os.path.join(originals_folder, fname)
            mapping_file = os.path.join(maps_folder, fname)
            crop_file = os.path.join(crops_folder, fname)
            box_file = os.path.join(boxes_folder, fname)

            crop_success = crop(
                original_file,
                mapping_file,
                crop_file,
                box_file,
                aspect_ratio,
                retained_attention,
                maximizer,
            )
            if crop_success:
                print("Cropped and boxed %s successfully" % fname)
            else:
                print("Cropping and boxing %s failed" % fname)
