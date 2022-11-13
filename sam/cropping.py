import os
import numpy as np
import cv2
import math


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


def find_rectangle(array2d, asp_ratio, keep_attention):
    array2d_hcum = np.pad(np.cumsum(array2d, axis=0), pad_width=[(1, 0), (0, 0)])
    img_h, img_w = array2d.shape
    img_area = img_h * img_w

    total_attention = np.sum(array2d)
    threshold_attention = keep_attention * total_attention

    # initialize
    y_start = 0
    min_height = 1
    y_finish = y_start + min_height
    best_area = img_area
    failedToFind = True

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
            if (box_area < best_area) or (
                box_area == best_area and attention_kept > best_attention
            ):
                best_area = box_area
                x, y, w, h = x_start, y_start, window_w, window_h
                best_attention = attention_kept
                failedToFind = False
            y_start += 1
        else:
            y_finish += 1

    attention_factor = best_attention / total_attention
    area_factor = w * h / img_area

    if failedToFind:
        return {}
    else:
        return {
            "coords": (x, y, w, h),
            "area": area_factor,
            "attention": attention_factor,
        }


def find_best_rectangle(
    salient_ndimage, a_r, min_attention, step=0.02, alpha=8, beta=3
):
    results = {}
    attention = 1
    count = 0
    while attention >= min_attention:
        attention -= step * (2**count)
        count += 1
        result = find_rectangle(salient_ndimage, a_r, attention)
        if result:
            score = alpha ** (result["attention"]) / beta ** (result["area"])
            results[score] = result.pop("coords")

    x, y, w, h = results[sorted(results)[-1]]
    return x, y, w, h


def crop(
    original_image_path,
    saliency_map_path,
    cropped_image_path,
    boxed_image_path,
    a_r,
    attention,
):
    salient_ndimage = cv2.imread(saliency_map_path, cv2.IMREAD_GRAYSCALE)

    x, y, w, h = find_best_rectangle(salient_ndimage, a_r, attention)

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
            )
            if crop_success:
                print("Cropped and boxed %s successfully" % fname)
            else:
                print("Cropping and boxing %s failed" % fname)
