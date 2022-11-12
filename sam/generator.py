import os
import numpy as np
from sam.config import (
    nb_gaussian,
    shape_r_gt,
    shape_c_gt,
    shape_r,
    shape_c,
    shape_r_out,
    shape_c_out,
    b_s,
)
from sam.utilities import (
    preprocess_images,
    preprocess_maps,
    preprocess_fixmaps,
)


def generator(
    images,
    maps,
    fixs,
):
    batch_size = b_s

    n_images = len(images)

    if n_images % batch_size != 0:
        raise Exception(
            f"The number (in your case: {n_images}) of training/validation images "
            "should be a multiple of the batch size. Please change the batch size."
        )

    gaussian = np.zeros((batch_size, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    img_yielded = 0
    while True:
        X = (
            preprocess_images(images[counter : counter + batch_size], shape_r, shape_c),
        )
        Y = preprocess_maps(
            maps[counter : counter + batch_size], shape_r_out, shape_c_out
        )
        Y_fix = preprocess_fixmaps(
            fixs[counter : counter + batch_size], shape_r_out, shape_c_out
        )
        yield [X, gaussian], [Y, Y, Y_fix]

        img_yielded += 1
        if img_yielded == n_images:
            break
        else:
            counter = (counter + batch_size) % n_images


def generator_test(fnames, images_path):
    batch_size = b_s
    images = [os.path.join(images_path, fname) for fname in fnames]

    n_images = len(images)

    gaussian = np.zeros((batch_size, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    img_yielded = 0
    while True:
        yield [
            [
                preprocess_images(
                    images[counter : counter + batch_size], shape_r, shape_c
                ),
                gaussian,
            ]
        ]
        img_yielded += 1
        if img_yielded == n_images:
            break
        else:
            counter = (counter + batch_size) % n_images
