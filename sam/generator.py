from __future__ import division

import os
import numpy as np
from sam.config import *
from sam.utilities import (
    preprocess_images,
    preprocess_maps,
    preprocess_fixmaps,
)


def generator(b_s, phase_gen="train"):
    if phase_gen == "train":
        images = [
            os.path.join(imgs_train_path, f)
            for fname in os.listdir(imgs_train_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        maps = [
            os.path.join(maps_train_path, fname)
            for fname in os.listdir(maps_train_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        fixs = [
            os.path.join(fixs_train_path)
            for fname in os.listdir(fixs_train_path)
            if fname.endswith(".mat")
        ]
    elif phase_gen == "val":
        images = [
            os.path.join(imgs_val_path, fname)
            for fname in os.listdir(imgs_val_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        maps = [
            os.path.join(maps_val_path, fname)
            for fname in os.listdir(maps_val_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        fixs = [
            os.path.join(fixs_val_path, fname)
            for fname in os.listdir(fixs_val_path)
            if fname.endswith(".mat")
        ]
    else:
        raise NotImplementedError

    images.sort()
    maps.sort()
    fixs.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter : counter + b_s], shape_r_out, shape_c_out)
        Y_fix = preprocess_fixmaps(
            fixs[counter : counter + b_s], shape_r_out, shape_c_out
        )
        yield [
            preprocess_images(images[counter : counter + b_s], shape_r, shape_c),
            gaussian,
        ], [Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)


def generator_test(b_s, imgs_test_path):
    images = [
        os.path.join(imgs_test_path, fname)
        for fname in os.listdir(imgs_test_path)
        if fname.endswith((".jpg", ".jpeg", ".png"))
    ]
    images.sort()

    gaussian = np.zeros((b_s, nb_gaussian, shape_r_gt, shape_c_gt))

    counter = 0
    while True:
        yield [
            preprocess_images(images[counter : counter + b_s], shape_r, shape_c),
            gaussian,
        ]
        counter = (counter + b_s) % len(images)
