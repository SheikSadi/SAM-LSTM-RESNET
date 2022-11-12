import cv2
import os

import tensorflow as tf
from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model
from keras.utils import get_file

from sam.utilities import postprocess_predictions
from sam.models import (
    kl_divergence,
    correlation_coefficient,
    nss,
    sam_resnet,
)
from sam.generator import generator, generator_test
from sam.cropping import batch_crop_images
from sam.config import (
    shape_r,
    shape_c,
    shape_r_gt,
    shape_c_gt,
    nb_gaussian,
    b_s,
    nb_epoch,
    aspect_ratio,
    retained_attention,
    DATASET_IMAGES_URL,
    DATASET_MAPS_URL,
    DATASET_FIXS_URL,
    SAM_RESNET_SALICON_2017_WEIGHTS,
)


tf.keras.backend.set_image_data_format(data_format="channels_first")


class SalMap:
    def __init__(self):
        self.x = Input((3, shape_r, shape_c))
        self.x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))
        self.model = Model(
            inputs=[self.x, self.x_maps], outputs=sam_resnet([self.x, self.x_maps])
        )
        self.model.compile(
            RMSprop(learning_rate=1e-4),
            loss=[kl_divergence, correlation_coefficient, nss],
        )

    def train(self, dataset_path, checkpoint_path, batch_size=b_s):
        imgs_path = os.path.join(dataset_path, "images")
        maps_path = os.path.join(dataset_path, "maps")
        fixs_path = os.path.join(dataset_path, "fixations")

        if not os.path.exists(imgs_path):
            print(f"Didn't find {imgs_path}")
            imgs_zip_path = get_file(
                origin=DATASET_IMAGES_URL,
                file_hash="2c72253ccb5288118864ebd2ab15a55e",
                hash_algorithm="md5",
                extract=True,
                cache_dir=os.getcwd(),
                cache_subdir=dataset_path,
            )
            imgs_path = imgs_zip_path.replace(".zip", "")

        if not os.path.exists(maps_path):
            print(f"Didn't find {maps_path}")
            maps_zip_path = get_file(
                origin=DATASET_MAPS_URL,
                file_hash="5218595acfeec3b9fc0a4964d0566360",
                hash_algorithm="md5",
                extract=True,
                cache_dir=os.getcwd(),
                cache_subdir=dataset_path,
            )
            maps_path = maps_zip_path.replace(".zip", "")

        if not os.path.exists(fixs_path):
            print(f"Didn't find {fixs_path}")
            fixs_zip_path = get_file(
                origin=DATASET_FIXS_URL,
                file_hash="0d6f4a54c3d36ccc85a74b1b4b40bed5",
                hash_algorithm="md5",
                extract=True,
                cache_dir=os.getcwd(),
                cache_subdir=dataset_path,
            )
            fixs_path = fixs_zip_path.replace(".zip", "")

        imgs_train_path = os.path.join(imgs_path, "train")
        maps_train_path = os.path.join(maps_path, "train")
        fixs_train_path = os.path.join(fixs_path, "train")

        imgs_val_path = os.path.join(imgs_path, "val")
        maps_val_path = os.path.join(maps_path, "val")
        fixs_val_path = os.path.join(fixs_path, "val")

        for path in [
            imgs_train_path,
            maps_train_path,
            fixs_train_path,
            imgs_val_path,
            maps_val_path,
            fixs_val_path,
        ]:
            if not os.path.exists(path):
                raise Exception(f"Didn't find the {path}! Can't start training...")

        print("Training SAM-ResNet")
        train_gen = generator(
            batch_size,
            imgs_train_path,
            maps_train_path,
            fixs_train_path,
        )
        validation_gen = generator(
            batch_size,
            imgs_val_path,
            maps_val_path,
            fixs_val_path,
        )
        if not os.path.exists(checkpoint_path):
            raise Exception(
                f"Directory: {checkpoint_path} not found, first make sure it exists. Then, try again!"
            )

        self.model.fit(
            train_gen,
            batch_size=b_s,
            epochs=nb_epoch,
            validation_data=validation_gen,
            callbacks=[
                EarlyStopping(patience=3),
                ModelCheckpoint(
                    os.path.join(
                        checkpoint_path, "sam-resnet-{epoch:02}-{val_loss:.4f}.pkl"
                    ),
                    monitor="val_loss",
                    verbose=0,
                    save_best_only=True,
                    save_weights_only=False,
                    mode="auto",
                    save_freq="epoch",
                ),
            ],
        )

    def load_weights(self, weights_dir):
        if weights_dir:
            fname = os.path.basename(weights_dir)
            if not weights_dir.startswith("/"):
                weights_dir = f"/{weights_dir}"
            dirname = os.path.dirname(weights_dir)
            cache_subdir = os.path.basename(dirname)
            cache_dir = os.path.dirname(dirname)
            if cache_dir == "/":
                cache_dir = os.getcwd()
        else:
            fname = "sam-resnet_salicon_weights.pkl"
            cache_dir = None
            cache_subdir = "weights"

        weights_dir = get_file(
            fname,
            SAM_RESNET_SALICON_2017_WEIGHTS,
            cache_subdir=cache_subdir,
            file_hash="92b5f89fd34a3968776a5c4327efb32c",
            cache_dir=cache_dir,
        )

        self.model.load_weights(weights_dir)

    def predict_maps(self, imgs_test_path="/samples", batch_size=b_s, weights_dir=None):
        self.load_weights(weights_dir)

        if imgs_test_path.startswith("/"):
            imgs_test_path = imgs_test_path.rsplit("/", 1)[1]
        # Output Folder Path
        if os.path.exists(imgs_test_path):
            self.imgs_test_path = imgs_test_path
        else:
            self.imgs_test_path = os.path.join(os.getcwd(), imgs_test_path)
        if not os.path.exists(self.imgs_test_path):
            raise Exception(
                f"Couldn't find the directory {imgs_test_path} or {self.imgs_test_path}"
            )

        maps_folder = os.path.join(os.path.dirname(self.imgs_test_path), "maps")
        if not os.path.exists(maps_folder):
            os.mkdir(maps_folder)

        file_names = [
            fname
            for fname in os.listdir(self.imgs_test_path)
            if fname.endswith((".jpg", ".jpeg", ".png"))
        ]
        file_names.sort()

        print("Predicting saliency maps for " + self.imgs_test_path)
        predictions = self.model.predict(
            generator_test(
                batch_size, imgs_test_path=self.imgs_test_path, img_fnames=file_names
            )
        )[0]

        for pred, fname in zip(predictions, file_names):
            image_path = os.path.join(self.imgs_test_path, fname)
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            res = postprocess_predictions(
                pred[0], original_image.shape[0], original_image.shape[1]
            )
            map_path = os.path.join(maps_folder, fname)
            cv2.imwrite(map_path, res.astype(int))

    def batch_crop(
        self,
        a_r=aspect_ratio,
        attention=retained_attention,
    ):
        originals_folder = self.imgs_test_path
        maps_folder = os.path.join(os.path.dirname(originals_folder), "maps")
        if not os.path.exists(maps_folder):
            raise Exception(
                f"Saliency mappings for the images in {originals_folder}"
                " must be present in {maps_folder}.\n"
                "Run this command - salmap.test(imgs_test_path=<original-images>) and try again!"
            )
        crops_folder = os.path.join(os.path.dirname(originals_folder), "crops")
        boxes_folder = os.path.join(os.path.dirname(originals_folder), "boxes")
        batch_crop_images(
            originals_folder, maps_folder, crops_folder, boxes_folder, a_r, attention
        )
