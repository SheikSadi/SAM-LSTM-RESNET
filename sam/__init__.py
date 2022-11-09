# TODO: Upgrade
import cv2
import os
import keras.backend as K

from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input
from keras.models import Model

from sam.utilities import postprocess_predictions
from sam.models import kl_divergence, correlation_coefficient, nss, sam_resnet #, sam_vgg
from sam.generator import generator, generator_test
from sam.cropping import batch_crop_images
from sam.config import *


class SalMap:
    RESNET = 1
    VGG = 0

    def __init__(self, weights_dir=None, version=RESNET):
        self.x = Input((3, shape_r, shape_c))
        self.x_maps = Input((nb_gaussian, shape_r_gt, shape_c_gt))
        self.version = version

    def compile(self):
        if self.version == self.RESNET:
            self.model = Model(
                inputs=[self.x, self.x_maps], outputs=sam_resnet([self.x, self.x_maps])
            )
            
            self.model.compile(
                RMSprop(learning_rate=1e-4), loss=[kl_divergence, correlation_coefficient, nss]
            )
        # elif self.version == self.VGG:
        #     self.model = Model(
        #         input=[self.x, self.x_maps], output=sam_vgg([self.x, self.x_maps])
        #     )
        #     print("Compiling SAM-VGG")
        #     self.model.compile(
        #         RMSprop(learning_rate=1e-4), loss=[kl_divergence, correlation_coefficient, nss]
        #     )
        else:
            raise NotImplementedError

    # def train(self):
    #     if nb_imgs_train % b_s != 0 or nb_imgs_val % b_s != 0:
    #         print(
    #             "The number of training and validation images should be a multiple of the batch size. Please change your batch size in config.py accordingly."
    #         )
    #         exit()

    #     if self.version == self.VGG:
    #         print("Training SAM-VGG")
    #         self.model.fit_generator(
    #             generator(b_s=b_s),
    #             nb_imgs_train,
    #             nb_epoch=nb_epoch,
    #             validation_data=generator(b_s=b_s, phase_gen="val"),
    #             nb_val_samples=nb_imgs_val,
    #             callbacks=[
    #                 EarlyStopping(patience=3),
    #                 ModelCheckpoint(
    #                     "weights.sam-vgg.{epoch:02d}-{val_loss:.4f}.pkl",
    #                     save_best_only=True,
    #                 ),
    #             ],
    #         )
    #     elif self.version == self.RESNET:
    #         print("Training SAM-ResNet")
    #         self.model.fit_generator(
    #             generator(b_s=b_s),
    #             nb_imgs_train,
    #             nb_epoch=nb_epoch,
    #             validation_data=generator(b_s=b_s, phase_gen="val"),
    #             nb_val_samples=nb_imgs_val,
    #             callbacks=[
    #                 EarlyStopping(patience=3),
    #                 ModelCheckpoint(
    #                     "weights.sam-resnet.{epoch:02d}-{val_loss:.4f}.pkl",
    #                     save_best_only=True,
    #                 ),
    #             ],
    #         )

    def load_weights(self,  weights_dir=None):
        if not weights_dir:
            weights_dir = os.path.join(os.getcwd(), "weights")
        
        if self.version == self.RESNET:
            resnet_weights_path = os.path.join(
                weights_dir, "sam-resnet_salicon_weights.pkl"
            )
            self.model.load_weights(resnet_weights_path)
        # elif self.version == self.VGG:
        #     vgg_weights_path = os.path.join(
        #         weights_dir, "sam-vgg_salicon_weights.pkl"
        #     )
        #     print("Loading SAM-VGG weights")
        #     self.model.load_weights(vgg_weights_path)


    def test(self, imgs_test_path="samples"):
        #FIXME: Upgrade
        # Output Folder Path
        imgs_test_path = os.path.join(os.getcwd(), imgs_test_path)
        
        maps_folder = os.path.join(os.getcwd(), "maps")
        if not os.path.exists(maps_folder):
            os.mkdir(maps_folder)

        file_names = [
            fname
            for fname in os.listdir(imgs_test_path)
            if (
                fname.endswith((".jpg", ".jpeg", ".png"))
                and fname not in os.listdir(maps_folder)
            )
        ]
        file_names.sort()
        nb_imgs_test = len(file_names)

        if nb_imgs_test % b_s != 0:
            print(
                "The number of test images should be a multiple of the batch size. Please change your batch size in config.py accordingly."
            )
            exit()
        
        print("Predicting saliency maps for " + imgs_test_path)
        predictions = self.model.predict(
            generator_test(b_s=b_s, imgs_test_path=imgs_test_path), nb_imgs_test
        )[0]

        for pred, fname in zip(predictions, file_names):
            image_path = os.path.join(imgs_test_path, fname)
            original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            res = postprocess_predictions(
                pred[0], original_image.shape[0], original_image.shape[1]
            )
            map_path = os.path.join(maps_folder, fname)
            cv2.imwrite(map_path, res.astype(int))

    def batch_crop(
        self,
        originals_folder="samples",
        maps_folder="maps",
        crops_folder="crops",
        boxes_folder="boxes",
        a_r=aspect_ratio,
        attention=retained_attention,
    ):
        batch_crop_images(
            originals_folder, maps_folder, crops_folder, boxes_folder, a_r, attention
        )
