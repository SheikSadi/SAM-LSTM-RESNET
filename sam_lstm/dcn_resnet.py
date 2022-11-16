"""
This code is part of the Keras ResNet-50 model
"""
import keras.backend as K
from keras.layers import (
    add,
    Input,
    Activation,
    Conv2D,
    MaxPooling2D,
    ZeroPadding2D,
    BatchNormalization,
)
from keras.models import Model
from keras.utils import get_file
from sam_lstm.config import TH_WEIGHTS_PATH_NO_TOP


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """
    UPGRADED - @SheikSadi
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        nb_filter1, kernel_size=(1, 1), name=conv_name_base + "2a", trainable=False
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter2,
        kernel_size,
        padding="same",
        name=conv_name_base + "2b",
        trainable=False,
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter3, kernel_size=(1, 1), name=conv_name_base + "2c", trainable=False
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c", trainable=False)(x)

    x = add([x, input_tensor])
    x = Activation("relu", trainable=False)(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """
    UPGRADED - @SheikSadi
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        nb_filter1,
        kernel_size=(1, 1),
        strides=strides,
        name=conv_name_base + "2a",
        trainable=False,
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter2,
        kernel_size,
        padding="same",
        name=conv_name_base + "2b",
        trainable=False,
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter3, kernel_size=(1, 1), name=conv_name_base + "2c", trainable=False
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c", trainable=False)(x)

    shortcut = Conv2D(
        nb_filter3,
        kernel_size=(1, 1),
        strides=strides,
        name=conv_name_base + "1",
        trainable=False,
    )(input_tensor)
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + "1", trainable=False
    )(shortcut)

    x = add([x, shortcut])
    x = Activation("relu", trainable=False)(x)
    return x


def conv_block_atrous(
    input_tensor, kernel_size, filters, stage, block, dilation_rate=(2, 2)
):
    """
    UPGRADED: @SheikSadi
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + "2a", trainable=False)(
        input_tensor
    )
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter2,
        kernel_size,
        padding="same",
        dilation_rate=dilation_rate,
        name=conv_name_base + "2b",
        trainable=False,
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c", trainable=False)(x)

    shortcut = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "1", trainable=False)(
        input_tensor
    )
    shortcut = BatchNormalization(
        axis=bn_axis, name=bn_name_base + "1", trainable=False
    )(shortcut)

    x = add([x, shortcut])
    x = Activation("relu", trainable=False)(x)
    return x


def identity_block_atrous(
    input_tensor, kernel_size, filters, stage, block, dilation_rate=(2, 2)
):
    """
    UPGRADED - @SheikSadi
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    bn_axis = 1

    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(
        nb_filter1, kernel_size=(1, 1), name=conv_name_base + "2a", trainable=False
    )(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(
        nb_filter2,
        kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        name=conv_name_base + "2b",
        trainable=False,
    )(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + "2c", trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2c", trainable=False)(x)

    x = add([x, input_tensor])
    x = Activation("relu", trainable=False)(x)
    return x


def dcn_resnet(input_tensor=None):
    input_shape = (3, None, None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor)
        else:
            img_input = input_tensor

    bn_axis = 1

    # conv_1
    x = ZeroPadding2D((3, 3), trainable=False)(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name="conv1", trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1", trainable=False)(x)
    x = Activation("relu", trainable=False)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", trainable=False)(x)

    # conv_2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="b")
    x = identity_block(x, 3, [64, 64, 256], stage=2, block="c")

    # conv_3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block="a", strides=(2, 2))
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="b")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="c")
    x = identity_block(x, 3, [128, 128, 512], stage=3, block="d")

    # conv_4
    x = conv_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="a", dilation_rate=(2, 2)
    )
    x = identity_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="b", dilation_rate=(2, 2)
    )
    x = identity_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="c", dilation_rate=(2, 2)
    )
    x = identity_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="d", dilation_rate=(2, 2)
    )
    x = identity_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="e", dilation_rate=(2, 2)
    )
    x = identity_block_atrous(
        x, 3, [256, 256, 1024], stage=4, block="f", dilation_rate=(2, 2)
    )

    # conv_5
    x = conv_block_atrous(
        x, 3, [512, 512, 2048], stage=5, block="a", dilation_rate=(4, 4)
    )
    x = identity_block_atrous(
        x, 3, [512, 512, 2048], stage=5, block="b", dilation_rate=(4, 4)
    )
    x = identity_block_atrous(
        x, 3, [512, 512, 2048], stage=5, block="c", dilation_rate=(4, 4)
    )

    # Create model
    model = Model(img_input, x)

    # Load weights
    weights_path = get_file(
        "resnet50_weights_th_dim_ordering_th_kernels_notop.h5",
        TH_WEIGHTS_PATH_NO_TOP,
        cache_subdir="weights",
        file_hash="f64f049c92468c9affcd44b0976cdafe",
    )
    model.load_weights(weights_path)

    return model
