"""
Upgraded @SheikSadi
"""
import os
import keras.backend as K
import numpy as np

from sam.config import *
# from sam.dcn_vgg import dcn_vgg
from sam.dcn_resnet import dcn_resnet
from sam.gaussian_prior import LearningPrior
from sam.attentive_convlstm import AttentiveConvLSTM

from keras.layers import Lambda, concatenate, Conv2D, UpSampling2D


def repeat(x):
    return K.reshape(
        K.repeat(K.batch_flatten(x), nb_timestep),
        (b_s, nb_timestep, 512, shape_r_gt, shape_c_gt),
    )


def repeat_shape(s):
    return (s[0], nb_timestep) + s[1:]


def upsampling(x):
    return UpSampling2D(
        size=upsampling_factor,
        interpolation="bilinear",
    )(x)


def upsampling_shape(s):
    return s[:2] + (s[2] * upsampling_factor, s[3] * upsampling_factor)


# KL-Divergence Loss
def kl_divergence(y_true, y_pred):
    max_y_pred = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    sum_y_pred = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    y_true /= sum_y_true + K.epsilon()
    y_pred /= sum_y_pred + K.epsilon()

    return 10 * K.sum(
        K.sum(y_true * K.log((y_true / (y_pred + K.epsilon())) + K.epsilon()), axis=-1),
        axis=-1,
    )


# Correlation Coefficient Loss
def correlation_coefficient(y_true, y_pred):
    max_y_pred = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    y_pred /= max_y_pred

    sum_y_true = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.sum(K.sum(y_true, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    sum_y_pred = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.sum(K.sum(y_pred, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )

    y_true /= sum_y_true + K.epsilon()
    y_pred /= sum_y_pred + K.epsilon()

    N = shape_r_out * shape_c_out
    sum_prod = K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
    sum_x = K.sum(K.sum(y_true, axis=2), axis=2)
    sum_y = K.sum(K.sum(y_pred, axis=2), axis=2)
    sum_x_square = K.sum(K.sum(K.square(y_true), axis=2), axis=2)
    sum_y_square = K.sum(K.sum(K.square(y_pred), axis=2), axis=2)

    num = sum_prod - ((sum_x * sum_y) / N)
    den = K.sqrt(
        (sum_x_square - K.square(sum_x) / N) * (sum_y_square - K.square(sum_y) / N)
    )

    return -2 * num / den


# Normalized Scanpath Saliency Loss
def nss(y_true, y_pred):
    max_y_pred = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.max(K.max(y_pred, axis=2), axis=2)),
                shape_r_out,
                axis=-1,
            )
        ),
        shape_c_out,
        axis=-1,
    )
    y_pred /= max_y_pred
    y_pred_flatten = K.batch_flatten(y_pred)

    y_mean = K.mean(y_pred_flatten, axis=-1)
    y_mean = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(
                K.expand_dims(K.expand_dims(y_mean)), shape_r_out, axis=-1
            )
        ),
        shape_c_out,
        axis=-1,
    )

    y_std = K.std(y_pred_flatten, axis=-1)
    y_std = K.repeat_elements(
        K.expand_dims(
            K.repeat_elements(K.expand_dims(K.expand_dims(y_std)), shape_r_out, axis=-1)
        ),
        shape_c_out,
        axis=-1,
    )

    y_pred = (y_pred - y_mean) / (y_std + K.epsilon())

    return -(
        K.sum(K.sum(y_true * y_pred, axis=2), axis=2)
        / K.sum(K.sum(y_true, axis=2), axis=2)
    )


# Gaussian priors initialization
def gaussian_priors_init(shape, name=None):
    means = np.random.uniform(low=0.3, high=0.7, size=shape[0] // 2)
    covars = np.random.uniform(low=0.05, high=0.3, size=shape[0] // 2)
    return K.variable(np.concatenate((means, covars), axis=0), name=name)

"""
def sam_vgg(x):
    # Dilated Convolutional Network
    dcn = dcn_vgg(input_tensor=x[0])

    # Attentive Convolutional LSTM
    att_convlstm = Lambda(repeat, repeat_shape)(dcn.output)
    att_convlstm = AttentiveConvLSTM(
        nb_filters_in=512, nb_filters_out=512, nb_filters_att=512, nb_cols=3, nb_rows=3
    )(att_convlstm)

    # Learned Prior (1)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
    concateneted = merge([att_convlstm, priors1], mode="concat", concat_axis=1)
    learned_priors1 = AtrousConvolution2D(
        512, 5, 5, border_mode="same", activation="relu", atrous_rate=(4, 4)
    )(concateneted)

    # Learned Prior (2)
    priors2 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
    concateneted = merge([learned_priors1, priors2], mode="concat", concat_axis=1)
    learned_priors2 = AtrousConvolution2D(
        512, 5, 5, border_mode="same", activation="relu", atrous_rate=(4, 4)
    )(concateneted)

    # Final Convolutional Layer
    outs = Convolution2D(1, 1, 1, border_mode="same", activation="relu")(
        learned_priors2
    )
    outs_up = Lambda(upsampling, upsampling_shape)(outs)

    return [outs_up, outs_up, outs_up]
"""

def sam_resnet(x):
    # Dilated Convolutional Network
    dcn = dcn_resnet(input_tensor=x[0])
    conv_feat = Conv2D(512, (3, 3), padding="same", activation="relu")(
        dcn.output
    )

    # Attentive Convolutional LSTM
    att_convlstm = Lambda(repeat, repeat_shape)(conv_feat)
    att_convlstm = AttentiveConvLSTM(
        nb_filters_in=512, nb_filters_out=512, nb_filters_att=512, nb_cols=3, nb_rows=3
    )(att_convlstm)

    # Learned Prior (1)
    priors1 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
    concateneted = concatenate([att_convlstm, priors1], axis=1)
    learned_priors1 = Conv2D(
        512, (5, 5), padding="same", activation="relu", dilation_rate=(4, 4)
    )(concateneted)

    # Learned Prior (2)
    priors2 = LearningPrior(nb_gaussian=nb_gaussian, init=gaussian_priors_init)(x[1])
    concateneted = concatenate([learned_priors1, priors2], axis=1)
    learned_priors2 = Conv2D(
        512, (5, 5), padding="same", activation="relu", dilation_rate=(4, 4)
    )(concateneted)

    # Final Convolutional Layer
    outs = Conv2D(1, (1, 1), padding="same", activation="relu")(
        learned_priors2
    )
    outs_up = Lambda(upsampling, upsampling_shape)(outs)

    return [outs_up, outs_up, outs_up]
