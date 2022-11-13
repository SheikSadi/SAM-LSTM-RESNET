"""
Upgraded @SheikSadi
"""
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.layers import Layer, InputSpec
from keras import initializers, regularizers, constraints


floatX = K.floatx()


class LearningPrior(Layer):
    def __init__(
        self,
        nb_gaussian,
        init="normal",
        weights=None,
        W_regularizer=None,
        activity_regularizer=None,
        W_constraint=None,
        **kwargs,
    ):
        self.nb_gaussian = nb_gaussian
        self.init = initializers.get(init)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        super(LearningPrior, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.initial_weights is None:
            self.W = self.add_weight(
                name=f"{self.name}_W",
                shape=(self.nb_gaussian * 4,),
                initializer=self.init,
                regularizer=self.W_regularizer,
                trainable=True,
                constraint=self.W_constraint,
            )
        else:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def get_output_shape_for(self, input_shape):
        self.b_s = input_shape[0]
        self.height = input_shape[2]
        self.width = input_shape[3]

        return self.b_s, self.nb_gaussian, self.height, self.width

    def call(self, x, mask=None):
        mu_x = K.clip(self.W[: self.nb_gaussian], 0.25, 0.75)
        mu_y = K.clip(self.W[self.nb_gaussian : self.nb_gaussian * 2], 0.35, 0.65)
        sigma_x = K.clip(self.W[self.nb_gaussian * 2 : self.nb_gaussian * 3], 0.1, 0.9)
        sigma_y = K.clip(self.W[self.nb_gaussian * 3 :], 0.2, 0.8)

        self.b_s = x.shape[0] if x.shape[0] else 1
        self.height = x.shape[2]
        self.width = x.shape[3]

        e = self.height / self.width
        e1 = (1 - e) / 2
        e2 = (1 + e) / 2

        x_t, y_t = tf.meshgrid(
            tf.cast(tf.linspace(0, 1, self.width), dtype=tf.float32),
            tf.cast(tf.linspace(e1, e2, self.height), dtype=tf.float32),
        )

        x_t = tf.repeat(tf.expand_dims(x_t, axis=2), self.nb_gaussian, axis=2)
        y_t = tf.repeat(tf.expand_dims(y_t, axis=2), self.nb_gaussian, axis=2)

        gaussian = (
            1
            / (2 * np.pi * sigma_x * sigma_y + K.epsilon())
            * tf.math.exp(
                -(
                    (x_t - mu_x) ** 2 / (2 * sigma_x**2 + K.epsilon())
                    + (y_t - mu_y) ** 2 / (2 * sigma_y**2 + K.epsilon())
                )
            )
        )

        gaussian /= tf.math.reduce_sum(gaussian, axis=[0, 1])

        gaussian = tf.repeat(tf.expand_dims(gaussian, axis=0), self.b_s, axis=0)

        output = tf.transpose(gaussian, perm=[0, 3, 1, 2])  # To NCHW format

        return output

    def get_config(self):
        config = {
            "nb_gaussian": self.nb_gaussian,
            "init": self.init.__name__,
            "W_regularizer": self.W_regularizer.get_config()
            if self.W_regularizer
            else None,
            "activity_regularizer": self.activity_regularizer.get_config()
            if self.activity_regularizer
            else None,
            "W_constraint": self.W_constraint.get_config()
            if self.W_constraint
            else None,
        }
        base_config = super(LearningPrior, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
