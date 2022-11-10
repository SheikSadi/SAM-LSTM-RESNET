import tensorflow as tf
import keras.backend as K
from keras.layers import Layer, InputSpec
from keras import initializers


class AttentiveConvLSTM(Layer):
    """
    att_convlstm = AttentiveConvLSTM(
        nb_filters_in=512, nb_filters_out=512, nb_filters_att=512, nb_cols=3, nb_rows=3
    )(att_convlstm)
    """

    def __init__(
        self,
        nb_filters_in,
        nb_filters_out,
        nb_filters_att,
        nb_rows,
        nb_cols,
        init="normal",
        inner_init="orthogonal",
        attentive_init="zero",
        **kwargs,
    ):
        self.nb_filters_in = nb_filters_in
        self.nb_filters_out = nb_filters_out
        self.nb_filters_att = nb_filters_att
        self.nb_rows = nb_rows
        self.nb_cols = nb_cols
        self.init = init
        self.inner_init = inner_init
        self.attentive_init = attentive_init
        super().__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        return input_shape[:1] + (self.nb_filters_out,) + input_shape[3:]

    def compute_mask(self, input, mask):
        return None

    def get_initial_states(self, x):
        # we will skip the time-axis (axis=1)
        initial_state_shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[4])
        initial_state = tf.zeros(initial_state_shape)
        initial_states = [initial_state for _ in self.states]

        return initial_states

    def build(self, input_shape):
        self.inputspec = [InputSpec(shape=input_shape)]
        self.states = [None, None]
        in_channels = input_shape[2]

        def init_conv_weights(out_channels, name, bias=True):
            _kernel = self.add_weight(
                f"{name}_kernel",
                shape=[out_channels, in_channels, self.nb_rows, self.nb_cols],
                trainable=True,
                initializer=initializers.get(self.init),
            )
            if bias:
                _bias = self.add_weight(
                    f"{name}_bias",
                    shape=[out_channels],
                    trainable=True,
                    initializer=initializers.get(self.init),
                )
            else:
                _bias = tf.zeros(shape=[out_channels])
            return _kernel, _bias

        self.Wa = init_conv_weights(self.nb_filters_att, "Wa")
        self.Ua = init_conv_weights(self.nb_filters_att, "Ua")
        self.Va = init_conv_weights(1, "Va", bias=False)

        self.Wi = init_conv_weights(self.nb_filters_out, "Wi")
        self.Ui = init_conv_weights(self.nb_filters_out, "Ui")

        self.Wf = init_conv_weights(self.nb_filters_out, "Wf")
        self.Uf = init_conv_weights(self.nb_filters_out, "Uf")

        self.Wc = init_conv_weights(self.nb_filters_out, "Wc")
        self.Uc = init_conv_weights(self.nb_filters_out, "Uc")

        self.Wo = init_conv_weights(self.nb_filters_out, "Wo")
        self.Uo = init_conv_weights(self.nb_filters_out, "Uo")

    def preprocess_input(self, x):
        return x

    @staticmethod
    def conv(kernel_bias, input):
        kernel, bias_ = kernel_bias
        kernelT = tf.transpose(kernel, perm=[3, 2, 1, 0])
        kernelT_x_input = tf.nn.conv2d(
            input, kernelT, strides=1, padding="SAME", data_format="NCHW"
        )

        bias_ = tf.expand_dims(tf.expand_dims(bias_, axis=1), axis=2)
        bias = tf.repeat(
            tf.repeat(bias_, repeats=input.shape[2], axis=1),
            repeats=input.shape[3],
            axis=2,
        )
        return kernelT_x_input + bias

    def step(self, X, states):
        sigmoid = tf.keras.activations.sigmoid
        tanh = tf.keras.activations.tanh
        conv = self.conv

        Ht_1 = states[0]
        Ct_1 = states[1]

        Zt = conv(self.Va, tanh(conv(self.Wa, X) + conv(self.Ua, Ht_1)))
        At = tf.repeat(
            tf.reshape(
                tf.nn.softmax(K.batch_flatten(Zt)),
                (X.shape[0], 1, X.shape[2], X.shape[3]),
            ),
            repeats=X.shape[1],
            axis=1,
        )

        Xt_ = X * At

        It = sigmoid(conv(self.Wi, Xt_) + conv(self.Ui, Ht_1))
        Ft = sigmoid(conv(self.Wf, Xt_) + conv(self.Uf, Ht_1))
        Ot = sigmoid(conv(self.Wo, Xt_) + conv(self.Uo, Ht_1))
        Gt = tanh(conv(self.Wc, Xt_) + conv(self.Uc, Ht_1))
        Ct = Ft * Ct_1 + It * Gt
        Ht = Ot * tanh(Ct)

        return Ht, [Ht, Ct]

    def get_constants(self, x):
        return []

    def call(self, x, mask=None):
        initial_states = self.get_initial_states(x)
        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        last_output, outputs, states = K.rnn(
            self.step,
            preprocessed_input,
            initial_states,
            go_backwards=False,
            mask=mask,
            constants=constants,
            unroll=False,
            input_length=x.shape[1],
        )

        if len(last_output.shape) == 3:
            last_output = K.expand_dims(last_output, axis=0)

        return last_output

    def get_config(self):
        config_to_serialize = dict(
            nb_filters_in=self.nb_filters_in,
            nb_filters_out=self.nb_filters_out,
            nb_filters_att=self.nb_filters_att,
            nb_rows=self.nb_rows,
            nb_cols=self.nb_cols,
            init=self.init,
            inner_init=self.inner_init,
            attentive_init=self.attentive_init,
        )
        config = super().get_config()
        config.update(config_to_serialize)
        return config
