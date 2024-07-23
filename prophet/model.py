import keras
import tensorflow as tf

tf.get_logger().setLevel("INFO")
from keras.layers import LSTM, Dense, Dropout, Input, Layer  # type: ignore
from keras.metrics import MeanAbsolutePercentageError  # type: ignore
from keras.metrics import RootMeanSquaredError  # type: ignore


@keras.saving.register_keras_serializable()
class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * x, axis=1)

        if self.return_attention:
            return [context, alpha]
        else:
            return context


@keras.saving.register_keras_serializable()
class AttentionLSTM(keras.Model):
    def __init__(self, time_steps, n_features, lr, **kwargs):
        super(AttentionLSTM, self).__init__(**kwargs)

        self.time_steps = time_steps
        self.n_features = n_features
        self.inputs = Input(shape=(self.time_steps, self.n_features))
        self.lstm1 = LSTM(units=64, return_sequences=True)
        self.dropout1 = Dropout(0.2)
        self.lstm2 = LSTM(units=64, return_sequences=True)
        self.dropout2 = Dropout(0.2)
        self.lstm3 = LSTM(units=64, return_sequences=True)
        self.dropout3 = Dropout(0.2)
        self.attention = Attention(return_attention=True)
        self.outputs = Dense(units=1)

        self.compile(lr)

    def compile(self, lr=1e-3):
        super(AttentionLSTM, self).compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="mean_absolute_error",
            metrics=[RootMeanSquaredError(), MeanAbsolutePercentageError()],
        )

    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dropout2(x)
        x = self.lstm3(x)
        x = self.dropout3(x)
        context, attention_weights = self.attention(x)
        outputs = self.outputs(context)

        return outputs
