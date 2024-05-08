import keras
import tensorflow as tf
import numpy as np
import seaborn as sns
from keras.layers import LSTM, Dense, Dropout, Input, Layer
from keras.models import Sequential
from keras.losses import mean_absolute_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from dataset import TimeSeriesDataset
import os

@keras.saving.register_keras_serializable()
class Attention(Layer):
    def __init__(self, return_attention=False, **kwargs):
        self.return_attention = return_attention
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        alpha = tf.nn.softmax(e, axis=1)
        context = tf.reduce_sum(alpha * x, axis=1)

        if self.return_attention:
            return [context, alpha]
        else:
            return context

def create_attention_lstm_model(time_steps, n_features):
    # Define the input shape
    inputs = Input(shape=(time_steps, n_features))

    # Add the LSTM layers
    lstm1 = LSTM(units=64, return_sequences=True)(inputs)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units=64, return_sequences=True)(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    lstm3 = LSTM(units=64, return_sequences=True)(dropout2)
    dropout3 = Dropout(0.2)(lstm3)

    # Add the Attention layer
    attention = Attention(return_attention=True)(dropout3)
    context, attention_weights = attention
    outputs = Dense(units=1)(context)

    # Create the model
    model = keras.models.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[keras.metrics.RootMeanSquaredError(), keras.metrics.MeanAbsolutePercentageError()])

    return model