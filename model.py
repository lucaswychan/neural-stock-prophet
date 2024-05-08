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

def get_prediction_trend(model, dataset: TimeSeriesDataset):
    predictions = model.predict(dataset.X)
    predictions = dataset.scaler.inverse_transform(predictions.reshape(-1, 1))
    
    return predictions

def visualize_prediction(model, dataset: TimeSeriesDataset, file_name):
    predictions = get_prediction_trend(model, dataset)
    y_test = dataset.scaler.inverse_transform(dataset.y.reshape(-1, 1))
    
    # Visualising the results
    plt.figure(figsize=(12, 6))
    plt.plot(dataset.time_index[dataset.time_steps:], y_test, color = 'red', label = 'Real Stock Price')
    plt.plot(dataset.time_index[dataset.time_steps:], predictions, color = 'blue', label = 'Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    
    file_path = f"Result/{file_name}.png"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)

    # Print the mae
    print(f"The mean absolute error is {round(mean_absolute_error(y_test, predictions), 3)}USD")


def visualize_timestep_importance(model, dataset: TimeSeriesDataset):
    # Create a new model to output the attention weights
    def create_attention_model(X):
        # Define the input shape
        inputs = Input(shape=(X.shape[1], dataset.n_features))

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

        # Create the model
        model = keras.models.Model(inputs=inputs, outputs=attention_weights)

        return model

    attention_model = create_attention_model(dataset.X)
    attention_model.set_weights(model.get_weights()[:-2])
    attention_weights_test = attention_model.predict(dataset.X)

    # Calculate the mean attention weights for each time step
    attention_weights_mean = np.mean(attention_weights_test, axis=0)

    # Visualize the importance of time steps
    plt.figure(figsize=(15, 6))
    sns.barplot(x=np.arange(1, attention_weights_mean.shape[0] + 1), y=attention_weights_mean[:, 0])
    plt.title('Importance of Time Steps')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    
    file_path = "Result/attention_weights.png"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)