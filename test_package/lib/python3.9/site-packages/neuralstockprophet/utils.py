import os

import matplotlib.pyplot as plt
import tensorflow as tf

from .dataset import TimeSeriesDataset

tf.get_logger().setLevel("INFO")
import keras
import numpy as np
import seaborn as sns
from keras.layers import LSTM, Dropout, Input  # type: ignore
from sklearn.metrics import mean_absolute_error

from .model import Attention

sns.set(style="darkgrid", font_scale=1.2)


def get_prediction_trend(model, dataset: TimeSeriesDataset):
    predictions = model.predict(dataset.X)
    predictions = dataset.scaler.inverse_transform(predictions.reshape(-1, 1))

    return predictions


def visualize_results(index, y_true, y_pred, title, file_name):
    plt.figure(figsize=(12, 9))
    plt.plot(index, y_true, color="red", label="Real Stock Price")
    plt.plot(index, y_pred, color="blue", label="Predicted Stock Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.text(
        0.5,
        -0.1,
        f"The mean absolute error is {round(mean_absolute_error(y_true, y_pred), 3)}USD",
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )

    file_path = f"Result/{file_name}.png"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)


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
    plt.figure(figsize=(15, 9))
    sns.barplot(
        x=np.arange(1, attention_weights_mean.shape[0] + 1),
        y=attention_weights_mean[:, 0],
    )
    plt.title("Importance of Time Steps")
    plt.xlabel("Time Step")
    plt.ylabel("Attention Weight")

    file_path = "Result/attention_weights.png"
    if os.path.exists(file_path):
        os.remove(file_path)
    plt.savefig(file_path)
