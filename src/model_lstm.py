"""LSTM architecture for 24-hour solar generation sequences."""
import tensorflow as tf


def build_lstm(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="hourly_features"),
        tf.keras.layers.LSTM(64, dropout=0.15, name="lstm_64"),
        tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
        tf.keras.layers.Dropout(0.10, name="dropout"),
        tf.keras.layers.Dense(1, name="next_hour_power"),
    ], name="lstm_regressor")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model
