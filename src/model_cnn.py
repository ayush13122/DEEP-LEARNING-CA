"""One-dimensional temporal CNN architecture for solar generation sequences."""
import tensorflow as tf


def build_cnn(input_shape: tuple[int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, name="hourly_features"),
        tf.keras.layers.Conv1D(32, 3, padding="causal", activation="relu", name="conv1d_32"),
        tf.keras.layers.Conv1D(64, 3, padding="causal", activation="relu", name="conv1d_64"),
        tf.keras.layers.GlobalAveragePooling1D(name="global_average_pool"),
        tf.keras.layers.Dense(32, activation="relu", name="dense_32"),
        tf.keras.layers.Dropout(0.15, name="dropout"),
        tf.keras.layers.Dense(1, name="next_hour_power"),
    ], name="cnn_1d_regressor")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse", metrics=["mae"])
    return model
