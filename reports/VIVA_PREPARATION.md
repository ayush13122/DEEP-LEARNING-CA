# Viva preparation

## What is Deep Learning?

Deep Learning uses neural networks with multiple learned transformations. Here, the networks learn a numeric mapping from a 24-hour multivariate history to one next-hour AC-power value.

## Why TensorFlow Keras?

TensorFlow/Keras provides concise, reproducible APIs for LSTM and Conv1D models, training callbacks, saved-model loading and CPU-compatible deployment.

## What is an LSTM?

An LSTM is a recurrent neural-network layer with gated memory. It processes hourly observations in order and can retain useful information from earlier hours in the lookback window.

## What is a 1D CNN?

A one-dimensional convolutional neural network applies filters along the time axis. It detects short local sequences, such as irradiance changes or power ramps, across the 24 hourly records.

## How are LSTM and CNN different?

The LSTM uses recurrent memory across ordered steps; the 1D-CNN uses shared local filters and pooling. Both receive the identical 24×27 input tensor and are evaluated on identical timestamps.

## What is a time-series window?

It is a consecutive ordered block of observations. One sample in this project contains 24 hourly feature rows ending at time *t* and predicts `target_ac_power` at *t+1*.

## Why use 24 hours as the lookback?

It includes a complete daily solar cycle and recent operating history while keeping the models compact enough for CPU training. It is a documented design choice, not evidence that exactly 24 is universally optimal.

## Why is the prediction one hour ahead?

The supplied supervised data already define `target_timestamp = timestamp + 1 hour`. The project preserves that Model 2 problem instead of shifting the target to 24 hours ahead.

## What are target, features, labels and sequence shape?

Features are the 27 permissible current/past variables, including observed current AC power. The label/target is next-hour AC power. Each model input has shape `(samples, 24, 27)`.

## What are epochs, batch size, loss, backpropagation and Adam?

An epoch is one pass through training data. Batch size is the number of samples used per parameter update. MSE is the training loss. Backpropagation calculates gradients, and Adam updates model weights using adaptive learning rates.

## What does EarlyStopping do?

It monitors validation loss and ends training when validation performance stops improving for the configured patience. `restore_best_weights=True` returns the best validation checkpoint rather than the final epoch.

## What is data leakage?

Leakage is use of information unavailable at forecast time. The project excludes `target_ac_power`, `target_timestamp`, and raw future timestamps from inputs. It fits scalers only on training data and never crosses missing-hour gaps.

## Why chronological splits?

Random time-series splits can put future regimes into training. Chronological splits preserve the practical direction of forecasting and make the 2017–January 2018 test period an honest later-period evaluation.

## Why fit scalers on training data only?

Fitting on validation or test data lets their distribution influence training transformations. Training-only fit prevents that indirect information leak.

## What are MAE, RMSE and R²?

MAE is average absolute error in kW. RMSE is the square-root of mean squared error, in kW, and penalises larger mistakes more strongly. R² measures variance explained relative to predicting the test mean; it is not classification accuracy.

## How do actual results support the conclusion?

Use `outputs/model_metrics.csv` in the viva. Compare LSTM and 1D-CNN on the same final test timestamps and identify the lower MAE/RMSE only after reading the saved results. Compare persistence separately as a sanity check.

## What are the limitations?

The data are from one historical plant and end in January 2018. Missing observations reduce valid windows; nighttime zero generation affects aggregate metrics; weather/reanalysis inputs are historical, not future operational forecasts; and no cross-plant or live-deployment generalisation is established.
