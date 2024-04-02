import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generating synthetic time series data
def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  # Wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # Wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # Noise
    return series[..., np.newaxis].astype(np.float32)

# Prepare data
n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
X_valid_scaled = scaler.transform(X_valid.reshape(-1, 1)).reshape(X_valid.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# LSTM expects data to be in [samples, time steps, features] format
n_features = 1

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(n_steps, n_features)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, validation_data=(X_valid_scaled, y_valid))

# Evaluate the model
mse = model.evaluate(X_test_scaled, y_test)
print(f'Test MSE: {mse}')

# Make predictions
predictions = model.predict(X_test_scaled)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(y_test[:, 0], label='Actual')
plt.plot(predictions[:, 0], label='Predicted')
plt.title('LSTM Time Series Prediction')
plt.legend()
plt.show()
