# Agritech-Project
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the data from CSV file into a DataFrame
data = pd.read_csv('/content/Guava.csv')  # Replace 'your_data.csv' with the path to your CSV file

# Extract the 'Modal Price (Rs./Quintal)' column
modal_price = data['Modal Price (Rs./Quintal)'].values.reshape(-1, 1)

# Scale the values using Min-Max scaling
scaler = MinMaxScaler()
scaled_modal_price = scaler.fit_transform(modal_price)

# Split data into training and testing sets
train_size = int(len(scaled_modal_price) * 0.8)
train_data, test_data = scaled_modal_price[:train_size], scaled_modal_price[train_size:]

# Prepare data for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # Number of past time steps to consider
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define and train the LSTM model with adjusted parameters
model = Sequential([
    LSTM(256, activation='relu', return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.3),
    LSTM(128, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Define callbacks
callbacks = [
    EarlyStopping(patience=15, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=10, min_lr=0.00001, verbose=1)
]

# Train the model with adjusted parameters
history = model.fit(X_train, y_train, epochs=70, batch_size=64, verbose=1, callbacks=callbacks, validation_split=0.1)

# Evaluate the model
mse = model.evaluate(X_test, y_test, verbose=0)
print("Mean Squared Error:", mse)

# Make predictions for future values
future_values = []
current_sequence = test_data[-seq_length:].reshape(1, seq_length, 1)
for i in range(len(test_data)):
    prediction = model.predict(current_sequence)[0]
    future_values.append(prediction)
    current_sequence = np.append(current_sequence[:,1:,:], prediction.reshape(1, 1, 1), axis=1)

# Inverse scaling to get actual predicted values
predicted_values = scaler.inverse_transform(np.array(future_values).reshape(-1, 1))

# Extract the 'Modal Price (Rs./Quintal)' column from the original data for comparison
actual_modal_price = data['Modal Price (Rs./Quintal)'].values[-len(predicted_values):].reshape(-1, 1)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_modal_price, predicted_values))
print("Root Mean Squared Error (Modal Price):", rmse)

# Calculate MAE
mae = mean_absolute_error(actual_modal_price, predicted_values)
print("Mean Absolute Error (Modal Price):", mae)
