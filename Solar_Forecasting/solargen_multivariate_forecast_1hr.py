'''
Solar Power Generation Forecasting using LSTM Neural Networks
------------------------------------------------------------
This script builds a deep learning pipeline to forecast hourly solar power generation
using multivariate time series data and LSTM networks. It includes data
loading, preprocessing, sequence creation (Moving Window), model training, evaluation, and result visualization.
'''
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import mean_squared_error
from keras.callbacks import EarlyStopping
from tensorflow.keras.layers import ConvLSTM2D

# %%
'''
Load and preprocess data
- Read CSV data
- Parse datetime
- Set datetime as index
- Sort and clean the data
'''
def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['datetime_beginning_utc'])
    df.set_index('datetime_beginning_utc', inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df.sort_index(inplace=True)
    # df.drop(columns=['datetime_beginning_ept', 'area'], inplace=True)

    return df
file_path = 'E:/Projects/Time_Series_Forecasting/Data/solar_gen_delaware.csv'
df = load_data(file_path)
print(df.head())

print(df.index.isna().sum())

#
df = df[~df.index.isna()]


# %%
'''
Visualize solar generation time series
'''
# Plot the specific column
plt.figure(figsize=(10, 5))
plt.plot(df['solar_generation_mw'])
plt.title('Solar Generation Data')
plt.xlabel('Time')
plt.ylabel('Solar Generation (MW)')
plt.grid(True)
plt.show()


# %%
'''
Split data into training and testing based on date
'''
train_start_date = '2022-01-01'
train_end_date = '2022-10-31'
test_start_date = '2022-11-01'
test_end_date = '2022-12-31'

train_df = df.loc[train_start_date:train_end_date]
test_df = df.loc[test_start_date:test_end_date]

print("\nTraining Data Shape:", train_df.shape)
print("\nTesting Data Shape:", test_df.shape)


# %%
'''
Scale the data using MinMaxScaler
'''
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_df)
test_scaled = scaler.transform(test_df)

import joblib
joblib.dump(scaler, 'solargen_multivariate_forecast_1hr_scaler.pkl')

# %%
'''
Create input-output sequences for LSTM
- n_past: number of past time steps used as input
'''
def create_sequences(data, n_past):
    X, y = [], []
    for i in range(n_past, len(data)):
        X.append(data[i - n_past:i, 0:data.shape[1]])
        y.append(data[i,0])
    return np.array(X), np.array(y)


# %%
n_past = 48
# Create sequences for training data
X_train, y_train = create_sequences(train_scaled, n_past)
# Create sequences for testing data
X_test, y_test = create_sequences(test_scaled, n_past)

print("\nShapes of Training Data:")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("\nShapes of Testing Data:")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

# %%
# Define the LSTM model
model = Sequential()
model.add(LSTM(128, activation='relu', 
               return_sequences=True, 
               input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.1))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(Dropout(0.1))
model.add(Dense(1)) # Adjust the number of output neurons to match the number of future hours you want to predict
model.compile(optimizer='adam', loss='mse')
model.summary()

# %%
# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
# Fit the model with the specified parameters
history = model.fit(X_train, y_train, 
                    epochs=50, batch_size=64, 
                    validation_split=0.1, verbose=1)

# Save the model
model.save('solargen_multivariate_forecast_1hr_model.keras')

# %%
# Plot training & validation loss values
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.show()


# %%
# Make predictions
train_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

print("\nShapes of Predictions:")
print("train_prediction shape:", train_prediction.shape)
print("test_prediction shape:", test_prediction.shape)

# %%
# test_prediction_copies = np.repeat(test_prediction, test_df.shape[1], axis=-1)
# print(test_prediction_copies.shape)

# test_prediction_inv = scaler.inverse_transform(np.reshape(test_prediction_copies, (len(test_prediction_copies), test_df.shape[1])))[:,0]

# y_test_inv = np.repeat(y_test, test_df.shape[1], axis=-1)
# y_test_inv = scaler.inverse_transform(np.reshape(y_test_inv, (len(y_test), test_df.shape[1])))[:,0]

train_prediction_inv = scaler.inverse_transform(np.concatenate((train_prediction, np.zeros((len(train_prediction), test_df.shape[1] - 1))), axis=1))[:,0]
test_prediction_inv = scaler.inverse_transform(np.concatenate((test_prediction, np.zeros((len(test_prediction), test_df.shape[1] - 1))), axis=1))[:,0]

y_train_inv = scaler.inverse_transform(np.concatenate((y_train.reshape(-1, 1), np.zeros((len(y_train), test_df.shape[1] - 1))), axis=1))[:,0]
y_test_inv = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((len(y_test), test_df.shape[1] - 1))), axis=1))[:,0]

print("\nShapes of Predictions:")
print("train_prediction_inv shape:", train_prediction_inv.shape)
print("test_prediction_inv shape:", test_prediction_inv.shape)
print("y_test_inv shape:", y_test_inv.shape)



# %%
# Calculate metrics
train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_prediction_inv))
test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_prediction_inv))

train_mae = np.mean(np.abs(y_train_inv - train_prediction_inv))
test_mae = np.mean(np.abs(y_test_inv - test_prediction_inv))

# train_mape = np.mean(np.abs((y_train_inv - train_prediction_inv) / y_train_inv)) * 100
# test_mape = np.mean(np.abs((y_test_inv - test_prediction_inv) / y_test_inv)) * 100

train_smape = np.mean(2.0 * np.abs(y_train_inv - train_prediction_inv) / (np.abs(y_train_inv) + np.abs(train_prediction_inv))) * 100
test_smape = np.mean(2.0 * np.abs(y_test_inv - test_prediction_inv) / (np.abs(y_test_inv) + np.abs(test_prediction_inv))) * 100

r2_score_train = 1 - (np.sum((y_train_inv - train_prediction_inv) ** 2) / np.sum((y_train_inv - np.mean(y_train_inv)) ** 2))
r2_score_test = 1 - (np.sum((y_test_inv - test_prediction_inv) ** 2) / np.sum((y_test_inv - np.mean(y_test_inv)) ** 2))

# Print metrics in tabular format
print("\nMetrics:")
print("-------------------------------------------------")
print("| Metric    | Train       | Test        |")
print("-------------------------------------------------")
print("| RMSE      | {:.2f}      | {:.2f}     |".format(train_rmse, test_rmse))
print("| MAE       | {:.2f}      | {:.2f}     |".format(train_mae, test_mae))
# print("| MAPE      | {:.2f}%     | {:.2f}%    |".format(train_mape, test_mape))
# print("| SMAPE     | {:.2f}%     | {:.2f}%    |".format(train_smape, test_smape))
print("| R2 Score  | {:.2f}      | {:.2f}     |".format(r2_score_train, r2_score_test))
print("-------------------------------------------------")


# %%
test_timestamps = test_df.index[n_past:]

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.plot(test_timestamps, y_test_inv, label='Actual')
plt.plot(test_timestamps,test_prediction_inv, label='Predicted')
plt.title('Actual vs Predicted')
plt.xlabel('Datetime')
plt.ylabel('PJME_MW')
plt.legend()
plt.show()

# %%
result_df = pd.DataFrame({'TimeStamp': test_timestamps, 
                          'Actual': y_test_inv, 
                          'Predicted': test_prediction_inv})

result_df.to_csv('solargen_multivariate_forecast_1hr_results.csv', index=False)

# %%
