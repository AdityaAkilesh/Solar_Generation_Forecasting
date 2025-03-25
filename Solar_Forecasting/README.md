# Solar Power Generation Forecasting using LSTM Neural Networks

## 📌 Overview

This project builds a deep learning pipeline to forecast **hourly solar power generation** using multivariate time series data. It leverages **Long Short-Term Memory (LSTM)** networks to model temporal dependencies in the data. The pipeline includes:

- Data loading & preprocessing
- Sequence generation using a moving window
- LSTM model training & validation
- Forecast visualization & evaluation

---

## ⚙️ Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/<your-username>/solar_generation_forecasting.git
   cd solar-generation-forecasting/Solar_Forecating
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Technologies Used

- **Python 3.9+**
- **Pandas** & **NumPy** – Data manipulation
- **Matplotlib** & **Seaborn** – Data Visualization
- **Scikit-learn** – Preprocessing, metrics
- **TensorFlow/Keras** – LSTM model architecture
- **Joblib** – Model saving/loading

---

## 🧠 Methodology

This project implements a structured deep learning workflow specifically tailored for multivariate time series forecasting of hourly solar power generation. The core of the methodology is based on a **stacked Long Short-Term Memory (LSTM)** architecture implemented using the **TensorFlow Keras API**. 

The process begins with parsing and indexing timestamps from the raw CSV data using **Pandas**, followed by removing null entries and normalizing all numeric features with **MinMaxScaler** from **Scikit-learn**. Input sequences are engineered using a sliding window of the previous 48 hours (`n_past = 48`), reshaped into 3D tensors suitable for LSTM models: `(samples, time_steps, features)`.

The model is a deep sequential LSTM network with two hidden layers: one with 128 units returning sequences, followed by a second with 64 units. **Dropout layers** (rate = 0.1) are used after each LSTM to prevent overfitting. The final **Dense layer** outputs a single forecasted value representing solar generation for the next hour.

The model is compiled using the **Adam optimizer** and trained to minimize **Mean Squared Error (MSE)**. **EarlyStopping** is implemented to halt training when the validation loss stops improving. Performance is quantitatively evaluated on both training and unseen test datasets using **RMSE**, **MAE**, **SMAPE**, and **R² Score**, and qualitatively via time series plots generated using **Matplotlib**. Results, including predictions and error metrics, are saved for further analysis and reproducibility.

### 1. **Data Preprocessing**

- Parse timestamps & sort data
- Handle missing values
- Normalize values using `MinMaxScaler`

### 2. **Sequence Generation**

- Moving window approach with 48 time steps (2 days of hourly data)
- Input shape for LSTM: `(samples, timesteps, features)`

### 3. **Model Architecture**

- LSTM layer with 128 units → Dropout
- LSTM layer with 64 units → Dropout
- Dense output layer (1 neuron)
- Loss function: Mean Squared Error
- Optimizer: Adam

### 4. **Evaluation Metrics**

- RMSE
- MAE
- SMAPE
- R² Score

### 5. **Visualization**

- Training/validation loss
- Actual vs Predicted plots

---

## 🗂 Directory Structure

```
solar-generation-forecasting/
|
├── Data/
│   └── solar_gen_delaware.csv              # Input dataset
├── Solar_Forecating/
│   ├── solargen_multivariate_forecast_1hr.py   # LSTM training and forecasting script
│   └── requirements.txt                    # Project dependencies
└── README.md                               # Project documentation
```

---

## 📈 Results

The model achieved strong performance in forecasting unseen data. Sample plots and metrics are generated and saved for evaluation. You can visualize and export predictions via the script.

---



