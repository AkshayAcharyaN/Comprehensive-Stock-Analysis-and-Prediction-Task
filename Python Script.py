# Data Validation and Cleaning

# Importing libraries  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the input CSV file
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data

file_path = 'C:\\Users\\abhic\\Desktop\\Assesement\\NSEI.csv'
data = read_csv(file_path)


# Validate data structure and format
def validate_data(data):
    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in expected_columns:
        if col not in data.columns:
            raise ValueError(f"Missing expected column: {col}")

    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = validate_data(data)

 

# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# Yes Data contain 14 Null values in "Open", "High", "Low", "Close", "Adj Close", "Volumne" 

# Imputing  missing values using Simple Imputer
from sklearn.impute import SimpleImputer
num_cols = data.select_dtypes(include=['float64', 'int64']).columns
imputer_mean = SimpleImputer(strategy='mean')
data[num_cols] = imputer_mean.fit_transform(data[num_cols])

# again we are checking for Null values 
print("Missing values per column:\n", data.isnull().sum())



# Visualize outliers using boxplots
for col in num_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=data[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
    

# Detecting using IQR Method
def detect_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data[column] < lower_bound) | (data[column] > upper_bound)]

# Detecting outliers in the 'Volume' column
outliers = detect_outliers(data, 'Volume')
if not outliers.empty:
    print(f"Outliers detected in column Volume:\n", outliers)

# Yes data contain Outliers in Volumne column. 

# Treating the Outliers using Winsorizer.
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', tail='both', fold=0.05)

data['Volume'] = winsor.fit_transform(data[['Volume']])

# again check with the outliers
plt.figure(figsize=(10, 5))
sns.boxplot(x=data["Volume"])
plt.title(f'Boxplot of {col}')
plt.show()

# 2. Stock Analysis:

# Calculate basic statistical measures
stats = data.describe()   # .describe() can give mean, median, mode, std Dev, Max and Min in single variable. 
print("Basic Statistical Measures:\n", stats)

# Visualization of Historical trend of Open Price
plt.figure(figsize=(15, 10))
plt.plot(data['Date'], data['Open'], label='Open Price')
plt.xlabel('Date')
plt.ylabel('Open Price')
plt.title('Historical Open Price')
plt.legend()
plt.show()

# Visualization of Historical trend of Close Price
plt.figure(figsize=(15, 10))
plt.plot(data['Date'], data['Close'], label='Close Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Historical Close Price')
plt.legend()
plt.show()

#Visualization of correlation between CLose and Open
plt.figure(figsize=(10, 6))
plt.scatter(data['Open'], data["Close"], color='blue', alpha=0.5)
plt.title('Scatter Plot of Open and Close Dates')
plt.xlabel('Open Dates')
plt.ylabel('Close Dates')
plt.grid(True)
plt.show()

# Prepare data for candlestick plot
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

df_candlestick = data[['Date', 'Open', 'High', 'Low', 'Close']]
df_candlestick['Date'] = mdates.date2num(df_candlestick['Date'].dt.to_pydatetime())

# Candlestick plot
fig, ax = plt.subplots(figsize=(25, 6))
candlestick_ohlc(ax, df_candlestick.values, width=0.6, colorup='g', colordown='r', alpha=0.8)
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y'))
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Candlestick Chart')
plt.show()

# Pairplot to see relationships and distributions
sns.pairplot(data)
plt.show()

# Heatmap for correlation
plt.figure(figsize=(10, 6))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()



# Model Buliding 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose


data.sort_values(by="Date", inplace=True)  # Ensure chronological order

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Determine if there are any seasonal patterns or cyclical trends in the stock's behavior
seasonal_decomposition = seasonal_decompose(data['Close'], model='multiplicative', period=30)
seasonal_decomposition.plot()
plt.show()

# Apply time series decomposition techniques to isolate seasonal, trend, and residual components
trend = seasonal_decomposition.trend
seasonal = seasonal_decomposition.seasonal
residual = seasonal_decomposition.resid

# Train the model using historical data
X_train = np.array(train_data.index).reshape(-1, 1)
y_train = train_data['Close']

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model's performance using appropriate metrics
X_test = np.array(test_data.index).reshape(-1, 1)
y_test = test_data['Close']
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f'RMSE: {rmse}, MAE: {mae}')

# Generate predictions for future stock prices based on the trained model
future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=30)
future_indices = np.array(range(len(data), len(data) + 30)).reshape(-1, 1)
future_predictions = model.predict(future_indices)

future_dates_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
print(future_dates_df)



# Fit the ARIMA model
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
# Assuming 'data' is the DataFrame containing your data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Split the data into train and test sets
train_size = int(len(data) * 0.9)
train, test = data[:train_size], data[train_size:]

# Build the ARIMA model
model = ARIMA(train['Close'], order=(5, 1, 0))
model_fit = model.fit()

# Make predictions
predictions = model_fit.predict(start=len(train), end=len(train)+len(test)-1, typ='levels')

# Evaluate the model
mse = mean_squared_error(test['Close'], predictions)
print(mse)
rmse = mse ** 0.5
print(f'RMSE: {rmse}')

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Close'], label='Train')
plt.plot(test.index, test['Close'], label='Test')
plt.plot(test.index, predictions, label='Predictions')
plt.legend()
plt.show()



# LSTM Model 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the dataset
# Assuming 'data' is the DataFrame containing your data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
dataset = data['Close'].values.reshape(-1, 1)

# Normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create sequences for the LSTM model
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Define the sequence length
X, y = create_sequences(scaled_data, seq_length)

# Split the data into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32)

# Make predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(data['Close'].values[train_size+seq_length:], predictions))
print(f'RMSE: {rmse}')

# Plot predictions
plt.figure(figsize=(12, 6))
plt.plot(data.index[train_size+seq_length:], data['Close'].values[train_size+seq_length:], label='Actual')
plt.plot(data.index[train_size+seq_length:], predictions, label='Predicted')
plt.legend()
plt.show()


# Deployment and Monitoring:

#    1. Deploy the model as a web service using a framework such as Flask or Django.
#    2. Use Docker to containerize the application for consistent deployment across environments.
#    3. Implement a CI/CD pipeline for automated testing and deployment.
#    4. Use cloud services (e.g., AWS, GCP, Azure) to host the application.
#    5. Set up logging and monitoring tools (e.g., Prometheus, Grafana) to track model performance and usage.

#   1. Monitor model predictions against actual stock prices to assess accuracy.
#   2. Track key performance metrics (e.g., RMSE, MAE) over time.
#   3. Set up alerts for performance degradation or anomalies in predictions.
#   4. Implement feedback loops to retrain the model with new data periodically.


