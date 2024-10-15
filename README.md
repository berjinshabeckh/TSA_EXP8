# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 15/10/24


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Step 1: Manually entering data
df = pd.read_csv('score.csv')

# Step 2: Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df['Scores'][:train_size], df['Scores'][train_size:]

# =================== Moving Average Model ===================

# Step 3: Fit a Moving Average (MA) model
# Using ARIMA with order (0,0,q) where q is the MA lag
ma_model = ARIMA(train, order=(0, 0, 1))  # MA(1) model
ma_model_fit = ma_model.fit()

# Step 4: Make predictions
ma_predictions = ma_model_fit.forecast(steps=len(test))

# Step 5: Evaluate MA model
ma_rmse = np.sqrt(mean_squared_error(test, ma_predictions))
print(f'Moving Average Model RMSE: {ma_rmse}')

# Step 6: Plot MA results
plt.figure(figsize=(10, 4))
plt.plot(test.values, label='Actual Scores')
plt.plot(ma_predictions, label='MA Predictions', linestyle='--')
plt.title('Moving Average Model')
plt.legend()
plt.show()

# =================== Exponential Smoothing Model ===================

# Step 7: Fit Exponential Smoothing model
es_model = ExponentialSmoothing(train, trend='add', seasonal=None, seasonal_periods=None)
es_model_fit = es_model.fit()

# Step 8: Make predictions using Exponential Smoothing
es_predictions = es_model_fit.forecast(steps=len(test))

# Step 9: Evaluate Exponential Smoothing model
es_rmse = np.sqrt(mean_squared_error(test, es_predictions))
print(f'Exponential Smoothing Model RMSE: {es_rmse}')

# Step 10: Plot Exponential Smoothing results
plt.figure(figsize=(10, 4))
plt.plot(test.values, label='Actual Scores')
plt.plot(es_predictions, label='ES Predictions', linestyle='--')
plt.title('Exponential Smoothing Model')
plt.legend()
plt.show()
```

### OUTPUT:
![download](https://github.com/user-attachments/assets/fb9a4533-fc81-4b1c-b37a-4e82cb3569d9)

![download](https://github.com/user-attachments/assets/d2dcb5a2-a655-44ca-aaf0-d215b13f9092)

![image](https://github.com/user-attachments/assets/9b066689-7380-4268-ab16-ec8d33a796ac)



### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
