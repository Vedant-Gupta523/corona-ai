# Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Settings
batch_size = 30
future_prediction_size = 30

# Get data from csv file
dataset = pd.read_csv("test_data.csv")

# Create scaled/unscaled datasets, divide into train and test data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_dataset = []
unscaled_dataset = []
for crash in list(dataset)[1:]:
    data = dataset.filter([crash])
    scaled_dataset.append(scaler.fit_transform((data.values)))
    unscaled_dataset.append(data.values)
for i in range(len(scaled_dataset)):
    scaled_dataset[i] = np.reshape(list(filter(lambda x: x==x, scaled_dataset[i])), (len(list(filter(lambda x: x==x, scaled_dataset[i]))), 1))
    unscaled_dataset[i] = np.reshape(list(filter(lambda x: x==x, unscaled_dataset[i])), (len(list(filter(lambda x: x==x, unscaled_dataset[i]))), 1))
train_data = scaled_dataset[:-1]
test_data = scaled_dataset[-1]
unscaled_test_data = unscaled_dataset[-1]

# Split data
x_train = []
x_test = []
y_train = []
y_unscaled_test = []
y_scaled_test = []

for crash in train_data:
    for i in range(batch_size, len(crash)):
        x_train.append(crash[i-batch_size:i, 0])
        y_train.append(crash[i, 0])

for i in range(batch_size, len(test_data)):
    x_test.append(test_data[i-batch_size:i, 0])
    y_unscaled_test.append(unscaled_test_data[i, 0])
    y_scaled_test.append(test_data[i, 0])
    
   
# Fitting SVR to the dataset
regressor = SVR(kernel = "rbf")
regressor.fit(x_train, y_train)

# Making predictions
y_pred = []
for test_case in x_test:
    y_pred.append(regressor.predict([test_case]))
y_pred = scaler.inverse_transform(y_pred)

# Future Predictions
x_future_test = x_test[-1][1:]
x_future_test = [np.append(x_future_test, y_scaled_test[-1])]
future_preds = []
for i in range(future_prediction_size):
    future_preds.append(regressor.predict([x_future_test[i]]))
    x_future_test.append(np.append(x_future_test[i][1:], future_preds[i]))
future_preds = scaler.inverse_transform(future_preds)
    

# Graphing predictions
plt.title("COVID-19 Crash Analysis")
plt.xlabel("Days from Crash")
plt.ylabel("S&P 500")
plt.plot([x for x in range(-99, len(y_pred) - 99)], y_pred, color = "orange")
plt.plot([x for x in range(-99, len(y_pred) - 99)], y_unscaled_test, linewidth=1)
plt.plot([x for x in range(len(y_pred) - 99, len(y_pred) - 99 + future_prediction_size)] , future_preds, color = "red")
plt.show()