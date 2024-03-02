#Machine Learning Model(Multiple Linear Regression) for Multiple Features available in Digits Datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
digits = datasets.load_digits()
# Checking what's inside the Dataset
print(digits.keys())

# For Multiple features available in the Dataset
digits_X = digits.data

# Extracting all elements except the last 30 for training
digits_X_train = digits_X[:-30]
# Extracting only the last 30 elements for testing
digits_X_test = digits_X[-30:]
# Selecting the target values corresponding to the training instances
digits_Y_train = digits.target[:-30]
# Used as the target values for the testing set
digits_Y_test = digits.target[-30:]

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a multi-dimensional Line from the data which will be saved in the Model
model.fit(digits_X_train, digits_Y_train)

# Predicting the values from the Created Model, whenever Features will be given
digits_Y_predicted = model.predict(digits_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(digits_Y_test, digits_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)
