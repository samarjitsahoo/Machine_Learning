#Machine Learning Model(Multiple Linear Regression) for Multiple Features available in Wine Datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
wine = datasets.load_wine()
# Checking what's inside the Dataset
print(wine.keys())

# For Multiple features available in the Dataset
wine_X = wine.data

# Extracting all elements except the last 30 for training
wine_X_train = wine_X[:-30]
# Extracting only the last 30 elements for testing
wine_X_test = wine_X[-30:]
# Selecting the target values corresponding to the training instances
wine_Y_train = wine.target[:-30]
# Used as the target values for the testing set
wine_Y_test = wine.target[-30:]

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a multi-dimensional Line from the data which will be saved in the Model
model.fit(wine_X_train, wine_Y_train)

# Predicting the values from the Created Model, whenever Features will be given
wine_Y_predicted = model.predict(wine_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(wine_Y_test, wine_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)
