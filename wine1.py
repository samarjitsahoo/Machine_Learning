#Machine Learning Model(Simple Linear Regression) for Single Feature in Wine Datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
wine = datasets.load_wine()

# Checking what's inside the Dataset
print(wine.keys())

# For single feature available in the Dataset
wine_X = wine.data[:, np.newaxis, 2]

# Printing a Numpy array(Arrays of array) in a single column
print(wine_X)

# Extracting all elements except the last 30 for training
wine_X_train = wine_X[:-30]

# Extracting only the last 30 elements for testing
wine_X_test = wine_X[-30:]

# Selecting the target values corresponding to the training instances
wine_Y_train = wine.target[:-30]

# Selecting the target values for the testing set
wine_Y_test = wine.target[-30:]

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a Line from the data which will be saved in the Model
model.fit(wine_X_train, wine_Y_train)

# Predicting the values from the line in the Graph, whenever Features will be given
wine_Y_predicted = model.predict(wine_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(wine_Y_test, wine_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)

# Scatter Points Plotting
plt.scatter(wine_X_test, wine_Y_test)

# Line Plotting
plt.plot(wine_X_test, wine_Y_predicted)
plt.show()
