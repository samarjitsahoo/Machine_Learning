#Machine Learning Model(Simple Linear Regression) for Single Feature in Iris Datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Datasets
iris = datasets.load_iris()
# Checking what's inside the Dataset
print(iris.keys())

# For single feature available in the Datasets
iris_X = iris.data[:, np.newaxis, 2]
# Printing a Numpy array(Arrays of array) in a single column
print(iris_X)

# Extracting all elements except the last 30 for training
iris_X_train = iris_X[:-30]
# Extracting only the last 30 elements for testing
iris_X_test = iris_X[-30:]
# Selecting the target values corresponding to the training instances
iris_Y_train = iris.target[:-30]
# Selecting the target values for the testing set
iris_Y_test = iris.target[-30:]

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a Line from the data which will be saved in the Model
model.fit(iris_X_train, iris_Y_train)

# Predicting the values from the line in the Graph, whenever Features will be given
iris_Y_predicted = model.predict(iris_X_test)

print("Mean Squared Error(Avg. of Sum of Squared Error): ", mean_squared_error(iris_Y_test, iris_Y_predicted))
print("Weights(m): ", model.coef_)
print("Intercept(b): ", model.intercept_)

# Scatter Points Plotting
plt.scatter(iris_X_test, iris_Y_test)
# Line Plotting
plt.plot(iris_X_test, iris_Y_predicted)
plt.show()
