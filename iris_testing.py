'''Machine Learning Model(Testing Linear Regression) for Single Feature in Iris Datasets
Given: X=1,2,3(Feature)
Given: Y=3,2,4(Label)'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
iris = datasets.load_iris()
# Checking what's inside the Dataset
print(iris.keys())

# For single feature available in the Dataset, taking X values
iris_X = np.array([[1], [2], [3]])
# Printing a Numpy array(Arrays of array) in a single column
print(iris_X)

iris_X_train = iris_X
iris_X_test = iris_X

# Selecting the target values corresponding to the training instances, taking Y values
iris_Y_train = np.array([3, 2, 4])
# Selecting the target values for the testing set, taking Y values
iris_Y_test = np.array([3, 2, 4])

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a Line from the data which will be saved in the Model
model.fit(iris_X_train, iris_Y_train)

# Predicting the values from the line in the Graph
iris_Y_predicted = model.predict(iris_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(iris_Y_test, iris_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)

# Scatter Points Plotting
plt.scatter(iris_X_test, iris_Y_test)
# Line Plotting
plt.plot(iris_X_test, iris_Y_predicted)
plt.show()
