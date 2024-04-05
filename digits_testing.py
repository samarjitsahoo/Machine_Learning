'''Machine Learning Model(Testing Linear Regression) for Single Feature in Digits Datasets
Given: X=1,2,3(Feature)
Given: Y=3,2,4(Label)'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
digits = datasets.load_digits()
# Checking what's inside the Dataset
print(digits.keys())

# For single feature available in the Dataset, taking X values
digits_X = np.array([[1], [2], [3]])
# Printing a Numpy array(Arrays of array) in a single column
print(digits_X)

digits_X_train = digits_X
digits_X_test = digits_X

# Selecting the target values corresponding to the training instances, taking Y values
digits_Y_train = np.array([3, 2, 4])
# Selecting the target values for the testing set, taking Y values
digits_Y_test = np.array([3, 2, 4])

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a Line from the data which will be saved in the Model
model.fit(digits_X_train, digits_Y_train)

# Predicting the values from the line in the Graph
digits_Y_predicted = model.predict(digits_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(digits_Y_test, digits_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)

# Scatter Points Plotting
plt.scatter(digits_X_test, digits_Y_test)
# Line Plotting
plt.plot(digits_X_test, digits_Y_predicted)
plt.show()
