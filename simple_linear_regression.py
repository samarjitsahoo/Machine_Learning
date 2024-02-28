#Machine Learning Model(Simple Linear Regression) for Single Feature in the Datasets

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

#Importing Datasets
diabetes=datasets.load_diabetes()
#Checking what's inside the Dataset
print(diabetes.keys())

#for single feature available in the Datasets
diabetes_X=diabetes.data[:,np.newaxis,2]
#Printing a Numpy array(Arrays of array) in a single column
print(diabetes_X)

# extracting all elements except the last 30 for training
diabetes_X_train=diabetes_X[:-30]
# extracting only the last 30 elements for testing
diabetes_X_test=diabetes_X[-30:]
# selecting the target values corresponding to the training instances
diabetes_Y_train=diabetes.target[:-30]
# selecting the target values for the testing set
diabetes_Y_test=diabetes.target[-30:]

#Creating a Linear Model and importing LinearRegression
model=linear_model.LinearRegression()

#Drawing a Line from the data which will be saved in the Model
model.fit(diabetes_X_train,diabetes_Y_train)

#Predicting the values from the line in the Graph, whenever Features will be given
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean Squared Error(Avg. of Sum of Squared Error): ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights(m): ",model.coef_)
print("Intercept(b): ",model.intercept_)

#Scatter Points Plotting
plt.scatter(diabetes_X_test,diabetes_Y_test)
#Line Plotting
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()