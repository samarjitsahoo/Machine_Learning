#Machine Learning Model(Multiple Linear Regression) for Multiple Features available in Diabetes Datasets

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

#Importing Datasets
diabetes=datasets.load_diabetes()
#Checking what's inside the Dataset
print(diabetes.keys())

#for Multiple features available in the Dataset
diabetes_X=diabetes.data

# extracting all elements except the last 30 for training
diabetes_X_train=diabetes_X[:-30]
# extracting only the last 30 elements for testing
diabetes_X_test=diabetes_X[-30:]
# selecting the target values corresponding to the training instances
diabetes_Y_train=diabetes.target[:-30]
# used as the target values for the testing set
diabetes_Y_test=diabetes.target[-30:]

#Creating a Linear Model and importing LinearRegression
model=linear_model.LinearRegression()

#Drawing a multi-dimensional Line from the data which will be saved in the Model
model.fit(diabetes_X_train,diabetes_Y_train)

#Predicting the values from the Created Model, whenever Features will be given
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean Squared Error(Avg. of Sum of Squared Error): ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights(m): ",model.coef_)
print("Intercept(b): ",model.intercept_)