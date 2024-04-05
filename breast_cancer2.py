#Machine Learning Model(Multiple Linear Regression) for Multiple Features available in Breast Cancer Datasets


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# Importing Dataset
breast_cancer = datasets.load_breast_cancer()
# Checking what's inside the Dataset
print(breast_cancer.keys())

# For Multiple features available in the Dataset
breast_cancer_X = breast_cancer.data

# Extracting all elements except the last 30 for training
breast_cancer_X_train = breast_cancer_X[:-30]
# Extracting only the last 30 elements for testing
breast_cancer_X_test = breast_cancer_X[-30:]
# Selecting the target values corresponding to the training instances
breast_cancer_Y_train = breast_cancer.target[:-30]
# Used as the target values for the testing set
breast_cancer_Y_test = breast_cancer.target[-30:]

# Creating a Linear Model and importing LinearRegression
model = linear_model.LinearRegression()

# Drawing a multi-dimensional Line from the data which will be saved in the Model
model.fit(breast_cancer_X_train, breast_cancer_Y_train)

# Predicting the values from the Created Model, whenever Features will be given
breast_cancer_Y_predicted = model.predict(breast_cancer_X_test)

print("Mean Squared Error (Avg. of Sum of Squared Error): ", mean_squared_error(breast_cancer_Y_test, breast_cancer_Y_predicted))
print("Weights (m): ", model.coef_)
print("Intercept (b): ", model.intercept_)
