#Machine Learning Model(Simple Linear Regression) for Single Feature in Breast Cancer Dataset

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

#Importing Datasets
breast_cancer=datasets.load_breast_cancer()
#Checking what's inside the Dataset
print(breast_cancer.keys())

#for single feature available in the Datasets
breast_cancer_X=breast_cancer.data[:,np.newaxis,2]
#Printing a Numpy array(Arrays of array) in a single column
print(breast_cancer_X)

# extracting all elements except the last 30 for training
breast_cancer_X_train=breast_cancer_X[:-30]
# extracting only the last 30 elements for testing
breast_cancer_X_test=breast_cancer_X[-30:]
# selecting the target values corresponding to the training instances
breast_cancer_Y_train=breast_cancer.target[:-30]
# selecting the target values for the testing set
breast_cancer_Y_test=breast_cancer.target[-30:]

#Creating a Linear Model and importing LinearRegression
model=linear_model.LinearRegression()

#Drawing a Line from the data which will be saved in the Model
model.fit(breast_cancer_X_train,breast_cancer_Y_train)

#Predicting the values from the line in the Graph, whenever Features will be given
breast_cancer_Y_predicted=model.predict(breast_cancer_X_test)

print("Mean Squared Error(Avg. of Sum of Squared Error): ",mean_squared_error(breast_cancer_Y_test,breast_cancer_Y_predicted))
print("Weights(m): ",model.coef_)
print("Intercept(b): ",model.intercept_)

#Scatter Points Plotting
plt.scatter(breast_cancer_X_test,breast_cancer_Y_test)
#Line Plotting
plt.plot(breast_cancer_X_test,breast_cancer_Y_predicted)
plt.show()