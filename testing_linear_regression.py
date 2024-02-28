'''Machine Learning Model(Testing Linear Regression) for Single Feature in the Datasets
Given: X=1,2,3(Feature)
Given: Y=3,2,4(Label)'''

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error

#Importing Datasets
diabetes=datasets.load_diabetes()
#Checking what's inside the Dataset
print(diabetes.keys())

#for single feature available in the Datasets, taking X values
diabetes_X=np.array([[1],[2],[3]])
#Printing a Numpy array(Arrays of array) in a single column
print(diabetes_X)

diabetes_X_train=diabetes_X
diabetes_X_test=diabetes_X

# selecting the target values corresponding to the training instances, taking Y values
diabetes_Y_train=np.array([3,2,4])
# selecting the target values for the testing set, taking Y values
diabetes_Y_test=np.array([3,2,4])

#Creating a Linear Model and importing LinearRegression
model=linear_model.LinearRegression()

#Drawing a Line from the data which will be saved in the Model
model.fit(diabetes_X_train,diabetes_Y_train)

#Predicting the values from the line in the Graph
diabetes_Y_predicted=model.predict(diabetes_X_test)

print("Mean Squared Error(Avg. of Sum of Squared Error): ",mean_squared_error(diabetes_Y_test,diabetes_Y_predicted))
print("Weights(m): ",model.coef_)
print("Intercept(b): ",model.intercept_)

#Scatter Points Plotting
plt.scatter(diabetes_X_test,diabetes_Y_test)
#Line Plotting
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()