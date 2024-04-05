'''Machine Learning Model(Simple Logistic Regression Classifier) for Single Feature in Iris Dataset
to predict whether the flower is Iris Virginica or not!
Attributes:-
1)sepal length in cm
2)sepal width in cm
3)petal length in cm
4)petal width in cm
Class:-
[0]-Iris-Setosa
[1]-Iris-Versicolour
[2]-Iris-Virginica'''

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt 
iris=datasets.load_iris()
# print(iris.keys())
# print(iris['data'])
# print(iris['data'].shape)
# print(iris['target'])
# print(iris['DESCR'])

#Taking Feature of Third column(Petal Length) only by Numpy Array Slicing
X=iris["data"][:,3:]
print(X)
#Taking Labels as Booleans and converting it to 0 0r 1 for False and True of Iris Virginica Presence
Y=(iris["target"]==2).astype(np.int64)
print(Y)

#Train a Logistic Regression Classifier
clf=LogisticRegression()

#Fitting values of X and Y in the Classifier
clf.fit(X,Y)

#Taking an example whether it is Iris Virginica or not by Petal Length
example=clf.predict(([[2.6]]))
print(example)

#Using Matplotlib to plot the visualization
#Linspace gives 1000 points between 0 and 3
#Reshape changes shape to 1D array
X_new=np.linspace(0,3,1000).reshape(-1,1)
print(X_new)

#Predict_proba predicts the probability(Actual value of Probability)
Y_prob=clf.predict_proba(X_new)
print(Y_prob)
plt.plot(X_new,Y_prob[:,1],"g-",label="virginica")
plt.show()