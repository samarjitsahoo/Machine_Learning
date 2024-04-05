'''Machine Learning Model(Multiple Logistic Regression Classifier) for Multiple Feature in Iris Dataset to predict whether the flower is Iris Virginica or not!
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

#Taking Features
X=iris["data"]
print(X)
#Taking Labels as Booleans and converting it to 0 or 1 for False and True of iris Virginica Presence
Y=(iris["target"]==2).astype(np.int64)
print(Y)

#Train a Logistic Regression Classifier
clf=LogisticRegression()

#Fitting values of X and Y in the Classifier
clf.fit(X,Y)

#Taking an example whether it is Iris Virginica or not by Petal Length
example=clf.predict(([[2.6,2.3,2.5,3.1]]))
print(example)