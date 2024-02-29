'''Machine Learning Model(K-Nearest-Neighbors) for Flower type Prediction
[0] for Iris-Setosa
[1] for Iris-Versicolor
[2] for Iris-Virginica'''

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading the Datasets
iris=datasets.load_iris()
#Spectating What's inside the Dataset
print(iris.keys())

#Loading Features and Labels of Dataset
features=iris.data
labels=iris.target

#Training the Classifier
clf=KNeighborsClassifier()

#Fitting all the loaded datas in the Classifier
clf.fit(features,labels)

#User Input
while True:
    try:
        a=float(input("Enter Sepal length in cm: "))
        b=float(input("Enter Sepal width in cm: "))
        c=float(input("Enter Petal length in cm: "))
        d=float(input("Enter Petal width in cm: "))
        break
    except ValueError:
        print("Invalid input. Please enter numerical values.")

#Predicting the Flower type
preds=clf.predict([[a,b,c,d]])
print(preds)