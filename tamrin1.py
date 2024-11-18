# -*- coding: utf-8 -*-
"""

@author: mhmd
"""


import pandas
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# q=pandas.read_excel("DataFake.csv")

q=datasets.load_iris()

x=q.data

y=q.target

print(x.shape)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)

print(x_train.shape)


print(y_train.shape)


knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

y_new=knn.predict(x_test)

print(y_new)

print(y_test)

s=knn.score(x_test, y_test)
print(s)

print(x_test.shape)

print(y_test.shape)


print(x_train.shape)

print(y_train.shape)

s1=knn.score(x_test, y_test)
print(s1)

