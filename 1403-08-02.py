# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:16:31 2024

@author: mhmd
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


q=datasets.load_iris()

x=q.data

y=q.target

print(x.shape)

print("-----------------")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

print(x_train.shape)

print("-----------------")

print(y_train.shape)

print("-----------------")

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train, y_train)

y_new=knn.predict(x_test)

print(y_new)
print("-----------------")

print(y_test)
print("-----------------")

s=knn.score(x_test, y_test)
print(s)
print("-----------------")

print(x_test.shape)
print("-----------------")

print(y_test.shape)
print("-----------------")


print(x_train.shape)
print("-----------------")

print(y_train.shape)
print("-----------------")

s1=knn.score(x_test, y_test)
print(s1)
print("-----------------")

