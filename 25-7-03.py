# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn import datasets
import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#q=datasets.load_iris();
q=datasets.load_breast_cancer();

x=q.data
y=q.target

print(q)

print(x)

print(y)

x_train,  x_test ,y_train , y_test=train_test_split(x,y, test_size=0.1,stratify=y)

#x1=[6.22, 13.4 ,5.4 ,2.3]

#x1=numpy.array(x1)

knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

ynew=knn.predict(x_test)

s=knn.score(x_test, y_test)

print(ynew)
print("-----------------")
print(y_train)

print(s)