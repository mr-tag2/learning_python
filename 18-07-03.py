# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:17:27 2024

@author: Mohammad
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy
import time

t1=time.time()

q=datasets.load_iris()

x=q.data
y=q.target

knn=KNeighborsClassifier(n_neighbors=3,metric='minkowski')
knn.fit(x, y)

x_new = [[6.1 ,2.6 ,5.6, 1.4]]
x_new=numpy.array(x_new)

y_new = knn.predict(x_new)

x_new2 = [[5.1 ,3.5 ,1.4, 0.2]]
x_new2=numpy.array(x_new2)

y_new2 = knn.predict(x_new2)


x_new3 = [[15.1 ,3.5 ,1.4, 0.2]]
x_new3=numpy.array(x_new3)

y_new3 = knn.predict(x_new3)


print(q)

print("-------------------------------------------")
print(x)

print("-------------------------------------------")
print(q.target)

print("-------------------------------------------")
print(x[0],y[0])

print("-------------------------------------------")
print(knn)

print("-------------------------------------------")
print(y_new)

print("-------------------------------------------")
print(y_new2)

print("-------------------------------------------")
print(y_new3)

t2=time.time()

print("-------------------------------------------")
print(t2-t1)


