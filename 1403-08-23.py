# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:13:55 2024

@author: mhmd
"""

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import numpy

q=datasets.load_breast_cancer()
x=q.data
y=q.target



knn=KNeighborsClassifier()
cvs=cross_val_score(knn, x, y, cv=5)
cvs1=cross_val_score(knn, x, y, cv=2)


print(x)

print('---------------------------------------')

print(cvs)
print(numpy.mean(cvs))
print('---------------------------------------')

print(cvs1)
print(numpy.mean(cvs1))


