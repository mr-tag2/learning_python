# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:10:53 2024

@author: mhmd
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier

q=datasets.load_diabetes()
x=q.data
y=q.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.9)

knn1=KNeighborsRegressor()

knn1.fit(x_train,y_train)
y_new = knn1.predict(x_test)
s1=knn1.score(x_test,y_test)

print(s1)

knn2=KNeighborsClassifier()
knn2.fit(x_train,y_train) 
y_new = knn2.predict(x_test)
s2=knn2.score(x_test,y_test)

print(s2)


print(q)