# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 10:24:43 2024

@author: mhmd
"""

import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier,RadiusNeighborsRegressor

q=pandas.read_csv("DataFake.csv")
q=pandas.DataFrame(q)

print(q)
print("------------------------------------------------")

y=q.loc[:]['27']
x=q.drop('27',axis=1)
print(x)
print(y)

print("------------------------------------------------")

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)



knn=KNeighborsRegressor(n_neighbors=3)
knn.fit(x_train, y_train)
y_new=knn.predict(x_test)
s=knn.score(x_test, y_test)
s1=knn.score(x_test, y_test)

print(y_new)
print(s1)

print("------------------------------------------------")

knn1=KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train, y_train)
y_new1=knn1.predict(x_test)
s1=knn1.score(x_test, y_test)
s11=knn1.score(x_test, y_test)

print(y_new1)
print(s11)

print("------------------------------------------------")


# knn5=RadiusNeighborsRegressor(radius=100000000)
# knn5.fit(x_train, y_train)
# y_new5=knn5.predict(x_test)
# s5=knn5.score(x_test, y_test)
# s15=knn5.score(x_test, y_test)

# print(y_new5)
# print(s15)

# print("------------------------------------------------")

# knn18=RadiusNeighborsClassifier(radius=100000000)
# knn18.fit(x_train, y_train)
# y_new18=knn18.predict(x_test)
# s18=knn18.score(x_test, y_test)
# s118=knn18.score(x_test, y_test)

# print(y_new18)
# print(s118)

# print("------------------------------------------------")

