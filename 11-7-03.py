# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:04:48 2024

@author: Mohammad
"""

import pandas
p=pandas.read_excel("11-7-03.xlsx")
print(p)

print("===============================================")

from sklearn import datasets

q=datasets.load_iris()

print(q)
print("-------------------------------------------")

print(q.data)
print("-------------------------------------------")

print(q.target)
print("-------------------------------------------")

print(q.DESCR)


print("===============================================")

q1=datasets.load_diabetes()

print(q1)
print("-------------------------------------------")

print(q1.data)
print("-------------------------------------------")

print(q1.target)
print("-------------------------------------------")

print(q1.DESCR)



print("===============================================")

q2=datasets.load_breast_cancer()

print(q2)
print("-------------------------------------------")

print(q2.data)
print("-------------------------------------------")

print(q2.data[1])
print("-------------------------------------------")

print(q2.target)
print("-------------------------------------------")

print(q2.DESCR)

















