# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from sklearn import datasets
import numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

q=datasets.load_breast_cancer()
x=q.data
y=q.target


# pGird={'n_neighbors':numpy.arange(1,21),'metric':numpy.array('minkowski',"hamming","cosine","chepyshev")}
pGird={'n_neighbors':numpy.arange(1,50),'metric':numpy.array(['correlation',"hamming","cosine"])}

x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.5, random_state=42, stratify=y)



# kNum=KNeighborsClassifier(n_neighbors=6, metric='minkowski')
# kNum.fit(x_train,y_train)
# y_predict=kNum.predict(x_test)
# k_Evaluate=kNum.score(x_test,y_test)
# print("Accuracy:", k_Evaluate)

# for i in range(1,11):
#  kNum=KNeighborsClassifier(n_neighbors=i, metric='minkowski')
#  kNum.fit(x_train,y_train)
#  y_predict=kNum.predict(x_test)
#  k_Evaluate=kNum.score(x_test,y_test)
#  print("Accuracy:",i, k_Evaluate)

kNum=KNeighborsClassifier()
gsc=GridSearchCV(kNum, pGird,cv=5)
gsc.fit(x,y)
p1=gsc.best_score_
p2=gsc.best_params_
p3=gsc.best_estimator_
p4=gsc.best_index_

print("best score : ",p1)
print("best params : ",p2)
print("best estimator : ",p3)
print("best index : ",p4)

# print(pGird)
