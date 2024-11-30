# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:10:17 2024

@author: mhmd
"""

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import adjusted_rand_score


x,y=load_iris(return_X_y=True)

print(x,y)

km=KMeans(n_clusters=3)
km.fit(x)
y_new=km.predict(x)
cc=km.cluster_centers_


xplot=x[:,0]

yplot=x[:,3]

plt.scatter(xplot, yplot,c=y,marker='*')
#plt.scatter(xplot, yplot,c=y,s=120,marker='*')

print(xplot,yplot)

print(cc)

print(y_new)

#plt.savefig('a.png')

score_1=adjusted_rand_score(y,y_new)

print(score_1)