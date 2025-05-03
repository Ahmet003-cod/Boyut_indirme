# -*- coding: utf-8 -*-
"""
Created on Sat May  3 15:54:46 2025

@author: Huzur Bilgisayar
"""

from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib import pyplot as plt


mnist=fetch_openml("mnist_784",version=1)
X=mnist.data
y=mnist.target.astype(int)

LDA=LinearDiscriminantAnalysis(n_components=2)

X_lda=LDA.fit_transform(X, y)

plt.figure()
plt.scatter(X_lda[:,0],X_lda[:,1],c=y,cmap="tab10",alpha=0.6)
plt.title("LDA of mnist")
plt.xlabel("LD1")
plt.ylabel("LD2")
plt.colorbar(label="digits")
plt.show()