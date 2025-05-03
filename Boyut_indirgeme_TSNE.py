# -*- coding: utf-8 -*-
"""
Created on Sat May  3 19:30:25 2025
@author: Huzur Bilgisayar
"""
from sklearn.datasets import fetch_openml
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
mnist=fetch_openml("mnist_784",version=1)

X,y=mnist.data,mnist.target
tsne=TSNE(n_components=2)
X_tsne=tsne.fit_transform(X)

plt.figure()

plt.scatter(X_tsne[:,0],X_tsne[:,1],c=y,cmap="tab10",alpha=0.8)
plt.title("TSNE Of MNIST Datasets")
plt.xlabel("T-SNE dimension 1")
plt.ylabel("T-SNE dimension 2")
