# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 16:42:19 2018

@author: hasee
"""

from sklearn import tree
#光滑是1，不光滑是0
features = [[140,1],[130,1],[150,0],[170,0]]
#苹果是0，橘子是1
labels = [0, 0, 1, 1]
#创建分类器
clf = tree.DecisionTreeClassifier()
clf =  clf.fit(features, labels)
print(clf.predict([[150,0]]))
