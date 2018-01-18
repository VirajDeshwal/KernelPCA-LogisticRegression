#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 01:15:54 2018

@author: virajdeshwal
"""
'''Feature Extraction technique is only for Linearly seperable dataset'''
'''We will be using KernelPCA on Non-linear dataset and model like LogisticRegression'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = pd.read_csv('Social_Network_Ads.csv')
#we are including the two index from our dataset and finding the corelation between them.

X = file.iloc[:,[2,3]].values
y= file.iloc[:,4].values


from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


#we need to do the feature scaling to get the accurate prediction.

from sklearn.preprocessing import StandardScaler

scaling = StandardScaler()

x_train = scaling.fit_transform(x_train)
x_test = scaling.fit_transform(x_test)

'''We have to apply KernelPCA just before applying LogisticRegression to the give Dataset.'''
#Applying KernelPCA
from sklearn.decomposition import KernelPCA

'''we have to define the no. of Principal Components for most variance. 
And check which components explain the most variance in the given dataset.
We want 2 independent variables but for now to check which are best. We will enter None.
And later replace it with the no. of top components'''


kpca = KernelPCA(n_components=2, kernel = 'rbf' )
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)


from sklearn.metrics import confusion_matrix


conf_matrix = confusion_matrix(y_test, y_pred)
print('\n\nThe Confusion Matrix for our KernelPCA Logistic Regression is:\n')
print(conf_matrix)
 
plt.imshow(conf_matrix)
plt.title('Graphical representation of Prediction of how many people will buy the SUV')
plt.xlabel('AGE')
plt.ylabel('Estimated Salary')
plt.show()




# Visualising the Training set results
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('\n\n\n Hence the accuracy of the Kernelized PCA for Logistic Regression is:',accuracy)
print('\n\n Done :)')
