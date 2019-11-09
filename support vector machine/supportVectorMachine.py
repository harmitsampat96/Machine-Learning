#importing library
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions

#import dataset
df = pd.read_csv("Iris.csv")
df.head()

#divide dataset into: train, test, X, Y
y = df[['Species']]
X = df[['SepalLengthCm', 'SepalWidthCm']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#creating svm classification object
model_lin = svm.SVC(kernel='linear', C=1, gamma=1)

model_rbf_c1 = svm.SVC(kernel='rbf', C=1, gamma=1)
model_rbf_c2 = svm.SVC(kernel='rbf', C=10, gamma=1)
model_rbf_c3 = svm.SVC(kernel='rbf', C=100, gamma=1)

model_rbf_g1 = svm.SVC(kernel='rbf', C=1, gamma=0.5)
model_rbf_g2 = svm.SVC(kernel='rbf', C=1, gamma=1)
model_rbf_g3 = svm.SVC(kernel='rbf', C=1, gamma=10)

#training model using training set
model_lin.fit(X_train, y_train)

model_rbf_c1.fit(X_train, y_train)
model_rbf_c2.fit(X_train, y_train)
model_rbf_c3.fit(X_train, y_train)

model_rbf_g1.fit(X_train, y_train)
model_rbf_g2.fit(X_train, y_train)
model_rbf_g3.fit(X_train, y_train)

#checking score
model_lin.score(X_train, y_train)

model_rbf_c1.score(X_train, y_train)
model_rbf_c2.score(X_train, y_train)
model_rbf_c3.score(X_train, y_train)

model_rbf_g1.score(X_train, y_train)
model_rbf_g2.score(X_train, y_train)
model_rbf_g3.score(X_train, y_train)

#predict output
y_pred_lin = model_lin.predict(X_test)

y_pred_rbf_c1 = model_rbf_c1.predict(X_test)
y_pred_rbf_c2 = model_rbf_c2.predict(X_test)
y_pred_rbf_c3 = model_rbf_c3.predict(X_test)

y_pred_rbf_g1 = model_rbf_g1.predict(X_test)
y_pred_rbf_g2 = model_rbf_g2.predict(X_test)
y_pred_rbf_g3 = model_rbf_g3.predict(X_test)

#printing accuracy
print('The linear accuracy is: ', accuracy_score(y_pred_lin, y_test))
print(confusion_matrix(y_test, y_pred_lin))
print(classification_report(y_test, y_pred_lin))

print('\nThe RBF (C=1, G=1) accuracy is: ', accuracy_score(y_pred_rbf_c1, y_test))
print(confusion_matrix(y_test, y_pred_rbf_c1))
print(classification_report(y_test, y_pred_rbf_c1))

print('\nThe RBF (C=10, G=1) accuracy is: ', accuracy_score(y_pred_rbf_c2, y_test))
print('The RBF (C=100, G=1) accuracy is: ', accuracy_score(y_pred_rbf_c3, y_test))

print('\nThe RBF (C=1, G=0.5) accuracy is: ', accuracy_score(y_pred_rbf_g1, y_test))
print('The RBF (C=1, G=1) accuracy is: ', accuracy_score(y_pred_rbf_g2, y_test))
print('The RBF (C=1, G=10) accuracy is: ', accuracy_score(y_pred_rbf_g3, y_test))

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C),
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, C=C))
models = (clf.fit(X, y) for clf in models)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel')

# Set-up 2x2 grid for plotting.
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

