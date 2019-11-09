import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

x = np.sort(np.random.rand(100, 1), axis=0)
y = np.cos(x+np.pi/2).ravel()
#print (x)
#print (y)

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1.0, gamma=0.7)
svr_lin = SVR(kernel='linear', C=1.0)
svr_poly = SVR(kernel='poly', C=1.0, degree=3)
y_rbf = svr_rbf.fit(x, y).predict(x)
y_lin = svr_lin.fit(x, y).predict(x)
y_poly = svr_poly.fit(x, y).predict(x)

print('The linear accuracy is: ', svr_lin.score(x, y))
print('The polynomial accuracy is: ', svr_poly.score(x, y))
print('The RBF accuracy is: ', svr_rbf.score(x, y))

lw = 2
plt.scatter(x, y, color='yellow', label='data')
plt.plot(x, y_rbf, color='blue', lw=lw, label='RBF model')
plt.plot(x, y_lin, color='red', lw=lw, label='Linear model')
plt.plot(x, y_poly, color='green', lw=lw, label='Polynomial model')
plt.xlabel('Data Values')
plt.ylabel('Model Values')
plt.title('Support Vector Regression for Cos Curve')
plt.legend()
plt.show()

