import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

x = np.sort(np.random.rand(100, 1), axis=0)
y = np.sin(x+np.pi).ravel()
#print (x)
#print (y)

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1.0, gamma=0.7)
svr_lin = SVR(kernel='linear', C=1.0)
svr_poly = SVR(kernel='poly', C=1.0, degree=3)
y_rbf = svr_rbf.fit(x, y).predict(x)
y_lin = svr_lin.fit(x, y).predict(x)
y_poly = svr_poly.fit(x, y).predict(x)

print('\nThe linear accuracy is: ', svr_lin.score(x, y))
print('\nThe polynomial accuracy is: ', svr_poly.score(x, y))
print('\nThe RBF accuracy is: ', svr_rbf.score(x, y))

lw = 2
plt.scatter(x, y, color='yellow', label='data')
plt.plot(x, y_rbf, color='blue', lw=lw, label='RBF model')
plt.plot(x, y_lin, color='red', lw=lw, label='Linear model')
plt.plot(x, y_poly, color='green', lw=lw, label='Polynomial model')
plt.xlabel('Data Values')
plt.ylabel('Model Values')
plt.title('Support Vector Regression for Sine Curve')
plt.legend()
plt.show()

