import pandas
# import scipy
import numpy
from numpy import array
from numpy.linalg import inv
from sklearn.preprocessing import Normalizer

# load CSV using Pandas
filename = 'mockData.csv'
names = ['name', 'sem1', 'sem2', 'sem3', 'sem4', 'sem5', 'sem6', 'sem7', 'sem8', 'dist', 'hour', 'tuition', 'hobby', 'gender', 'sep_room', 'competitive', 'higher_stud', 'campus', 'extra', 'cet_score']
data = pandas.read_csv(filename, names=names)

# print(data.shape)

# separate array into input and output components
X = data.values[1:, 1:8]
Y = data.values[1:, 8]

X = X.astype(float)
Y = Y.astype(float)

b = inv(X.T.dot(X)).dot(X.T).dot(Y)
print('co-efficients: ', b)

x = array([7.42, 8.31, 7.48, 7.04, 6.68, 7.35, 8.46])
# x = x.reshape((len(x), 1))

# predict using coefficients
y = x.dot(b)
print('predicted pointer: ', y)

# CROSS-VALIDATION

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.85)
regression = LinearRegression()
r2 = regression.fit(X_train, y_train)
y_pred = regression.predict(X_test)
print(y_pred)
#print(y_test)
accuracy = r2_score(y_test, y_pred)*100
print("Accuracy is :" , accuracy)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(regression, X_train, y_train, cv=4, scoring='neg_mean_squared_error')
print(scores)
print("The mean score is: " , scores.mean())

plt.scatter(y_pred, y_test , s=30, c='r', marker='+', zorder=10)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.show()
