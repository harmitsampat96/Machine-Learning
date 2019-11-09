# Importing libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.plotly as py
import plotly
import plotly.graph_objs as go

plotly.tools.set_credentials_file(username='harmitsampat96', api_key='y9W3QVS0fv8ihgqf9gL2')

# Reading and pre-processing the data set
df = pd.read_csv('D:/Dataset.csv')
df['Daily distance of travel to college in km (write for two way journey)'] = df[
    'Daily distance of travel to college in km (write for two way journey)'].fillna(
    (df['Daily distance of travel to college in km (write for two way journey)'].mean()))
df['Separate room for study(yes=1 /no=0)'] = df['Separate room for study(yes=1 /no=0)'].fillna(
    (df['Separate room for study(yes=1 /no=0)'].mean()))
df['Pointer of Sem VII marks'] = df['Pointer of Sem VII marks'].fillna((df['Pointer of Sem VII marks'].mean()))

# Splitting the data set
X = df.iloc[:, 2:16].values
y = df.iloc[:, 16].values

# Fitting the data and finding the covariance matrix
X_std = StandardScaler().fit_transform(X)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# SVD and selecting the principal components
u, s, v = np.linalg.svd(X_std.T)
for ev in eig_vecs:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort()
eig_pairs.reverse()

# Choosing the PC
traces = []
legend = {0: False, 1: False, 2: False, 3: True}
tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
trace1 = go.Bar(x=['PC %s' % i for i in range(1, 5)], y=var_exp, showlegend=False)
trace2 = go.Scatter(x=['PC %s' % i for i in range(1, 5)], y=cum_var_exp, name='cumulative explained variance')
data = go.Data([trace1, trace2])

fig = go.Figure(data=data)
py.plot(fig)

# Reshaping the dimensions
matrix_w = np.hstack((eig_pairs[0][1].reshape(14, 1), eig_pairs[1][1].reshape(14, 1), eig_pairs[2][1].reshape(14, 1),
                      eig_pairs[3][1].reshape(14, 1)))
print('Matrix W:\n', matrix_w)
Y = X_std.dot(matrix_w)
print(Y)
