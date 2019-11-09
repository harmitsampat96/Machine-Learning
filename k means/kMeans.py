# Importing the libraries
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importing the Iris dataset with pandas
variables = pd.read_csv('iris.csv')
Y = variables[['sepal_length']]
X = variables[['sepal_width']]

# Finding the optimum number of clusters for k-means classification
Nc = range(1, 151)
kmeans = [KMeans(n_clusters=i) for i in Nc]

print("kmeans : \n")

score = [kmeans[i].fit(Y).score(Y) for i in range(len(kmeans))]

print("score : \n")

pl.plot(Nc, score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()

# Using PCA
pca = PCA(n_components=1).fit(Y)
pca_d = pca.transform(Y)
pca_c = pca.transform(X)

# Clustering
kmeans = KMeans(n_clusters=3)
kmeansoutput = kmeans.fit(Y)

print("kmeansoutput : \n")

pl.figure('3 Cluster K-Means')
pl.scatter(pca_c[:, 0], pca_d[:, 0], c=kmeansoutput.labels_)
pl.xlabel('Sepal Length')
pl.ylabel('Sepal Width')
pl.title('3 Cluster K-Means')
pl.show()
