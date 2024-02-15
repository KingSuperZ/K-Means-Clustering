"""Clustering data points using k-means"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

### 1D KMeans Clustering (Unsupervised Learning)
X,y = make_blobs(n_samples = 100, n_features = 1, centers = 2, cluster_std = 0.5) # Creating a random set of 100 data points
alg = KMeans(n_clusters = 2) # Sets the number of clusters to two
alg.fit(X) # Creates the clusters using the randomized data
ypred = alg.labels_ # Answers of the clustering operation

## This section is used to plot the clustered datapoints into different colors
zeros = np.zeros(len(X)) # Creates a list of zeros
plt.scatter(X, zeros, c = ypred, s = 5) # Creates the figure which shows the different clusters
plt.yticks([]) # Removes the y-axis numbers for cleanliness


### 2D KMeans CLustering (Unsupervised Learning)
X2,y = make_blobs(n_samples = 1000, n_features = 2, centers = 2, cluster_std = 0.5) # Creating a random set of 100 data points
alg = KMeans(n_clusters = 2) # Sets the number of clusters to two
alg.fit(X2) # Creates the clusters using the randomized data
ypred2 = alg.labels_ # Answers of the clustering operation 

## This section is used to plot the clustered datapoints into different colors
xcord = X2[:,0] # Stores the x coordinates
ycord = X2[:,1] # Stores the y coordinates
plt.figure() # Allows the creation of a separate figure
plt.scatter(xcord, ycord, c = ypred2) # Creates the final figure
