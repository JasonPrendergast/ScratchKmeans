import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
#from sklearn.cluster import KMeans
style.use('ggplot')

#ORIGINAL:

#X = np.array([[1, 2],
#              [1.5, 1.8],
#              [5,8],
#              [8,8],
#              [1,0.6],
#              [9,11],
#              [1,3],
#              [8,9],
#              [0,3],
#              [5,4],
#             [6,4]])


#plt.scatter(X[:, 0],X[:, 1], s=150, linewidths = 5, zorder = 10)
#plt.show()
#colors = ['r','g','b','c','k','o','y']

class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data):
#       crate empty centroids dictionary (dict)
        self.centroids = {}

        for i in range(self.k):
            #assign first k data points as data to as first centroids
            self.centroids[i] = data[i]

        #empty the classification dictionary
        for i in range(self.max_iter):
            self.classifications = {}
            #Create k classification
            for i in range(self.k):
                self.classifications[i] = []
#create the new centroids, as well as measuring the movement of the centroids.
#If that movement is less than our tolerance (self.tol)
            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                   break
                
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

