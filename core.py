import time
import sys
import random
import numpy as np
from stream import afficheur
import sklearn.cluster.k_means_ as cluster

class core():
   
  def __init__(self):
    self.init='k-means++'
  def kmean(self, n_clusters, n_init, X):
  
    #Compute clustering with Means
    k_means = cluster.KMeans(init=self.init, n_clusters=n_clusters, n_init=n_init)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    
    return k_means,t_batch
    
    
