import random

import sys

from threading import Thread
import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import time

import matplotlib.pyplot as plt



class Afficheur(Thread):
 t0 = time.time()
 t_mini_batch = time.time() - t0
 #Plot result
 
 
 def __init__(self,test,mbk,k_means,X,n_clusters,t_batch):
 
  
  self.t0 = time.time()
  
  self.t_mini_batch = time.time() - self.t0
  #Plot result
  
  self.colors = ['#4EACC5', '#FF9C34', '#4E9A06']
  self.k_means=k_means
  self.mbk=mbk
  self.X=X
  self.n_clusters= n_clusters
  self.t_batch=t_batch
  
  self.test=test
  print('%s'%self.test)
  Thread.__init__(self)
 def run(self):
  
  self.colors = ['#4EACC5', '#FF9C34', '#4E9A06']
  # KMeans
  k_means_cluster_centers = np.sort(self.k_means.cluster_centers_, axis=0)
  k_means_labels = pairwise_distances_argmin(self.X, k_means_cluster_centers)
  for k, col in zip(range(self.n_clusters), self.colors):
      my_members = k_means_labels == k
      cluster_center = k_means_cluster_centers[k]

  self.t_mini_batch = time.time() - self.t0
  # MiniBatchKMeans
  mbk_means_cluster_centers = np.sort(self.mbk.cluster_centers_, axis=0)
  mbk_means_labels = pairwise_distances_argmin(self.X, mbk_means_cluster_centers)
  order = pairwise_distances_argmin(k_means_cluster_centers,
                             mbk_means_cluster_centers)
  for k, col in zip(range(self.n_clusters), self.colors):
   my_members = mbk_means_labels == order[k]
   cluster_center = mbk_means_cluster_centers[order[k]]


  # Initialise the different array to all False
  different = (mbk_means_labels == 4)

  
  for k in range(self.n_clusters):
   different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
      
  identic = np.logical_not(different)


  n_diff =len(self.X[different,])
  print('Clustering \'s difference: %d'%n_diff)
  ratio = n_diff/len(mbk_means_labels == 4)
  print('Difference \'s ratio: %f'%ratio)
  
  

 def update(self,mbk,X,n_clusters,t_batch):


  self.mbk=mbk
  self.X=X
  self.n_clusters= n_clusters
  self.t_batch=t_batch
 # We want to have the same colors for the same cluster from the
  # MiniBatchKMeans and the KMeans algorithm. Let's pair the cluster centers per
  # closest one.
  k_means_cluster_centers = np.sort(self.k_means.cluster_centers_, axis=0)
  mbk_means_cluster_centers = np.sort(self.mbk.cluster_centers_, axis=0)
  k_means_labels = pairwise_distances_argmin(self.X, k_means_cluster_centers)
  mbk_means_labels = pairwise_distances_argmin(self.X, mbk_means_cluster_centers)
  order = pairwise_distances_argmin(k_means_cluster_centers,
                             mbk_means_cluster_centers)

  # MiniBatchKMeans
  for k, col in zip(range(self.n_clusters), self.colors):
   my_members = mbk_means_labels == order[k]
   cluster_center = mbk_means_cluster_centers[order[k]]
  

  # Initialise the different array to all False
  different = (mbk_means_labels == 4)

  for k in range(self.n_clusters):
   different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
      
  identic = np.logical_not(different)

  


  n_diff =len(self.X[different,])
  print('Clustering \'s difference: %d'%n_diff)
  ratio = n_diff/len(mbk_means_labels == 4)
  print('Difference \'s ratio: %f'%ratio)
  
 

