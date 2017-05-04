import time
import sys
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
from core import core 
from datagen import datagen
from stream import afficheur
import sklearn.cluster.k_means_ as cluster
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.utils.sparsefuncs import mean_variance_axis

#export PATH=~/anaconda3/bin:$PATH

if __name__ == '__main__':

  #config
  n_init=100
  init='k-means++'
  n_iter=int(sys.argv[1])
  datapath = 'data/'
  try:
    if (sys.argv[2] == '-l' ):
      with open(datapath + sys.argv[3], newline='') as csvfile:
        csvdata = csv.reader(csvfile, delimiter='"', quotechar='|')
        print('reading csv...')
        index=0
        for row in csvdata:
          print(', '.join(row))
          index += 1
    elif(sys.argv[3] == '-l'):
      with open(datapath + sys.argv[4], newline='') as csvfile:
        csvdata = csv.reader(csvfile, delimiter='"', quotechar='|')
        print('reading csv...')
        index=0
        for row in csvdata:
          print(', '.join(row))
          index += 1
    else:
      #Generate sample data
      np.random.seed(0)
      generator = datagen()
      X, labels_true, batch_size, n_samples, n_clusters = generator.genesis()
     

  #Compute clustering with K Mean
  core = core()
  k_means,t_batch = core.kmean(n_clusters, n_init, X)

  #Compute clustering with Minibatch K Mean
  mbk, t_mini_batch = core.mbkmean(sys.argv,n_clusters, n_init, batch_size, n_iter, n_samples, labels_true, k_means, X)
 

  with open(datapath + 'datatest.csv', newline='') as csvfile:
    csvdata = csv.reader(csvfile, delimiter='"', quotechar='|')
    for row in csvdata:
      print(', '.join(row))

  #plotting
   
  try:
   if sys.argv[2] == '-p' or sys.argv[2] == '-pp' or sys.argv[2] == '-s' or sys.argv[3] == '-pp' or sys.argv[3] == '-p':
    print('plotting')
    #Plot result
    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ['#4EACC5', '#FF9C34', '#4E9A06']
    
    # Attend que les threads se terminent
    #thread_1.join()
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    mbk_means_cluster_centers = np.sort(mbk.cluster_centers_, axis=0)
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)
    order = pairwise_distances_argmin(k_means_cluster_centers,mbk_means_cluster_centers)
    
    # KMeans
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
         markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
         markeredgecolor='k', markersize=6)
    ax.set_title('KMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8,  'train time: %.2fs\ninertia: %f' % (
        t_batch, k_means.inertia_))

    
    # MiniBatchKMeans
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
     my_members = mbk_means_labels == order[k]
     cluster_center = mbk_means_cluster_centers[order[k]]
     ax.plot(X[my_members, 0], X[my_members, 1], 'w',
      markerfacecolor=col, marker='.')
     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
      markeredgecolor='k', markersize=6)
    ax.set_title('MiniBatchKMeans')
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(-3.5, 1.8, 'train time: %.2fs\ninertia: %f' %
      (t_mini_batch, mbk.inertia_))

    # Initialise the different array to all False
    different = (mbk_means_labels == 4)
    nbK = np.arange(n_clusters)
    err = np.arange(n_clusters)
    nbL = np.arange(n_clusters)
    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
     different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
     i = 0
     for s in mbk_means_labels:
      if s == labels_true[i] :
       nbK[k] += 1 
      if labels_true[i] == k:
       nbL[k] +=1
      i += 1
       
     err[k] = nbK[k]/ nbL[k] 
        
    identic = np.logical_not(different)

    ax.plot(X[identic, 0], X[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.')

    ax.plot(X[different, 0], X[different, 1], 'w',markerfacecolor='r', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())
     
    n_diff =len(X[different,])
    for k in range(n_clusters):
     print('Error cluster %d : %f'%(k ,(nbK[k]/ nbL[k])))
     
    print('Clustering \'s difference: %d'%n_diff)
    ratio = n_diff/len(mbk_means_labels == 4)
    print('Difference \'s ratio: %f'%ratio)
   
    
    if sys.argv[2] == '-p' or sys.argv[2] == '-pp'or sys.argv[2] == '-s' or sys.argv[3] == '-pp' or sys.argv[3] == '-p' :
     plt.show()
    if (sys.argv[3] == '-f') and (sys.argv[4] != None):
     print('Saving...')
     plt.savefig(sys.argv[4])
     print('Done !')
    elif sys.argv[3] == '-f' and sys.argv[4] == None:
     print('Usage : -f PATH/NAME')

  except IndexError:
   pass 
   
  






