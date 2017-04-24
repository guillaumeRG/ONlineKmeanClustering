import time
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from plot import Afficheur
import sklearn.cluster.k_means_ as cluster
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

#export PATH=~/anaconda3/bin:$PATH

def dataGen(centers,n_Samples,cluster_std):
   return make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
   
def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol

if __name__ == '__main__':
  
  
  
  #config
  n_init=100
  init='k-means++'
  n_iter=int(sys.argv[1])
  n_samples=100000
  cluster_std=0.7
  #Generate sample data
  np.random.seed(0)
  batch_size = 750
  centers = [[2, 2], [-2, -2], [2, -2]]
  n_clusters = len(centers)
  X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
  print(X.shape)
  print(labels_true)



  #Compute clustering with Means
  k_means = cluster.KMeans(init='k-means++', n_clusters=3, n_init=10)
  t0 = time.time()
  k_means.fit(X)
  t_batch = time.time() - t0



 
  #to do with online MBK_mean  
  #Compute clustering with MiniBatchKMeans
  mbk = cluster.MiniBatchKMeans(init=init, n_clusters=3, batch_size=batch_size,n_init=10, max_no_improvement=n_iter, verbose=0)
  t0 = time.time()
  
  try:
   if sys.argv[2] == '-pp' or sys.argv[3] == '-pp':
    thread_1 = Afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters,t_batch)
    thread_1.start()
  except IndexError:
   pass
   
   

  try:
   if sys.argv[2] == '-pp':
    # Sample a minibatch from the full dataset
    for iteration_idx in range(n_iter-1):
     mbk=mbk.partial_fit(X)
     thread_1.update(mbk)
    thread_1.stop()
     
   elif sys.argv[2] == '-o':
    n_batches = int(np.ceil(float(n_samples) / batch_size))
    max_iter = 100
    n_iter = int(max_iter * n_batches)
    tol=0
    _, n_features = X.shape
    old_center_buffer = np.zeros(n_features, dtype=X.dtype)
    
    #  print('self.max_iter %d , n_batches %d '%(n_iter,n_batches))
    if sys.argv[3] == '-pp': 
     #init state
     
     random_state = check_random_state(None)
     init_size = 3 * batch_size
     if init_size > n_samples:
      init_size = n_samples
     validation_indices = random_state.randint(0, n_samples, init_size)
     X_valid = X[validation_indices]
     x_squared_norms = row_norms(X, squared=True)
     x_squared_norms_valid = x_squared_norms[validation_indices]
     counts = np.zeros(n_clusters, dtype=np.int32)
     best_inertia = None
     cluster_centers = None
     for init_idx in range(n_init):
      cluster_centers = cluster._init_centroids(X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms, init_size=init_size)
      batch_inertia, centers_squared_diff = cluster._mini_batch_step(X_valid, x_squared_norms[validation_indices], cluster_centers_, counts, old_center_buffer, False, distances=None, verbose=False)

      _, inertia = cluster._labels_inertia(X_valid, x_squared_norms_valid, cluster_centers)
      if best_inertia is None or inertia < best_inertia:
       mbk.cluster_centers_ = cluster_centers
       mbk.counts_ = counts
       best_inertia = inertia
       print('best inertia %d' %best_inertia)
       
       
     convergence_context = {}
     for iteration_idx in range(n_iter):
      mbk=mbk.partial_fit(X)
      tol = _tolerance(X, tol)    
      thread_1.update(mbk)

      # Monitor convergence and do early stopping if necessary
      if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=mbk.verbose):
       break
     thread_1.stop()
   
    elif sys.argv[3] == '-p':
     for iteration_idx in range(n_iter):
      mbk=mbk.partial_fit(X)
      tol = _tolerance(X, tol)
     
      # Monitor convergence and do early stopping if necessary
      if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=False):
       break
   
   elif sys.argv[2] == '-n':
    mbk=mbk.fit(X)
   else:   
    # Sample a minibatch from the full dataset
    for iteration_idx in range(n_iter-1):
     mbk=mbk.partial_fit(X)
     
  except IndexError:
   pass
  
  try:
   if sys.argv[2] == '-p' or sys.argv[2] == '-pp' or sys.argv[3] == '-pp' or sys.argv[3] == '-p':
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

    t_mini_batch = time.time() - t0
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
     i=0
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
   
    
    if sys.argv[2] == '-p' or sys.argv[2] == '-pp' or sys.argv[3] == '-pp' or sys.argv[3] == '-p':
     plt.show()
    if (sys.argv[3] == '-f') and (sys.argv[4] != None):
     print('Saving...')
     plt.savefig(sys.argv[4])
     print('Done !')
    elif sys.argv[3] == '-f' and sys.argv[4] == None:
     print('Usage : -f PATH/NAME')
    
   
    
    
   
  except IndexError:
   pass 
   
   






