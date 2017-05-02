import time
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from core import core
from datagen import datagen
from stream import afficheur
import sklearn.cluster.k_means_ as cluster
from sklearn.metrics.pairwise import pairwise_distances_argmin
import scipy.sparse as sp
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

#export PATH=~/anaconda3/bin:$PATH
   
def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol
def input_num(prompt):
    """
    prompt the user for a numeric input
    prompt again if the input is not numeric
    return an integer or a float
    """
    while True:
        # strip() removes any leading or trailing whitespace
        num_str = input(prompt).strip()
        if num_str == 'stop':
            return 'stop'
        num_flag = True
        for c in num_str:
            # check for non-numerics
            if c not in '+-.0123456789':
                num_flag = False
        if num_flag:
            break
    # a float contains a period (US)
    if '.' in num_str:
        return float(num_str)
    else:
        return int(num_str)

if __name__ == '__main__':


 


  #config
  n_init=100
  init='k-means++'
  n_iter=int(sys.argv[1])
  #Generate sample data
  np.random.seed(0)
  generator = datagen()
  X, labels_true, batch_size, n_samples, n_clusters = generator.genesis()

  #Compute clustering with Means
  core = core()
  k_means,t_batch = core.kmean(n_clusters, n_init, X)



 
  #to do with online MBK_mean  
  #Compute clustering with MiniBatchKMeans
  mbk = cluster.MiniBatchKMeans(init=init, n_clusters=3, batch_size=batch_size,n_init=10, max_no_improvement=n_iter, verbose=0)
  
  #INIT THREADs
  try:
   if sys.argv[2] == '-pp' or sys.argv[3] == '-pp':
    thread_1 = afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters,t_batch)
    thread_1.start()
  except IndexError:
   pass
   
  try:
   if sys.argv[2] == '-s':
    
    #init state
    n_batches = int(np.ceil(float(n_samples) / batch_size))
    max_iter = 100
    
   
    tol=0
    _, n_features = X.shape
    old_center_buffer = np.zeros(n_features, dtype=X.dtype)
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
     batch_inertia, centers_squared_diff = cluster._mini_batch_step(X_valid, x_squared_norms[validation_indices], cluster_centers, counts, old_center_buffer, False, distances=None, verbose=False)
     _, inertia = cluster._labels_inertia(X_valid, x_squared_norms_valid, cluster_centers)
     if best_inertia is None or inertia < best_inertia:
      mbk.cluster_centers_ = cluster_centers
      mbk.counts_ = counts
      best_inertia = inertia
      print('best inertia %d' %best_inertia)
       
    while(True):
     thread_1 = afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters,t_batch)
     thread_1.start()
     t0 = time.time()
    
    
     for iteration_idx in range(n_iter):
      minibatch_indices = random_state.randint(0, n_samples, batch_size)
      mbk=mbk.partial_fit(X[minibatch_indices])
      thread_1.update(mbk)
     t_mini_batch = time.time() - t0
     thread_1.stop()
     thread_1.join()
     n_iter = input_num("Iterations suivante : ")
     if n_iter == "stop":
      break
     if isinstance(n_iter, int) == False:
      print('error integer is required !!! type %s'%type(n_iter))
      break
   
  except IndexError:
   pass   
  try:
   if sys.argv[2] == '-pp':
    random_state = check_random_state(None) 
    t0 = time.time()
    # Sample a minibatch from the full dataset
    for iteration_idx in range(n_iter-1):
     minibatch_indices = random_state.randint(0, n_samples, batch_size)
     mbk=mbk.partial_fit(X[minibatch_indices])
     thread_1.update(mbk)
    t_mini_batch = time.time() - t0
    thread_1.stop()
    
  except IndexError:
   pass   
       
  try:
   if sys.argv[2] == '-p':
    tol=0
    random_state = check_random_state(None)
    t0 = time.time()
    for iteration_idx in range(n_iter):
     minibatch_indices = random_state.randint(0, n_samples, batch_size)
     mbk=mbk.partial_fit(X[minibatch_indices])
     tol = _tolerance(X, tol)
    t_mini_batch = time.time() - t0
   
  except IndexError:
   pass
    
  try: 
   if sys.argv[2] == '-n':
    t0 = time.time()
    mbk=mbk.fit(X)
    t_mini_batch = time.time() - t0
    
   
  except IndexError:
   pass
   
  try :
   if sys.argv[2] == None:
    random_state = check_random_state(None)   
    # Sample a minibatch from the full dataset
    t0 = time.time()
    for iteration_idx in range(n_iter-1):
     minibatch_indices = random_state.randint(0, n_samples, self.batch_size) 
     mbk=mbk.partial_fit(X,minibatch_indices=minibatch_indices) 
    t_mini_batch = time.time() - t0  
  except IndexError:
   pass

  try:  
   if sys.argv[2] == '-o':
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
      batch_inertia, centers_squared_diff = cluster._mini_batch_step(X_valid, x_squared_norms[validation_indices], cluster_centers, counts, old_center_buffer, False, distances=None, verbose=False)
      _, inertia = cluster._labels_inertia(X_valid, x_squared_norms_valid, cluster_centers)
      if best_inertia is None or inertia < best_inertia:
       mbk.cluster_centers_ = cluster_centers
       mbk.counts_ = counts
       best_inertia = inertia
       print('best inertia %d' %best_inertia)
       
    
     convergence_context = {}
     mbk.batch_inertia = batch_inertia
     mbk.centers_squared_diff = centers_squared_diff
     t0 = time.time()
     for iteration_idx in range(n_iter):
      minibatch_indices = random_state.randint(0, n_samples, batch_size)
      mbk=mbk.partial_fit(X[minibatch_indices])
      tol = _tolerance(X, tol)    
      thread_1.update(mbk)

      # Monitor convergence and do early stopping if necessary
      if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=mbk.verbose):
       t_mini_batch = time.time() - t0
       thread_1.stop()
       break
     
   
    elif sys.argv[3] == '-p':
     random_state = check_random_state(None)
     convergence_context = {}
     t0 = time.time()
     for iteration_idx in range(n_iter):
      minibatch_indices = random_state.randint(0, n_samples, batch_size)
      mbk=mbk.partial_fit(X[minibatch_indices])
      tol = _tolerance(X, tol)
     
      # Monitor convergence and do early stopping if necessary
      if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=False):
       t_mini_batch = time.time() - t0
      
       break
  except IndexError:
   pass
  
  
  
  
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
   
   






