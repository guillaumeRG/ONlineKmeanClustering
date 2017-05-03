import time
import sys
import random
import numpy as np
from stream import afficheur
import sklearn.cluster.k_means_ as cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms
import scipy.sparse as sp

class core():
   
  def __init__(self):
    self.init='k-means++'
  def _tolerance(self, X, tol):
    """Return a tolerance which is independent of the dataset"""
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol  
    
  def input_num(self,prompt):
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
         
  def kmean(self, n_clusters, n_init, X):
  
    #Compute clustering with Means
    k_means = cluster.KMeans(init=self.init, n_clusters=n_clusters, n_init=n_init)
    t0 = time.time()
    k_means.fit(X)
    t_batch = time.time() - t0
    
    return k_means,t_batch
  
  def mbkmean(self, options,  n_clusters, n_init, batch_size, n_iter, n_samples, labels_true, k_means, X):
    #to do with online MBK_mean  
    #Compute clustering with MiniBatchKMeans
    mbk = cluster.MiniBatchKMeans(init=self.init, n_clusters=n_clusters, batch_size=batch_size,n_init=10, max_no_improvement=n_iter, verbose=0)
    
    #INIT THREADs
    try:
     if options[2] == '-pp' or options[3] == '-pp':
      thread_1 = afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters)
      thread_1.start()
    except IndexError:
     pass
     
    try:
     if options[2] == '-s':
      
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
       
       cluster_centers = cluster._init_centroids(X, n_clusters, self.init, random_state=random_state, x_squared_norms=x_squared_norms, init_size=init_size)
       batch_inertia, centers_squared_diff = cluster._mini_batch_step(X_valid, x_squared_norms[validation_indices], cluster_centers, counts, old_center_buffer, False, distances=None, verbose=False)
       _, inertia = cluster._labels_inertia(X_valid, x_squared_norms_valid, cluster_centers)
       if best_inertia is None or inertia < best_inertia:
        mbk.cluster_centers_ = cluster_centers
        mbk.counts_ = counts
        best_inertia = inertia
        print('best inertia %d' %best_inertia)
         
      while(True):
       thread_1 = afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters)
       thread_1.start()
       t0 = time.time()
       
       
       for iteration_idx in range(n_iter):
        minibatch_indices = random_state.randint(0, n_samples, batch_size)
        mbk=mbk.partial_fit(X[minibatch_indices])
        thread_1.update(mbk)
       t_mini_batch = time.time() - t0
       thread_1.stop()
       thread_1.join()
       
       
       
       n_iter = self.input_num("Iterations suivante : ")
       
       if n_iter == "stop":
        return mbk, t_mini_batch
        break
       if isinstance(n_iter, int) == False:
        print('error integer is required !!! type %s'%type(n_iter))
        break
     
    except IndexError:
     pass   
    try:
     if options[2] == '-pp':
     
      random_state = check_random_state(None) 
      t0 = time.time()
      # Sample a minibatch from the full dataset
      for iteration_idx in range(n_iter-1):
       minibatch_indices = random_state.randint(0, n_samples, batch_size)
       mbk=mbk.partial_fit(X[minibatch_indices])
      
       thread_1.update(mbk)
      t_mini_batch = time.time() - t0
      thread_1.stop()
      thread_1.join()
      return mbk, t_mini_batch
      
    except IndexError:
     pass   
         
    try:
     if options[2] == '-p':
      
      random_state = check_random_state(None)
      t0 = time.time()
      for iteration_idx in range(n_iter):
       minibatch_indices = random_state.randint(0, n_samples, batch_size)
       mbk=mbk.partial_fit(X[minibatch_indices])
       
      t_mini_batch = time.time() - t0
      return mbk, t_mini_batch
     
    except IndexError:
     pass
      
    try: 
     if options[2] == '-n':
      t0 = time.time()
      mbk=mbk.fit(X)
      t_mini_batch = time.time() - t0
      return mbk, t_mini_batch
      
     
    except IndexError:
     pass
     
    try :
     if options[2] == None:
      random_state = check_random_state(None)   
      # Sample a minibatch from the full dataset
      t0 = time.time()
      for iteration_idx in range(n_iter-1):
       minibatch_indices = random_state.randint(0, n_samples, self.batch_size) 
       mbk=mbk.partial_fit(X,minibatch_indices=minibatch_indices) 
      t_mini_batch = time.time() - t0  
      return mbk, t_mini_batch
    except IndexError:
     pass

    try:  
     if options[2] == '-o':
      n_batches = int(np.ceil(float(n_samples) / batch_size))
      max_iter = 100
      
      n_iter = int(max_iter * n_batches)
      tol=0
      _, n_features = X.shape
      old_center_buffer = np.zeros(n_features, dtype=X.dtype)
      try:
        #  print('self.max_iter %d , n_batches %d '%(n_iter,n_batches))
        if options[3] == '-pp': 
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
          cluster_centers = cluster._init_centroids(X, n_clusters, self.init, random_state=random_state, x_squared_norms=x_squared_norms, init_size=init_size)
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
          tol = self._tolerance(X, tol)    
          thread_1.update(mbk)

          # Monitor convergence and do early stopping if necessary
          if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=mbk.verbose):
           t_mini_batch = time.time() - t0
           thread_1.stop()
           thread_1.join()
           return mbk, t_mini_batch
           break
         
       
        elif options[3] == '-p':
         random_state = check_random_state(None)
         convergence_context = {}
         t0 = time.time()
         for iteration_idx in range(n_iter):
          minibatch_indices = random_state.randint(0, n_samples, batch_size)
          mbk=mbk.partial_fit(X[minibatch_indices])
          tol = self._tolerance(X, tol)
         
          # Monitor convergence and do early stopping if necessary
          if cluster._mini_batch_convergence(mbk, iteration_idx, n_iter, tol, n_samples,mbk.centers_squared_diff, mbk.batch_inertia, convergence_context,verbose=False):
            t_mini_batch = time.time() - t0
            return mbk, t_mini_batch
            break
      except TypeError:
        pass
    except IndexError:
      pass
    
      
  
    

