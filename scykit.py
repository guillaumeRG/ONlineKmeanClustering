import time
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from plot import Afficheur
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets.samples_generator import make_blobs

#export PATH=~/anaconda3/bin:$PATH

def dataGen(centers,n_Samples,cluster_std):
   return make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)


if __name__ == '__main__':
  
  
  
  #config
  n_iter=int(sys.argv[1])
  n_samples=30000
  cluster_std=0
  #Generate sample data
  np.random.seed(0)
  batch_size = 45
  centers = [[6, 6], [-5, -5], [5, -5]]
  n_clusters = len(centers)
  X, labels_true = make_blobs(n_samples=n_samples, centers=centers, cluster_std=cluster_std)
  print(X.shape)
  print(labels_true)



  #Compute clustering with Means
  k_means = KMeans(init='k-means++', n_clusters=3, n_init=10)
  t0 = time.time()
  k_means.fit(X)
  t_batch = time.time() - t0



 
  #to do with online MBK_mean  
  #Compute clustering with MiniBatchKMeans
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=batch_size,n_init=10, max_no_improvement=10, verbose=0)

  t0 = time.time()
  mbk=mbk.partial_fit(X)
  try:
   if sys.argv[2] == '-pp':
    thread_1 = Afficheur('starting threads',labels_true,mbk,k_means,X,n_clusters,t_batch)
    thread_1.start()
  except IndexError:
   pass
  # Lancement des threads
  
 
  #thread_1.run(k_means,mbk,X,n_clusters,t_batch)
  try:
   if sys.argv[2] == '-pp':
    # Sample a minibatch from the full dataset
    for iteration_idx in range(n_iter-1):
     mbk=mbk.partial_fit(X)
     thread_1.update(mbk)
    thread_1.stop()
   else:   
   
    # Sample a minibatch from the full dataset
    for iteration_idx in range(n_iter-1):
     mbk=mbk.partial_fit(X)
     
  except IndexError:
   pass
  
  try:
   if sys.argv[2] == '-p' or sys.argv[2] == '-pp':
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

    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
     different += ((k_means_labels == k) != (mbk_means_labels == order[k]))
        
    identic = np.logical_not(different)

    ax.plot(X[identic, 0], X[identic, 1], 'w',markerfacecolor='#bbbbbb', marker='.')

    ax.plot(X[different, 0], X[different, 1], 'w',markerfacecolor='r', marker='.')
    ax.set_title('Difference')
    ax.set_xticks(())
    ax.set_yticks(())
     
    n_diff =len(X[different,])
    print('Clustering \'s difference: %d'%n_diff)
    ratio = n_diff/len(mbk_means_labels == 4)
    print('Difference \'s ratio: %f'%ratio)
    
    if (sys.argv[3] == '-f') and (sys.argv[4] != None):
     print('Saving...')
     plt.savefig(sys.argv[4])
     print('Done !')
    elif sys.argv[3] == '-f' and sys.argv[4] == None:
     print('Usage : -f PATH/NAME')
    plt.show()
   
  except IndexError:
   pass 






