from sklearn.datasets.samples_generator import make_blobs
class datagen():


  def __init__(self):

    self.n_samples=100000
    self.cluster_std=0.7
    self.batch_size = 750
    self.centers = [[2, 2], [-2, -2], [2, -2]]
    self.n_clusters = len(self.centers)
   
    
  
  def genesis(self):
    #Generate sample data
    self.X, self.labels_true = make_blobs(n_samples=self.n_samples, centers=self.centers, cluster_std=self.cluster_std)
    print(self.X.shape)
    print(self.labels_true)
    return self.X, self.labels_true, self.batch_size, self.n_samples, self.n_clusters
    
  def get(self):
    return self.X, self.labels_true, self.batch_size, self.n_samples, self.n_clusters
  
