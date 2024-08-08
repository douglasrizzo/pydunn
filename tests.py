import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics.pairwise import euclidean_distances

import pydunn


def test_toy() -> None:
  """Test Dunn index on toy example."""
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 14], [14, 10], [14, 14]])
  labels = [0, 0, 0, 0, 1, 1, 1, 1]
  distances = euclidean_distances(data)

  print("#### Distances ####")
  for cdist_method in pydunn.ClusterDistanceMethod:
    print(cdist_method, "\n", pydunn.inter_cluster_distances(labels, distances, cdist_method))
  print("\n#### Diameters ####")
  for diameter_method in pydunn.DiameterMethod:
    print(diameter_method, pydunn.compute_cluster_diameters(labels, distances, diameter_method))

  print("\n\n#### Dunn ####")
  for diameter_method in pydunn.DiameterMethod:
    for cdist_method in pydunn.ClusterDistanceMethod:
      print(diameter_method, cdist_method, pydunn.dunn(labels, distances, diameter_method, cdist_method))


def test_iris() -> None:
  """Test Dunn index on Iris dataset."""
  data = load_iris()
  kmeans = KMeans(n_clusters=3)
  c = data["target"]
  x = data["data"]
  k = kmeans.fit_predict(x)
  d = euclidean_distances(x)

  for diameter_method in pydunn.DiameterMethod:
    for cdist_method in pydunn.ClusterDistanceMethod:
      dund = pydunn.dunn(c, d, diameter_method, cdist_method)
      dunk = pydunn.dunn(k, d, diameter_method, cdist_method)
      print(diameter_method, cdist_method, dund, dunk)
