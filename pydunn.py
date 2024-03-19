from enum import Enum

import numpy as np
from sklearn.preprocessing import LabelEncoder


class DiameterMethod(Enum):
  """Cluster diameter computation methods."""

  MEAN_CLUSTER = 1
  FARTHEST = 2


class ClusterDistanceMethod(Enum):
  """Inter cluster distance computation methods."""

  NEAREST = 1
  FARTHEST = 2


def inter_cluster_distances(
  labels: list[int], distances: np.ndarray, method: ClusterDistanceMethod = ClusterDistanceMethod.NEAREST
) -> np.ndarray:
  """Compute inter-cluster distances based on the given labels and distances using the specified method.

  Args:
    labels (list[int]): The cluster labels for each data point.
    distances (np.ndarray): The pairwise distances between data points.
    method (ClusterDistanceMethod, optional): The method to use for calculating inter-cluster distances. Defaults to
    ClusterDistanceMethod.NEAREST.

  Returns:
    np.ndarray: The inter-cluster distances matrix.
  """
  c_labels = np.unique(labels)
  n_clusters = len(c_labels)
  cluster_distances = np.full((n_clusters, n_clusters), float("inf") if method == ClusterDistanceMethod.NEAREST else 0)

  np.fill_diagonal(cluster_distances, 0)

  for i, c1 in enumerate(c_labels):
    for c2 in c_labels[i + 1 :]:
      if method == ClusterDistanceMethod.NEAREST:
        cluster_distances[c1, c2] = cluster_distances[c2, c1] = distances[labels == c1][:, labels == c2].min()
      else:
        cluster_distances[c1, c2] = cluster_distances[c2, c1] = distances[labels == c1][:, labels == c2].max()
  return cluster_distances


def compute_cluster_diameters(
  labels: list[int], distances: np.ndarray, method: DiameterMethod = DiameterMethod.FARTHEST
) -> dict[int, float]:
  """Compute cluster diameters based on the given labels, distances, and diameter computation method.

  Parameters:
    labels (list[int]): List of cluster labels
    distances (np.ndarray): 2D array of distances between data points
    method (DiameterMethod, optional): Method for computing cluster diameters, defaults to DiameterMethod.FARTHEST

  Returns:
    dict[int, float]: A dictionary containing the computed diameters for each cluster
  """
  labels = np.array(labels, dtype=int)
  if method == DiameterMethod.MEAN_CLUSTER:
    diameters = {c: distances[labels == c][:, labels == c].sum() for c in np.unique(labels)}
    for c in np.unique(labels):
      c_cize = sum(labels == c)
      diameters[c] /= c_cize * (c_cize - 1)

  elif method == DiameterMethod.FARTHEST:
    diameters = {c: distances[labels == c][:, labels == c].max() for c in np.unique(labels)}

  return diameters


def dunn(
  labels: list[int],
  distances: np.ndarray,
  diameter_method: DiameterMethod = DiameterMethod.FARTHEST,
  cdist_method: ClusterDistanceMethod = ClusterDistanceMethod.NEAREST,
) -> float:
  r"""Compute the Dunn index, the ratio of the minimum inter-cluster distance to the maximum cluster diameter.

  The index is defined as:

  .. math:: D = \min_{i = 1 \ldots n_c; j = i + 1\ldots n_c} \left\lbrace \frac{d \left( c_i,c_j \right)}{\max_{k = 1 \ldots n_c} \left(diam \left(c_k \right) \right)} \right\rbrace

  where :math:`d(c_i,c_j)` represents the distance between clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)`
  is the diameter of cluster :math:`c_k`.

  - Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their
  closest elements.
  - Cluster diameter can be defined as the mean distance between all elements in the cluster, between
  all elements to the cluster centroid, or as the distance between the two furthest elements.

  The higher the value of the resulting Dunn index, the better the clustering result is considered, since higher values
  indicate that clusters are compact (small :math:`diam(c_k)`) and far apart (large :math:`d \left( c_i,c_j \right)`).

  Parameters:
    labels (list[int]): The list of labels for each data point.
    distances (np.ndarray): The array of distances between data points.
    diameter_method (DiameterMethod, optional): The method to calculate the cluster diameter. Defaults to
    DiameterMethod.FARTHEST.
    cdist_method (ClusterDistanceMethod, optional): The method to calculate the inter-cluster distances. Defaults to
    ClusterDistanceMethod.NEAREST.

  Returns:
    float: The ratio of the minimum inter-cluster distance to the maximum cluster diameter.

  References:
    Dunn JC. Well-Separated Clusters and Optimal Fuzzy Partitions. Journal of Cybernetics. 1974 Jan;4(1):95-104.
  """
  labels = LabelEncoder().fit_transform(labels)
  ic_distances = inter_cluster_distances(labels, distances, cdist_method)
  min_distance = min(ic_distances[ic_distances.nonzero()])
  max_diameter = max(compute_cluster_diameters(labels, distances, diameter_method).values())
  return min_distance / max_diameter


if __name__ == "__main__":
  from sklearn.metrics.pairwise import euclidean_distances

  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 14], [14, 10], [14, 14]])
  labels = [0, 0, 0, 0, 1, 1, 1, 1]
  distances = euclidean_distances(data)

  print("#### Distances ####")
  for cdist_method in ClusterDistanceMethod:
    print(cdist_method, "\n", inter_cluster_distances(labels, distances, cdist_method))
  print("\n\n#### Diameters ####")
  for diameter_method in DiameterMethod:
    print(diameter_method, compute_cluster_diameters(labels, distances, diameter_method))

  print("\n\n#### Dunn ####")
  for diameter_method in DiameterMethod:
    for cdist_method in ClusterDistanceMethod:
      print(diameter_method, cdist_method, dunn(labels, distances, diameter_method, cdist_method))
