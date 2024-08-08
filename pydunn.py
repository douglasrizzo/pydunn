from enum import Enum

import numpy as np


class DiameterMethod(Enum):
  """Cluster diameter computation methods."""

  MEAN_CLUSTER = 1
  FARTHEST = 2


class ClusterDistanceMethod(Enum):
  """Inter cluster distance computation methods."""

  NEAREST = 1
  FARTHEST = 2


def inter_cluster_distances(
  labels: list[int],
  distances: np.ndarray,
  method: ClusterDistanceMethod = ClusterDistanceMethod.NEAREST,
) -> np.ndarray:
  """Compute inter-cluster distances based on the given labels and distances using the specified method.

  Parameters
  ----------
  labels : list[int]
      The cluster labels for each data point.
  distances : np.ndarray
      The pairwise distances between data points.
  method : ClusterDistanceMethod, optional
      The method to use for calculating inter-cluster distances. Defaults to
      ClusterDistanceMethod.NEAREST.

  Returns
  -------
  np.ndarray
      The inter-cluster distances matrix, a symmetric matrix.
  """
  __validate_distance_matrix(distances)
  labels = np.array(labels, dtype=int)
  c_labels = np.unique(labels)
  n_clusters = len(c_labels)

  # create matrix of cluster distances, it is convenient to fill it with infinity values when working with nearest
  # cluster distance method
  cluster_distances = np.full(
    (n_clusters, n_clusters),
    float("inf") if method == ClusterDistanceMethod.NEAREST else 0,
  )
  np.fill_diagonal(cluster_distances, 0)

  cluster_pairs = ((c1, c2) for i, c1 in enumerate(c_labels) for c2 in c_labels[i + 1 :])
  # distances[labels == c1] returns the distance between all data points in cluster c1 and all other data points
  # distances[labels == c1][:, labels == c2] returns the distances between data points in clusters c1 and c2
  for c1, c2 in cluster_pairs:
    # for nearest cluster distance method, get the min(), for farthest cluster distance method get the max()
    c_dist = (
      distances[labels == c1][:, labels == c2].min()
      if method == ClusterDistanceMethod.NEAREST
      else distances[labels == c1][:, labels == c2].max()
    )
    cluster_distances[c1, c2] = cluster_distances[c2, c1] = c_dist
  return cluster_distances


def compute_cluster_diameters(
  labels: list[int],
  distances: np.ndarray,
  method: DiameterMethod = DiameterMethod.FARTHEST,
) -> dict[int, float]:
  """Compute cluster diameters based on the given labels, distances, and diameter computation method.

  Parameters
  ----------
  labels : list[int]
      List of cluster labels
  distances : np.ndarray
      Array of distances between data points
  method : DiameterMethod, optional
      Method for computing cluster diameters, default is DiameterMethod.FARTHEST

  Returns
  -------
  dict[int, float]
      Dictionary containing the computed diameters for each cluster.
  """
  __validate_distance_matrix(distances)
  # convert cluster labels to numpy array to use it as a boolean mask, which does not work with lists
  labels = np.array(labels, dtype=int)
  if method == DiameterMethod.MEAN_CLUSTER:
    # for mean cluster diameter method, sum the distances between all data points in the same cluster
    # and divide it by the number of possible pairs data points in the cluster
    diameters = {c: distances[labels == c][:, labels == c].sum() for c in np.unique(labels)}
    for c in np.unique(labels):
      c_cize = sum(labels == c)
      # because we are summing the full symmetric matrix, we need to divide by n*(n-1) and not (n*(n-1))/2
      diameters[c] /= c_cize * (c_cize - 1)

  # for farthest cluster diameter method, get the maximum distance between all data points in the same cluster
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

  .. math::

     D = \min_{i = 1 \ldots n_c; j = i + 1\ldots n_c} \left\lbrace \frac{d \left( c_i,c_j \right)}
      {\max_{k = 1 \ldots n_c} \left(diam \left(c_k \right) \right)} \right\rbrace

  where :math:`d(c_i,c_j)` represents the distance between clusters :math:`c_i` and :math:`c_j`, and :math:`diam(c_k)`
  is the diameter of cluster :math:`c_k`.

  - Inter-cluster distance can be defined in many ways, such as the distance between cluster centroids or between their
  closest elements.
  - Cluster diameter can be defined as the mean distance between all elements in the cluster, between
  all elements to the cluster centroid, or as the distance between the two furthest elements.

  The higher the value of the resulting Dunn index, the better the clustering result is considered, since higher values
  indicate that clusters are compact (small :math:`diam(c_k)`) and far apart (large :math:`d \left( c_i,c_j \right)`).

  Parameters
  ----------
  labels : list[int]
      The list of labels for each data point.
  distances : np.ndarray
      The array of distances between data points.
  diameter_method : DiameterMethod, optional
      The method to calculate the cluster diameter. Defaults to DiameterMethod.FARTHEST.
  cdist_method : ClusterDistanceMethod, optional
      The method to calculate the inter-cluster distances. Defaults to ClusterDistanceMethod.NEAREST.

  Returns
  -------
  float
      The ratio of the minimum inter-cluster distance to the maximum cluster diameter.

  References
  ----------
  Dunn JC. Well-Separated Clusters and Optimal Fuzzy Partitions.
  Journal of Cybernetics. 1974 Jan;4(1):95-104.
  """
  __validate_distance_matrix(distances)
  # encode labels as integers starting from 0
  label_map = {old_label: new_label for new_label, old_label in enumerate(set(labels))}
  labels = [label_map[old_label] for old_label in labels]

  # get the minimum inter-cluster distance and the maximum cluster diameter
  ic_distances = inter_cluster_distances(labels, distances, cdist_method)
  min_distance = min(ic_distances[ic_distances.nonzero()])
  max_diameter = max(compute_cluster_diameters(labels, distances, diameter_method).values())

  # compute and return the Dunn index
  return min_distance / max_diameter


def __validate_distance_matrix(distances: np.ndarray) -> None:
  """Validate a distance matrix.

  Parameters
  ----------
  distances : ndarray
      The matrix of distances to be validated.

  Raises
  ------
  AssertionError
      If the distance matrix is not 2-dimensional, not square, or not symmetric.
  """
  assert distances.ndim == 2, "Distance matrix must be 2-dimensional."  # noqa: PLR2004
  assert distances.shape[0] == distances.shape[1], "Distance matrix must be square."
  assert np.allclose(distances, distances.T, rtol=1e-05, atol=1e-08), "Distance matrix must be symmetric."
