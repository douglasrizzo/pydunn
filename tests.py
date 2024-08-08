import logging

import numpy as np
import validclust
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris

import pydunn

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(module)s(%(lineno)d)]: %(message)s]")
logger = logging.getLogger()


def test_toy() -> None:
  """Test Dunn index on toy example."""
  data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 14], [14, 10], [14, 14]])
  labels = [0, 0, 0, 0, 1, 1, 1, 1]
  distances = squareform(pdist(data))

  for cdist_method in pydunn.ClusterDistanceMethod:
    logger.info(
      "Distance method %s: %s",
      cdist_method.name,
      pydunn.inter_cluster_distances(labels, distances, cdist_method),
    )
  for diameter_method in pydunn.DiameterMethod:
    logger.info(
      "Diameter method %s: %s",
      diameter_method,
      pydunn.compute_cluster_diameters(labels, distances, diameter_method),
    )
  for diameter_method in pydunn.DiameterMethod:
    for cdist_method in pydunn.ClusterDistanceMethod:
      logger.info(
        "%s, %s: %s",
        diameter_method,
        cdist_method,
        pydunn.dunn(labels, distances, diameter_method, cdist_method),
      )


def test_iris() -> None:
  """Test Dunn index on Iris dataset, compare with existing implementation on PyPI."""
  data = load_iris()
  c = data["target"]
  x = data["data"]
  d = squareform(pdist(x))

  assert validclust.dunn(d, c) == pydunn.dunn(c, d)

  for diameter_method in pydunn.DiameterMethod:
    for cdist_method in pydunn.ClusterDistanceMethod:
      dunn_labels = pydunn.dunn(c, d, diameter_method, cdist_method)
      logger.info("%s, %s: %s", diameter_method, cdist_method, dunn_labels)


if __name__ == "__main__":
  test_toy()
  test_iris()
