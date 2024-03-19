# Dunn index in Python

An implementation of the Dunn index in Python. It sat for ages as a GitHub Gist page, but now I've transferred it to a proper repo so people can report issues more easily.

```py
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pydunn import dunn

# data points and labels
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 14], [14, 10], [14, 14]])
labels = [0, 0, 0, 0, 1, 1, 1, 1]
distances = euclidean_distances(data)

# compute the Dunn index
print("\n\n#### Dunn ####")
for diameter_method in DiameterMethod:
for cdist_method in ClusterDistanceMethod:
    dunn_index = dunn(labels, distances, diameter_method, cdist_method)
    print(diameter_method, cdist_method, dunn_index)
```

As a bonus, you can also compute inter-cluster distances and cluster diameters separately.

```py
from pydunn import inter_cluster_distances, compute_cluster_diameters

# compute inter-cluster distances
print("#### Distances ####")
for cdist_method in ClusterDistanceMethod:
    print(cdist_method, "\n", inter_cluster_distances(labels, distances, cdist_method))

# compute cluster diameters
print("\n\n#### Diameters ####")
for diameter_method in DiameterMethod:
    print(diameter_method, compute_cluster_diameters(labels, distances, diameter_method))
```
