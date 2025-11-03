import hnswlib
import numpy as np

# Create some dummy data
dim = 4
num_elements = 100

data = np.random.random((num_elements, dim)).astype(np.float32)

# Create your space (this is where your counter lives)
#space = hnswlib.L2Space(dim)
#space.reset_distance_computation_count()

# Create index
index = hnswlib.Index(space='l2', dim=dim)
index.init_index(max_elements=num_elements, ef_construction=10, M=16)
index.add_items(data)

print("After build:", index.get_distance_computation_count())

# Search for a single vector
query = data[0].reshape(1, -1)
labels, distances = index.knn_query(query, k=5)
time_ns = index.get_distance_time_ns()
time_ms = time_ns / 1e6
print("Distance computation time: {:.4f} ms".format(time_ms))
print(index.get_metric_hops())

print("After first search:", index.get_distance_computation_count())

# Search again
labels, distances = index.knn_query(query, k=5)
print("After second search:", index.get_distance_computation_count())

# Reset and verify reset works
index.reset_distance_computation_count()
print("After reset:", index.get_distance_computation_count())

