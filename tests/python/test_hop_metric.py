import hnswlib
import numpy as np

dim = 4
num_elements = 10

# Create space + index
p = hnswlib.Index(space='l2', dim=dim)
p.init_index(max_elements=num_elements, ef_construction=10, M=16)

# Add some data
data = np.float32(np.random.rand(num_elements, dim))
p.add_items(data)

# Do a query (this should invoke your modified search code)
labels, distances = p.knn_query(data[0], k=3)
print(p.get_metric_hops())
p.reset_metric_hops()
print(p.get_metric_hops())

print("Query result:", labels, distances)

