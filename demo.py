import numpy as np
a = np.random.rand(3, 4, 5)
b = np.moveaxis(a, 2, 0)
print(b.shape)  # (4, 5, 3)