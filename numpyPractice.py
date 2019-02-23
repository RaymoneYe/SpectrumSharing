import numpy as np
import math
import matplotlib.pyplot as plt

b = np.array([[1, 2]])
print(b.shape)
b = np.expand_dims(b, axis=-1)
print(b.shape)

