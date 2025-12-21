# Created by Micah
# Date: 12/15/25
# Time: 7:17 AM
# Project: NumpyNetwork
# File: Transform.py

import numpy as np
import numpy.random
import pandas as pd
import matplotlib.pyplot as plt

# draw 100 random weights
w = np.random.randn(10000)

print("mean:", w.mean())
print("std :", w.std())

# histogram
plt.hist(w, bins=50, density=True)
plt.xlabel("weight value")
plt.ylabel("density")
plt.title("Histogram of 100 samples from N(0,1)")
plt.show()

def pixels_to_vectors(i: int):
    pass

def make_shear_matricies():
    pass

def apply_shear_matricies():
    pass

def vectors_to_pixels(i: int):
    pass





for i in range(8,18,1):
    print(i/2)
