import numpy as np
from npy_append_array import NpyAppendArray
file_name=f"./data/loss_epoch/test.npy"

arr1 = np.array([[1, 2]])
arr2 = np.array([[3, 4]])

with NpyAppendArray(file_name) as npaa:
    npaa.append(arr1)
    npaa.append(arr2)
    npaa.append(arr2)

data = np.load(file_name, mmap_mode="r")

print(data)