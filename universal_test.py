import numpy as np
from npy_append_array import NpyAppendArray
file_name=f"./data/model/FRQI_27-033923.npy"


data = np.load(file_name, mmap_mode="r")

print(data)