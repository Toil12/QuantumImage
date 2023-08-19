from npy_append_array import NpyAppendArray
import numpy as np
file_name="test_npyy.npy"
# path1=f"data/loss_epoch/{name1}"

array1=np.array([[1,2]])
array2=np.array([[5,6]])

npaa=NpyAppendArray(f"./data/loss_epoch/{file_name}")


npaa.append(array1)
npaa.append(array2)

npaa.close()

