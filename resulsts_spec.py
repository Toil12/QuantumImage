from matplotlib import pyplot as plt
import numpy as np

file_name="CNN_15-193938"
path1=f"data/loss_epoch/{file_name}_loss.npy"
# path1=f"data/loss_epoch/{name1}"

array=np.load(path1)

x=range(len(array))
# x=range(50)
t_acc=array[:,0]
loss=array[:,1]
v_acc=array[:,2]

print(f'The largest train acc is {max(t_acc)}')
print(f'The largest test acc is {max(v_acc)}')
print(f'The lowest loss is {min(loss)}')