from matplotlib import pyplot as plt
import numpy as np
import math

file_name="CNN_15-193938"
path1=f"data/loss_epoch/{file_name}_loss.npy"
# path1=f"data/loss_epoch/{name1}"

array=np.load(path1)

x=range(len(array))
# x=range(50)
t_acc=array[:,0]
loss=array[:,1]
v_acc=array[:,2]
print(v_acc)

for id,data in enumerate(t_acc):
    avg=0
    gap=3
    for i in range(id+1,id+4):
        avg=avg+(t_acc[i]/gap)
    print(avg)
    if (abs(avg-t_acc[id])/t_acc[id])<=0.05 and t_acc[id]>=0.87:
        print(id+1)
        break
    else:
        continue