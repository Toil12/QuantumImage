from matplotlib import pyplot as plt
import numpy as np

file_name="angle_02-130937_loss.npy"
path1=f"data/loss_epoch/{file_name}"
# path1=f"data/loss_epoch/{name1}"

array=np.load(path1)

x=range(len(array))
t_acc=array[:,0]
loss=array[:,1]
v_acc=array[:,2]

plt.plot(x,t_acc)
plt.plot(x,v_acc,color='r')
# plt.plot(x,loss,color='y')
plt.show()

