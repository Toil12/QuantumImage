import numpy
from matplotlib import pyplot as plt
import numpy as np
import math

file_name="CNN_15-193938"
path1=f"data/loss_epoch/angle_14-155241_loss.npy"
path2=f"data/loss_epoch/amplitude_03-205916_loss.npy"
path3=f"data/loss_epoch/FRQI_31-194237_loss.npy"
path4=f"data/loss_epoch/CNN_15-193938_loss.npy"
# path1=f"data/loss_epoch/{name1}"

array1=np.load(path1)[:,1]
array2=np.load(path2)[:,1]
array3=np.load(path3)[:,1]
array4=np.load(path4)[:,1]

# t_acc=array[:,0]
# loss=array[:,1]
# v_acc=array[:,2]
D=[array1[13:-1],array2[3:-1],array3[10:-1],array4[1:-1]]


# plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()
ax.violinplot(D,
                  showmeans=False,
                  showmedians=True)

ax.yaxis.grid(True)
ax.set_xticks([y + 1 for y in range(len(D))],
              labels=['Angle','Amplitude','FRQI','CNN'])
ax.set_ylabel('Loss value')
ax.set_title('Loss Distribution')
fig.savefig('lossDistribution.png')
plt.show()