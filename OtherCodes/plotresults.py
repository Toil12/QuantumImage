from matplotlib import pyplot as plt
import numpy as np

file_name="FRQI_31-194237"
path1=f"data/loss_epoch/{file_name}_loss.npy"
# path1=f"data/loss_epoch/{name1}"
array=np.load(path1)
x=range(len(array))
# x=range(50)
t_acc=array[:,0]
loss=array[:,1]
v_acc=array[:,2]
fig=plt.figure()
ax=fig.add_subplot(111)

haed=0
if file_name.split('_')[0]=="angle":
    head="Angle"
elif file_name.split('_')[0]=="amplitude":
    head="Amplitude"
else:
    head=file_name.split('_')[0]

if file_name.split('_')[0]=="FRQI":
    x=range(50)
    t_acc = array[:50, 0]
    loss = array[:50, 1]
    v_acc = array[:50, 2]

ax.set_title(f"{head} Embedding")

lns1=ax.plot(x,t_acc,'-b',label='train accuracy')
lns2=ax.plot(x,v_acc,'-r',label='test accuracy')

ax2=ax.twinx()
lns3=ax2.plot(x,loss,':y',label='loss')

ax.set_xlabel('Epoch')
ax.set_ylabel(r"Accuracy Percentage")

ax2.set_ylabel('Loss')

ax.set_ylim(0,1)
ax2.set_ylim(0,0.6)

lns=lns1+lns2+lns3
labs = [l.get_label() for l in lns]
ax.legend(lns,labs,loc=0)
# ax2.legend(loc=0)
# plt.plot(x,t_acc)
# plt.plot(x,v_acc,color='r')
# plt.plot(x,loss,color='y')
plt.savefig(f'data/image/{file_name}.png')
plt.show()



