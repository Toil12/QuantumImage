import matplotlib.pyplot as plt


x_data = ('Angle', 'Amplitude', 'FRQI', 'CNN')
train_acc = [0.985,0.97,0.96,1]
test_acc = [1,1,0.92,1]
loss=[0.062,0.09,0.19,0.00001]

# plot
fig, ax = plt.subplots()

ax.bar(x_data,loss,color='orange')
ax.bar_label(ax.containers[0], label_type='edge')
ax.set_ylabel("Loss Value")
ax.set_ylim(0,0.2)
ax.set_title("Classification Performance")
# ax2.set_zorder(0)
#
ax2=ax.twinx()
lns1 = ax2.plot(x_data, train_acc, '-', label = 'train acc',color='b')
lns2 = ax2.plot(x_data, test_acc, '-', label = 'test acc',color='r')
lns = lns1+lns2
labs = [l.get_label() for l in lns]
ax2.legend(lns, labs, loc='center left')
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0.9, 1.01)
# ax.set_zorder(1)
#
plt.show()
fig.savefig("performance.png")