import matplotlib.pyplot as plt
import numpy as np


x_data = ('Angle', 'Amplitude', 'FRQI', 'CNN')
train_acc = [0.985,0.97,0.96,1]
test_acc = [1,1,0.92,1]
loss=[0.062,0.09,0.19,0.00001]

# plot
fig, ax = plt.subplots()

x=np.arange(len(x_data))
width=0.35
rest1 = ax.bar(x-width/2, train_acc, width,label = 'train accuracy')
rest2 = ax.bar(x+width/2, test_acc, width,label = 'test accuracy')




ax.set_title("Classification Accuracy Performance (Best)")
ax.set_ylabel("Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(x_data)
ax.set_xlabel("Embedding Method")
ax.set_ylim(0.9, 1.01)
ax.legend(loc='lower left')

plt.bar_label(rest1, label_type='edge')
plt.bar_label(rest2, label_type='edge')

plt.show()
fig.savefig(f'data/comparision/accuracy_per.png')
