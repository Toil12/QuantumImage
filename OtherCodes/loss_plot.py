import matplotlib.pyplot as plt


x_data = ('Angle', 'Amplitude', 'FRQI', 'CNN')
train_acc = [0.985,0.97,0.96,1]
test_acc = [1,1,0.92,1]
loss=[0.062,0.09,0.19,0.00001]
colors = ['r','b','orange','g']
# plot
fig, ax = plt.subplots()
ax.set_title("Classification Loss Performance (Best)")

bar=ax.bar(x_data,loss,color=colors)
ax.set_ylabel("Loss")
ax.set_xlabel("Embedding Method")
ax.set_ylim(0,0.2)
ax.bar_label(bar)

plt.show()
fig.savefig(f'data/comparision/loss_per.png')
