import matplotlib.pyplot as plt

data = [13,3,10]
labels=["Angle","Amplitude","FRQI"]
color=['red','blue','orange']
file_name="convergence"

fig, ax = plt.subplots()
bar_container = ax.bar(labels, data,color=color)
ax.set(ylabel='Epochs', title='Convergence Performance')
ax.bar_label(bar_container)


plt.savefig(f'data/image/{file_name}.png')
plt.show()