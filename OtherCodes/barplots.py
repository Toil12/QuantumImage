import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y_data = [ 14,4,11,2 ]
x_data = ('Angle', 'Amplitude', 'FRQI', 'CNN')

# 柱状图颜色
colors = ['r','b','orange','g']

# 柱状图
bar = plt.bar(x_data, y_data, 0.5, color=colors)

# 设置标题
ax.set_title('Convergence Performance',fontsize=14,y=1.05)
# 设置坐标轴标题
ax.set_ylabel("Epochs",rotation=90)
ax.set_xlabel("Embedding Method")


plt.bar_label(bar, label_type='edge')
plt.show()
fig.savefig(f'data/comparision/convergence.png')

