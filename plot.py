import matplotlib.pyplot as plt
import numpy as np
# loss1=[0.25811085945022455,
#        0.2553117220145149,
#        0.25266225530295955,
#        0.24949225580791207,
#        0.2447250246806588,
#        0.24434726520149083,
#        0.24115845883846507,
#        0.23698690477220338,
#        0.2338118519361571,
#        0.23002761210759676,
#        0.2309171793216781,
#        0.23017683166526223,
#        0.22861823497411968,
#        0.2256717333882834,
#        0.2229016189002682,
#        0.2185483603359516,
#        0.2186988903889111,
#        0.21812255708249836,
#        0.21324940304277473,
#        0.21370662697298823
#        ]
# loss2=[0.24701643166668485,
#        0.24557743566097529,
#        0.23692368420562265,
#        0.2350689592327689,
#        0.2319834170623185,
#        0.2256885941537883,
#        0.22410136195250727,
#        0.218766296393015,
#        0.21638153311788355,
#        0.21583500149437845,
#        0.21139940261342346,
#        0.20899010574853427,
#        0.20779289823497804,
#        0.20429789303215556,
#        0.20480494651032205,
#        0.20130097532611121,
#        0.2022761722014696]
# plt.ion()
# figure,ax=plt.subplots()
# lines=ax.plot([],[])
# ax.set_autoscale_on(True)
# ax.grid()
loss1=np.load(f"./data/loss_epoch/FRQI_17-103207_loss.npy")
# loss2=np.load(f"./data/FRQI_27-171924.npy")

# print(loss1)
# print(loss1_acc,loss1_val)
# for n in range(len(loss1)):
#       x=np.arange(len(loss1))
# y1=list(range(len(loss1_acc)))
# y2=list(range(len(loss1_val)))
# plt.plot(y1, loss1_acc, 'r')
# plt.plot(y2, loss1_val, 'b')
# plt.ylabel("accuracy")
# plt.xlabel("epoch")
# plt.show()
#
# plt.clf()

loss1_acc=loss1[:,1]
loss2_acc=loss2[:,1]
for n in range(len(loss1)):
      x=np.arange(len(loss1))
y1=list(range(len(loss1_acc)))
y2=list(range(len(loss2_acc)))
plt.plot(y1, loss1_acc, 'r')
plt.plot(y2, loss2_acc, 'b')
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

plt.clf()



