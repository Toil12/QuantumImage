import math
import time

import numpy as np
import matplotlib.pyplot as plt
from math import pi

import torch
from torchvision import datasets, transforms
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import pennylane as qml

dev = qml.device("default.qubit", wires=7,shots=2**20)

def Net():
    pass
@qml.qnode(dev, interface='torch')
def encode_circuit(qubits_num,section_number,parameters):
    # print(qubits_num,section_number,parameters)
    wires=[x for x in range(qubits_num)]
    # set H gates
    for wire in wires[0:-1]:
        qml.Hadamard(wire)
    # for each pixel set circuit
    for i in range(section_number):
        a=parameters[i]
        # print(a)
        i_b=str(np.binary_repr(i,width=qubits_num-1))[::-1]
        for pos,bit in enumerate(i_b):
            if bit=='0':
                qml.PauliX(pos)
        qml.ctrl(qml.RY, wires[0:-1], control_values=[1]*(qubits_num-1))(a*2, wires=qubits_num-1)
        for pos, bit in enumerate(i_b):
            if bit == '0':
                qml.PauliX(pos)
        qml.Barrier(wires)
    return qml.counts()
    # print(circuit)

def get_images(n_samples:int=100,r:int=8,c:int=8):
    # Concentrating on the first 100 samples
    n_samples = n_samples
    rows=r
    cols=c
# training data
    X_train = datasets.MNIST(root='./data', train=True, download=True,
                             transform=transforms.Compose([transforms.Grayscale(),
                                                           transforms.ToTensor(),
                                                           # transforms.Normalize((0.1307,), (0.3081,)),
                                                           transforms.Resize([rows, cols])]))

    # Leaving only labels 0 and 1
    idx = np.append(np.where(X_train.targets == 0)[0][:n_samples],
                    np.where(X_train.targets == 1)[0][:n_samples])

    X_train.data = X_train.data[idx]
    X_train.targets = X_train.targets[idx]

    train_loader = torch.utils.data.DataLoader(X_train, batch_size=1 , shuffle=False)

# test data
    n_samples = 50

    X_test = datasets.MNIST(root='./data', train=False, download=True,
                            transform=transforms.Compose([transforms.Grayscale(),
                                                          transforms.ToTensor()]))

    idx = np.append(np.where(X_test.targets == 0)[0][:n_samples],
                    np.where(X_test.targets == 1)[0][:n_samples])

    X_test.data = X_test.data[idx]
    X_test.targets = X_test.targets[idx]

    test_loader = torch.utils.data.DataLoader(X_test, batch_size=1, shuffle=True)
    n_samples_show = 6
    return train_loader,test_loader,rows*cols

def parameter_name(id):
    return "piexls_{}".format(str(id))

def recover_image(quantam_encode:list,show=True):
    image = quantam_encode
    recover_dict = {}
    for i in range(2 ** 6):
        recover_dict[i] = [0, 0]

    piexls_counts = 0
    for item in image.items():
        print(item)

        encode = item[0][::-1]
        count = int(item[1])
        pos = int(encode[1:],2)
        color = int(encode[0])
        piexls_counts += 1
        # print(pos,color)
        recover_dict[pos][color] += count

    image = np.zeros([8, 8])
    for key, val in recover_dict.items():
        raw = key // 8
        column = key % 8
        color_gray = val[1] / max((sum(val)),1)
        image[raw][column] = color_gray
    image=torch.tensor(image)
    if show:
        plt.imshow(image, cmap='gray')
        print("now show recovered image")
        print(image)
        plt.savefig('images/q.jpg')
        plt.show()

def show_recoverd_image(file_name):
    r=np.load(file_name)
    print(r)
    recover_image(r)

def quantum_image_encode(data:np.ndarray,image_size:int=64):
    image=data
    # print(image)
# fill circuit with angles
    count = 0
    # print(circ.parameters)
    test_number = 1
    for i in range(test_number):
        time_start = time.time()
# set angles ans parameters
        image_sequence=image.ravel()
        # print(image_sequence)
# pre processing
        for pos in range(len(image_sequence)):
            image_sequence[pos]=image_sequence[pos]*pi/2
        print("complete image is",image_sequence)
        q_counts = encode_circuit(qubits_num=7,
                                  section_number=image_size,
                                  parameters=image_sequence)
        time_end = time.time()
        spend_time = time_end - time_start
        # output=sorted(q_counts.items(), key=lambda x:x[1])
        print("running parameterized test{} bind paramaeters spends {} s ".format(i, spend_time))
        # print(output)

        return q_counts


        # count += spend_time
    # print("average reused time is approximately {} to the original time".format(count / test_number))

if __name__ == '__main__':
    # print(encode_circuit(7,64,math.pi))
    # print(qml.draw(encode_circuit)(7,64,"x"))
    sample_number=200
    train_loader, test_loader,image_size=get_images(n_samples=sample_number)
    train_images_q=np.empty([0],dtype=bool)
    train_loader_q=[]

    count=1
    targets=[]
    start_time=time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        targets.append(target)
        image=data[0][0]
#image visualization
        # plt.imshow(image,cmap='gray')
        # plt.savefig('images/original.jpg')
        # plt.show()
#quantum encode
        encode_image=quantum_image_encode(data=image,
                                          image_size=image_size)
        image_name=str(batch_idx)+"_"+str(count)
        # np.save("images/encode_images/{}.npy".format(image_name),encode_image)
        print(" store image {}, {} left".format(count, sample_number - count))
        count+=1
        print(encode_image)
        recover_image(encode_image)
        break
#
#     np.save("images/targets.npy",targets)
#     end_time=time.time()
#     print("spend {} s".format(end_time-start_time))
#     # r=np.load("images/encode_images/0_0.npy")
#     # print(r)
# #     show_recoverd_image("images/encode_images/0_1.npy")



