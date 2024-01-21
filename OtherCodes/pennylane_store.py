import time
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
import matplotlib.pyplot as plt
from math import pi
import torch
from torchvision import datasets, transforms
import pennylane as qml

dev = qml.device("default.qubit", wires=7,shots=2**20)

def encode_circuit(qubits_num, section_number, x):
    # print(qubits_num,section_number,parameters)
    wires=[w for w in range(qubits_num)]
    # set H gates
    for wire in wires[0:-1]:
        qml.Hadamard(wire)
    # for each pixel set circuit
    for i in range(section_number):
        # print("here is inside the circuit",i,x[i])
        a=x[i]
        # print(i,a)
        i_b=str(np.binary_repr(i,width=qubits_num-1))[::-1]
        for pos,bit in enumerate(i_b):
            if bit=='0':
                qml.PauliX(pos)
        qml.ctrl(qml.RY, wires[0:-1], control_values=[1]*(qubits_num-1))(2*a, wires=qubits_num-1)
        for pos, bit in enumerate(i_b):
            if bit == '0':
                qml.PauliX(pos)
        qml.Barrier(wires)
    # print("encode done")

def ansatz_layer(W,qubits_num:int=7):
    # print("weights inside the circuit",W)
    for i in range(qubits_num):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
        if i !=qubits_num-1:
            qml.CNOT(wires=[i, i+1])
        else:
            qml.CNOT(wires=[i,0])

@qml.qnode(dev, interface='autograd')
def circuit(weights,x,qubits_n=7,section_n=64):
    # print("get w as ",weights)
    # print("input image in circuit is",x)
    encode_circuit(qubits_num=qubits_n,
                   section_number=section_n,
                   x=x)
    for W in weights:
        ansatz_layer(W)
    result=[qml.expval(qml.PauliZ(i)) for i in range(qubits_n)]
    return result

def variational_classifier(weights, bias, x):
    parity_result=circuit(weights,x)
    result=0
    for i in range(len(parity_result)):
        bit=parity_result[i]
        if i==0:
            if bit<0:
                result=0
            else:
                result=1
        if bit <0:
            result=result^0
        elif bit>0:
            result=result^1
    r=-1 if result==0 else 1
    return r+bias
    # return circuit(weights, x) + bias

def square_loss(labels, predictions):
    # print("labels and predicitons are ", labels, predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):
    # print("labels and predicitons in acc are ", labels, predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

def cost(weights, bias, X, Y):
    # print(X,Y)
    predictions = [variational_classifier(weights, bias, X)]
    # print("pre in cost is",predictions)
    return square_loss(Y, predictions)

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
        encode = item[0]
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
        print(image)
        plt.savefig('images/q.jpg')
        plt.show()

def show_recoverd_image(file_name):
    r=np.load(file_name)
    print(r)
    recover_image(r)

def image_preprocessing(data:np.ndarray,image_size:int=64):
    image = data
    # print(image)
    # fill circuit with angles
    # print(circ.parameters)
    test_number = 1
    for i in range(test_number):
        # set angles ans parameters
        image_sequence = image.ravel()
        # pre processing
        for pos in range(len(image_sequence)):
            image_sequence[pos] = image_sequence[pos] * pi / 2
        # print("complete image is",image_sequence)
        return image_sequence


if __name__ == '__main__':
    # print(encode_circuit(7,64,math.pi))
    # print(qml.draw(encode_circuit)(7,64,"x"))
    sample_number=100
    train_loader, test_loader,image_size=get_images(n_samples=sample_number)
    train_images_q=np.empty([0],dtype=bool)
    train_loader_q=[]

    count=1
    targets=[]
    start_time=time.time()
    # initialize weights and bias
    weights_init = 0.01 * np.random.randn(1, 7, 3, requires_grad=True)
    bias_init = np.array(0.01, requires_grad=True)
    opt = NesterovMomentumOptimizer(0.5)
    weights = weights_init
    bias = bias_init
    #start training
    for epoch in range(100):
        for batch_idx, (data, target) in enumerate(train_loader):
            targets.append(target)
            image=data[0][0]

            if target[0]==0:
                target[0]=-1
            # print("target is ", target[0])
            # print(targets)
    #image visualization
            # plt.imshow(image,cmap='gray')
            # plt.savefig('images/original.jpg')
            # plt.show()
    #quantum encode
            encode_image=image_preprocessing(image)
            image_name=str(batch_idx)+"_"+str(count)
            # np.save("images/encode_images/{}.npy".format(image_name),encode_image)
            print(" store image {}, {} left".format(count, sample_number - count))
            count+=1
            # print(encode_image)
            # recover_image(encode_image)
            # feeding data

            weights, bias, _, _ = opt.step(cost, weights, bias, encode_image, target)
            predictions = [np.sign(variational_classifier(weights, bias, encode_image))]
            print("prediction is ", predictions)
            acc = accuracy(target, predictions)
            print("one train finish")
            print(
                "Image: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                    epoch + 1, cost(weights, bias, encode_image, target), acc
                )
            )
            break
    #
    #     np.save("images/targets.npy",targets)
    #     end_time=time.time()
    #     print("spend {} s".format(end_time-start_time))
    #     # r=np.load("images/encode_images/0_0.npy")
    #     # print(r)
    # #     show_recoverd_image("images/encode_images/0_1.npy")



