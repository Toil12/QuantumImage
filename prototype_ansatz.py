import math
import time
import random

import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer

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
    encode_circuit(qubits_num=qubits_n,
                   section_number=section_n,
                   x=x)
    for W in weights:
        ansatz_layer(W)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    # print("labels and predicitons are ", labels, predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2

    loss = loss / len(labels)
    return loss

def accuracy(labels, predictions):
    print("labels and predicitons in acc are ", labels, predictions)
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)

    return loss

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, X)]
    print("pre in cost is",predictions)
    return square_loss(Y, predictions)


if __name__ == '__main__':
    X=[[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 1.2358, 1.2312, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000, 1.4453, 1.3968, 0.8751, 0.0000, 0.0000, 0.0000, 0.0000, 1.0152,
       0.4778, 0.0000, 0.0000, 0.8662, 0.0000, 0.0000, 0.0000, 1.2778, 0.0000,
       0.0000, 0.1586, 0.5267, 0.0000, 0.0000, 0.0000, 0.9471, 0.0000, 0.6884,
       0.4839, 0.0000, 0.0000, 0.0000, 0.0000, 1.4091, 1.4241, 0.0000, 0.0000,
       0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
       0.0000]]
    X=np.asarray(X,requires_grad=False)
    Y=[[0]]
    for i in range(len(Y[0])):
        if Y[0][i]==0:
            Y [0][i]= -1
    Y = np.asarray(Y, requires_grad=False)
    # specs_func = qml.specs(encode_circuit)
    # print(qml.draw(encode_circuit)(7,64,test))
    # print(encode_circuit(7,64,test))
    # print(specs_func(7,64,test))

#     print(qml.draw(circuit)(weights[0], X))
#     print(circuit(weights[0], X))
# #

    print("X = {}, Y = {}".format(X, Y))
    print("...")

    weights_init = 0.01 * np.random.randn(1, 7, 3, requires_grad=True)
    bias_init = np.array(0.0, requires_grad=True)

    opt = NesterovMomentumOptimizer(0.5)
    batch_size = 5

    weights = weights_init
    bias = bias_init
    for it in range(25):

        # Update the weights by one optimizer step
        # batch_index = np.random.randint(0, len(X), (batch_size,))
        # print(batch_index)
        # X_batch = X[batch_index]
        # Y_batch = Y[batch_index]
        # print(X,Y)
        weights, bias, _, _ = opt.step(cost, weights, bias, X[0], Y[0])

        # Compute accuracy
        predictions = [np.sign(variational_classifier(weights, bias, X[0]))]
        print("prediction is ",predictions)
        acc = accuracy(Y[0], predictions)

        print(
            "Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(
                it + 1, cost(weights, bias, X[0], Y[0]), acc
            )
        )