from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from npy_append_array import NpyAppendArray
from math import pi
from torchvision import datasets, transforms


import argparse
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import time
import logging

# file_name=time.strftime("%d-%H%M%S", time.localtime())
device="lightning.qubit"
dev_7 = qml.device(device, wires=7, shots=1000)
from npy_append_array import NpyAppendArray

@qml.qnode(dev_7, diff_method='parameter-shift')
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
    return qml.probs()


if __name__ == '__main__':
    x=np.random.randn(64,)
    specs_func=qml.specs(encode_circuit)
    print(specs_func(7,64,x))

    qml.drawer.use_style("solarized_light")
    fig, ax = qml.draw_mpl(encode_circuit)(7,8,x)
    fig.savefig("frqi.png")
