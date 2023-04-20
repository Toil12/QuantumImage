import math
import time
import random

import numpy as np
import pennylane as qml
from qiskit import Aer
import matplotlib.pyplot as plt
from math import pi
dev = qml.device("default.qubit", wires=7,shots=2**20)

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
    print("done")
    return qml.expval(qml.PauliZ(0))
if __name__ == '__main__':
    test=[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 1.2358, 1.2312, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000, 1.4453, 1.3968, 0.8751, 0.0000, 0.0000, 0.0000, 0.0000, 1.0152,
        0.4778, 0.0000, 0.0000, 0.8662, 0.0000, 0.0000, 0.0000, 1.2778, 0.0000,
        0.0000, 0.1586, 0.5267, 0.0000, 0.0000, 0.0000, 0.9471, 0.0000, 0.6884,
        0.4839, 0.0000, 0.0000, 0.0000, 0.0000, 1.4091, 1.4241, 0.0000, 0.0000,
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
        0.0000]
    # for i in range(64):
    #     test.append(random.random())
    specs_func = qml.specs(encode_circuit)
    print(qml.draw(encode_circuit)(7,64,test))
    print(encode_circuit(7,64,test))
    print(specs_func(7,64,test))
