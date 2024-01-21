import time
import os
import math

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS
from torch import cat
from torch.nn import (
    Module,
    Conv2d,
    Linear,
    Dropout2d,
    NLLLoss,
    MaxPool2d,
    Flatten,
    Sequential,
    ReLU,
    CrossEntropyLoss,
    MSELoss
)

# Importing standard Qiskit libraries and configuring account
import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import RYGate,HGate
from qiskit.circuit import Parameter
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.neural_networks import SamplerQNN,EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap

image_path= "../images/encode_images"
target_path= "../images/targets.npy"

def data_reader(image_path=image_path,target_path=target_path):
    data=[]
    targets=np.load(target_path,allow_pickle=True)
    # print(targets[0])

    for image_name in os.listdir(image_path):
        sequence=image_name.split(".")[0].split('_')[1]
        # print(sequence)
        d=[np.load("{}/{}".format(image_path,image_name)),targets[int(sequence)-1]]
        data.append(d)
    # print(data)
    return data

def counts2state(data,size=7):
    recover_list=[0 for i in range(size)]
    for d in data:
        encode=d[0][::-1]
        count=int(d[1])
        for bit in range(size):
            if encode[bit]=='0':
                recover_list[bit]+=count
    s=sum(recover_list)
    recover_list=[math.sqrt(x/s) for x in recover_list]
    recover_list=[math.acos(x)*2 for x in recover_list]
    recover_list=torch.tensor(recover_list)
    return torch.tensor(recover_list,dtype=torch.float64)

def ansatz_circuit(inputs,weights,q_number,c_number):
    qubits_number=q_number
    classical_nuber=c_number

    print(f"input parameters: {[str(item) for item in inputs.params]}")
    print(f"weight parameters: {[str(item) for item in weights.params]}")

    circuit = QuantumCircuit(qubits_number,classical_nuber)
    for i in range(qubits_number):
        circuit.rx(inputs[i],i)
        circuit.ry(weights[i], i)
    for i in range(qubits_number):
        if i==qubits_number-1:
            circuit.cx(i,0)
        else:
            circuit.cx(i,i+1)
    circuit.measure(6,0)
    # print(circuit)
    return circuit

def ansatz_instance(q_number,c_number):
    qubits_number=q_number
    classical_nuber=c_number
    inputs = ParameterVector("input", qubits_number)
    weights = ParameterVector("weight", qubits_number)
    circuit=ansatz_circuit(inputs,weights,qubits_number,classical_nuber)

    parity = lambda x: "{:b}".format(x).count("1") % 2
    sampler_qnn = SamplerQNN(circuit=circuit,
                             input_params=inputs,
                             weight_params=weights,
                             interpret=parity,
                             output_shape=2,
                             input_gradients=True
                             )
    # print(sampler_qnn)


    print(
        f"Number of input features for SamplerQNN: {sampler_qnn.num_inputs}")
    print(
        f"Number of trainable weights for SamplerQNN: {sampler_qnn.num_weights}")


    return sampler_qnn


class Net(Module):
    def __init__(self,qnn,initial_weights):
        super().__init__()
        self.qnn=TorchConnector(qnn,initial_weights=initial_weights)

    def forward(self,x):
        x=self.qnn(x)
        return x[0:1].detach().requires_grad_(True)

def run1(data):
    weights = np.random.randn(qubits_number) / np.sqrt(qubits_number)
    qnn = ansatz_instance(qubits_number, classical_number)
    print(qnn.circuit)

    model = Net(qnn, weights)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_func = nn.BCELoss()

    # Start training
    epochs = 10  # Set number of epochs
    loss_list = []  # Store loss history
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        print("epoch {} starts".format(epoch))
        start_time = time.time()
        for idx, d in enumerate(data):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            image = counts2state(d[0])
            target = torch.tensor(d[1], dtype=torch.float64)
            output = model(image)  # Forward pass
            # print(output,target)
            loss = loss_func(output, target)  # Calculate loss
            # print(loss)
            # for name,parms in model.named_parameters():
            #     print(name,parms,parms.requires_grad,parms.grad)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
        end_time = time.time()
        print("epoch {} spends {} s".format(epoch, end_time - start_time))

def run2(data):
    weights = np.random.randn(qubits_number) / np.sqrt(qubits_number)
    feature_map = ZZFeatureMap(7)
    ansatz = RealAmplitudes(7, reps=1)
    qc = QuantumCircuit(7)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

    class Net2(Module):
        def __init__(self, qnn):
            super().__init__()
            self.qnn = TorchConnector(qnn)  # Apply torch connector, weights chosen
            # uniformly at random from interval [-1,1].
            self.fc3 = Linear(1, 1)  # 1-dimensional output from QNN

        def forward(self, x):
            x = self.qnn(x)  # apply QNN
            x = self.fc3(x)
            return cat((x, 1 - x), -1)

    model = Net2(qnn)

    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    loss_func = NLLLoss()

    # Start training
    epochs = 10  # Set number of epochs
    loss_list = []  # Store loss history
    model.train()  # Set model to training mode

    for epoch in range(epochs):
        total_loss = []
        print("epoch {} starts".format(epoch))
        start_time = time.time()
        for idx, d in enumerate(data):
            optimizer.zero_grad(set_to_none=True)  # Initialize gradient
            image = torch.tensor(counts2state(d[0]),dtype=torch.float64)
            target = torch.tensor(d[1])
            output = model(image).resize(1,2) # Forward pass
            output=torch.tensor(output,dtype=torch.float64,requires_grad=True)
            # print(output,target)
            loss = loss_func(output, target)  # Calculate loss
            # print(loss)
            # for name,parms in model.named_parameters():
            #     print(name,parms,parms.requires_grad,parms.grad)
            loss.backward()  # Backward pass
            optimizer.step()  # Optimize weights
            total_loss.append(loss.item())  # Store loss
        loss_list.append(sum(total_loss) / len(total_loss))
        print("Training [{:.0f}%]\tLoss: {:.4f}".format(100.0 * (epoch + 1) / epochs, loss_list[-1]))
        end_time = time.time()
        print("epoch {} spends {} s".format(epoch, end_time - start_time))


if __name__ == '__main__':
    data=data_reader()
    qubits_number=7
    classical_number=1
    run1(data)
    # run2(data)

#training
    state=counts2state(data[0][0])
    target=data[0][1]
    input=state
#building part

    # net=Net(qnn,weights)
    # print(net(input))
    #
    # print(type(input),type(weights))
    # print(input,weights)
    # model = TorchConnector(qnn, initial_weights=weights)
    # r=model(input)
    # print(r)
    # weights = np.random.randn(qubits_number) / np.sqrt(qubits_number)
    # qnn = ansatz_instance(qubits_number, classical_number)
    # sampler_qnn_forward_batched = qnn.forward(input,weights)
    # print(
    #     f"Forward pass result for SamplerQNN: {sampler_qnn_forward_batched}.  \nShape: {sampler_qnn_forward_batched.shape}"
    # )
    # # print(type(sampler_qnn_forward_batched[0][0]),weights,input)
    #
    # sampler_qnn_input_grad, sampler_qnn_weight_grad = qnn.backward(
    #     input, weights
    # )
    # print(
    #     f"Input gradients for SamplerQNN: {sampler_qnn_input_grad}.  \nShape: {sampler_qnn_input_grad.shape}"
    # )
    # print(
    #     f"Weight gradients for SamplerQNN: {sampler_qnn_weight_grad}.  \nShape: {sampler_qnn_weight_grad.shape}"
    # )
    #

