import time

import numpy as np
import matplotlib.pyplot as plt
from math import pi
from IPython.display import clear_output
# import torch libray
import torch
from torchvision import datasets, transforms
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
# Importing standard Qiskit libraries and configuring account
import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit.circuit.library.standard_gates import RYGate
from qiskit import transpile, assemble
from qiskit import QuantumCircuit,QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library.standard_gates import RYGate,HGate
from qiskit.circuit import Parameter

from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B

def Net():
    pass

def encode_circuit(circuit, registers, section_number,parameters):
    circuit.h(registers[0:-1])

    for i in range(section_number):
        a=parameters[i]
        i_b=str(np.binary_repr(i,width=registers.size-1))[::-1]
        for pos,bit in enumerate(i_b):
            if bit=='0':
                circuit.x(registers[pos])

        CCRY = RYGate(a * 2).control(len(registers) - 1)
        circuit.append(CCRY, registers)

        for pos, bit in enumerate(i_b):
            if bit == '0':
                circuit.x(registers[pos])

        circuit.barrier(registers)
    # circuit.measure_all()
    # print(circuit)
    return circuit

def circuit_instance(image_size):
    # initialize quantum circuit
    time_start = time.time()
    time_end = time.time()
    qubits_number = 7
    classical_number = 7
    qr = QuantumRegister(qubits_number)
    circ = QuantumCircuit(qr)
    image_size = image_size
    parameters = []
    for i in range(image_size):
        parameters.append(Parameter(str(parameter_name(i))))
    # print(parameters)
    time_start=time.time()
    circuit = encode_circuit(circ, qr, image_size, parameters)
    time_end=time.time()
    build_time = time_end - time_start
    print("building fixed circuit spend ", build_time, " s")
    print("image size is {} pixels".format(image_size))
    print("circuits deptsh is: ", circ.depth())
    return parameters,circuit

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
    for item in image:
        encode = item[0]
        count = int(item[1])
        # print(type(item))
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

def quantum_image_encode(data:np.ndarray,circ,parameters,image_size:int=64):
    image=data
# fill circuit with angles
    count = 0
    # print(circ.parameters)
    test_number = 1
    for i in range(test_number):
        time_start = time.time()
# set angles ans parameters
        circuit=circ
        ps={}
        image_sequence=image.ravel()
# pre processing
        for pos in range(len(parameters)):
            ps[parameters[pos]]=(image_sequence[pos]).item()*pi/2
        circuits = circuit.bind_parameters(ps)
        # if i==0:
        #     print(circuits)
        # circuit.draw()
        backend = Aer.get_backend('aer_simulator')
        t_qc2 = transpile(circuits, backend)
        qobj = assemble(t_qc2, shots=2**20)
        result = backend.run(qobj).result()
        q_counts = result.get_counts()
        time_end = time.time()
        spend_time = time_end - time_start
        output=sorted(q_counts.items(), key=lambda x:x[1])
        print("running parameterized test{} bind paramaeters spends {} s ".format(i, spend_time))
        print(output)

        return output


        # count += spend_time
    # print("average reused time is approximately {} to the original time".format(count / test_number))

def callback1(weights, obj_func_eval):
    print("measure is: {}".format(obj_func_eval))

if __name__ == '__main__':
    sample_number=200
    train_loader, test_loader,image_size=get_images(n_samples=sample_number)
    train_images_q=np.empty([0],dtype=bool)
    train_loader_q=[]
    q_parameters,feature_map=circuit_instance(image_size)
    ansatz=RealAmplitudes(7, reps=1)
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        loss="cross_entropy",
        optimizer=COBYLA(maxiter=30),
        callback=callback1,
    )
    print(vqc.circuit)
    vqc.circuit.draw(output="mpl")
    objective_func_vals = []
    # count=1
    targets=[]
    # start_time=time.time()
    # print(train_loader)
    datas=[]
    for batch_idx, (data, target) in enumerate(train_loader):
        datas.append(data[0][0].tolist())
        target=target.tolist()
        if target==0:
            targets.append([0,0,0,0,0,0,0])
        else:
            targets.append([0, 0, 0, 0, 0, 0,1])

    datas=np.asarray(datas)
    targets=np.asarray(targets)
    print(datas.shape,targets.shape)
    vqc.fit(datas,targets)
    #     break
    # fit classifier to data
    # print(datas.tolist(),[0,1])


        # return to default figsize
        # plt.rcParams["figure.figsize"] = (6, 4)

        # score classifier
        # vqc.score(X, y_one_hot)




