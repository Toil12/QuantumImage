import pennylane as qml
dev = qml.device("default.qubit", wires=2, shots=1000)

@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    return qml.counts(qml.PauliZ(0)), qml.counts(qml.PauliZ(1))

k=circuit()
print(k)