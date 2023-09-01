from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit

import pennylane as qml

from numpy import pi, array


def read_data(filename):

    labels, sentences = [], []
    
    with open(filename) as file:
        for line in file:
            t = int(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences


def qc_measurements(diagrams, weights):

    measurements = []

    for i, d in enumerate(diagrams):
        probs = [0,0]
        measured_qubits = []
        qubit = 0
        qiskit_qc = tk_to_qiskit(d.to_tk())
        num_wires = qiskit_qc.data[0].qubits[0].register.size
        wires = [i for i in range(num_wires)]
        for inst in qiskit_qc.data:
                if inst.operation.name == 'measure': measured_qubits.append(*[qubit.index for qubit in inst.qubits])

        dev = qml.device('default.qubit', wires=wires)

        @qml.qnode(dev)
        
        def qlm_qc(qiskit_qc, wires):

            for inst in qiskit_qc.data:
                
                if inst.operation.name == 'measure': continue
                
                if len(inst.operation.params)==0:
                    if inst.operation.name=='h':
                        eval('qml.Hadamard(wires={})'.format([qubit.index for qubit in inst.qubits]))
                    else:
                        eval('qml.{}(wires={})'.format(inst.operation.name.upper(), [qubit.index for qubit in inst.qubits]))

                else:
                    if str(inst.operation.params[0])[0] == '-':
                        eval('qml.{}(phi={}, wires={})'.format(inst.operation.name.upper(),
                                                               2*pi*weights[str(inst.operation.params[0])[6:]],
                                                               [qubit.index for qubit in inst.qubits]))
                    else:
                        eval('qml.{}(phi={}, wires={})'.format(inst.operation.name.upper(),
                                                               2*pi*weights[str(inst.operation.params[0])[5:]],
                                                               [qubit.index for qubit in inst.qubits]))
            return qml.probs(wires=wires)

        for j in range(num_wires):
                if j not in measured_qubits: qubit = j
        
        measures = qlm_qc(qiskit_qc, wires)
        indices = array([pow(2,num_wires)-1-pow(2,j), pow(2,num_wires)-1])
        for i in range(2): probs[i] = measures[indices[i]]
        measurements.append(array(probs))

    return measurements
