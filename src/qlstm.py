import torch
import torch.nn as nn

import pennylane as qml

from numpy import pi, zeros

def construct_ising_zz_spin_chain_circuit(inputs, weights, wires):
    if len(wires)%2 == 0:
        cut1 = len(wires)
        cut2 = 2*len(wires)
        cut3 = 3*len(wires)-2
        cut4 = 4*len(wires)-4

    if len(wires)%2 == 1:
        cut1 = len(wires)-1
        cut2 = 2*len(wires)-2
        cut3 = 3*len(wires)-3
        cut4 = 4*len(wires)-4
    for wire in wires: qml.Hadamard(wires=[wire])
    for i, layer in enumerate(weights):
        qml.templates.AngleEmbedding(inputs[:cut1], wires=wires[:cut1])
        qml.templates.AngleEmbedding(inputs[cut1:cut2], wires=wires[:cut1], rotation='Y')
        qml.templates.AngleEmbedding(layer[:cut1], wires=wires[:cut1])
        for j in [2*k for k in range(len(wires)//2)]:
            qml.IsingZZ(pi/2, wires=[wires[j], wires[j+1]])
        if len(wires) > 2:
            qml.templates.AngleEmbedding(inputs[cut2:cut3], wires=wires[1:cut1-pow(-1,len(wires)%2)])
            qml.templates.AngleEmbedding(inputs[cut3:cut4], wires=wires[1:cut1-pow(-1,len(wires)%2)], rotation='Y')
            qml.templates.AngleEmbedding(layer[cut1:], wires=wires[1:cut1-pow(-1,len(wires)%2)])
            for j in [2*k+1 for k in range((len(wires)-1)//2)]:
                qml.IsingZZ(pi/2, wires=[wires[j], wires[j+1]])

class QLSTM(nn.Module):
    def __init__(self,
                input_size,
                hidden_size,
                n_qubits=4,
                n_qlayers=1,
                ising=False,
                probs=False,
                batch_first=True,
                return_sequences=False,
                return_state=False,
                backend="default.qubit"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.probs = probs
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget)
        self.dev_input = qml.device(self.backend, wires=self.wires_input)
        self.dev_update = qml.device(self.backend, wires=self.wires_update)
        self.dev_output = qml.device(self.backend, wires=self.wires_output)

        def _circuit_forget(inputs, weights):
            if ising==True:
                construct_ising_zz_spin_chain_circuit(inputs, weights, self.wires_forget)
                if probs: return qml.probs(wires=self.wires_forget)
                return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

            for i, layer in enumerate(weights):
                qml.templates.AngleEmbedding(inputs[:self.n_qubits], wires=self.wires_forget)
                qml.templates.AngleEmbedding(inputs[self.n_qubits:], wires=self.wires_forget, rotation='Y')
                qml.templates.BasicEntanglerLayers(layer[None], wires=self.wires_forget, rotation=qml.RX)
            if probs: return qml.probs(wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        self.qlayer_forget = qml.QNode(_circuit_forget, self.dev_forget, interface="torch")

        def _circuit_input(inputs, weights):
            if ising==True:
                construct_ising_zz_spin_chain_circuit(inputs, weights, self.wires_input)
                if probs: return qml.probs(wires=self.wires_input)
                return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

            for i, layer in enumerate(weights):
                qml.templates.AngleEmbedding(inputs[:self.n_qubits], wires=self.wires_input)
                qml.templates.AngleEmbedding(inputs[self.n_qubits:], wires=self.wires_input, rotation='Y')
                qml.templates.BasicEntanglerLayers(layer[None], wires=self.wires_input, rotation=qml.RX)
            if probs: return qml.probs(wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input, self.dev_input, interface="torch")

        def _circuit_update(inputs, weights):
            if ising==True:
                construct_ising_zz_spin_chain_circuit(inputs, weights, self.wires_update)
                if probs: return qml.probs(wires=self.wires_update)
                return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

            for i, layer in enumerate(weights):
                qml.templates.AngleEmbedding(inputs[:self.n_qubits], wires=self.wires_update)
                qml.templates.AngleEmbedding(inputs[self.n_qubits:], wires=self.wires_update, rotation='Y')
                qml.templates.BasicEntanglerLayers(layer[None], wires=self.wires_update, rotation=qml.RX)
            if probs: return qml.probs(wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update, self.dev_update, interface="torch")

        def _circuit_output(inputs, weights):
            if ising==True:
                construct_ising_zz_spin_chain_circuit(inputs, weights, self.wires_output)
                if probs: return qml.probs(wires=self.wires_output)
                return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

            for i, layer in enumerate(weights):
                qml.templates.AngleEmbedding(inputs[:self.n_qubits], wires=self.wires_output)
                qml.templates.AngleEmbedding(inputs[self.n_qubits:], wires=self.wires_output, rotation='Y')
                qml.templates.BasicEntanglerLayers(layer[None], wires=self.wires_output, rotation=qml.RX)
            if probs: return qml.probs(wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        self.qlayer_output = qml.QNode(_circuit_output, self.dev_output, interface="torch")

        if ising==True: weight_shapes = {"weights": (n_qlayers, 2*(n_qubits-1))}
        else: weight_shapes = {"weights": (n_qlayers, n_qubits)}

        if ising==True: self.clayer_in = torch.nn.Linear(self.concat_size, 4*(n_qubits-1))
        else: self.clayer_in = torch.nn.Linear(self.concat_size, 2*n_qubits)

        print(weight_shapes)

        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)

    def forward(self, x, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)
            # match qubit dimension
            y_t = self.clayer_in(v_t)
            y_t = 2 * pi * y_t / torch.max(y_t)

            if self.probs:
                sf = nn.Softmax(dim=0)
                f_t = torch.sigmoid(self.clayer_out(sf(self.VQC['forget'](y_t)[:,[pow(2,i) for i in range(self.n_qubits)]])))  # forget block
                i_t = torch.sigmoid(self.clayer_out(sf(self.VQC['input'](y_t)[:,[pow(2,i) for i in range(self.n_qubits)]])))  # input block
                g_t = torch.tanh(self.clayer_out(sf(self.VQC['update'](y_t)[:,[pow(2,i) for i in range(self.n_qubits)]])))  # update block
                o_t = torch.sigmoid(self.clayer_out(sf(self.VQC['output'](y_t)[:,[pow(2,i) for i in range(self.n_qubits)]]))) # output block
            else:
                f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
                i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
                g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
                o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t)))  # output block
            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        return hidden_seq, (h_t, c_t)
