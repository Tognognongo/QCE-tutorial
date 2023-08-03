import torch.nn as nn
import torch.nn.functional as F

from qlstm import QLSTM

class QLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 n_qubits=0, n_qlayers=1, ising=False, probs=False, backend='default.qubit'):
        super(QLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_qlayers = n_qlayers

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim) # From vector with word indeces to embeddings dimension

        # The LSTM takes word embeddings as inputs,
        # and outputs hidden states with dimensionality hidden_dim.

        if n_qubits > 0:
            #print(f'Classifier will use Quantum LSTM running on backend {backend}')
            self.lstm = QLSTM(embedding_dim, hidden_dim, n_qubits=n_qubits, n_qlayers=n_qlayers, ising=ising, probs=probs, backend=backend)
        else:
            #print('Classifier will use Classical LSTM')
            self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden dimension to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_logits = self.hidden2tag(lstm_out[-1])
        tag_scores = F.softmax(tag_logits, dim=1)
        
        return tag_scores
