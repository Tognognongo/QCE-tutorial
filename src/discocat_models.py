from __future__ import annotations

from collections.abc import Callable
from typing import Any

from discopy.quantum import Circuit, Id, Measure
from discopy.tensor import Diagram
import numpy as np

from lambeq.training.quantum_model import QuantumModel

import pennylane as qml

from discocat_aux import qc_measurements
'''
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
'''
class DisCoCatClassifier(QuantumModel):

    def __init__(self, backend_config: dict[str, Any]) -> None:
        
        super().__init__()

        fields = ('backend', 'compilation', 'shots')
        missing_fields = [f for f in fields if f not in backend_config]
        if missing_fields:
            raise KeyError('Missing arguments in backend configuation. '
                           f'Missing arguments: {missing_fields}.')
        self.backend_config = backend_config

    def get_diagram_output(self, diagrams: list[Diagram]) -> np.ndarray:
       
        if len(self.weights) == 0 or not self.symbols:
            raise ValueError('Weights and/or symbols not initialised. '
                             'Instantiate through '
                             '`TketModel.from_diagrams()` first, '
                             'then call `initialise_weights()`, or load '
                             'from pre-trained checkpoint.')

        qiskit_weights = dict(zip([str(symbol) for symbol in [*self.symbols]], [*self.weights]))
        '''
        Model training is parallelized by splitting the number of circuits simulated per MPI process.
        
        jobs_per_rank = len(diagrams) // size
        leftover = len(diagrams) % size
        if rank > size-leftover-1: jobs_per_rank += 1
        jobsizes = comm.allgather(jobs_per_rank)
        starts = list(sum(jobsizes[:i]) for i in range(len(jobsizes)))
        rank_diagrams = diagrams[starts[rank]:starts[rank]+jobsizes[rank]]

        rank_measurements = qc_measurements(rank_diagrams, qiskit_weights)
        measurements = comm.allgather(rank_measurements)
        '''
        measurements = qc_measurements(diagrams, qiskit_weights)

        self.backend_config['backend'].empty_cache()
        if len(diagrams) == 1:
            result = self._normalise_vector(measurements[0])
            return result.reshape(1, *result.shape)
        return np.array([self._normalise_vector(m) for m in measurements])

    def forward(self, x: list[Diagram]) -> np.ndarray:

        return self.get_diagram_output(x)
