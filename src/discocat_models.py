from __future__ import annotations

from collections.abc import Callable
from typing import Any

from discopy.quantum import Circuit, Id, Measure
from discopy.tensor import Diagram
import numpy as np

from lambeq.training.quantum_model import QuantumModel

import pennylane as qml

from discocat_aux import qc_measurements

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

        measurements = qc_measurements(diagrams, qiskit_weights)

        self.backend_config['backend'].empty_cache()
        if len(diagrams) == 1:
            result = self._normalise_vector(measurements[0])
            return result.reshape(1, *result.shape)
        return np.array([self._normalise_vector(m) for m in measurements])

    def forward(self, x: list[Diagram]) -> np.ndarray:

        return self.get_diagram_output(x)
