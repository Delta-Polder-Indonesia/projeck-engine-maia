from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort


@dataclass
class OnnxPolicyModel:
    """Thin wrapper around ONNX Runtime session for policy-only inference."""

    session: ort.InferenceSession
    input_name: str
    output_name: str

    @classmethod
    def from_path(cls, model_path: str) -> "OnnxPolicyModel":
        providers = ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return cls(session=session, input_name=input_name, output_name=output_name)

    def predict(self, board_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run([self.output_name], {self.input_name: board_tensor})
        return np.asarray(outputs[0]).reshape(-1)
