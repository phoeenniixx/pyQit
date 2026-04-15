from pyqit.base import _PyQitObject
from pyqit.utils import _is_torch


class ClassifierMixin(_PyQitObject):
    _tags = {
        "estimator_type": "classifier",
    }

    def predict_step(self, X):
        raw_output = self.forward(X)

        is_torch = _is_torch(raw_output)

        if self.n_classes == 2:
            if is_torch:
                preds = (raw_output >= 0.5).int()
            else:
                preds = (raw_output >= 0.5).astype(int)
        else:
            if raw_output.ndim > 1:
                if is_torch:
                    preds = raw_output.argmax(dim=1)
                else:
                    preds = raw_output.argmax(axis=1)
            else:
                if is_torch:
                    preds = raw_output.argmax(dim=0)
                else:
                    preds = raw_output.argmax(axis=0)
        if is_torch:
            import torch

            return torch.atleast_1d(preds)
        else:
            import numpy as np

            return np.atleast_1d(preds)
