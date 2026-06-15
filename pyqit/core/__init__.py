"""A module for core functionality."""

from pyqit.core.callbacks import HistoryCallback
from pyqit.core.config import get_backend, set_backend
from pyqit.core.embeddings import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding
from pyqit.core.losses import cross_entropy_loss, hinge_loss, mse_loss
from pyqit.core.measurements import measure_expval_x, measure_expval_z, measure_probs
from pyqit.core.pipeline import PipelineStage, QuantumPipeline
from pyqit.core.trainer import Trainer

__all__ = [
    "Trainer",
    "cross_entropy_loss",
    "mse_loss",
    "hinge_loss",
    "AmplitudeEmbedding",
    "AngleEmbedding",
    "IQPEmbedding",
    "measure_probs",
    "measure_expval_z",
    "measure_expval_x",
    "HistoryCallback",
    "set_backend",
    "get_backend",
    "QuantumPipeline",
    "PipelineStage",
]
