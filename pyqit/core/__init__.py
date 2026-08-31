"""A module for core functionality."""

from pyqit.core.callbacks import (
    BaseCallback,
    EarlyStopping,
    HistoryCallback,
    ModelCheckpoint,
)
from pyqit.core.config import get_backend, set_backend, set_seed
from pyqit.core.embeddings import AmplitudeEmbedding, AngleEmbedding, IQPEmbedding
from pyqit.core.losses import (
    BaseLoss,
    CrossEntropyLoss,
    HingeLoss,
    MSELoss,
    get_loss_fn,
)
from pyqit.core.measurements import measure_expval_x, measure_expval_z, measure_probs
from pyqit.core.pipeline import PipelineStage, QuantumPipeline
from pyqit.core.trainer import Trainer, TrainingHistory

__all__ = [
    "Trainer",
    "TrainingHistory",
    "BaseLoss",
    "MSELoss",
    "HingeLoss",
    "CrossEntropyLoss",
    "get_loss_fn",
    "AmplitudeEmbedding",
    "AngleEmbedding",
    "IQPEmbedding",
    "measure_probs",
    "measure_expval_z",
    "measure_expval_x",
    "BaseCallback",
    "HistoryCallback",
    "ModelCheckpoint",
    "EarlyStopping",
    "set_backend",
    "get_backend",
    "set_seed",
    "QuantumPipeline",
    "PipelineStage",
]
