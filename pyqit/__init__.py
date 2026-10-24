from importlib.metadata import version as _version
import logging

__version__ = _version("pyqit")

logging.getLogger("pyqit").addHandler(logging.NullHandler())

from pyqit.core.config import get_backend, get_dtype, set_backend, set_dtype, set_seed
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule

__all__ = [
    "DataModule",
    "Trainer",
    "__version__",
    "get_backend",
    "get_dtype",
    "set_backend",
    "set_dtype",
    "set_seed",
]
