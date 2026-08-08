import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

from pyqit.core.config import get_backend, set_backend, set_seed

__all__ = ["get_backend", "set_backend", "set_seed"]
