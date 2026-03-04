from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Unified interface for all models in a pipeline stage.
    Every stage must implement forward() and fit().
    """

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def fit(self, X, y, **kwargs):
        pass

    def __call__(self, X):
        return self.forward(X)