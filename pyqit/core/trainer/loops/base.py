"""Base class for backend training loops."""

from abc import abstractmethod
import warnings

from pyqit.base.base_object import _PyQitObject


class BaseTrainingLoop(_PyQitObject):
    """One backend's fit mechanics.

    The Trainer has already seeded, set the DataModule up, printed the summary
    and run any diagnostic before a loop is asked to do anything.

    Tags
    ----
    backend : str
        The ``pyqit.set_backend`` value this loop serves; the registry keys on it.
    rejects : tuple of str
        Trainer parameters this backend cannot honour. Passing one a non-default
        value raises.
    warns : tuple of str
        Trainer parameters this backend ignores but still trains correctly
        without. Passing one a non-default value warns.
    reserved_backend_kwargs : tuple of str
        Keys the loop derives from Trainer settings, so ``backend_kwargs`` may
        not set them.

    Parameters
    ----------
    trainer : Trainer
        The Trainer whose settings this loop runs under.
    reporter : Reporter
        Console output for the run.
    """

    _tags = {
        "object_type": "training_loop",
        "backend": None,
        "rejects": (),
        "warns": (),
        "reserved_backend_kwargs": (),
    }

    def __init__(self, trainer, reporter):
        self.trainer = trainer
        self.reporter = reporter
        super().__init__()

        self._validate_config()

    def _is_set(self, param: str, defaults: dict) -> bool:
        """Whether the user moved ``param`` off its constructor default."""
        if not hasattr(self.trainer, param):
            return False
        return getattr(self.trainer, param) != defaults.get(param)

    def _validate_config(self) -> None:
        """Raise or warn on Trainer settings this backend cannot honour."""
        defaults = self.trainer.get_param_defaults()
        backend = self.get_tag("backend")

        for param in self.get_tag("rejects") or ():
            if self._is_set(param, defaults):
                raise ValueError(
                    f"Trainer({param}=...) is not supported on the {backend!r} backend."
                )

        for param in self.get_tag("warns") or ():
            if self._is_set(param, defaults):
                warnings.warn(
                    f"Trainer({param}=...) is ignored on the {backend!r} backend.",
                    UserWarning,
                    # _validate_config -> __init__ -> get_training_loop ->
                    # Trainer.fit -> the caller, who is the one to point at.
                    stacklevel=5,
                )

    def _resolve_backend_kwargs(self) -> dict:
        """``backend_kwargs`` after checking it sets no reserved key."""
        extra = dict(getattr(self.trainer, "backend_kwargs", None) or {})
        reserved = set(self.get_tag("reserved_backend_kwargs") or ())
        clashes = sorted(reserved & set(extra))
        if clashes:
            raise ValueError(
                f"backend_kwargs may not set {clashes}: the Trainer derives "
                f"{'them' if len(clashes) > 1 else 'it'} from its own settings. "
                f"Reserved keys: {sorted(reserved)}."
            )
        return extra

    @staticmethod
    def _emit(callbacks: list, hook: str, state) -> None:
        """Call ``hook`` on every callback with ``state``."""
        for callback in callbacks:
            getattr(callback, hook)(state)

    @abstractmethod
    def fit(self, model, datamodule, state, callbacks: list) -> None:
        """Train ``model``, calling ``callbacks`` on the shared hooks.

        Parameters
        ----------
        model : BaseModel
            The model to train, mutated in place.
        datamodule : DataModule
            Already set up by the Trainer.
        state : LoopState
            Shared state; the loop fills ``epoch`` and ``metrics`` each epoch
            and honours ``stop``.
        callbacks : list of BaseCallback
            Fired on ``on_fit_start``, ``on_epoch_end`` and ``on_fit_end``.
        """

    @classmethod
    def get_test_params(cls):
        return []
