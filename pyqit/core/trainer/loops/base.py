"""Base class for backend training loops.

A loop owns one backend's fitting mechanics and nothing else: the Trainer has
already seeded, set the DataModule up, printed the summary and run the
diagnostic before a loop is asked to do anything.

Which Trainer settings a backend cannot honour is declared through the
``rejects`` and ``warns`` tags rather than through control flow inside ``fit``,
for the same reason losses declare ``backends``: the policy is then visible on
the class, and a new backend declares its own instead of editing shared code.
"""

from abc import abstractmethod
import warnings

from pyqit.base.base_object import _PyQitObject


class BaseTrainingLoop(_PyQitObject):
    """One backend's fit mechanics.

    Tags
    ----
    backend : str
        The ``pyqit.set_backend`` value this loop serves. The registry keys on it.
    rejects : dict of {str: str}
        Trainer parameters this backend cannot honour at all. Passing one a
        non-default value raises, because the alternative is a run that quietly
        ignored what the user asked for.
    warns : dict of {str: str}
        Parameters this backend degrades on but still trains correctly under.
        These warn rather than raise, since the resulting model is still valid.
    reserved_backend_kwargs : tuple of str
        Keys the loop derives from Trainer settings and therefore refuses to let
        ``backend_kwargs`` overwrite.
    """

    _tags = {
        "object_type": "training_loop",
        "backend": None,
        "rejects": {},
        "warns": {},
        "reserved_backend_kwargs": (),
    }

    def __init__(self, trainer, reporter):
        self.trainer = trainer
        self.reporter = reporter
        super().__init__()

        self._validate_config()

    def _is_set(self, param: str, defaults: dict) -> bool:
        """Whether the user moved ``param`` off its default.

        Comparing against the default rather than against ``None`` means a
        backend only complains about settings that were actually asked for, so
        constructing a Trainer with defaults is never noisy.
        """
        if not hasattr(self.trainer, param):
            return False
        return getattr(self.trainer, param) != defaults.get(param)

    def _validate_config(self) -> None:
        # skbase already derives these from the constructor signature; deriving
        # them again with inspect would also force an import of Trainer here.
        defaults = self.trainer.get_param_defaults()
        backend = self.get_tag("backend")

        for param, reason in (self.get_tag("rejects") or {}).items():
            if self._is_set(param, defaults):
                raise ValueError(
                    f"Trainer({param}=...) is not supported on the {backend!r} "
                    f"backend. {reason}"
                )

        for param, reason in (self.get_tag("warns") or {}).items():
            if self._is_set(param, defaults):
                warnings.warn(
                    f"Trainer({param}=...) is only partly honoured on the "
                    f"{backend!r} backend. {reason}",
                    UserWarning,
                    # _validate_config -> __init__ -> get_training_loop ->
                    # Trainer.fit -> the caller, who is the one to point at.
                    stacklevel=5,
                )

    def _resolve_backend_kwargs(self) -> dict:
        """``backend_kwargs`` after checking it does not fight the Trainer."""
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
    def _emit(callbacks, hook: str, state) -> None:
        for callback in callbacks:
            getattr(callback, hook)(state)

    @abstractmethod
    def fit(self, model, datamodule, state, callbacks) -> None:
        """Train ``model``, calling ``callbacks`` on the shared hooks."""

    @classmethod
    def get_test_params(cls):
        return []
