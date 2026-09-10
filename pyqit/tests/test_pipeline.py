import copy

import numpy as np
import pytest

import pyqit
from pyqit.core.callbacks import BaseCallback
from pyqit.core.pipeline import PipelineStage, QuantumPipeline
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.tests.scenarios import make_scenario
from pyqit.tests.test_datamodule import _record_inputs
from pyqit.tests.test_trainer import BACKENDS, _require, _to_numpy


def _data(n_samples=20, n_features=2):
    scenario = make_scenario(n_samples=n_samples, n_features=n_features, seed=0)
    return scenario["X"], scenario["y"]


def _vqc(n_qubits=2, **kwargs):
    return VQCClassifier(n_qubits=n_qubits, n_layers=1, **kwargs)


def _weights(model):
    return {k: np.array(_to_numpy(v)) for k, v in model.weights.items()}


def _changed(before, model):
    after = _weights(model)
    return any(not np.allclose(before[k], after[k]) for k in before)


def _trainer(**kwargs):
    return Trainer(max_epochs=1, verbose=0, **kwargs)


@pytest.mark.parametrize("backend", BACKENDS)
def test_frozen_backbone_trains_only_the_head_then_predicts_raw_rows(backend):
    """The tutorial's flow: a multi-class backbone's outputs feed a binary head."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_features=3)
    backbone, head = _vqc(3, n_classes=3), _vqc(3)
    backbone_before, head_before = _weights(backbone), _weights(head)
    pipe = QuantumPipeline(
        [PipelineStage(backbone, trainable=False), PipelineStage(head)]
    )

    pipe.fit(DataModule(X, y), trainers=_trainer(), fit_mode="frozen_backbone")

    assert not _changed(backbone_before, backbone)
    assert _changed(head_before, head)
    preds = pipe.predict(X[:5], return_format="numpy")
    assert len(preds) == 5
    assert set(np.unique(preds)) <= {0, 1}


def test_sequential_greedy_trains_every_stage_except_frozen_ones():
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data()
    first, middle, last = _vqc(), _vqc(), _vqc()
    before = [_weights(m) for m in (first, middle, last)]
    pipe = QuantumPipeline(
        [
            PipelineStage(first),
            PipelineStage(middle, trainable=False),
            PipelineStage(last),
        ]
    )

    pipe.fit(DataModule(X, y), trainers=_trainer())

    assert [_changed(b, m) for b, m in zip(before, (first, middle, last))] == [
        True,
        False,
        True,
    ]


def test_predict_on_raw_rows_feeds_the_first_stage_the_preprocessed_split():
    """predict() re-applies the normalizer and prescaling fitted during fit."""
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data(n_samples=30)
    pipe = QuantumPipeline([_vqc(), _vqc()])
    dm = DataModule(X * 10 + 5, y, normalize="minmax", seed=1)
    pipe.fit(dm, trainers=_trainer())
    raw_test = DataModule(X * 10 + 5, y, seed=1).setup().X_test
    seen = _record_inputs(pipe[0].model)

    pipe.predict(raw_test)

    np.testing.assert_allclose(np.concatenate(seen), dm.X_test)


@pytest.mark.xfail(
    strict=True,
    reason="non-first stages re-prescale their whole input, so passthrough "
    "columns are multiplied by pi a second time; see notes.md",
)
def test_passthrough_hands_the_head_the_backbones_input_unchanged():
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data()
    backbone, head = _vqc(2), _vqc(3)
    seen = _record_inputs(head)
    pipe = QuantumPipeline([PipelineStage(backbone, passthrough=True), head])
    dm = DataModule(X, y).setup(
        n_qubits=backbone.n_qubits, encoder=type(backbone.embedding_obj)
    )

    pipe.forward(dm.X_train)

    np.testing.assert_allclose(seen[0][:, :2], dm.X_train)


@pytest.mark.parametrize(
    "n_classes, aggregation",
    [
        (2, "mean"),
        (2, "vote"),
        (3, "mean"),
        pytest.param(
            3,
            "vote",
            marks=pytest.mark.xfail(
                strict=True,
                reason="vote rounds each class probability rather than voting on "
                "labels, returning an (n, n_classes) 0/1 matrix; see notes.md",
            ),
        ),
    ],
)
def test_ensemble_of_identical_models_predicts_like_the_model(n_classes, aggregation):
    """The mean or majority of copies of one model is that model's own answer."""
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data()
    model = _vqc(n_classes=n_classes)
    pipe = QuantumPipeline(
        [model, copy.deepcopy(model), copy.deepcopy(model)],
        mode="ensemble",
        aggregation=aggregation,
    )

    alone = Trainer(verbose=0).predict(
        model, DataModule(X, y, split=(0.0, 0.0, 1.0)), return_format="numpy"
    )

    np.testing.assert_array_equal(
        np.ravel(pipe.predict(X, return_format="numpy")), np.ravel(alone)
    )


@pytest.mark.xfail(
    strict=True,
    raises=IndexError,
    reason="DataModule prescaling truncates to the first stage's n_qubits before "
    "input_slice is applied; see notes.md",
)
def test_first_stage_can_read_a_column_subset_of_wider_data():
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data(n_features=4)
    first = _vqc(2)
    pipe = QuantumPipeline([PipelineStage(first, input_slice=[2, 3]), _vqc()])
    pipe.fit(DataModule(X, y), trainers=_trainer())
    seen = _record_inputs(first)

    pipe.predict(X[:3])

    np.testing.assert_allclose(np.concatenate(seen), X[:3, [2, 3]] * np.pi)


def test_fine_tuning_a_clone_leaves_the_original_untouched():
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data()
    pipe = QuantumPipeline([_vqc(), _vqc()])
    pipe.fit(DataModule(X, y), trainers=_trainer())
    fitted = [_weights(stage.model) for _, stage in pipe.steps]

    clone = pipe.clone()
    clone.fit(DataModule(X, y), trainers=_trainer())

    assert not any(_changed(w, s.model) for w, (_, s) in zip(fitted, pipe.steps))
    assert all(_changed(w, s.model) for w, (_, s) in zip(fitted, clone.steps))


class _EpochCounter(BaseCallback):
    def __init__(self):
        super().__init__()
        self.epochs = 0

    def on_epoch_end(self, state):
        self.epochs += 1


def test_each_stage_trains_under_the_trainer_named_for_it():
    _require("pennylane")
    pyqit.set_seed(0)
    X, y = _data()
    backbone_epochs, head_epochs = _EpochCounter(), _EpochCounter()
    pipe = QuantumPipeline([("backbone", _vqc()), ("head", _vqc())])

    pipe.fit(
        DataModule(X, y),
        trainers={
            "head": Trainer(max_epochs=3, callbacks=[head_epochs], verbose=0),
            "backbone": Trainer(max_epochs=1, callbacks=[backbone_epochs], verbose=0),
        },
    )

    assert (backbone_epochs.epochs, head_epochs.epochs) == (1, 3)
