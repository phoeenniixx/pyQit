import copy

import numpy as np
import pytest

import pyqit
from pyqit.core.embeddings import AmplitudeEmbedding
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
        [PipelineStage(backbone, trainable=False), PipelineStage(head)],
        fit_mode="frozen_backbone",
    )

    _trainer().fit(pipe, DataModule(X, y))

    assert not _changed(backbone_before, backbone)
    assert _changed(head_before, head)
    preds = pipe.predict(X[:5], return_format="numpy")
    assert len(preds) == 5
    assert set(np.unique(preds)) <= {0, 1}


@pytest.mark.parametrize("backend", BACKENDS)
def test_sequential_greedy_trains_every_stage_except_frozen_ones(backend):
    _require(backend)
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

    _trainer().fit(pipe, DataModule(X, y))

    assert [_changed(b, m) for b, m in zip(before, (first, middle, last))] == [
        True,
        False,
        True,
    ]


@pytest.mark.parametrize("backend", BACKENDS)
def test_predict_on_raw_rows_feeds_the_first_stage_the_preprocessed_split(backend):
    """predict() re-applies the fitted normalizer; each stage is then prescaled."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_samples=30)
    pipe = QuantumPipeline([_vqc(), _vqc()])
    dm = DataModule(X * 10 + 5, y, normalize="minmax", seed=1)
    _trainer().fit(pipe, dm)
    raw_test = DataModule(X * 10 + 5, y, seed=1).setup().X_test
    seen = _record_inputs(pipe[0].model)

    pipe.predict(raw_test)

    np.testing.assert_allclose(np.concatenate(seen), dm.X_test * np.pi, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_trainer_predict_feeds_a_pipeline_what_pipeline_predict_does(backend):
    """Handing the pipeline to Trainer.predict must not skip its prescaling."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data()
    first = _vqc()
    pipe = QuantumPipeline([first, _vqc()])
    seen = _record_inputs(first)

    pipe.predict(X)
    via_pipeline = np.concatenate(seen)
    seen.clear()
    Trainer(verbose=0).predict(pipe, DataModule(X, y, split=(0.0, 0.0, 1.0)))

    np.testing.assert_allclose(np.concatenate(seen), via_pipeline)


@pytest.mark.parametrize("backend", BACKENDS)
def test_pipeline_test_agrees_with_trainer_predict_on_the_test_split(backend):
    """The pipeline's own accuracy must match counting its predictions by hand."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_samples=40)
    dm = DataModule(X, y, split=(0.6, 0.2, 0.2), batch_size=8)
    pipe = QuantumPipeline(
        [PipelineStage(_vqc(), trainable=False), PipelineStage(_vqc())],
        fit_mode="frozen_backbone",
    )
    _trainer().fit(pipe, dm)

    metrics = Trainer(verbose=0).test(pipe, dm)
    val_metrics = Trainer(verbose=0).validate(pipe, dm)

    preds = np.ravel(_to_numpy(Trainer(verbose=0).predict(pipe, dm)))
    expected_acc = np.mean(preds.astype(int) == dm.y_test.astype(int).ravel())
    assert metrics["test_acc"] == pytest.approx(expected_acc)
    assert set(val_metrics) == {"val_loss", "val_acc"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_passthrough_hands_the_head_the_backbones_input_unchanged(backend):
    _require(backend)
    pyqit.set_seed(0)
    X, _ = _data()
    backbone, head = _vqc(2), _vqc(3)
    backbone_saw, head_saw = _record_inputs(backbone), _record_inputs(head)
    pipe = QuantumPipeline([PipelineStage(backbone, passthrough=True), head])

    pipe.predict(X)

    np.testing.assert_allclose(head_saw[0][:, :2], backbone_saw[0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_amplitude_encoded_stages_receive_unit_norm_states(backend):
    """Both the first stage and a downstream one are padded to 2**n and normalized."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_features=4)
    backbone = _vqc(2, encoder=AmplitudeEmbedding, n_classes=3)
    head = _vqc(2, encoder=AmplitudeEmbedding)
    backbone_saw, head_saw = _record_inputs(backbone), _record_inputs(head)
    pipe = QuantumPipeline(
        [PipelineStage(backbone, trainable=False), head], fit_mode="frozen_backbone"
    )

    _trainer().fit(pipe, DataModule(X, y))
    pipe.predict(X[:5])

    assert backbone_saw and head_saw
    for batch in backbone_saw + head_saw:
        assert batch.shape[1] == 4
        np.testing.assert_allclose(np.linalg.norm(batch, axis=1), 1.0, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "n_classes, aggregation",
    [
        (2, "mean"),
        (2, "vote"),
        (3, "mean"),
        (3, "vote"),
    ],
)
def test_ensemble_of_identical_models_predicts_like_the_model(
    n_classes, aggregation, backend
):
    """The mean or majority of copies of one model is that model's own answer."""
    _require(backend)
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_first_stage_can_read_a_column_subset_of_wider_data(backend):
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_features=4)
    first = _vqc(2)
    pipe = QuantumPipeline([PipelineStage(first, input_slice=[2, 3]), _vqc()])
    _trainer().fit(pipe, DataModule(X, y))
    seen = _record_inputs(first)

    pipe.predict(X[:3])

    np.testing.assert_allclose(np.concatenate(seen), X[:3, [2, 3]] * np.pi, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fine_tuning_a_clone_leaves_the_original_untouched(backend):
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data()
    pipe = QuantumPipeline([_vqc(), _vqc()])
    _trainer().fit(pipe, DataModule(X, y))
    fitted = [_weights(stage.model) for _, stage in pipe.steps]

    clone = pipe.clone()
    _trainer().fit(clone, DataModule(X, y))

    assert not any(_changed(w, s.model) for w, (_, s) in zip(fitted, pipe.steps))
    assert all(_changed(w, s.model) for w, (_, s) in zip(fitted, clone.steps))
