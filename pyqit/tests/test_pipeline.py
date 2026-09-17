import copy

import numpy as np
import pennylane as qml
import pytest

import pyqit
from pyqit.ansatzes import CNOTLadderAnsatz
from pyqit.core.embeddings import AmplitudeEmbedding, HadamardAngleEmbedding
from pyqit.core.pipeline import PipelineStage, QuantumPipeline
from pyqit.core.trainer import Trainer
from pyqit.data.datamodule import DataModule
from pyqit.models.classification.vqc import VQCClassifier
from pyqit.models.layers import DenseClassifier, DenseLayer, QuantumLayer
from pyqit.tests.scenarios import make_scenario
from pyqit.tests.test_datamodule import _record_inputs
from pyqit.tests.test_trainer import BACKENDS, _mse_gradient, _require, _to_numpy


def _data(n_samples=20, n_features=2):
    scenario = make_scenario(n_samples=n_samples, n_features=n_features, seed=0)
    return scenario["X"], scenario["y"]


def _vqc(n_qubits=2, **kwargs):
    return VQCClassifier(n_qubits=n_qubits, n_layers=1, **kwargs)


def _as_input(X, backend):
    if backend != "torch":
        return X
    import torch

    return torch.as_tensor(X, dtype=torch.float32)


def _weights(model):
    return {k: np.array(_to_numpy(v)) for k, v in model.weights.items()}


def _changed(before, model):
    after = _weights(model)
    return any(not np.allclose(before[k], after[k]) for k in before)


def _trainer(**kwargs):
    return Trainer(max_epochs=1, verbose=0, **kwargs)


def _hybrid(n_features=3, **quantum_stage):
    """Dense, circuit, dense: the hybrid network, as three jointly trained stages."""
    return QuantumPipeline(
        [
            ("pre", DenseLayer(n_features, 2, activation="tanh")),
            PipelineStage(
                QuantumLayer(n_qubits=2, n_layers=1), name="q", **quantum_stage
            ),
            ("head", DenseClassifier(4 if quantum_stage.get("passthrough") else 2)),
        ],
        fit_mode="joint",
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_joint_hybrid_learns_and_trains_the_stage_before_a_frozen_circuit(backend):
    """The loss sits after the last stage, so reaching `pre` means crossing `q`."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_samples=40, n_features=3)
    pipe = _hybrid(trainable=False)
    before = {name: _weights(stage.model) for name, stage in pipe.steps}
    dm = DataModule(X, y, normalize="minmax", split=(0.8, 0.0, 0.2))

    history = Trainer(max_epochs=8, learning_rate=0.1, verbose=0, batch_size=40).fit(
        pipe, dm
    )

    moved = {name: _changed(before[name], stage.model) for name, stage in pipe.steps}
    assert moved == {"pre": True, "q": False, "head": True}
    assert history.train_loss[-1] < history.train_loss[0]
    probs = np.ravel(_to_numpy(pipe.forward(_as_input(dm.X_test, backend))))
    assert Trainer(verbose=0, batch_size=40).test(pipe, dm)["test_loss"] == (
        pytest.approx(np.mean((probs - dm.y_test.ravel()) ** 2), rel=1e-5)
    )
    preds = Trainer(verbose=0).predict(
        pipe, dm.for_prediction(X[:5]), return_format="numpy"
    )
    assert preds.shape == (5,)
    assert set(np.unique(preds)) <= {0, 1}


def test_dressed_circuit_stage_equals_the_reference_circuit_of_mari_et_al():
    """`quantum_net` of PennyLane's transfer-learning demo, on the same features."""
    _require("pennylane")
    pyqit.set_seed(0)
    n, depth = 3, 2
    circuit = QuantumLayer(
        n_qubits=n,
        n_layers=depth,
        ansatz=CNOTLadderAnsatz,
        encoder=HadamardAngleEmbedding,
    )
    head = DenseClassifier(n)
    head_saw = _record_inputs(head)
    features = np.tanh(np.random.default_rng(0).normal(size=(4, n)))
    q_weights = np.asarray(circuit.weights["main_circuit.weights"])

    @qml.qnode(qml.device("default.qubit", wires=n))
    def quantum_net(q_in):
        for w in range(n):
            qml.Hadamard(wires=w)
        for w in range(n):
            qml.RY(q_in[w], wires=w)
        for k in range(depth):
            for i in range(0, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for i in range(1, n - 1, 2):
                qml.CNOT(wires=[i, i + 1])
            for w in range(n):
                qml.RY(q_weights[k, w], wires=w)
        return [qml.expval(qml.PauliZ(w)) for w in range(n)]

    Trainer(verbose=0).predict(
        QuantumPipeline([circuit, head]),
        DataModule(features, np.zeros(4), split=(0.0, 0.0, 1.0)),
    )

    reference = np.array([quantum_net(row * np.pi / 2) for row in features])
    np.testing.assert_allclose(head_saw[0], reference, atol=1e-9)


def test_joint_gradient_is_the_same_under_autograd_and_torch():
    """Two autodiff engines agree on every stage's gradient, passthrough included."""
    X, y = _data(n_samples=6, n_features=3)
    grads = []
    for backend in BACKENDS:
        _require(backend)
        pyqit.set_seed(0)
        grads.append(_mse_gradient(_hybrid(passthrough=True), X, y))

    assert np.abs(grads[0][:6]).min() > 0
    np.testing.assert_allclose(grads[0], grads[1], atol=1e-6)


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

    dm = DataModule(X, y)
    _trainer().fit(pipe, dm)

    assert not _changed(backbone_before, backbone)
    assert _changed(head_before, head)
    preds = Trainer(verbose=0).predict(
        pipe, dm.for_prediction(X[:5]), return_format="numpy"
    )
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
    """for_prediction() carries the fitted normalizer; each stage is then prescaled."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_samples=30)
    pipe = QuantumPipeline([_vqc(), _vqc()])
    dm = DataModule(X * 10 + 5, y, normalize="minmax", seed=1)
    _trainer().fit(pipe, dm)
    raw_test = DataModule(X * 10 + 5, y, seed=1).setup().X_test
    seen = _record_inputs(pipe[0].model)

    Trainer(verbose=0).predict(pipe, dm.for_prediction(raw_test))

    np.testing.assert_allclose(np.concatenate(seen), dm.X_test * np.pi, rtol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_trainer_predict_prescales_a_pipelines_first_stage(backend):
    """The pipeline's DataModule is never prescaled, so the pipeline must do it."""
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data()
    first = _vqc()
    pipe = QuantumPipeline([first, _vqc()])
    seen = _record_inputs(first)

    Trainer(verbose=0).predict(pipe, DataModule(X, y, split=(0.0, 0.0, 1.0)))

    np.testing.assert_allclose(np.concatenate(seen), X * np.pi, rtol=1e-6)


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
    X, y = _data()
    backbone, head = _vqc(2), _vqc(3)
    backbone_saw, head_saw = _record_inputs(backbone), _record_inputs(head)
    pipe = QuantumPipeline([PipelineStage(backbone, passthrough=True), head])

    Trainer(verbose=0).predict(pipe, DataModule(X, y, split=(0.0, 0.0, 1.0)))

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

    dm = DataModule(X, y)
    _trainer().fit(pipe, dm)
    Trainer(verbose=0).predict(pipe, dm.for_prediction(X[:5]))

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

    together = Trainer(verbose=0).predict(
        pipe, DataModule(X, y, split=(0.0, 0.0, 1.0)), return_format="numpy"
    )

    np.testing.assert_array_equal(np.ravel(together), np.ravel(alone))


@pytest.mark.parametrize("backend", BACKENDS)
def test_first_stage_can_read_a_column_subset_of_wider_data(backend):
    _require(backend)
    pyqit.set_seed(0)
    X, y = _data(n_features=4)
    first = _vqc(2)
    pipe = QuantumPipeline([PipelineStage(first, input_slice=[2, 3]), _vqc()])
    dm = DataModule(X, y)
    _trainer().fit(pipe, dm)
    seen = _record_inputs(first)

    Trainer(verbose=0).predict(pipe, dm.for_prediction(X[:3]))

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
