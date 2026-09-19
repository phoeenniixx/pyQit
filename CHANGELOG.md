# Changelog

Generated from merged pull requests with `build_tools/changelog.py`; see that file's
docstring for how classification works.

## 0.1.0

First release on PyPI. PyQit puts a Trainer, a DataModule and model classes on top of
PennyLane QNodes, so training a variational circuit is a `fit` call. PyTorch and
PyTorch Lightning are optional: `pyqit.set_backend("torch")` moves the same code onto
Lightning.

What is added:

- **Models.** `VQCClassifier`, `VQCRegressor`, `DataReuploadingClassifier` and the
  hybrid `DressedQuantumClassifier`. `QuantumLayer`, `DenseLayer` and
  `DenseClassifier` are the stages hybrids are built from.
- **Circuits.** Embeddings `AngleEmbedding`, `HadamardAngleEmbedding`,
  `AmplitudeEmbedding`, `IQPEmbedding` and `ZZFeatureMap`; ansatzes `SELAnsatz`,
  `RealAmplitudesAnsatz`, `EfficientSU2Ansatz`, `SimplifiedTwoDesignAnsatz`,
  `BasicEntanglerAnsatz` and `CNOTLadderAnsatz`; measurements `measure_probs`,
  `measure_expval_z`, `measure_expval_x` and `measure_parity_z`.
- **Training.** `Trainer` with per-weight-group learning rates, `diff_method`
  selection, and a `check_bp=True` barren-plateau pre-flight. `EarlyStopping`,
  `ModelCheckpoint` and `HistoryCallback` work on both backends; checkpoints hold
  weights, optimizer state and history, and resume from any of them.
- **Data.** `DataModule` splits, normalizes on the train split only, and prescales
  inputs for the model's embedding. `DataModule.save`/`load` and `for_prediction`
  reuse a fitted DataModule on new rows.
- **Pipelines.** `QuantumPipeline` chains stages sequentially, as an ensemble, or
  trains them jointly as one hybrid network.
- **Losses.** `MSELoss`, `HingeLoss`, `CrossEntropyLoss`, or any callable.
- **Devices.** Any PennyLane device name, `lightning.qubit` and the PennyLane-Qiskit
  plugin included, via the `qiskit` extra.

### Bug fixes

- Bugfixes ([#35](https://github.com/phoeenniixx/pyQit/pull/35)) by [@phoeenniixx](https://github.com/phoeenniixx)
- bugfixes and removal of dead code ([#20](https://github.com/phoeenniixx/pyQit/pull/20)) by [@phoeenniixx](https://github.com/phoeenniixx)

### Enhancements

- add diff_method, run the nb and parameter shift debug ([#46](https://github.com/phoeenniixx/pyQit/pull/46)) by [@phoeenniixx](https://github.com/phoeenniixx)
- update tutorials and add tags collection ([#43](https://github.com/phoeenniixx/pyQit/pull/43)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Add dm checkpointing, update model checkpoint, chack_bp debug for joint pipeline ([#42](https://github.com/phoeenniixx/pyQit/pull/42)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Add pipeline support for hybrid models ([#40](https://github.com/phoeenniixx/pyQit/pull/40)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Add new Models ([#38](https://github.com/phoeenniixx/pyQit/pull/38)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add new anastz ([#37](https://github.com/phoeenniixx/pyQit/pull/37)) by [@phoeenniixx](https://github.com/phoeenniixx)
- refactor vqc and add ZZFeatureMap ([#36](https://github.com/phoeenniixx/pyQit/pull/36)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Improve API ([#34](https://github.com/phoeenniixx/pyQit/pull/34)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add test and validate API end points to trainer and pipeline ([#33](https://github.com/phoeenniixx/pyQit/pull/33)) by [@phoeenniixx](https://github.com/phoeenniixx)
- debugging and add tests for datamodule and pipeline ([#31](https://github.com/phoeenniixx/pyQit/pull/31)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add readthedocs ([#28](https://github.com/phoeenniixx/pyQit/pull/28)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add pennylane tensor as return_format option ([#24](https://github.com/phoeenniixx/pyQit/pull/24)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Refactor Trainer ([#21](https://github.com/phoeenniixx/pyQit/pull/21)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Major - 2 ([#17](https://github.com/phoeenniixx/pyQit/pull/17)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Major updates - 2 ([#3](https://github.com/phoeenniixx/pyQit/pull/3)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add vectorization ([#1](https://github.com/phoeenniixx/pyQit/pull/1)) by [@phoeenniixx](https://github.com/phoeenniixx)

### Documentation

- update docs to use tags ([#44](https://github.com/phoeenniixx/pyQit/pull/44)) by [@phoeenniixx](https://github.com/phoeenniixx)
- update docs ([#41](https://github.com/phoeenniixx/pyQit/pull/41)) by [@phoeenniixx](https://github.com/phoeenniixx)
- update docs ([#39](https://github.com/phoeenniixx/pyQit/pull/39)) by [@phoeenniixx](https://github.com/phoeenniixx)
- add new pyqit icon ([#29](https://github.com/phoeenniixx/pyQit/pull/29)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Add docs, community md etc ([#25](https://github.com/phoeenniixx/pyQit/pull/25)) by [@phoeenniixx](https://github.com/phoeenniixx)
- Add a new callbacks nb and fix a cross-entropy bug found due to new pennylane ver ([#23](https://github.com/phoeenniixx/pyQit/pull/23)) by [@phoeenniixx](https://github.com/phoeenniixx)
- update nbs, Readme and add workflows to run nbs ([#22](https://github.com/phoeenniixx/pyQit/pull/22)) by [@phoeenniixx](https://github.com/phoeenniixx)

### Maintenance

- [Dependabot](deps): Update scikit-base requirement from <1.1.0 to <1.2.0 ([#18](https://github.com/phoeenniixx/pyQit/pull/18)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps): Bump actions/setup-python from 6 to 7 ([#16](https://github.com/phoeenniixx/pyQit/pull/16)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps): Bump actions/checkout from 6 to 7 ([#15](https://github.com/phoeenniixx/pyQit/pull/15)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps): Update scikit-base requirement from <0.14.0 to <1.1.0 ([#12](https://github.com/phoeenniixx/pyQit/pull/12)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps-dev): Update setuptools requirement from >=70.0.0 to >=82.0.1 ([#11](https://github.com/phoeenniixx/pyQit/pull/11)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps-dev): Update torch requirement from !=2.0.1,<3.0.0,>=2.0.0 to !=2.0.1,>=2.11.0,<3.0.0 ([#10](https://github.com/phoeenniixx/pyQit/pull/10)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps): Update scikit-learn requirement from <2.0,>=1.2 to >=1.7.2,<2.0 ([#9](https://github.com/phoeenniixx/pyQit/pull/9)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps): Update pandas requirement from <3.1.0,>=1.3.0 to >=2.3.3,<3.1.0 ([#8](https://github.com/phoeenniixx/pyQit/pull/8)) by [@app/dependabot](https://github.com/app/dependabot)
- [Dependabot](deps-dev): Update ipywidgets requirement from <9.0.0,>=8.0.1 to >=8.1.8,<9.0.0 ([#7](https://github.com/phoeenniixx/pyQit/pull/7)) by [@app/dependabot](https://github.com/app/dependabot)

### Needs a label

- Minor update - 1 ([#13](https://github.com/phoeenniixx/pyQit/pull/13)) by [@phoeenniixx](https://github.com/phoeenniixx)

### Contributors

[@phoeenniixx](https://github.com/phoeenniixx)
