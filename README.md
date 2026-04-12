# PyQit

## Current Plan (sub to change any time >_<)

- ansatz -> `QuantumPipeline` wrapper that connects the ansatz and DL model (if any) -> `Trainer` (with/w/o `torch`)
- Data preprocessing techniques (no idea how to do that rn) - maybe using a `lightning` type of data module (thanks a lot `lightning`). But as `lightning` has a core dep of `torch`, I have to reinvent(?) it ig
- any ansatz could go with any DL backbone (ideally, not sure how much is feasible - tbd)
- `torch` is used with `lightning` as a imp soft-dep now!!

*Will add better vignettes once i have my ideas consolidated in my mind*

> ***Have a look at a basic tutorial [here](https://github.com/phoeenniixx/pyQit/tree/main/doc/tutorials/vqc.ipynb) (This is how the flow would look like in future, still working on making data module and Trainer etc more "user friendly" and expressive)***

for some idea how it might look like:
```python
# no torch
pyqit.set_backend("pennylane")
qml_model= QMLmodel(...) # may use their own ansatz?
dm = DataModule(...)
trainer = Trainer(...)
trainer.fit(qml_model, dm)
trainer.predict(qml_model, dm_new, return_format = "numpy") # or "torch" for torch tensors if torch is backend,
# should i add pennylane tensors as well? good question!
```
### Using Pipeline
```python
pyqit.set_backend("torch")
dm = DataModule(...)
model_a = QMLmodel(**params)
model_b = QMLmodel(**params) # or DLModel for that matter
trainer = Trainer( max_epochs=10, learning_rate=0.01)
pipeline = QuantumPipeline(
            [
                PipelineStage(model_a, name="stage_1", trainable=trainable_a),
                PipelineStage(model_b, name="stage_2", trainable=True),
            ],
            mode="sequential",
        )
pipeline.fit(datamodule=dm, trainers=trainer, fit_mode="sequential_greedy")
preds = pipeline.predict(X_new, batch_size=8)
```
You can also train just QMLmodel using `Trainer`
here `anyQMLmodel` and `DLmodel` can be implemented by the user themselves or use the implemented ones from the package
Then package would also have a complete model zoo.
