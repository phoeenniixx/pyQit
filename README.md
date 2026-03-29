# PyQit

## Current Plan (sub to change any time >_<)

- anastz -> `QuantumPipeline` wrapper that connects the anastz and DL model (if any) -> `Trainer` (with/w/o `torch`)
- Data preprocessing techniques (no idea how to do that rn)
- any anastz could go with any DL backbone (ideally, not sure how much is feasible - tbd)

```python
QMLmodel = anyQMLmodel()
DLmodel = DLmodel()
Qm = QuantumPipeline([
        DLmodel,
        anyQMLmodel,
    ])
trainer = Trainer()
trainer.fit(Qm) # uses torch
```
And if no DLmodel then
```python
QMLmodel = anyQMLmodel()
Qm = QuantumPipeline([
        anyQMLmodel,
        anyQMLmodel,
    ])
trainer = Trainer()
trainer.fit(Qm) # uses pennylane only
```
You can also train just QMLmodel using `Trainer`
here `anyQMLmodel` and `DLmodel` can be implemented by the user himself or use the implemented ones from the package
Then package would also have a complete model zoo.

User would simply call
```python
myModel= hybridmodel()
trainer=Trainer()
trainer.fit(myModel)
```
These would be the official model zoo
