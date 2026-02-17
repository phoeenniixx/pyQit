# PyQit

## Current Plan (sub to change any time >_<)

- anastz -> `QuantumModel` wrapper that connects the anastz and DL model (if any) -> `Trainer` (with/w/o `torch`)
- Data preprocessing techniques (no idea how to do that rn)
- any anastz could go with any DL backbone (ideally, not sure how much is feasible - tbd)

```python
anastz = anyanastz()
DLmodel = DLmodel()
Qm = QuantumModel(anastz, DLmodel)
trainer = Trainer()
trainer.fit(Qm) # uses torch
```
And if no DLmodel then
```python
anastz = anyanastz()
Qm = QuantumModel(anastz)
trainer = Trainer()
trainer.fit(Qm) # uses pennylane only
```
here `anastz` and `DLmodel` can be implemented by the user himself or use the implemented ones from the package
Then package would also have a complete model zoo. That would contain anatsz/hybrid models wrapped using `QuantumModel`.

User would simply call
```python
myModel= hybridmodel() #QuantumModel object
trainer=Trainer()
trainer.fit(myModel)
```
These would be the official model zoo


