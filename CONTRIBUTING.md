# Contributing

PyQit is `0.1.0b1`, unstable, and mostly one maintainer plus dependabot. Issues and PRs
are welcome, and this doubles as a note-to-self for keeping things consistent.

## Setup

```bash
git clone https://github.com/phoeenniixx/pyqit.git
cd pyqit
pip install -e ".[dev]"              # base + test tooling
pip install -e ".[dev,all_extras]"   # + torch, lightning, matplotlib, rich
pre-commit install
```

`all_extras` is what CI runs against, so a PR touching torch/lightning/matplotlib/rich
needs it locally to actually exercise that code. Anything new in that area also needs to
degrade gracefully with only `[dev]` installed, since CI runs both.

## Before opening a PR

```bash
python -m pytest
pre-commit run --all-files       # ruff, ruff-format, nbQA on notebooks
```

Run the notebooks if you touched anything they import. `run-notebook-tutorials` runs them
in CI too, `--inplace`, so the same command regenerates their stored outputs locally
before you commit.

## Adding an ansatz, embedding, model, or loss

There's no registration step. A class with the right `object_type` tag is picked up by
`all_objects()` and enrolled in the test suite automatically, as long as it implements
`get_test_params()`. One catch: skbase's discovery walk skips any module whose name
starts with `_`, so a class defined there is silently never found. Put it somewhere
else.

## Soft dependencies

torch, lightning, matplotlib, and rich are optional. Guard any import of them with
`skbase.utils.dependencies._check_soft_dependencies(..., severity="none")`, not
`_safe_import`. `_safe_import` falls back to a `MagicMock`, and a missing dependency
then returns mock objects instead of raising, which fails in stranger ways somewhere
downstream instead of at the import.

## Labeling PRs

`build_tools/changelog.py` builds the changelog from merged PRs, sorted into Bug fixes,
Enhancements, Documentation, and Maintenance by GitHub label first, a keyword guess from
the title second. Add one of `bug`, `enhancement`, `documentation`, `maintenance` when
you open or merge a PR. An unlabeled PR that also doesn't match a keyword lands in a
"Needs a label" bucket rather than getting force-fit into one of the four; the fix is to
label it and rerun the script, not to hand-edit the output.

## Commit messages

This repo squash-merges, so a PR's title becomes its one commit message on `main`. Write
the title as you want it to read in the changelog and commit history.
