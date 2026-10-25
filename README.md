# DAFTED

DAFTED is a multimodal transformer for patient-level data that combines **tabular attributes** (e.g. clinical
records, both numerical and categorical) with **time-series attributes** (e.g. curves measured over a cardiac cycle,
possibly for several acquisition views).

Each attribute is turned into a token: tabular attributes go through an FT-Transformer-style feature tokenizer, and
time series through a dedicated tokenizer. Each modality is then encoded by its own transformer. The DAFTED encoder
splits the tabular tokens into a part shared with the time series and a part unique to the tabular data, and fuses the
modalities with cross-attention. Training combines the supervised prediction losses with a decoupling loss and a
supervised inter-sample contrastive loss.

The code is built on [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) and configured with
[Hydra](https://hydra.cc/). It was developed on the private CARDINAL echocardiography dataset, which is **not**
distributed here. The repository instead ships a **synthetic dataset**, so that you can run everything end-to-end and
plug in your own data.

## Installation

DAFTED requires Python 3.10. Dependencies are managed with [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/Jeremstym/DAFTED.git
cd DAFTED
poetry install
```

By default, PyTorch is installed from the CUDA 11.8 wheel index (see `[[tool.poetry.source]]` in `pyproject.toml`).
If you need another CUDA version or a CPU-only build, edit that source before running `poetry install`.

## Quick start: synthetic data

Train DAFTED from scratch on the synthetic dataset, then evaluate it on the test set and write predictions:

```bash
poetry run dafted-runner +experiment=synthetic/dafted
```

This runs in a few minutes on a laptop. Results are saved in Hydra's output directory (`outputs/<date>/<time>/` by
default):

- `checkpoints/`: model checkpoints;
- `lightning_logs/`: TensorBoard logs (`tensorboard --logdir outputs`);
- `predictions/`: per-subset classification scores (`*_categorical_scores.csv`) and a 2D embedding of the learned
  representations (`features_wrt_*.png`).

Any configuration value can be overridden from the command line, e.g.:

```bash
poetry run dafted-runner +experiment=synthetic/dafted trainer.max_steps=2000 task.embed_dim=64 data.batch_size=128
```

## Using your own data

DAFTED never hard-codes the variables of a dataset. Plugging in a new dataset takes four steps.

### 1. Describe your dataset in a schema

A schema is a YAML file that lists the views, the time-series attributes and the tabular attributes (with their type)
of your dataset. Copy [`dafted/data/schemas/synthetic.yaml`](dafted/data/schemas/synthetic.yaml) and adapt it:

```yaml
views: [A4C, A2C]                  # Views for which time series are available (can be empty)
time_series_attrs: [gls, lv_area]  # Time-series attributes, available for each view (can be empty)
tabular_attrs:
  age: numerical
  sex: [M, F]                      # Categorical: list of labels, encoded by their index
  diagnosis: [healthy, mild, severe]
ordinal_attrs: [diagnosis]         # Categorical attributes whose labels are ordered
```

Then point the `DAFTED_SCHEMA` environment variable to your schema. The simplest way is a `.env` file at the root of
the repository, which is loaded automatically:

```bash
DAFTED_SCHEMA=/path/to/my_schema.yaml
```

The runner logs the schema in use at startup. In the code, the schema is exposed by
[`dafted/data/schema.py`](dafted/data/schema.py) as the `TabularAttribute`, `TimeSeriesAttribute` and `View` enums.

### 2. Write a data module

A data module is a Lightning `DataModule` that inherits from `dafted.core.data.data_module.VitalDataModule` and
yields one dict per patient:

```python
{
    "id": "patient_0001",
    "age": np.array(63.0, dtype=np.float32),        # numerical attribute, np.nan if missing
    "sex": np.array(1, dtype=np.int64),             # categorical attribute: index of the label, -1 if missing
    "A4C": {"gls": np.ndarray(shape=(64,), dtype=np.float32), ...},  # time series of each view
    ...
}
```

[`dafted/data/synthetic.py`](dafted/data/synthetic.py) is a complete, documented example of the contract a data
module must fulfil (input shapes, prediction dataloaders, etc.). Start from it.

### 3. Add a data config

Create `dafted/config/data/<my_dataset>.yaml` that instantiates your data module, following
[`dafted/config/data/synthetic.yaml`](dafted/config/data/synthetic.yaml).

### 4. Add an experiment config

Copy [`dafted/config/experiment/synthetic/dafted.yaml`](dafted/config/experiment/synthetic/dafted.yaml), set
`override /data` to your data config, list the attributes to use, and pick the target(s) to predict under
`task.predict_losses`. Then run:

```bash
poetry run dafted-runner +experiment=<my_dataset>/dafted
```

## Repository structure

```
dafted/
├── config/            # Hydra configs
│   ├── data/          #   data modules (synthetic, cardinal)
│   ├── experiment/    #   ready-to-run experiments (synthetic/, cardinal/)
│   └── task/          #   models, tokenizers, optimizers, ...
├── core/              # Generic training framework: runner, Lightning system/data module, utilities
├── data/
│   ├── schema.py      # Dataset schema: attributes and their types, read from a YAML file
│   ├── schemas/       # Example schema of the synthetic dataset
│   ├── synthetic.py   # Synthetic data module (template for your own data)
│   └── cardinal/      # Prediction writer (metrics, embedding plots)
├── models/            # Tokenizers, transformers (DAFTED encoder, FT-Transformer, ...), baselines, losses
├── tasks/cardinal/    # Lightning tasks: multimodal representation learning, records-only prediction
└── runner.py          # Entry point (`dafted-runner`)
IRENE/                 # Third-party IRENE implementation, used as a baseline (see IRENE/LICENSE)
```

## Logging

Experiments log to [Comet](https://www.comet.com/) by default (`logger=comet/online`), which needs the
`COMET_API_KEY` environment variable (it can go in `.env`). To log locally instead, use TensorBoard with
`logger=tensorboard`, as the synthetic experiment does.

## CARDINAL experiments

The configs under `dafted/config/experiment/cardinal/` reproduce the experiments on the CARDINAL dataset. They need
access to the private data and to its data module, which are not part of this repository.

## License and acknowledgements

This project is released under the [Apache-2.0 license](LICENSE). The generic training framework in `dafted/core/` is
adapted from the [vital](https://github.com/vitalab/vital) library (Apache-2.0). The IRENE baseline in `IRENE/` comes
from its [original implementation](https://github.com/RL4M/IRENE) and keeps its own license.


The refactoring of this repository for its public release (dataset schema, synthetic data module, decoupling from
the private CARDINAL code) and this README were developed with the help of
[Claude Code](https://claude.com/claude-code), Anthropic's AI coding assistant.

<!--
## Citation

If you use this code, please cite:

```bibtex
@article{TODO,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
}
```
-->
