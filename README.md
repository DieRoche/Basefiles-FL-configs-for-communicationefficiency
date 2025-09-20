# Federated Learning Communication Metrics Utilities

This repository contains helper modules that make it easier to configure and
analyse communication-efficient federated learning experiments.  The
`metrics.py` module adds utilities for summarising loss/accuracy statistics and
for estimating the bandwidth used to broadcast the global state and ingest
client updates.

## Repository structure

Although the communication tracking helpers live in `metrics.py`, the project
also ships several reference components that can be reused when assembling a
federated learning pipeline:

- `config.py` collects baseline experiment settings and hyper-parameters that
  can be adapted for individual runs.
- `data_utils.py` provides dataset preparation helpers that normalise client
  partitions and central evaluation splits.
- `ResNet18.py` contains a torchvision-inspired CNN backbone that can serve as
  the shared model in image classification experiments.
- `effnet.py` mirrors the EfficientNet-style architecture that some
  communication efficient baselines employ.

These modules are intentionally lightweight so that they can be imported into
your own training scripts or modified to suit custom research scenarios.

## Tracking communication traffic

The :class:`~metrics.CommunicationMetricsTracker` class centralises the logic for
collecting round level metrics.  It automatically keeps running totals for
uploads and downloads and augments a user supplied report dictionary with
summary statistics such as the mean/standard deviation of training losses.

```python
from metrics import CommunicationMetricsTracker

tracker = CommunicationMetricsTracker()
report = tracker.summarise_round(
    training_losses=[0.5, 0.42, 0.47],
    client_accuracies=[0.81, 0.79, 0.83],
    server_accuracies=[0.80, 0.82],
    global_state=server_state_dict,
    participating_updates=[client_1_update, client_2_update],
)
```

The tracker uses the :func:`~metrics.tensor_dict_bytes` helper to estimate the
number of bytes transferred by serialising the provided mappings.  Nested
mappings and tensor containers are supported out of the box.

## Working with quantised models

When model parameters are quantised before being transmitted, the default byte
count estimation based on tensor element sizes might not be accurate.  To
address this, both :func:`~metrics.tensor_dict_bytes` and
:class:`~metrics.CommunicationMetricsTracker` expose two knobs:

- `dtype_size_overrides`: a mapping that specifies custom element sizes (in
  bytes) for individual dtypes.  Keys can be dtype objects or their string
  representations, which allows passing entries such as
  `{torch.float32: 1, "torch.quint8": 1}`.
- `default_tensor_element_size`: a fallback element size in bytes that is used
  when a tensor does not expose size information and no dtype specific override
  exists.

For example, if the global model is quantised to 8-bit values before being sent
out to clients you can instantiate the tracker like this:

```python
tracker = CommunicationMetricsTracker(
    dtype_size_overrides={"torch.float32": 1},
    default_tensor_element_size=1,
)
```

Quantised tensors produced by PyTorch are detected automatically and their
integer representation is used when estimating the payload.  The overrides above
are therefore only necessary when additional custom compression schemes are in
use.

Run `python -m compileall .` after editing the module to ensure there are no
syntax errors.
