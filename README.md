# Basefiles-FL-configs-for-communicationefficiency

## Overview
This repository contains the minimal scaffolding used to run communication-efficiency experiments for federated learning (FL) on image classification tasks. It serves as a starting point for organizing configuration files, experiment scripts, and generated results when benchmarking different FL strategies under varying communication budgets.

## Repository layout
Currently the repository includes the following top-level files:

| Path | Description |
| --- | --- |
| `README.md` | Project overview and guidance on how to structure additional configuration, source, and report files for communication-efficient FL experiments. |

When populating the project for a specific study, it is recommended to add:

* **`configs/`** – YAML or JSON configuration files describing datasets, model architectures, federated aggregation strategies, and communication constraints.
* **`experiments/`** – Notebooks or scripts that launch FL runs using the configurations and log metrics.
* **`reports/`** – Generated summaries, plots, and tables capturing the key communication and performance metrics for each experiment.
* **`artifacts/`** – Optional directory for storing trained model checkpoints or intermediate tensors needed for reproducibility.

Structuring the repository in this way keeps configurations, code, and outputs organized and makes it easier to compare experiment variants.

## Report expectations
Communication-efficiency reports should include both accuracy/loss statistics and traffic measurements gathered across rounds. A typical report dictionary is expected to contain the following derived values:

```python
report["training_loss_lowest"] = training_loss_mean - training_loss_std
report["training_loss_highest"] = training_loss_mean + training_loss_std
report["acc_clients_lowest"] = acc_clients_mean - acc_clients_std
report["acc_clients_highest"] = acc_clients_mean + acc_clients_std
report["acc_servers_lowest"] = acc_servers_mean - acc_servers_std
report["acc_servers_highest"] = acc_servers_mean + acc_servers_std

download_traffic = tensor_dict_bytes(global_state)
upload_traffic = sum(tensor_dict_bytes(update) for update in participating_updates)
total_upload_traffic += upload_traffic
total_download_traffic += download_traffic
report["upload_traffic"] = upload_traffic
report["download_traffic"] = download_traffic
report["overall_traffic"] = total_upload_traffic + total_download_traffic
```

These fields make it possible to understand not only the central tendency of model performance but also the variability across clients/servers and the cumulative communication overhead. Tracking both the loss/accuracy bounds and the upload/download traffic is essential for evaluating how design choices impact the overall efficiency of the federated learning system.

## Next steps
To extend this baseline, add your configuration files, scripts, and experiment documentation to the respective directories. Ensure that each report includes the metrics above so downstream analysis tools can consume a consistent schema.
