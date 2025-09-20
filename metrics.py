"""Utilities for aggregating training statistics and estimating communication traffic.

This module contains helper functionality that can be used when running
federated-learning style experiments.  The central entry point is the
:class:`CommunicationMetricsTracker` which can be used to accumulate the
communication volume across multiple rounds while summarising statistics of the
metrics reported by participating clients.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import math
import numpy as np


def _lookup_size_override(
    dtype: Any, overrides: Optional[Mapping[Any, int]]
) -> Optional[int]:
    """Return an override element size for ``dtype`` if available."""

    if not overrides:
        return None

    candidates = [dtype]

    # ``numpy`` dtypes expose a ``name`` attribute while ``torch`` dtypes can be
    # turned into strings that uniquely identify them (e.g. ``torch.float32``).
    name = getattr(dtype, "name", None)
    if name is not None:
        candidates.append(name)
    if dtype is not None:
        candidates.append(str(dtype))

    for candidate in candidates:
        if candidate in overrides:
            return int(overrides[candidate])
    return None


def _tensor_like_nbytes(
    value: Any,
    *,
    dtype_size_overrides: Optional[Mapping[Any, int]] = None,
    default_element_size: Optional[int] = None,
) -> int:
    """Return the number of bytes required to serialise ``value``.

    The helper understands ``torch.Tensor`` instances, :class:`numpy.ndarray`
    objects, builtin ``bytes``/``bytearray`` values and (nested) containers that
    yield one of those types.  Optional ``dtype_size_overrides`` and
    ``default_element_size`` arguments allow callers to adjust the calculation
    for scenarios such as model quantisation.  The function falls back to the
    ``nbytes`` attribute for objects such as tensors living on special device
    wrappers.  ``RuntimeError`` is raised for unsupported values so that
    mistakes become visible during development instead of silently producing
    incorrect traffic estimations.
    """

    if value is None:
        return 0

    # ``torch`` is an optional dependency.  Importing locally avoids paying the
    # cost when it is not installed while still allowing us to handle tensors.
    try:  # pragma: no cover - import guard is trivial to reason about.
        import torch  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - exercised when torch missing.
        torch = None  # type: ignore

    if torch is not None and isinstance(value, torch.Tensor):
        override = _lookup_size_override(getattr(value, "dtype", None), dtype_size_overrides)
        if override is not None:
            return int(value.nelement()) * override

        if getattr(value, "is_quantized", False):
            try:
                int_repr = value.int_repr()
            except (RuntimeError, AttributeError):
                int_repr = None

            if int_repr is not None:
                base = int(int_repr.nelement()) * int(int_repr.element_size())
                # Account for quantization metadata.  Per-tensor quantisation stores
                # ``scale`` and ``zero_point`` values while per-channel additionally
                # tracks the axis.  We conservatively count float32/int32 values.
                try:
                    scales = value.q_per_channel_scales()
                    zero_points = value.q_per_channel_zero_points()
                    metadata = (
                        int(scales.nelement()) * int(scales.element_size())
                        + int(zero_points.nelement()) * int(zero_points.element_size())
                        + 4  # channel axis as a 32-bit integer
                    )
                except (RuntimeError, AttributeError):
                    # Per-tensor quantisation exposes scalar accessors instead of
                    # tensors.  We model them as 32-bit values.
                    metadata = 8

                return base + metadata

        element_size = None
        try:
            element_size = int(value.element_size())
        except (AttributeError, TypeError):
            element_size = None

        if element_size is None or element_size == 0:
            element_size = _lookup_size_override(type(value), dtype_size_overrides)

        if element_size is None:
            element_size = default_element_size

        if element_size is None:
            raise RuntimeError(
                "Unable to determine element size for tensor while estimating traffic"
            )

        return int(value.nelement()) * int(element_size)

    if isinstance(value, np.ndarray):
        override = _lookup_size_override(value.dtype, dtype_size_overrides)
        if override is not None:
            return int(value.size) * override
        return int(value.nbytes)

    if hasattr(value, "nbytes") and isinstance(value.nbytes, (int, float)):
        return int(value.nbytes)

    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)

    if isinstance(value, Mapping):
        return tensor_dict_bytes(
            value,
            dtype_size_overrides=dtype_size_overrides,
            default_element_size=default_element_size,
        )

    if isinstance(value, (list, tuple)):
        return sum(
            _tensor_like_nbytes(
                v,
                dtype_size_overrides=dtype_size_overrides,
                default_element_size=default_element_size,
            )
            for v in value
        )

    raise RuntimeError(
        "Unsupported value encountered while estimating tensor dictionary size: "
        f"{type(value)!r}"
    )


def tensor_dict_bytes(
    tensors: Mapping[Any, Any],
    *,
    dtype_size_overrides: Optional[Mapping[Any, int]] = None,
    default_element_size: Optional[int] = None,
) -> int:
    """Return the total number of bytes consumed by a tensor dictionary.

    Parameters
    ----------
    tensors:
        Mapping that contains tensors or other supported container objects.
    dtype_size_overrides:
        Optional mapping that specifies custom byte sizes for individual
        ``dtype`` values.  Keys can be the actual dtype instances or their
        string representations.
    default_element_size:
        Fallback element size (in bytes) used when a dtype specific override is
        not available and no direct size information can be inferred.

    Returns
    -------
    int
        Estimated number of bytes when serialising the mapping.
    """

    return sum(
        _tensor_like_nbytes(
            value,
            dtype_size_overrides=dtype_size_overrides,
            default_element_size=default_element_size,
        )
        for value in tensors.values()
    )


def _mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    """Return the mean and standard deviation of ``values``.

    ``numpy`` is used for the actual computation to keep the implementation
    concise.  The helper gracefully handles empty and single element sequences by
    returning ``math.nan``/``0.0`` respectively.
    """

    if not values:
        return math.nan, math.nan
    if len(values) == 1:
        return float(values[0]), 0.0

    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _prepare_sequence(values: Iterable[Union[float, int]]) -> List[float]:
    """Materialise ``values`` into a list of floats."""

    return [float(v) for v in values]


@dataclass
class CommunicationMetricsTracker:
    """Track statistics for federated learning rounds.

    The tracker keeps cumulative upload and download traffic and provides a
    convenience :meth:`summarise_round` method that collects descriptive
    statistics for training loss and accuracies.  ``dtype_size_overrides`` and
    ``default_tensor_element_size`` allow fine tuning the traffic estimation for
    situations where tensors are transmitted using custom precisions (for
    example after quantisation).
    """

    total_upload_traffic: int = 0
    total_download_traffic: int = 0
    report_defaults: Mapping[str, Any] = field(default_factory=dict)
    dtype_size_overrides: Mapping[Any, int] = field(default_factory=dict)
    default_tensor_element_size: Optional[int] = 4

    def summarise_round(
        self,
        *,
        training_losses: Iterable[Union[float, int]],
        client_accuracies: Iterable[Union[float, int]],
        server_accuracies: Iterable[Union[float, int]],
        global_state: Mapping[Any, Any],
        participating_updates: Iterable[Mapping[Any, Any]],
        report: Optional[MutableMapping[str, Any]] = None,
    ) -> MutableMapping[str, Any]:
        """Summarise statistics for a single FL communication round.

        Parameters
        ----------
        training_losses, client_accuracies, server_accuracies:
            Numeric measurements collected during the round.  Iterables are
            materialised internally to allow for generator inputs.
        global_state:
            Mapping representing the state distributed to clients.
        participating_updates:
            Iterable of updates returned by the participating clients.
        report:
            Optional mapping that is updated with the computed statistics.

        Returns
        -------
        MutableMapping[str, Any]
            The mapping that contains the combined statistics.
        """

        report_mapping: MutableMapping[str, Any]
        if report is None:
            report_mapping = dict(self.report_defaults)
        else:
            report_mapping = report

        losses = _prepare_sequence(training_losses)
        acc_clients = _prepare_sequence(client_accuracies)
        acc_servers = _prepare_sequence(server_accuracies)

        training_loss_mean, training_loss_std = _mean_and_std(losses)
        acc_clients_mean, acc_clients_std = _mean_and_std(acc_clients)
        acc_servers_mean, acc_servers_std = _mean_and_std(acc_servers)

        report_mapping.update(
            {
                "training_loss_mean": training_loss_mean,
                "training_loss_std": training_loss_std,
                "acc_clients_mean": acc_clients_mean,
                "acc_clients_std": acc_clients_std,
                "acc_servers_mean": acc_servers_mean,
                "acc_servers_std": acc_servers_std,
                "training_loss_lowest": training_loss_mean - training_loss_std,
                "training_loss_highest": training_loss_mean + training_loss_std,
                "acc_clients_lowest": acc_clients_mean - acc_clients_std,
                "acc_clients_highest": acc_clients_mean + acc_clients_std,
                "acc_servers_lowest": acc_servers_mean - acc_servers_std,
                "acc_servers_highest": acc_servers_mean + acc_servers_std,
            }
        )

        download_traffic = tensor_dict_bytes(
            global_state,
            dtype_size_overrides=self.dtype_size_overrides,
            default_element_size=self.default_tensor_element_size,
        )
        upload_traffic = sum(
            tensor_dict_bytes(
                update,
                dtype_size_overrides=self.dtype_size_overrides,
                default_element_size=self.default_tensor_element_size,
            )
            for update in participating_updates
        )

        self.total_upload_traffic += upload_traffic
        self.total_download_traffic += download_traffic

        report_mapping.update(
            {
                "upload_traffic": upload_traffic,
                "download_traffic": download_traffic,
                "overall_traffic": self.total_upload_traffic + self.total_download_traffic,
                "total_upload_traffic": self.total_upload_traffic,
                "total_download_traffic": self.total_download_traffic,
            }
        )

        return report_mapping
