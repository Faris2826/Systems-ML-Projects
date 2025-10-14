# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""General utils for GenAI model testing"""

import torch
import threading
import psutil
import time
import os
import json
import collections


def _format_bytes(rawbytes: int, unit: str = "GB") -> float:
    """Helper function to format collected statistics to specified unit"""
    if unit == "KB":
        return rawbytes / 1024
    if unit == "MB":
        return rawbytes / (1024**2)
    if unit == "GB":
        return rawbytes / (1024**3)

    raise ValueError("Unsupported byte unit")


class ResourceProfiler:
    """Context manager to monitor resource consumption"""

    def __init__(
        self, sampling_frequency: float = 1.0, disable_constant_sampling: bool = False
    ):
        self.cuda_memory_allocated = []
        self.cuda_memory_reserved = []
        self.cpu_memory_usage = []
        self.disable_constant_sampling = disable_constant_sampling

        if not self.disable_constant_sampling:
            self._stop_event = threading.Event()
            self.sampling_frequency = sampling_frequency

    def _monitor_memory(self):
        while not self._stop_event.is_set():
            self.cuda_memory_allocated.append(torch.cuda.memory_allocated())
            self.cuda_memory_reserved.append(torch.cuda.memory_reserved())
            self.cpu_memory_usage.append(psutil.virtual_memory().used)
            time.sleep(self.sampling_frequency)

    def __enter__(self):
        # pylint: disable=attribute-defined-outside-init
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.perf_counter()

        if not self.disable_constant_sampling:
            self.thread = threading.Thread(target=self._monitor_memory)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # pylint: disable=attribute-defined-outside-init
        self.stop_time = time.perf_counter()

        if not self.disable_constant_sampling:
            self._stop_event.set()
            self.thread.join()

    def runtime(self) -> float:
        """Report time spent inside context manager"""
        return self.stop_time - self.start_time

    def peak_ram_usage(self, unit: str = "GB") -> float:
        """Report peak RAM usage inside context manager"""
        return _format_bytes(max(self.cpu_memory_usage), unit)

    def min_ram_usage(self, unit: str = "GB") -> float:
        """Report min RAM usage inside context manager"""
        return _format_bytes(min(self.cpu_memory_usage), unit)

    def peak_cuda_usage(self, unit: str = "GB") -> float:
        """Report peak CUDA usage inside context manager"""
        return _format_bytes(torch.cuda.max_memory_allocated(), unit)

    def min_cuda_usage(self, unit: str = "GB") -> float:
        """Report min CUDA usage inside context manager"""
        return _format_bytes(min(self.cuda_memory_allocated), unit)

    def cuda_memory_usage(self, unit: str = "GB") -> tuple[float, ...]:
        """Report raw CUDA usage data inside context manager"""
        return tuple(_format_bytes(mem, unit) for mem in self.cuda_memory_allocated)

    def ram_memory_usage(self, unit: str = "GB") -> tuple[float, ...]:
        """Report raw RAM usage data inside context manager"""
        return tuple(_format_bytes(mem, unit) for mem in self.cpu_memory_usage)

    def as_dict(self, unit: str = "GB"):
        """Report all collected stats as dict"""
        if self.disable_constant_sampling:
            return {
                "runtime": self.runtime(),
                "peak_cuda_usage": self.peak_cuda_usage(unit),
            }

        return {
            "runtime": self.runtime(),
            "peak_ram_usage": self.peak_ram_usage(unit),
            "min_ram_usage": self.min_ram_usage(unit),
            "peak_cuda_usage": self.peak_cuda_usage(unit),
            "min_cuda_usage": self.min_cuda_usage(unit),
            "cuda_memory_usage": self.cuda_memory_usage(unit),
            "cpu_memory_usage": self.ram_memory_usage(unit),
        }


def recursive_update(d, u):
    """Internal helper function to update nested dict"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def write_stats_to_disk(
    filename, model_cls, model_params, quant_recipe_cls, quant_recipe_params, stats
):
    """Helper function to write collected stats to disk, only overwriting newly collected fields"""

    # Check if the file exists
    if os.path.exists(filename):
        # Open the file and load the existing data
        with open(filename, "r") as f:
            data = json.load(f)
    else:
        # If the file does not exist, create an empty dictionary
        data = {}

    quant_params_string_formatted = ", ".join(
        [f"{key}={value}" for key, value in quant_recipe_params.items()]
    )
    model_params_string_formatted = ", ".join(
        [f"{key}={value}" for key, value in model_params.items()]
    )

    # Update the dictionary with x
    x = {f"{quant_params_string_formatted}": stats}
    x = {f"{quant_recipe_cls.__name__}": x}
    x = {f"{model_params_string_formatted}": x}
    x = {f"{model_cls.__name__}": x}
    recursive_update(data, x)

    # Write the updated dictionary back to the file
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
