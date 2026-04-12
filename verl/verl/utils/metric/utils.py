# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Metrics utils.
"""

from typing import Any, Dict, List

import numpy as np


def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean, max, or min of each list.
    The reduce operation is determined by the key name:
    - If the key contains "max", np.max is used
    - If the key contains "min", np.min is used
    - Otherwise, np.mean is used

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its reduced value.

    Example:
        >>> metrics = {
        ...     "loss": [1.0, 2.0, 3.0],
        ...     "accuracy": [0.8, 0.9, 0.7],
        ...     "max_reward": [5.0, 8.0, 6.0],
        ...     "min_error": [0.1, 0.05, 0.2]
        ... }
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8, "max_reward": 8.0, "min_error": 0.05}
    """
    def reduce_value(val, mode="mean"):
        if isinstance(val, (int, float)):   # 标量
            return val
        if val is None or len(val) == 0:    # 空 list
            return 0
        if mode == "max":
            return np.max(val)
        if mode == "min":
            return np.min(val)
        if mode == "median":
            return np.median(val)
        if mode == "q1":
            return np.percentile(val, 25)
        if mode == "q3":
            return np.percentile(val, 75)
        return np.mean(val)                 # 默认 mean

    for key, val in metrics.items():
        if "max" in key:
            metrics[key] = reduce_value(val, "max")
        elif "min" in key:
            metrics[key] = reduce_value(val, "min")
        elif "q1" in key:
            metrics[key] = reduce_value(val, "q1")
        elif "median" in key:
            metrics[key] = reduce_value(val, "median")
        elif "q3" in key:
            metrics[key] = reduce_value(val, "q3")
        else:
            metrics[key] = reduce_value(val, "mean")
    return metrics
