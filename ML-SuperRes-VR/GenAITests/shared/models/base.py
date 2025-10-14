# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""LLM base class for GenAI test framework"""

from abc import abstractmethod, ABC
from pathlib import Path
from transformers import PreTrainedTokenizer, PreTrainedModel


class LLM(ABC):
    @classmethod
    @abstractmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        """Instantiate model"""
        pass

    @staticmethod
    @abstractmethod
    def instantiate_tokenizer(model_id: str) -> PreTrainedTokenizer:
        """Instantiate model tokenizer"""

    @staticmethod
    def get_quantsim_config() -> str:
        """Get default QuantSim config"""
        config_path = Path(__file__).parent / "config/default_config.json"
        return str(config_path.resolve())
