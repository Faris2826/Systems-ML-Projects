# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Phi-3.5 model class"""

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.phi3 import modeling_phi3

from .base import LLM


class Phi_3(LLM):
    """Generic Phi-3"""

    DEFAULT_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
        return modeling_phi3.Phi3ForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
