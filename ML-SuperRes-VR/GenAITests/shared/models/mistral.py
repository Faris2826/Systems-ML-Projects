# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Mistral model class"""

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.mistral import modeling_mistral

from .base import LLM


class Mistral_03(LLM):
    """Mistral Instruct v0.3"""

    DEFAULT_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        if small_model:
            llm_config.num_hidden_layers = 2
        return modeling_mistral.MistralForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
