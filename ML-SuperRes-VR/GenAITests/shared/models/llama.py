# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Llama model class"""

from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
from transformers.models.llama import modeling_llama

from .base import LLM


class Llama_32(LLM):
    """Generic LLaMa 3.2"""

    DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

    @classmethod
    def instantiate_model(
        cls, model_id: str, small_model: bool = False
    ) -> PreTrainedModel:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        llm_config = AutoConfig.from_pretrained(
            model_id, trust_remote_code=True, attn_implementation="eager"
        )
        if small_model:
            llm_config.num_hidden_layers = 2
        return modeling_llama.LlamaForCausalLM.from_pretrained(
            model_id, config=llm_config
        )

    @classmethod
    def instantiate_tokenizer(cls, model_id: str) -> PreTrainedTokenizer:
        if model_id is None:
            model_id = cls.DEFAULT_MODEL_ID

        return AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )
