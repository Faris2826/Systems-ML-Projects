.. _ptq-adascale:

#########
AdaScale
#########

Context
=======

AdaScale is a PTQ technique which improves the accuracy of the quantized model by computing optimal quantization parameters for weights. AdaScale is based on FlexRound: https://arxiv.org/abs/2306.00317 and integrates `Learnable Weight Clipping` from OmniQuant: https://arxiv.org/abs/2308.13137.

AdaScale introduces trainable parameters (gamma, beta, s2, s3) in the weight quantizers of every supported module and performs BKD (Blockwise Knowledge Distillation) by comparing quantized output of every supported block with its FP32 equivalent.

From AdaScale perspective, a block is defined as a non-leaf module which takes in one activation input tensor and outputs one activation tensor. AdaScale also requires blocks to be contiguous to perform optimization.

Warning: This feature is currently experimental.

Workflow
========

Prerequisites
-------------

To use AdaScale, you must:

- Use PyTorch. AdaScale does not support other frameworks yet
- Load a pre-trained model
- Create a dataloader for the model
- Choose a model which has contiguous blocks, and each block taking in one activation input and outputting one activation tensor. Example block: LlamaDecoderLayer in LlamaModel

Procedure
---------

Setup
~~~~~

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [setup]
            :end-before: # End of [setup]

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [prepare-dataloader]
            :end-before: # End of [prepare-dataloader]

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 1
~~~~~~

Use AIMET's :ref:`quantization simulation<quantsim-index>` to create a QuantSimModel object.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [create-sim]
            :end-before: # End of [create-sim]

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


Step 2
~~~~~~

Apply AdaScale to decide optimal quantization encodings for parameters of supported layers.
It is recommended to use a minimum of 1500 iterations when applying AdaScale regardless of the dataloader batch size.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [apply-adascale]
            :end-before: # End of [apply-adascale]

    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 3
~~~~~~

Compute encodings for remaining parameters of the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [compute_encodings]
            :end-before: # End of [compute_encodings]
    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 4
~~~~~~

Evaluate the quantized model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [evaluation]
            :end-before: # End of [evaluation]
    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

Step 5
~~~~~~

If the resulting quantized accuracy is satisfactory, export the model.

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. literalinclude:: ../snippets/torch/apply_adascale.py
            :language: python
            :start-after: # [export]
            :end-before: # End of [export]
    .. tab-item:: ONNX
        :sync: onnx

        Not supported.

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: PyTorch
        :sync: torch

        .. include:: ../apiref/torch/adascale.rst
            :start-after: # start-after
    .. tab-item:: ONNX
        :sync: onnx

        Not supported.


