.. include:: ../../abbreviation.txt

.. _featureguide-litemp:

####################
Lite mixed precision
####################

Lite Mixed Precision (Lite-MP) automatically identifies the layers most sensitive to quantization and selectively increases the precision (e.g., to ``INT16`` or ``FP16``) for a configurable percentage of those layers.

Lite-MP is only available in ``aimet-onnx``.

In ``aimet-onnx``, Lite-MP enables mixed precision through the following steps:

Step 1: QuantAnalyzer

This component profiles the model by enabling quantization one layer at a time and measures the PSNR (Peak Signal-to-Noise Ratio) between the original FP32 output(s) and the quantized output(s). It then generates a sensitivity report that ranks layers based on their impact on model accuracy.

Step 2: Precision Adjustment via Lite-MP

Using the sensitivity report, AIMET Lite-MP sorts layers by their PSNR degradation. The layers with the highest sensitivity—those causing the greatest drop in PSNR—are selected and converted to higher precision to preserve model performance.

.. image:: ../../images/LiteMixedPrecisionWorkflow.png

Lite-MP can quickly determine mixed-precision configuration, balancing the model accuracy with inference latency.

Workflow
========

Let’s take an example for applying Lite-MP to a MobileNetV2 model.

Prerequisites
-------------

1. Download ImageNet dataset

.. code-block:: bash

    wget -P ./imagenet_dataset https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
    wget -P ./imagenet_dataset https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar

If you already have imagenet dataset locally that you would like to use, simply replace dataset path from `imagenet_dataset` later.

2. Load PyTorch model and dataset

.. note::

    The examples below use a pre-trained MobileNetV2 model. You can also load your model instead.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # imports start
            :end-before: # imports end

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Load the model
            :end-before:  # End of loading the model

        Optionally simplify the exported onnx graph before quantization. This is not strictly required but
        may improve accuracy and runtime performance.

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Prepare model with onnx-simplifier
            :end-before:  # End of prepare model

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Set up dataloader
            :end-before:  # End of setting up dataloader

        Evaluate FP32 model

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Evaluate FP32 model accuracy
            :end-before:  # End of FP32 evaluation

        .. rst-class:: script-output

            .. code-block:: none

                fp32 accuracy: 0.6885

Step 1: Creating a QuantSim model
---------------------------------

Use AIMET to create a :class:`QuantizationSimModel`. AIMET inserts
fake quantization operations in the model graph and configures them.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Create QuantSim object
            :end-before:  # End of creating QuantSim object

Step 2: Computing encodings
---------------------------

Next, call :func:`compute_encodings` to pass representative
data through the quantized model. The quantizers in the quantized model use the observed inputs
to initialize their quantization encodings. "Encodings" refers to the scale and offset quantization parameters.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Compute quantization encodings
            :end-before:  # End of computing quantization encodings

Step 3: Evaluation of w8a8 base precision
-----------------------------------------

Next, evaluate the :class:`QuantizationSimModel` to measure the model’s accuracy after quantization.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Evaluate quantized accuracy
            :end-before:  # End of quantized accuracy

        .. rst-class:: script-output

            .. code-block:: none

                quantized accuracy (w8a8): 0.6836

Step 4: Perform sensitivity analysis
------------------------------------

Lite-MP requires only a small number of samples during per-layer analysis to evaluate layer sensitivity to quantization

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Perform sensitivity analysis
            :end-before: # End of sensitivity analysis

Step 5: Apply precision adjustment
----------------------------------

Convert the most sensitive layers to higher precision (e.g., ``INT16`` or ``float16``) to recover accuracy.
In this example, the least sensitive layers remain in ``W8A8`` precision, while the weights and output quantizers of the most
sensitive layers are flipped to ``INT16`` precision.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Apply precision adjustment
            :end-before: # End of precision adjustment

Step 6: Recompute encodings
---------------------------

Recalibrate the mixed precision profile.

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Recompute quantization encodings
            :end-before: # End of recompute quantization encodings

Step 7: Evaluation of w8a8_mixed precision
------------------------------------------

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. literalinclude:: ../../snippets/onnx/apply_litemp.py
            :language: python
            :start-after: # Reevaluate model's accuracy after quantization
            :end-before: # End of re-evaluation

        .. rst-class:: script-output

            .. code-block:: none

                quantized accuracy (w8a8_mixed): 0.6865

API
===

.. tab-set::
    :sync-group: platform

    .. tab-item:: ONNX
        :sync: onnx

        .. include:: ../../apiref/onnx/litemp.rst
           :start-after: # start-after
