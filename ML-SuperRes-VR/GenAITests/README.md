# AIMET GenAI Test Framework

AIMET's GenAI framework provides an easy way to evaluate large models against quantization techniques provided by both
AIMET-ONNX and AIMET-Torch. You can use a config file and have the framework take care of running it all for you, or
you can use the utilities provided to write an ad-hoc script that highlight all the quantization settings used.

## Prerequisites

- Either AIMET-Torch or AIMET-ONNX
- pytest
- Huggingface transformers and datasets

## Setting up a YAML config file

This is an example config file that tells the framework to: instantiate Llama 3.2, quantize it using regular per-channel
quantization, and then evaluate the perplexity and TinyMMLU score of the quantized model. Please note that the same
config files can be used between Torch and ONNX (assuming that the specified quantization technique is available in
both AIMET-Torch and AIMET-ONNX)

```yaml
model:
  name: Llama_32
  sequence_length: 2048
  context_length: 4096
dataset:
  name: Wikitext
  split: train
recipe:
  name: PCQ
metrics:
  - name: TinyMMLU
  - name: PPL
```

There are four mandatory sections in each config file, as outlined below.

### `model`

This section allows users to specify which model to use. Supported models include but are not limited to Llama 3.2,
Qwen 2.5, and Phi 3.5. You can specify which model family you would like via the `name` argument as shown above. If you
would like to know the exact string that should be used, please use the class name as specified in `GenAITests/shared/models`.
Although each family has a default HuggingFace model ID, you can customize this by specifying the `model_id` field. For
example, the framework can be instructed to use Llama 3.2 3B instead of Llama 3.2 1B (the default for the Llama 3.2 family)
as follows:

```yaml
model:
  name: Llama_32
  model_id: meta-llama/Llama-3.2-3B
  sequence_length: 2048
  context_length: 4096
```

Users should also specify sequence length (the number of tokens the model can consume on a single inference), and
context length (the maximum number of tokens the model can consume and retain context).

Please note that the `model` section, along with the `name`, `sequence_length`, and `context_length` fields are mandatory.

### `dataset`

This section allows users to specify which dataset should be used for applying the specified quantization technique. It
has two fields: `name` and `split`, which are both mandatory. `name` refers to the class name of the dataset as it is
implemented in `GenAITests/shared/helpers/datasets.py`. `split` is, as the name suggests, which split of the dataset
to use.

### `recipe`

This section allows users to specify which quantization technique should be applied to the model. There is only one
mandatory field here - `name` - although individual quantization techniques may accept different parameters based on
their implementations. `name` refers to the class name of the desired technique as it is implemented in
`GenAITests/torch/helpers/quant_recipes.py` or `GenAITests/torch/helpers/quant_recipes.py`. Please also consult these
files to learn more about what parameters can be controlled via the YAML config file.

For example, a user who wants to run AdaScale using AIMET-Torch could also specify the number of iterations that the
algorith uses as follows:

```yaml
recipe:
  name: AdaScale
  num_iterations: 3000
```

### `metric`

This section allows users to specify which evaluation metrics to run after applying the specified quantization technique.
Users can specify which metrics they would like to run by adding a list in this section. Each entry in the list only
has one mandatory field - `name` - referring to the class name of the desired metric as specified in
`GenAITests/shared/helpers/metrics.py`, although additional parameters can be provided as well.

For example, a user would like to run PPL evaluation and MMLU evaluation can do so as follows:

```yaml
  - name: MMLU
  - name: PPL
```

## Starting the framework with a YAML config file

1. Set up your PYTHONPATH with `bash GenAITests/update_pythonpath.sh`. This only needs to be run once
2. Invoke the genAI test framework either in Torch or ONNX
    * Torch: `pytest -s GenAITests/torch/test_genai.py --config <path to config file>`
    * ONNX: `pytest -s GenAITests/onnx/test_genai.py --config <path to config file>`


## Writing an ad-hoc script

If you would rather have a single script that accomplishes some task using the framework rather than using the YAML config,
most utilities are designed to be flattened such that all huggingface API calls and AIMET API calls are visible from the
top level. For an example on how to do this, please consult `GenAITests/torch/example_custom_script.py`

## How it all works

Under the hood, inference on Torch and ONNX models is done with the same driver code - contained in
`GenAITests/shared/models/generator.py`. Essentially, the `Generator` class is used to restore the regular HuggingFace API
to models with static shape requirements. The framework follows the same set of steps (with minor differences) in both
Torch and ONNX which are:
1. Instantiate the model
    * For Torch models, this just involves pulling the model from HuggingFace and wrapping in an IO class that makes it JIT traceable
    * For ONNX models, this includes the same steps as Torch plus calling `torch.onnx.export`, loading the model, and wrapping it in a class that mimics Torch model semantics
2. Instantiate the tokenizer
3. Instantiate a `QuantizationSimModel` using the loaded model and tokenizer
4. Create a `Generator` object using the `QuantizationSimModel` and tokenizer
5. Load and tokenize the user-specified dataset
6. Apply the user-specified quantization technique (using the `QuantizationSimModel`, `Generator`, and dataset)
7. Run the user-specified evals (using the `Generator`)
