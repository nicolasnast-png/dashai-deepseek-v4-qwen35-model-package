# dashai-deepseek-v4-qwen35-model-package

A [DashAI](https://github.com/DashAISoftware/DashAI) plugin that adds **DeepSeek-V4-Pro Qwen3.5 9B (GGUF, IQ4_XS)** as a text generation model. It runs locally with llama.cpp and works in DashAI's generative chat and **RAG** pipelines.

The model is a Qwen3.5-9B fine-tune distilled from DeepSeek-V4-Pro reasoning traces, published by [Jackrong](https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF) (Apache-2.0, base `unsloth/Qwen3.5-9B`). This plugin downloads the `DeepSeek-V4-Pro-Qwen3.5-9B-MTP-IQ4_XS.gguf` file (5.36 GB).

This is a community plugin. It is not affiliated with DeepSeek, the Qwen team or the original model author.

## What the plugin adds

- **Component `DeepSeekV4ProQwen35_9B`:** a `GGUFTextGenerationModel`, downloadable from DashAI like the built-in GGUF models. It uses the standard GGUF parameters (`max_tokens`, `temperature`, `frequency_penalty`, `context_window`, `device`).
- **Thinking mode disabled:** the model is a reasoning model and by default writes a long `<think>…</think>` block before answering. The plugin renders the model's own chat template with `enable_thinking=False`, which makes the model answer directly. It also strips any leftover `<think>` block. This keeps RAG answers clean and saves generation time.

## Requirements

- **DashAI:** a version that provides `GGUFTextGenerationModel` (0.11.0 or later; tested on the `develop` branch, commit `ac94dc73c`).
- **llama.cpp:** must support the `qwen35` architecture. `llama-cpp-python` 0.3.34 (shipped with DashAI `develop`) and the DashAI 0.11.0 bundle both support it.
- **Memory:** about 6 GB of disk, and 8 GB or more of free RAM for CPU inference.

## Install

**From DashAI:** open the plugin store, search for `dashai-deepseek-v4-qwen35-model-package` and install it.

**Manually:** install into DashAI's plugin directory, then restart DashAI:

```bash
pip install --no-deps --target ~/.DashAI/plugins/py3.12 dashai-deepseek-v4-qwen35-model-package
```

Then download the weights from the model card in DashAI, or with the API:

```bash
curl -X POST http://localhost:8000/api/v1/component/DeepSeekV4ProQwen35_9B/download
```

## Use in a RAG session

Set the generation model of a RAG session to `DeepSeekV4ProQwen35_9B`. Example API payload:

```json
{"generation_model": {"component": "DeepSeekV4ProQwen35_9B",
  "params": {"max_tokens": 600, "temperature": 0.0, "frequency_penalty": 0.1,
             "context_window": 10000, "device": "CPU"}}}
```

On a Ryzen 9 5900HX (CPU only), generation runs at about 3 tokens/s. A RAG answer over 10 retrieved chunks takes about 2 minutes.

## Español

Plugin comunitario para DashAI que agrega el modelo **DeepSeek-V4-Pro Qwen3.5 9B (GGUF IQ4_XS)** con el modo de razonamiento desactivado, pensado para RAG.

1. Instálalo desde la tienda de plugins de DashAI, o con `pip install --no-deps --target ~/.DashAI/plugins/py3.12 dashai-deepseek-v4-qwen35-model-package`.
2. Reinicia DashAI.
3. Descarga el modelo (5,36 GB) desde DashAI.

Se usó en una validación piloto del módulo RAG de DashAI con MIRACL-es y NoMIRACL-es.

## License

- **Plugin code:** Apache-2.0 (see [LICENSE](LICENSE)).
- **Model weights:** Apache-2.0, under the terms of the [original model card](https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF).
