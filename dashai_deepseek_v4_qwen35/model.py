"""DeepSeek-V4-Pro-Qwen3.5-9B-MTP (IQ4_XS) as a DashAI GGUF text-generation model.

Reuses DashAI's ``GGUFTextGenerationModel`` (download via HuggingFace, loading
via llama.cpp) and only overrides ``generate`` so that the model answers
WITHOUT its reasoning block: the GGUF chat template is rendered with
``enable_thinking=False`` (the template then emits an empty
``<think>\\n\\n</think>`` block). Any residual ``<think>...</think>`` text is
stripped from the output as a safeguard.
"""

import logging
import re
from typing import List

from DashAI.back.core.utils import MultilingualString
from DashAI.back.models.hugging_face.gguf_text_generation_base import (
    GGUFTextGenerationModel,
    GGUFTextGenerationSchema,
)

log = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?(</think>|$)", flags=re.DOTALL)
_STOP = ["<|im_end|>", "<|endoftext|>"]


def _strip_think(text: str) -> str:
    """Remove any ``<think>...</think>`` block (or an unterminated one)."""
    return _THINK_RE.sub("", text).strip()


def _chatml_fallback(messages: list) -> str:
    """Plain ChatML rendering with thinking disabled (used if Jinja fails)."""
    parts = [
        f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n"
        for m in messages
    ]
    parts.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(parts)


class DeepSeekV4ProQwen35_9B(GGUFTextGenerationModel):  # noqa: N801
    """DeepSeek-V4-Pro-Qwen3.5-9B-MTP GGUF checkpoint (IQ4_XS quantization).

    Qwen3.5-9B fine-tune (base ``unsloth/Qwen3.5-9B``) distilled from
    DeepSeek-V4-Pro reasoning traces, by Jackrong. Apache-2.0.
    Thinking mode is disabled for RAG use.

    References
    ----------
    - https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF
    """

    REPO_ID = "Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF"
    GGUF_PATTERN = "DeepSeek-V4-Pro-Qwen3.5-9B-MTP-IQ4_XS.gguf"
    DOWNLOAD_SIZE_BYTES = 5357876320
    SCHEMA = GGUFTextGenerationSchema
    COLOR: str = "#4d6bfe"
    DISPLAY_NAME = MultilingualString(
        en="DeepSeek-V4-Pro Qwen3.5 9B (IQ4_XS)",
        es="DeepSeek-V4-Pro Qwen3.5 9B (IQ4_XS)",
        pt="DeepSeek-V4-Pro Qwen3.5 9B (IQ4_XS)",
        de="DeepSeek-V4-Pro Qwen3.5 9B (IQ4_XS)",
        zh="DeepSeek-V4-Pro Qwen3.5 9B (IQ4_XS)",
    )
    DESCRIPTION = MultilingualString(
        en=(
            "Qwen3.5 9B fine-tuned on DeepSeek-V4-Pro reasoning traces, loaded "
            "as an IQ4_XS GGUF (~5.4 GB) for CPU inference with llama.cpp. "
            "Thinking mode is disabled so answers are direct (suited for RAG). "
            "https://huggingface.co/Jackrong/DeepSeek-V4-Pro-Qwen3.5-9B-MTP-GGUF"
        ),
        es=(
            "Qwen3.5 9B ajustado con trazas de razonamiento de DeepSeek-V4-Pro, "
            "cargado como GGUF IQ4_XS (~5,4 GB) para inferencia en CPU con "
            "llama.cpp. El modo de razonamiento está desactivado para que las "
            "respuestas sean directas (adecuado para RAG)."
        ),
        pt=(
            "Qwen3.5 9B ajustado com traços de raciocínio do DeepSeek-V4-Pro, "
            "carregado como GGUF IQ4_XS (~5,4 GB) para inferência em CPU com "
            "llama.cpp. O modo de raciocínio está desativado (adequado para RAG)."
        ),
        de=(
            "Qwen3.5 9B, feinabgestimmt auf DeepSeek-V4-Pro-Reasoning-Traces, "
            "als IQ4_XS-GGUF (~5,4 GB) für CPU-Inferenz mit llama.cpp geladen. "
            "Der Denkmodus ist deaktiviert (geeignet für RAG)."
        ),
        zh=(
            "基于 DeepSeek-V4-Pro 推理轨迹微调的 Qwen3.5 9B，以 IQ4_XS GGUF "
            "（约 5.4 GB）格式通过 llama.cpp 在 CPU 上推理。已关闭思考模式，适用于 RAG。"
        ),
    )

    def _render_prompt(self, messages: list) -> str:
        """Render the GGUF chat template with ``enable_thinking=False``."""
        template = (getattr(self.model, "metadata", None) or {}).get(
            "tokenizer.chat_template"
        )
        if not template:
            return _chatml_fallback(messages)
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def _raise(msg):
                raise ValueError(msg)

            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals["raise_exception"] = _raise
            return env.from_string(template).render(
                messages=messages,
                add_generation_prompt=True,
                enable_thinking=False,
                bos_token="",
                eos_token="<|im_end|>",
            )
        except Exception:  # noqa: BLE001
            log.exception("Chat template rendering failed; using ChatML fallback")
            return _chatml_fallback(messages)

    def generate(self, prompt: list) -> List[str]:
        """Generate a reply (no reasoning block) for an OpenAI-style chat prompt."""
        text = self._render_prompt(prompt)
        output = self.model.create_completion(
            prompt=text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            frequency_penalty=self.frequency_penalty,
            stop=_STOP,
        )
        return [_strip_think(output["choices"][0]["text"])]
