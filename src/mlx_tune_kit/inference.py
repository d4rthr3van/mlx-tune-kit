"""MLX-VLM generation adapter with imports deferred until use."""

from __future__ import annotations

from typing import Any

from mlx_tune_kit.config import AppConfig

__all__ = ["generate_response", "load_model"]


def load_model(config: AppConfig, adapter_path: str | None = None) -> tuple[Any, Any]:
    try:
        from mlx_vlm import load
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError(
            "MLX-VLM could not initialize. Run inference on Apple Silicon with Metal available."
        ) from exc
    kwargs: dict[str, Any] = {"revision": config.model.revision}
    if config.model.trust_remote_code:
        kwargs["processor_config"] = {"trust_remote_code": True}
    return load(config.model.id, adapter_path=adapter_path, **kwargs)


def generate_response(
    model: Any,
    processor: Any,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
) -> str:
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template

    prompt = apply_chat_template(
        processor,
        model.config,
        messages,
        add_generation_prompt=True,
        num_images=0,
        enable_thinking=False,
    )
    return generate(
        model,
        processor,
        prompt,
        max_tokens=max_tokens,
        temperature=0,
        verbose=False,
    ).text.strip()
