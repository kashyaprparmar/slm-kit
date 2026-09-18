"""Heavy-worker tokenizer rendering and loss-mask implementation."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.datasets.adapters import CanonicalRecord, DatasetSchemaError, canonicalize


class TokenizerDataError(ValueError):
    pass


def _ids(value) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TokenizerDataError("Tokenizer did not return a one-dimensional token sequence.")
    return [int(item) for item in value]


def _render_messages(tokenizer, messages: list[dict], *, generation_prompt=False) -> str:
    if not getattr(tokenizer, "chat_template", None):
        raise TokenizerDataError(
            "This conversational dataset requires a chat template. Select a tokenizer with a native template or provide an explicit custom template."
        )
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt,
        )
    except Exception as exc:
        raise TokenizerDataError(f"The selected chat template could not render this conversation: {exc}") from exc


def _tokenize_rendered(tokenizer, rendered: str) -> list[int]:
    encoded = tokenizer(rendered, add_special_tokens=False)
    return _ids(encoded["input_ids"] if hasattr(encoded, "__getitem__") else encoded.input_ids)


def render_preference_row(
    row: Mapping[str, Any] | Any,
    tokenizer,
    *,
    chat_template: str | None = None,
) -> dict[str, str]:
    """Render one canonical preference record into exact prompt/completion strings."""
    if chat_template is not None:
        if not chat_template.strip():
            raise TokenizerDataError("A custom chat template cannot be blank.")
        tokenizer.chat_template = chat_template
    try:
        record = canonicalize(dict(row) if isinstance(row, Mapping) else row)
    except DatasetSchemaError as exc:
        raise TokenizerDataError(str(exc)) from exc
    if record.kind != "preference":
        raise TokenizerDataError(f"Preference training requires preference rows, got {record.kind}.")
    prompt_messages = [message.model_dump() for message in record.prompt]
    prompt = _render_messages(tokenizer, prompt_messages, generation_prompt=True)

    def continuation(messages) -> str:
        rendered = _render_messages(
            tokenizer,
            [*prompt_messages, *(message.model_dump() for message in messages)],
        )
        if not rendered.startswith(prompt):
            raise TokenizerDataError(
                "The chat template's preference prompt is not an exact prefix of the completed branch."
            )
        return rendered[len(prompt):]

    return {"prompt": prompt, "chosen": continuation(record.chosen), "rejected": continuation(record.rejected)}


def render_kto_row(
    row: Mapping[str, Any] | Any,
    tokenizer,
    *,
    chat_template: str | None = None,
) -> dict[str, str | bool]:
    """Render one canonical KTO record without changing its binary label."""
    if chat_template is not None:
        if not chat_template.strip():
            raise TokenizerDataError("A custom chat template cannot be blank.")
        tokenizer.chat_template = chat_template
    try:
        record = canonicalize(dict(row) if isinstance(row, Mapping) else row)
    except DatasetSchemaError as exc:
        raise TokenizerDataError(str(exc)) from exc
    if record.kind != "kto":
        raise TokenizerDataError(f"KTO requires KTO rows, got {record.kind}.")
    prompt_messages = [message.model_dump() for message in record.prompt]
    prompt = _render_messages(tokenizer, prompt_messages, generation_prompt=True)
    completed = _render_messages(tokenizer, [*prompt_messages, record.response.model_dump()])
    if not completed.startswith(prompt):
        raise TokenizerDataError(
            "The chat template's KTO prompt is not an exact prefix of the completed response."
        )
    return {"prompt": prompt, "completion": completed[len(prompt):], "label": record.desirable}


def render_and_tokenize(
    row: Mapping[str, Any] | Any,
    tokenizer,
    *,
    max_length: int,
    loss_policy: str = "full_sequence",
    chat_template: str | None = None,
) -> dict:
    """Return rendered text, tokens and exact training labels for one record."""
    if isinstance(row, Mapping) and not isinstance(row, dict):
        row = dict(row)
    try:
        record: CanonicalRecord = canonicalize(row)
    except DatasetSchemaError as exc:
        raise TokenizerDataError(str(exc)) from exc
    if chat_template is not None:
        if not chat_template.strip():
            raise TokenizerDataError("A custom chat template cannot be blank.")
        tokenizer.chat_template = chat_template
    if record.kind == "text":
        if loss_policy != "full_sequence":
            raise TokenizerDataError(f"{loss_policy} requires conversational prompt/assistant data.")
        rendered = record.text or ""
        encoded = tokenizer(rendered, add_special_tokens=True)
        input_ids = _ids(encoded["input_ids"])
        labels = list(input_ids)
    else:
        messages = [message.model_dump() for message in record.messages]
        rendered = _render_messages(tokenizer, messages)
        input_ids = _tokenize_rendered(tokenizer, rendered)
        if loss_policy == "full_sequence":
            labels = list(input_ids)
        elif loss_policy == "completion_only":
            prompt = _render_messages(tokenizer, messages[:-1], generation_prompt=True)
            prompt_ids = _tokenize_rendered(tokenizer, prompt)
            if input_ids[:len(prompt_ids)] != prompt_ids:
                raise TokenizerDataError(
                    "This chat template's generation prompt is not an exact prefix of its training rendering; completion-only masking would be incorrect."
                )
            labels = [-100] * len(prompt_ids) + input_ids[len(prompt_ids):]
        elif loss_policy == "assistant_only":
            try:
                encoded = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False,
                    return_dict=True, return_assistant_tokens_mask=True,
                )
                input_ids = _ids(encoded["input_ids"])
                mask = encoded.get("assistant_masks")
                if mask is None:
                    mask = encoded.get("assistant_tokens_mask")
                mask = _ids(mask)
            except Exception as exc:
                raise TokenizerDataError(
                    "Assistant-only loss requires a chat template with assistant generation masks. Choose completion-only/full-sequence loss or another template."
                ) from exc
            if len(mask) != len(input_ids) or not any(mask):
                raise TokenizerDataError("The chat template returned no usable assistant token mask.")
            labels = [token if enabled else -100 for token, enabled in zip(input_ids, mask, strict=True)]
        else:
            raise TokenizerDataError(f"Unknown loss policy: {loss_policy}.")
    original_length = len(input_ids)
    input_ids = input_ids[:max_length]
    labels = labels[:max_length]
    if not any(label != -100 for label in labels):
        raise TokenizerDataError("Truncation removed every target token; increase max sequence length or shorten the prompt.")
    return {
        "rendered": rendered,
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "loss_mask": [label != -100 for label in labels],
        "original_length": original_length,
        "truncated": original_length > max_length,
        "source_format": record.source_format,
    }
