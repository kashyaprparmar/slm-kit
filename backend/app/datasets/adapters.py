"""Canonical, import-light schema boundary shared by inspection and workers.

Adapters do not render templates, tokenize, normalize Unicode, or mutate input.
Advanced formats are recognized but never silently passed to the SFT worker.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.capabilities import Capability, SupportState


class DatasetSchemaError(ValueError):
    pass


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str

    @field_validator("content")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Message content must not be empty.")
        return value


class CanonicalRecord(BaseModel):
    schema_version: Literal[1] = 1
    kind: Literal["text", "conversation"]
    source_format: str
    text: str | None = None
    messages: list[Message] = Field(default_factory=list)

    def content_text(self) -> str:
        return self.text if self.kind == "text" else "\n".join(message.content for message in self.messages)

    def training_pair(self) -> tuple[list[dict[str, str]], str]:
        if self.kind != "conversation" or not self.messages:
            raise DatasetSchemaError("An evaluation/training pair requires a conversation ending in an assistant response.")
        return ([message.model_dump() for message in self.messages[:-1]], self.messages[-1].content)

    def storage_row(self) -> dict[str, Any]:
        if self.kind == "text":
            return {"text": self.text}
        return {"messages": [message.model_dump() for message in self.messages]}


PAIR_FORMATS = {
    "instruction": ("instruction", "output"),
    "prompt_response": ("prompt", "response"),
    "question_answer": ("question", "answer"),
    "prompt_completion": ("prompt", "completion"),
}
GATED_FORMATS = {
    "preference": "Preference data requires a preference-training worker; it cannot be used as SFT data.",
    "pretokenized": "Pre-tokenized data requires tokenizer fingerprint and label/mask validation before training.",
    "tool_calling": "Tool data requires model chat-template and worker support; it cannot be flattened to ordinary chat.",
}


class DatasetAdapterCapabilities(BaseModel):
    schema_version: Literal[1] = 1
    format_id: str
    detection_priority: Literal["gated", "structured", "fallback"]
    training_stages: dict[str, Capability]


@runtime_checkable
class DatasetAdapter(Protocol):
    """Schema boundary used by inspection, preparation, and training workers."""

    format_id: str
    detection_priority: Literal["gated", "structured", "fallback"]

    def capabilities(self) -> DatasetAdapterCapabilities: ...

    def detect(self, row: Mapping[str, Any]) -> bool: ...

    def validate(self, row: Mapping[str, Any]) -> list[str]: ...

    def canonicalize(self, row: Mapping[str, Any]) -> CanonicalRecord: ...

    def preview(self, row: Mapping[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BuiltinDatasetAdapter:
    format_id: str
    detector: Callable[[Mapping[str, Any]], bool]
    detection_priority: Literal["gated", "structured", "fallback"] = "structured"
    trainable: bool = True

    def capabilities(self) -> DatasetAdapterCapabilities:
        supported = Capability(state=SupportState.SUPPORTED, reason="Canonicalization is implemented.")
        unsupported = Capability(
            state=SupportState.UNSUPPORTED,
            reason=GATED_FORMATS.get(self.format_id, "This format is not accepted by that training stage."),
        )
        return DatasetAdapterCapabilities(
            format_id=self.format_id,
            detection_priority=self.detection_priority,
            training_stages={
                "from_scratch_pretraining": supported if self.format_id == "text" else unsupported,
                "continued_pretraining": supported if self.format_id == "text" else unsupported,
                "supervised_fine_tuning": supported if self.trainable and self.format_id != "text" else unsupported,
                "alignment": unsupported,
            },
        )

    def detect(self, row: Mapping[str, Any]) -> bool:
        return self.detector(row)

    def validate(self, row: Mapping[str, Any]) -> list[str]:
        try:
            self.canonicalize(row)
        except DatasetSchemaError as exc:
            return [str(exc)]
        return []

    def canonicalize(self, row: Mapping[str, Any]) -> CanonicalRecord:
        if not self.trainable:
            raise DatasetSchemaError(GATED_FORMATS[self.format_id])
        return _canonicalize_schema(dict(row), self.format_id)

    def preview(self, row: Mapping[str, Any]) -> dict[str, Any]:
        return self.canonicalize(row).storage_row()


def _messages(row: Mapping[str, Any]) -> Any:
    return row.get("messages", row.get("conversations"))


DATASET_ADAPTERS: tuple[DatasetAdapter, ...] = (
    BuiltinDatasetAdapter("preference", lambda row: any(key in row for key in ("chosen", "rejected")), "gated", False),
    BuiltinDatasetAdapter("pretokenized", lambda row: any(key in row for key in ("input_ids", "attention_mask", "labels")), "gated", False),
    BuiltinDatasetAdapter(
        "tool_calling",
        lambda row: "tools" in row or (
            isinstance(_messages(row), list)
            and any(
                isinstance(message, Mapping)
                and (
                    "tool_calls" in message
                    or "tool_call_id" in message
                    or (message.get("role") or message.get("from")) in ("tool", "function")
                )
                for message in _messages(row)
            )
        ),
        "gated",
        False,
    ),
    *(BuiltinDatasetAdapter(name, lambda row, keys=keys: all(key in row for key in keys)) for name, keys in PAIR_FORMATS.items()),
    BuiltinDatasetAdapter("messages", lambda row: "messages" in row),
    BuiltinDatasetAdapter("sharegpt", lambda row: "conversations" in row),
    BuiltinDatasetAdapter("text", lambda row: "text" in row, "fallback"),
)


def get_dataset_adapter(format_id: str) -> DatasetAdapter:
    try:
        return next(adapter for adapter in DATASET_ADAPTERS if adapter.format_id == format_id)
    except StopIteration as exc:
        raise DatasetSchemaError(f"No dataset adapter is registered for '{format_id}'.") from exc


def detect_schema(row: Mapping[str, Any] | Any) -> str:
    if not isinstance(row, Mapping):
        raise DatasetSchemaError("Each record must be an object.")
    row = dict(row)
    gated = [adapter for adapter in DATASET_ADAPTERS if adapter.detection_priority == "gated" and adapter.detect(row)]
    if gated:
        return gated[0].format_id
    candidates = [
        adapter.format_id
        for adapter in DATASET_ADAPTERS
        if adapter.detection_priority == "structured" and adapter.detect(row)
    ]
    if not candidates:
        candidates = [
            adapter.format_id
            for adapter in DATASET_ADAPTERS
            if adapter.detection_priority == "fallback" and adapter.detect(row)
        ]
    if len(candidates) != 1:
        raise DatasetSchemaError(
            "Ambiguous schema; explicitly map one prompt/answer pair or conversation column."
            if candidates else "Unrecognized schema. Map text, messages, or a complete prompt/answer pair."
        )
    return candidates[0]


def _text(row: dict, key: str, *, optional: bool = False) -> str:
    value = row.get(key, "") if optional else row.get(key)
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise DatasetSchemaError(f"'{key}' must be {'a string' if optional else 'a non-empty string'}.")
    return value


def canonicalize(row: Mapping[str, Any] | Any, columns: dict[str, str] | None = None) -> CanonicalRecord:
    if not isinstance(row, Mapping):
        raise DatasetSchemaError("Each record must be an object.")
    row = dict(row)
    if columns:
        missing = [source for source in columns.values() if source not in row]
        if missing:
            raise DatasetSchemaError(f"Mapped source columns are missing: {missing}.")
        row = {target: row[source] for target, source in columns.items()}
    schema = detect_schema(row)
    return get_dataset_adapter(schema).canonicalize(row)


def _canonicalize_schema(row: dict[str, Any], schema: str) -> CanonicalRecord:
    if schema == "text":
        return CanonicalRecord(kind="text", source_format=schema, text=_text(row, "text"))
    if schema in PAIR_FORMATS:
        prompt_key, answer_key = PAIR_FORMATS[schema]
        prompt, answer = _text(row, prompt_key), _text(row, answer_key)
        if row.get("input") and row.get("context"):
            raise DatasetSchemaError("Both input and context are populated; explicitly map the intended context.")
        context = _text(row, "input" if "input" in row else "context", optional=True)
        messages = [{"role": "user", "content": prompt + (f"\n\n{context}" if context else "")},
                    {"role": "assistant", "content": answer}]
    else:
        messages = row["messages" if schema == "messages" else "conversations"]
        if not isinstance(messages, list) or not messages:
            raise DatasetSchemaError("Conversation must be a non-empty list.")
        if schema == "sharegpt":
            roles = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
            converted = []
            for message in messages:
                if not isinstance(message, Mapping):
                    raise DatasetSchemaError("Every conversation turn must be an object.")
                message = dict(message)
                role = message.get("from", message.get("role"))
                converted.append({"role": roles.get(role) if isinstance(role, str) else None,
                                  "content": message.get("value", message.get("content"))})
            messages = converted
        else:
            converted = []
            for message in messages:
                if not isinstance(message, Mapping):
                    raise DatasetSchemaError("Every conversation turn must be an object.")
                converted.append(dict(message))
            messages = converted
    try:
        turns = [Message.model_validate(message) for message in messages]
    except ValueError as exc:
        raise DatasetSchemaError("Each turn requires a supported role and non-empty string content; map extra message fields explicitly.") from exc
    conversation = turns[1:] if turns[0].role == "system" else turns
    if not conversation or conversation[-1].role != "assistant":
        raise DatasetSchemaError("Training conversations must end with an assistant response.")
    for index, message in enumerate(conversation):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise DatasetSchemaError(f"Invalid role order at conversation turn {index + 1}: expected {expected}.")
    return CanonicalRecord(kind="conversation", source_format=schema, messages=turns)


class SchemaSample(BaseModel):
    line: int
    source_format: str | None = None
    canonical: CanonicalRecord | None = None
    error: str | None = None


class SchemaReport(BaseModel):
    schema_version: Literal[1] = 1
    sampled_rows: int
    sample_limit: int
    sample_only: Literal[True] = True
    formats: list[str]
    valid: bool
    samples: list[SchemaSample]


def inspect_rows(rows, limit: int = 20) -> SchemaReport:
    from itertools import islice

    samples = []
    for line, row in islice(rows, limit):
        sample = SchemaSample(line=line)
        try:
            sample.source_format = detect_schema(row)
            sample.canonical = canonicalize(row)
        except DatasetSchemaError as exc:
            sample.error = str(exc)
        samples.append(sample)
    return SchemaReport(sampled_rows=len(samples), sample_limit=limit,
                        formats=sorted({s.source_format for s in samples if s.source_format}),
                        valid=bool(samples) and all(s.error is None for s in samples), samples=samples)
