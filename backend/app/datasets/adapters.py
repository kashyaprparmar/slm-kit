"""Canonical, import-light schema boundary shared by inspection and workers.

Adapters do not render templates, tokenize, normalize Unicode, or mutate input.
Advanced formats are recognized but never silently passed to the SFT worker.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from app.capabilities import Capability, SupportState
from app.domain import TrainingStage


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


def _validate_prompt(messages: list[Message], *, label: str) -> None:
    dialogue = messages[1:] if messages and messages[0].role == "system" else messages
    if not dialogue:
        raise ValueError(f"{label} must contain a user turn.")
    for index, message in enumerate(dialogue):
        expected = "user" if index % 2 == 0 else "assistant"
        if message.role != expected:
            raise ValueError(f"{label} turn {index + 1} must have the {expected} role.")
    if dialogue[-1].role != "user":
        raise ValueError(f"{label} must end with a user turn.")


def _validate_continuation(messages: list[Message], *, label: str) -> None:
    for index, message in enumerate(messages):
        expected = "assistant" if index % 2 == 0 else "user"
        if message.role != expected:
            raise ValueError(f"{label} turn {index + 1} must have the {expected} role.")
    if messages[-1].role != "assistant":
        raise ValueError(f"{label} must end with an assistant turn.")


class CanonicalRecord(BaseModel):
    """Base for versioned canonical records.

    Existing text/conversation rows retain their v1 on-disk shape. Advanced
    record types use v2 and include an explicit discriminator when stored.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: int
    kind: str
    source_format: str

    def content_text(self) -> str:
        raise DatasetSchemaError(f"'{self.kind}' does not expose plain training text.")

    def training_pair(self) -> tuple[list[dict[str, str]], str]:
        raise DatasetSchemaError(
            "An evaluation/training pair requires a conversation ending in an assistant response."
        )

    def storage_row(self) -> dict[str, Any]:
        raise NotImplementedError

    def canonical_fingerprint(self) -> str:
        """Stable semantic identity independent of source columns/format."""
        encoded = json.dumps(
            self.storage_row(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class RawTextRecord(CanonicalRecord):
    schema_version: Literal[1] = 1
    kind: Literal["text"] = "text"
    text: str = Field(min_length=1)

    def content_text(self) -> str:
        return self.text

    def storage_row(self) -> dict[str, Any]:
        return {"text": self.text}


class ConversationRecord(CanonicalRecord):
    schema_version: Literal[1] = 1
    kind: Literal["conversation"] = "conversation"
    messages: list[Message] = Field(min_length=2)

    @model_validator(mode="after")
    def valid_dialogue(self) -> ConversationRecord:
        _validate_prompt(self.messages[:-1], label="Conversation prompt")
        if self.messages[-1].role != "assistant":
            raise ValueError("Conversation must end with an assistant response.")
        return self

    def content_text(self) -> str:
        return "\n".join(message.content for message in self.messages)

    def training_pair(self) -> tuple[list[dict[str, str]], str]:
        return ([message.model_dump() for message in self.messages[:-1]], self.messages[-1].content)

    def storage_row(self) -> dict[str, Any]:
        return {"messages": [message.model_dump() for message in self.messages]}


class MediaReference(BaseModel):
    """Non-loaded media identity for future multimodal adapters."""

    model_config = ConfigDict(extra="forbid", strict=True)
    media_type: Literal["image", "audio", "video", "file"]
    uri: str = Field(min_length=1)
    mime_type: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class PreferenceRecord(CanonicalRecord):
    schema_version: Literal[2] = 2
    kind: Literal["preference"] = "preference"
    prompt: list[Message] = Field(min_length=1)
    chosen: list[Message] = Field(min_length=1)
    rejected: list[Message] = Field(min_length=1)
    media: list[MediaReference] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def distinct_branches(self) -> PreferenceRecord:
        _validate_prompt(self.prompt, label="Preference prompt")
        _validate_continuation(self.chosen, label="Chosen continuation")
        _validate_continuation(self.rejected, label="Rejected continuation")
        if self.chosen == self.rejected:
            raise ValueError("Chosen and rejected continuations must differ.")
        return self

    def content_text(self) -> str:
        return "\n".join(message.content for message in [*self.prompt, *self.chosen, *self.rejected])

    def storage_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"source_format"})


class KTORecord(CanonicalRecord):
    schema_version: Literal[2] = 2
    kind: Literal["kto"] = "kto"
    prompt: list[Message] = Field(min_length=1)
    response: Message
    desirable: bool
    media: list[MediaReference] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def assistant_response(self) -> KTORecord:
        _validate_prompt(self.prompt, label="KTO prompt")
        if self.response.role != "assistant":
            raise ValueError("KTO response must have the assistant role.")
        return self

    def content_text(self) -> str:
        return "\n".join(message.content for message in [*self.prompt, self.response])

    def storage_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"source_format"})


class ToolFunctionCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    arguments: dict[str, JsonValue] = Field(default_factory=dict)


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolFunctionCall


class ToolDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    name: str = Field(min_length=1)
    description: str | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ToolMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None


class ToolConversationRecord(CanonicalRecord):
    schema_version: Literal[2] = 2
    kind: Literal["tool_conversation"] = "tool_conversation"
    messages: list[ToolMessage] = Field(min_length=2)
    tools: list[ToolDefinition] = Field(min_length=1)
    media: list[MediaReference] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def content_text(self) -> str:
        return "\n".join(message.content or "" for message in self.messages)

    def storage_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"source_format"})


class MultimodalRecord(CanonicalRecord):
    """Metadata-only contract; no current training stage accepts this kind."""

    schema_version: Literal[2] = 2
    kind: Literal["multimodal"] = "multimodal"
    messages: list[Message] = Field(default_factory=list)
    text: str | None = None
    media: list[MediaReference] = Field(min_length=1)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    def content_text(self) -> str:
        return self.text or "\n".join(message.content for message in self.messages)

    def storage_row(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"source_format"})


CanonicalDatasetRecord = Annotated[
    RawTextRecord
    | ConversationRecord
    | PreferenceRecord
    | KTORecord
    | ToolConversationRecord
    | MultimodalRecord,
    Field(discriminator="kind"),
]


PAIR_FORMATS = {
    "instruction": ("instruction", "output"),
    "prompt_response": ("prompt", "response"),
    "question_answer": ("question", "answer"),
    "prompt_completion": ("prompt", "completion"),
}
GATED_FORMATS = {
    "preference": "Preference data is accepted only by DPO/IPO/ORPO/SimPO alignment objectives.",
    "kto": "KTO data is accepted only by the KTO alignment objective.",
    "pretokenized": "Pre-tokenized data requires tokenizer fingerprint and label/mask validation before training.",
    "tool_calling": "Tool data requires model chat-template and worker support; it cannot be flattened to ordinary chat.",
    "multimodal": "Multimodal metadata is recognized, but no current dataset or training adapter supports it.",
}


class DatasetAdapterCapabilities(BaseModel):
    schema_version: Literal[1] = 1
    format_id: str
    detection_priority: Literal["gated", "structured", "fallback"]
    canonicalization: Capability
    training_stages: dict[str, Capability]


class AdapterValidationIssue(BaseModel):
    code: str
    message: str
    path: str | None = None


class AdapterValidationReport(BaseModel):
    valid: bool
    issues: list[AdapterValidationIssue] = Field(default_factory=list)


class DatasetPreview(BaseModel):
    schema_version: Literal[1] = 1
    source_format: str
    validation: AdapterValidationReport
    canonical: CanonicalDatasetRecord | None = None
    storage: dict[str, Any] | None = None
    fingerprint: str | None = None


@runtime_checkable
class DatasetAdapter(Protocol):
    """Schema boundary used by inspection, preparation, and training workers."""

    format_id: str
    detection_priority: Literal["gated", "structured", "fallback"]

    def capabilities(self) -> DatasetAdapterCapabilities: ...

    def detect(self, row: Mapping[str, Any]) -> bool: ...

    def validate(self, row: Mapping[str, Any]) -> AdapterValidationReport: ...

    def canonicalize(self, row: Mapping[str, Any]) -> CanonicalDatasetRecord: ...

    def preview(self, row: Mapping[str, Any]) -> DatasetPreview: ...


@dataclass(frozen=True)
class BuiltinDatasetAdapter:
    format_id: str
    detector: Callable[[Mapping[str, Any]], bool]
    detection_priority: Literal["gated", "structured", "fallback"] = "structured"
    canonicalization_supported: bool = True
    sft_trainable: bool = True
    alignment_trainable: bool = False

    def capabilities(self) -> DatasetAdapterCapabilities:
        supported = Capability(state=SupportState.SUPPORTED, reason="Canonicalization is implemented.")
        unsupported = Capability(
            state=SupportState.UNSUPPORTED,
            reason=GATED_FORMATS.get(self.format_id, "This format is not accepted by that training stage."),
        )
        canonicalization = supported if self.canonicalization_supported else unsupported
        return DatasetAdapterCapabilities(
            format_id=self.format_id,
            detection_priority=self.detection_priority,
            canonicalization=canonicalization,
            training_stages={
                TrainingStage.FROM_SCRATCH_PRETRAINING.value: (
                    supported if self.format_id == "text" else unsupported
                ),
                TrainingStage.CONTINUED_PRETRAINING.value: (
                    supported if self.format_id == "text" else unsupported
                ),
                TrainingStage.SUPERVISED_FINE_TUNING.value: (
                    supported if self.sft_trainable and self.format_id != "text" else unsupported
                ),
                TrainingStage.ALIGNMENT.value: (
                    supported if self.alignment_trainable else unsupported
                ),
            },
        )

    def detect(self, row: Mapping[str, Any]) -> bool:
        return self.detector(row)

    def validate(self, row: Mapping[str, Any]) -> AdapterValidationReport:
        try:
            self.canonicalize(row)
        except DatasetSchemaError as exc:
            code = "unsupported_format" if not self.canonicalization_supported else "invalid_record"
            return AdapterValidationReport(
                valid=False,
                issues=[AdapterValidationIssue(code=code, message=str(exc))],
            )
        return AdapterValidationReport(valid=True)

    def canonicalize(self, row: Mapping[str, Any]) -> CanonicalDatasetRecord:
        if not self.canonicalization_supported:
            raise DatasetSchemaError(GATED_FORMATS[self.format_id])
        try:
            return _canonicalize_schema(dict(row), self.format_id)
        except ValueError as exc:
            # Canonical model invariants must use the same line-aware error
            # contract as adapter parsing, including preference/KTO failures.
            raise DatasetSchemaError(str(exc)) from exc

    def preview(self, row: Mapping[str, Any]) -> DatasetPreview:
        try:
            record = self.canonicalize(row)
        except DatasetSchemaError as exc:
            code = "unsupported_format" if not self.canonicalization_supported else "invalid_record"
            validation = AdapterValidationReport(
                valid=False,
                issues=[AdapterValidationIssue(code=code, message=str(exc))],
            )
            return DatasetPreview(source_format=self.format_id, validation=validation)
        validation = AdapterValidationReport(valid=True)
        return DatasetPreview(
            source_format=self.format_id,
            validation=validation,
            canonical=record,
            storage=record.storage_row(),
            fingerprint=record.canonical_fingerprint(),
        )


def _messages(row: Mapping[str, Any]) -> Any:
    return row.get("messages", row.get("conversations"))


def _has_multimodal_content(row: Mapping[str, Any]) -> bool:
    if any(key in row for key in ("image", "images", "audio", "video", "media")):
        return True
    messages = _messages(row)
    return isinstance(messages, list) and any(
        isinstance(message, Mapping) and isinstance(message.get("content"), list)
        for message in messages
    )


class DatasetAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: list[DatasetAdapter] = []

    def __iter__(self):
        return iter(self._adapters)

    def register(self, adapter: DatasetAdapter) -> None:
        if any(current.format_id == adapter.format_id for current in self._adapters):
            raise ValueError(f"Dataset adapter '{adapter.format_id}' is already registered.")
        self._adapters.append(adapter)

    def get(self, format_id: str) -> DatasetAdapter:
        try:
            return next(adapter for adapter in self._adapters if adapter.format_id == format_id)
        except StopIteration as exc:
            raise DatasetSchemaError(f"No dataset adapter is registered for '{format_id}'.") from exc

    def detect(self, row: Mapping[str, Any]) -> DatasetAdapter:
        gated = [
            adapter for adapter in self._adapters
            if adapter.detection_priority == "gated" and adapter.detect(row)
        ]
        if len(gated) > 1:
            formats = ", ".join(adapter.format_id for adapter in gated)
            raise DatasetSchemaError(f"Ambiguous advanced schema markers: {formats}.")
        if gated:
            return gated[0]
        candidates = [
            adapter for adapter in self._adapters
            if adapter.detection_priority == "structured" and adapter.detect(row)
        ]
        if not candidates:
            candidates = [
                adapter for adapter in self._adapters
                if adapter.detection_priority == "fallback" and adapter.detect(row)
            ]
        if len(candidates) != 1:
            raise DatasetSchemaError(
                "Ambiguous schema; explicitly map one prompt/answer pair or conversation column."
                if candidates
                else "Unrecognized schema. Map text, messages, or a complete prompt/answer pair."
            )
        return candidates[0]


dataset_adapters = DatasetAdapterRegistry()
for _adapter in (
    BuiltinDatasetAdapter(
        "preference",
        lambda row: any(key in row for key in ("chosen", "rejected")),
        "gated",
        canonicalization_supported=True,
        sft_trainable=False,
        alignment_trainable=True,
    ),
    BuiltinDatasetAdapter(
        "kto",
        lambda row: "desirable" in row,
        "gated",
        canonicalization_supported=True,
        sft_trainable=False,
        alignment_trainable=True,
    ),
    BuiltinDatasetAdapter(
        "pretokenized",
        lambda row: any(key in row for key in ("input_ids", "attention_mask", "labels")),
        "gated",
        canonicalization_supported=False,
        sft_trainable=False,
    ),
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
        canonicalization_supported=False,
        sft_trainable=False,
    ),
    BuiltinDatasetAdapter(
        "multimodal",
        _has_multimodal_content,
        "gated",
        canonicalization_supported=False,
        sft_trainable=False,
    ),
    *(BuiltinDatasetAdapter(name, lambda row, keys=keys: all(key in row for key in keys)) for name, keys in PAIR_FORMATS.items()),
    BuiltinDatasetAdapter("messages", lambda row: "messages" in row),
    BuiltinDatasetAdapter("sharegpt", lambda row: "conversations" in row),
    BuiltinDatasetAdapter("text", lambda row: "text" in row, "fallback"),
):
    dataset_adapters.register(_adapter)

# Compatibility name retained for existing extensions/tests; the registry is iterable.
DATASET_ADAPTERS = dataset_adapters


def register_dataset_adapter(adapter: DatasetAdapter) -> None:
    dataset_adapters.register(adapter)


def get_dataset_adapter(format_id: str) -> DatasetAdapter:
    return dataset_adapters.get(format_id)


def detect_schema(row: Mapping[str, Any] | Any) -> str:
    if not isinstance(row, Mapping):
        raise DatasetSchemaError("Each record must be an object.")
    return dataset_adapters.detect(dict(row)).format_id


def _text(row: dict, key: str, *, optional: bool = False) -> str:
    value = row.get(key, "") if optional else row.get(key)
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise DatasetSchemaError(f"'{key}' must be {'a string' if optional else 'a non-empty string'}.")
    return value


def _alpaca_history(row: dict[str, Any]) -> list[dict[str, str]]:
    if "history" not in row:
        return []
    history = row["history"]
    if not isinstance(history, list):
        raise DatasetSchemaError("'history' must be a list of [user, assistant] pairs.")
    messages: list[dict[str, str]] = []
    for index, pair in enumerate(history, start=1):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise DatasetSchemaError(f"History item {index} must be a [user, assistant] pair.")
        user, assistant = pair
        if not isinstance(user, str) or not user.strip():
            raise DatasetSchemaError(f"History item {index} user text must be a non-empty string.")
        if not isinstance(assistant, str) or not assistant.strip():
            raise DatasetSchemaError(f"History item {index} assistant text must be a non-empty string.")
        messages.extend((
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ))
    return messages


def canonicalize(
    row: Mapping[str, Any] | Any,
    columns: dict[str, str] | None = None,
) -> CanonicalDatasetRecord:
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
        return RawTextRecord(source_format=schema, text=_text(row, "text"))
    if schema == "preference":
        return PreferenceRecord(
            source_format=schema,
            prompt=_preference_messages(row.get("prompt"), default_role="user", label="prompt"),
            chosen=_preference_messages(row.get("chosen"), default_role="assistant", label="chosen"),
            rejected=_preference_messages(row.get("rejected"), default_role="assistant", label="rejected"),
            metadata=dict(row.get("metadata") or {}),
        )
    if schema == "kto":
        desirable = row.get("desirable")
        if not isinstance(desirable, bool):
            raise DatasetSchemaError("'desirable' must be a boolean.")
        response = _preference_messages(
            row.get("response", row.get("completion")),
            default_role="assistant",
            label="response",
        )
        if len(response) != 1:
            raise DatasetSchemaError("KTO response must contain exactly one assistant turn.")
        return KTORecord(
            source_format=schema,
            prompt=_preference_messages(row.get("prompt"), default_role="user", label="prompt"),
            response=response[0],
            desirable=desirable,
            metadata=dict(row.get("metadata") or {}),
        )
    if schema in PAIR_FORMATS:
        prompt_key, answer_key = PAIR_FORMATS[schema]
        prompt, answer = _text(row, prompt_key), _text(row, answer_key)
        if schema != "instruction" and any(key in row for key in ("system", "history")):
            raise DatasetSchemaError(
                "Top-level system/history fields are supported only by instruction/output data; map them explicitly."
            )
        if row.get("input") and row.get("context"):
            raise DatasetSchemaError("Both input and context are populated; explicitly map the intended context.")
        context = _text(row, "input" if "input" in row else "context", optional=True)
        messages = []
        if schema == "instruction" and "system" in row:
            messages.append({"role": "system", "content": _text(row, "system")})
        if schema == "instruction":
            messages.extend(_alpaca_history(row))
        messages.extend((
            {"role": "user", "content": prompt + (f"\n\n{context}" if context else "")},
            {"role": "assistant", "content": answer},
        ))
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
    return ConversationRecord(source_format=schema, messages=turns)


def _preference_messages(value: Any, *, default_role: str, label: str) -> list[Message]:
    if isinstance(value, str):
        raw = [{"role": default_role, "content": value}]
    elif isinstance(value, list):
        raw = value
    else:
        raise DatasetSchemaError(f"'{label}' must be a non-empty string or message list.")
    if not raw:
        raise DatasetSchemaError(f"'{label}' must not be empty.")
    try:
        return [Message.model_validate(message) for message in raw]
    except ValueError as exc:
        raise DatasetSchemaError(
            f"'{label}' messages require supported roles and non-empty string content."
        ) from exc


class SchemaSample(BaseModel):
    line: int
    source_format: str | None = None
    canonical: CanonicalDatasetRecord | None = None
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
            if not isinstance(row, Mapping):
                raise DatasetSchemaError("Each record must be an object.")
            adapter = dataset_adapters.detect(dict(row))
            sample.source_format = adapter.format_id
            preview = adapter.preview(row)
            sample.canonical = preview.canonical
            if not preview.validation.valid:
                sample.error = "; ".join(issue.message for issue in preview.validation.issues)
        except DatasetSchemaError as exc:
            sample.error = str(exc)
        samples.append(sample)
    return SchemaReport(sampled_rows=len(samples), sample_limit=limit,
                        formats=sorted({s.source_format for s in samples if s.source_format}),
                        valid=bool(samples) and all(s.error is None for s in samples), samples=samples)
