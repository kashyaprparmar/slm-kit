from copy import deepcopy

import pytest
from pydantic import ValidationError

from app.datasets.adapters import (
    DatasetAdapterRegistry,
    DatasetSchemaError,
    KTORecord,
    MediaReference,
    Message,
    MultimodalRecord,
    PreferenceRecord,
    ToolCall,
    ToolConversationRecord,
    ToolDefinition,
    ToolFunctionCall,
    ToolMessage,
    canonicalize,
    detect_schema,
    get_dataset_adapter,
)


def test_alpaca_system_history_and_current_turn_are_preserved_exactly():
    row = {
        "system": " उत्तर दो ",
        "history": [["पहला प्रश्न", "पहला उत्तर"], [" second ", " two "]],
        "instruction": "Current",
        "input": "context",
        "output": " final ",
    }
    original = deepcopy(row)
    record = canonicalize(row)
    assert [message.model_dump() for message in record.messages] == [
        {"role": "system", "content": " उत्तर दो "},
        {"role": "user", "content": "पहला प्रश्न"},
        {"role": "assistant", "content": "पहला उत्तर"},
        {"role": "user", "content": " second "},
        {"role": "assistant", "content": " two "},
        {"role": "user", "content": "Current\n\ncontext"},
        {"role": "assistant", "content": " final "},
    ]
    assert row == original


@pytest.mark.parametrize(
    "row,match",
    [
        ({"instruction": "Q", "output": "A", "system": None}, "system"),
        ({"instruction": "Q", "output": "A", "history": "bad"}, "history"),
        ({"instruction": "Q", "output": "A", "history": [["Q"]]}, "History item 1"),
        ({"instruction": "Q", "output": "A", "history": [["", "A"]]}, "user text"),
        ({"prompt": "Q", "response": "A", "system": "S"}, "supported only"),
    ],
)
def test_malformed_or_lossy_instruction_metadata_is_rejected(row, match):
    with pytest.raises(DatasetSchemaError, match=match):
        canonicalize(row)


@pytest.mark.parametrize(
    "row,format_id",
    [
        ({"input_ids": [1], "labels": [1]}, "pretokenized"),
        ({"messages": [{"role": "assistant", "tool_calls": []}], "tools": []}, "tool_calling"),
        ({"messages": [{"role": "user", "content": [{"type": "image"}]}]}, "multimodal"),
    ],
)
def test_advanced_formats_are_detected_and_previewed_as_unsupported(row, format_id):
    assert detect_schema(row) == format_id
    adapter = get_dataset_adapter(format_id)
    assert not adapter.capabilities().canonicalization.allowed
    assert not any(capability.allowed for capability in adapter.capabilities().training_stages.values())
    preview = adapter.preview(row)
    assert not preview.validation.valid
    assert preview.validation.issues[0].code == "unsupported_format"
    assert preview.canonical is None


@pytest.mark.parametrize(
    "row,format_id",
    [
        ({"prompt": "Q", "chosen": "A", "rejected": "B"}, "preference"),
        ({"prompt": "Q", "response": "A", "desirable": True}, "kto"),
    ],
)
def test_alignment_formats_are_canonical_and_stage_gated(row, format_id):
    adapter = get_dataset_adapter(format_id)
    capabilities = adapter.capabilities()
    assert capabilities.canonicalization.allowed
    assert capabilities.training_stages["alignment"].allowed
    assert not capabilities.training_stages["supervised_fine_tuning"].allowed
    preview = adapter.preview(row)
    assert preview.validation.valid
    assert preview.canonical is not None


def test_conflicting_advanced_markers_are_not_guessed():
    with pytest.raises(DatasetSchemaError, match="Ambiguous advanced"):
        detect_schema({"prompt": "Q", "chosen": "A", "rejected": "B", "desirable": True})


def test_registry_rejects_duplicate_format_ids():
    registry = DatasetAdapterRegistry()
    adapter = get_dataset_adapter("text")
    registry.register(adapter)
    assert registry.get("text") is adapter
    with pytest.raises(ValueError, match="already registered"):
        registry.register(adapter)


def test_canonical_fingerprint_is_semantic_and_stable():
    pair = canonicalize({"response": "A", "prompt": "Q"})
    chat = canonicalize({"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]})
    assert pair.source_format != chat.source_format
    assert pair.storage_row() == chat.storage_row()
    assert pair.canonical_fingerprint() == chat.canonical_fingerprint()


def test_preference_and_kto_v2_contracts_enforce_core_invariants():
    prompt = [Message(role="user", content="Q")]
    chosen = [Message(role="assistant", content="A")]
    rejected = [Message(role="assistant", content="B")]
    preference = PreferenceRecord(
        source_format="fixture",
        prompt=prompt,
        chosen=chosen,
        rejected=rejected,
    )
    assert preference.schema_version == 2
    assert preference.storage_row()["kind"] == "preference"
    with pytest.raises(ValidationError, match="must differ"):
        PreferenceRecord(
            source_format="fixture",
            prompt=prompt,
            chosen=chosen,
            rejected=chosen,
        )
    with pytest.raises(ValidationError, match="assistant role"):
        KTORecord(
            source_format="fixture",
            prompt=prompt,
            response=Message(role="user", content="A"),
            desirable=True,
        )


def test_tool_and_multimodal_v2_contracts_preserve_structured_metadata():
    call = ToolCall(
        id="call-1",
        function=ToolFunctionCall(name="weather", arguments={"city": " पुणे "}),
    )
    tool_record = ToolConversationRecord(
        source_format="fixture",
        tools=[ToolDefinition(name="weather", parameters={"type": "object"})],
        messages=[
            ToolMessage(role="user", content="Weather?"),
            ToolMessage(role="assistant", tool_calls=[call]),
            ToolMessage(role="tool", content="Sunny", tool_call_id="call-1"),
            ToolMessage(role="assistant", content="It is sunny."),
        ],
    )
    assert tool_record.storage_row()["messages"][1]["tool_calls"][0]["function"]["arguments"] == {"city": " पुणे "}

    media = MediaReference(
        media_type="image",
        uri="dataset://images/1.png",
        mime_type="image/png",
        metadata={"width": 640, "height": 480},
    )
    multimodal = MultimodalRecord(
        source_format="fixture",
        messages=[Message(role="user", content="Describe the image")],
        media=[media],
    )
    assert multimodal.storage_row()["media"][0]["metadata"]["width"] == 640
    assert multimodal.canonical_fingerprint() == multimodal.canonical_fingerprint()
