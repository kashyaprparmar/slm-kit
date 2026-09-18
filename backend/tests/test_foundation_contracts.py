from app.api.runs import backends
from app.backends.base import RegisteredBackendSelector, load_builtin_backends
from app.core.events import (
    ArtifactEvent,
    ErrorEvent,
    ProfileEvent,
    ProgressEvent,
    ResourceEvent,
    WarningEvent,
    dump_event,
    parse_event,
)
from app.datasets.adapters import DATASET_ADAPTERS, DatasetAdapter, get_dataset_adapter
from app.domain import RunConfig, TaskType, TrainingStage
from app.models.capabilities import FAMILY_RULES, ModelFamilyAdapter, resolve_capabilities
from app.serving.providers import transformers


def test_run_config_exposes_stage_without_changing_serialized_contract():
    config = RunConfig(task="finetune", output_name="test")
    assert config.schema_version == 1
    assert config.training_stage == TrainingStage.SUPERVISED_FINE_TUNING
    assert config.operation.stage == TrainingStage.SUPERVISED_FINE_TUNING
    assert "training_stage" not in config.model_dump()


def test_registered_backend_descriptor_is_selection_and_api_source():
    load_builtin_backends()
    config = RunConfig(task="finetune", output_name="test", backend="transformers")
    selection = RegisteredBackendSelector().select(config)
    assert selection.capabilities.supports_task(TaskType.FINETUNE)
    assert selection.backend.supported_methods == selection.capabilities.supported_methods

    api_descriptor = next(item for item in backends() if item["name"] == "transformers")
    assert api_descriptor["methods"] == sorted(method.value for method in selection.capabilities.supported_methods)
    assert api_descriptor["method_capabilities"] == selection.capabilities.model_dump(mode="json")["methods"]


def test_dataset_registry_implements_shared_adapter_contract():
    assert all(isinstance(adapter, DatasetAdapter) for adapter in DATASET_ADAPTERS)
    adapter = get_dataset_adapter("prompt_response")
    record = adapter.canonicalize({"prompt": "Hello", "response": "Hi"})
    assert record.messages[-1].content == "Hi"
    assert adapter.validate({"prompt": "Hello", "response": "Hi"}).valid
    assert adapter.preview({"prompt": "Hello", "response": "Hi"}).storage["messages"][-1]["content"] == "Hi"
    assert adapter.capabilities().training_stages["supervised_fine_tuning"].allowed


def test_model_family_adapter_drives_structured_capabilities():
    capabilities = resolve_capabilities(
        "org/model",
        {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]},
        {"chat_template": "{{ messages }}"},
        dependencies={},
    )
    assert capabilities.family == "Qwen"
    assert capabilities.template_capabilities.native.allowed
    assert capabilities.peft_methods["qlora"].requires_quantized_base
    assert capabilities.quantization_matrix["gguf"].operations == ["export", "serve"]
    assert isinstance(FAMILY_RULES[0], ModelFamilyAdapter)


def test_serving_descriptor_derives_legacy_operations():
    descriptor = transformers.capabilities()
    assert descriptor.enabled_operations == ["serve", "test", "chat", "scratch", "adapter"]


def test_extended_events_round_trip_on_the_shared_wire_protocol():
    events = [
        ProgressEvent(current=1, total=2),
        ResourceEvent(resources={"vram_mb": 512}),
        ArtifactEvent(kind="adapter", path="output/adapter"),
        WarningEvent(code="tight_fit", message="VRAM headroom is low."),
        ProfileEvent(name="tokens", values={"p95": 1024}),
        ErrorEvent(code="worker_failed", message="Worker stopped."),
    ]
    assert [parse_event(dump_event(event)).type for event in events] == [event.type for event in events]
