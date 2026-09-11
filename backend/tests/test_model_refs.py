from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from app.model_refs import ModelReferenceError, _classify_path, resolve_model_ref, run_ref


class ModelReferenceTests(unittest.TestCase):
    def test_run_ref_is_stable(self):
        self.assertEqual(run_ref(42), "run:42")

    def test_local_adapter_uses_declared_base(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "org/base-model"}), encoding="utf-8"
            )
            model = _classify_path(path, str(path))
            self.assertEqual(model.kind, "adapter")
            self.assertEqual(model.base_model, "org/base-model")
            self.assertEqual(model.adapter_path, str(path))

    def test_scratch_checkpoint_requires_tokenizer_files(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "model.pt").touch()
            (path / "arch.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ModelReferenceError, "incomplete"):
                _classify_path(path, str(path))

    def test_remote_hf_id_remains_lazy(self):
        model = resolve_model_ref("org/model-1b")
        self.assertEqual(model.kind, "transformers")
        self.assertEqual(model.load_ref, "org/model-1b")


if __name__ == "__main__":
    unittest.main()
