import unittest
from pathlib import Path

from app.backends.unsloth_backend import _UNSLOTH_DEFAULT_TARGET_MODULES, _unsloth_target_modules

BACKEND_ROOT = Path(__file__).resolve().parents[1]


class GpuConstraintTests(unittest.TestCase):
    def test_docker_stack_is_pinned_as_one_compatibility_matrix(self):
        constraints = (BACKEND_ROOT / "constraints-gpu.txt").read_text(encoding="utf-8")
        dockerfile = (BACKEND_ROOT / "Dockerfile").read_text(encoding="utf-8")

        expected = {
            "torch==2.11.0",
            "torchvision==0.26.0",
            "torchaudio==2.11.0",
            "xformers==0.0.35",
            "torchao==0.18.0",
            "transformers==4.57.6",
            "peft==0.20.0",
        }
        self.assertTrue(expected.issubset(set(constraints.splitlines())))
        self.assertIn("pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime", dockerfile)
        self.assertIn("Qwen3ForCausalLM", dockerfile)

    def test_unsloth_receives_explicit_default_projection_modules(self):
        # PEFT's "all-linear" shorthand is not valid for Unsloth: it iterates
        # the string as characters. An empty UI field must retain automatic
        # targeting without sending that shorthand to Unsloth.
        self.assertEqual(_unsloth_target_modules([]), list(_UNSLOTH_DEFAULT_TARGET_MODULES))
        self.assertEqual(_unsloth_target_modules(["q_proj"]), ["q_proj"])


if __name__ == "__main__":
    unittest.main()
