from __future__ import annotations

import unittest

from app.main import app


class ApiContractTests(unittest.TestCase):
    def test_lifecycle_routes_are_registered(self):
        paths = set(app.openapi()["paths"])
        required = {
            "/api/eval/run",
            "/api/registry/model-options",
            "/api/registry/deploy",
            "/api/registry/deployment",
            "/api/registry/inspect",
            "/api/registry/preflight",
            "/api/serving/providers",
            "/api/serving/ollama/import",
            "/api/system/diagnostics",
            "/api/system/services",
            "/api/system/database",
            "/api/projects",
            "/api/projects/{project_id}",
            "/api/datasets/{dataset_id}/schema",
            "/api/datasets/{dataset_id}/versions",
            "/api/dataset-versions/{version_id}/tokenizer-profile",
            "/api/dataset-versions/{version_id}/quality-profile",
        }
        self.assertFalse(required - paths)


if __name__ == "__main__":
    unittest.main()
