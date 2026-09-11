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
            "/api/serving/providers",
            "/api/serving/ollama/import",
            "/api/system/diagnostics",
        }
        self.assertFalse(required - paths)


if __name__ == "__main__":
    unittest.main()
