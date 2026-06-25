import importlib.util
import json
import unittest
from pathlib import Path


def load_smoke_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "agent_readiness_smoke.py"
    spec = importlib.util.spec_from_file_location("agent_readiness_smoke", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class AgentReadinessSmokeTest(unittest.TestCase):
    def test_static_smoke_writes_validation_result(self):
        smoke = load_smoke_module()
        self.addCleanup(lambda: None)
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "validation_result.json"
            exit_code = smoke.main(["--mode", "static", "--output", str(output)])

            self.assertEqual(exit_code, 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["product"], "cuVSLAM")
            self.assertEqual(payload["mode"], "static")
            self.assertEqual(payload["status"], "pass")
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertTrue(payload["checks"])
            self.assertTrue(all(check["status"] == "pass" for check in payload["checks"]))
