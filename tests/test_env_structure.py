"""环境分层与公共契约的结构回归测试。"""

import ast
import importlib
from pathlib import Path
import unittest


class TestEnvStructure(unittest.TestCase):
    def test_modules_are_grouped_by_responsibility(self):
        modules = (
            "env.entities.region", "env.entities.user", "env.entities.uav",
            "env.workload.dag_generator",
            "env.communication.channel_model", "env.communication.routing",
            "env.runtime.event_runtime", "env.runtime.dag_runtime",
            "env.runtime.task_runtime", "env.runtime.channel_queue",
            "env.runtime.server_queue", "env.runtime.slot_result",
        )
        for module in modules:
            with self.subTest(module=module):
                try:
                    loaded = importlib.import_module(module)
                except ModuleNotFoundError:
                    self.fail(f"缺少分类后的模块 {module}")
                self.assertEqual(loaded.__name__, module)

    def test_simulator_exposes_existing_slot_lifecycle(self):
        try:
            module = importlib.import_module("env.simulator")
        except ModuleNotFoundError:
            self.fail("现有仿真编排应迁入 env.simulator")
        self.assertEqual(module.Simulator.__module__, "env.simulator")
        for method in ("begin_slot", "end_slot", "refresh_slot_topology",
                       "create_scheduling_runtime"):
            self.assertTrue(callable(getattr(module.Simulator, method)))

    def test_shared_identifiers_have_one_definition(self):
        try:
            types = importlib.import_module("env.types")
        except ModuleNotFoundError:
            self.fail("基础标识应由 env.types 定义")
        for name in ("EntityKind", "EntityRef", "DirectedChannelKey", "TaskKey"):
            with self.subTest(name=name):
                self.assertEqual(getattr(types, name).__module__, "env.types")

    def test_simulation_requests_belong_to_env(self):
        try:
            contracts = importlib.import_module("env.contracts")
        except ModuleNotFoundError:
            self.fail("仿真输入应由 env.contracts 定义")
        for name in ("DAGRequest", "PlacementDecision"):
            with self.subTest(name=name):
                self.assertEqual(getattr(contracts, name).__module__, "env.contracts")

    def test_env_does_not_import_methods(self):
        env_dir = Path(__file__).resolve().parents[1] / "env"
        violations = []
        for path in env_dir.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    modules = [node.module or ""]
                elif isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    continue
                if any(module == "methods" or module.startswith("methods.") for module in modules):
                    violations.append(f"{path.relative_to(env_dir)}:{node.lineno}")
        self.assertEqual(violations, [], "env 不能反向依赖 methods")


if __name__ == "__main__":
    unittest.main()
