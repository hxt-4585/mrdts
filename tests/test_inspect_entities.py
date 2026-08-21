"""实体属性检查脚本的测试。"""

import io
import unittest
from contextlib import redirect_stdout

from scripts.inspect_entities import inspect_default_entities


class TestInspectEntities(unittest.TestCase):
    def test_prints_all_default_entity_groups_and_key_attributes(self):
        """脚本应直接输出三类实体及其关键属性。"""
        output = io.StringIO()

        with redirect_stdout(output):
            inspect_default_entities()

        text = output.getvalue()
        self.assertIn("User", text)
        self.assertIn("MasterUAV", text)
        self.assertIn("MemberUAV", text)
        self.assertIn("positions", text)
        self.assertIn("region_ids", text)
        self.assertIn("core_frequencies", text)
        self.assertIn("bs_index", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
