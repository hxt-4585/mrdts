"""打印默认场景中 User、MasterUAV 和 MemberUAV 实例的全部当前属性。

用法：
    python scripts/inspect_entities.py
"""

import os
import sys
from dataclasses import is_dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.randomness import RandomStreams
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User


def _format_value(value):
    """将常见实体属性格式化为完整且易读的文本。"""
    if isinstance(value, np.ndarray):
        return (
            f"ndarray(shape={value.shape}, dtype={value.dtype})\n"
            f"{np.array2string(value, threshold=value.size, max_line_width=120)}"
        )
    if is_dataclass(value):
        return repr(value)
    return repr(value)


def _print_entity(name, entity, property_names=()):
    """打印实例存储字段及指定的只读属性。"""
    print(f"\n{'=' * 20} {name} {'=' * 20}")
    for attribute_name, value in vars(entity).items():
        print(f"{attribute_name}: {_format_value(value)}")
    for property_name in property_names:
        print(f"{property_name}: {_format_value(getattr(entity, property_name))}")


def inspect_default_entities():
    """生成默认实体并直接打印其所有当前属性。"""
    region = Region(rng=RandomStreams.from_config().region)
    region.generate()
    users = User(rng=RandomStreams.from_config().user)
    users.generate_from_region(region)
    masters = MasterUAV()
    masters.generate_from_region(region)
    members = MemberUAV()
    members.generate_from_region_and_user(region, users, masters)

    _print_entity("User", users, property_names=("num_users",))
    _print_entity("MasterUAV", masters, property_names=("num_uavs",))
    _print_entity(
        "MemberUAV",
        members,
        property_names=("num_uavs", "member_uav_count", "bs_index"),
    )


if __name__ == "__main__":
    inspect_default_entities()
