"""由实验总 seed 派生随机流：固定空间场景，各回合生成随机 DAG。"""

from dataclasses import asdict, dataclass, replace

from env.communication.channel_model import ChannelModel
from env.entities.region import Region
from env.entities.uav import MasterUAV, MemberUAV
from env.entities.user import User
from env.settings import ChannelConfig, DAGConfig, RegionConfig, SchedulingConfig, UAVConfig, UserConfig
from env.simulator import Simulator
from env.workload.dag_generator import DAGGenerator


def resolved_settings(config):
    return {
        "region": RegionConfig.default(),
        "user": replace(UserConfig.default(), total_users=config.users),
        "dag": replace(DAGConfig.default(), n=config.dag_nodes),
        "uav": UAVConfig.default(), "channel": ChannelConfig.default(),
        "scheduling": SchedulingConfig.default(),
    }


@dataclass
class Scene:
    region: Region
    users: User
    masters: MasterUAV
    simulator: Simulator
    generator: DAGGenerator

    def workload(self, slot):
        return tuple((slot * self.users.num_users + user_id, user_id, self.generator.generate_single_dag())
                     for user_id in range(self.users.num_users))


def build_scene(settings, *, randomness):
    region = Region(settings["region"], rng=randomness.region)
    region.generate()
    users = User(settings["user"], rng=randomness.user)
    users.generate_from_region(region)
    masters = MasterUAV(settings["uav"])
    masters.generate_from_region(region)
    members = MemberUAV(settings["uav"])
    members.generate_from_region_and_user(region, users, masters)
    simulator = Simulator(members, ChannelModel(settings["channel"]), users.config.transmit_power,
                          users.config.core_frequency, settings["scheduling"])
    return Scene(region, users, masters, simulator,
                 DAGGenerator(settings["dag"], rng=randomness.dag, py_rng=randomness.dag_python))


def settings_snapshot(settings):
    return {name: asdict(value) for name, value in settings.items()}
