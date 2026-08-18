"""MRDTS environment configuration.

The numerical defaults reproduce the single-region JTPDTS baseline.  Values
that belong specifically to the multi-region extension are intentionally
optional until the scenario scale is fixed.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class TimeConfig:
    """Two-phase timing used by one environment step."""

    trajectory_duration_s: float = 0.5
    task_processing_duration_s: float = 1.0
    max_time_slots: int = 500

    @property
    def slot_duration_s(self) -> float:
        """Return the complete slot duration: trajectory plus processing."""
        return self.trajectory_duration_s + self.task_processing_duration_s


@dataclass
class MapConfig:
    """Physical deployment area for the JTPDTS baseline region."""

    width_m: float = 600.0
    height_m: float = 600.0


@dataclass
class GroundDeviceConfig:
    """Ground-device resources and per-slot DAG generation."""

    baseline_devices_per_region: int = 10
    cpu_cores: int = 1
    cpu_frequency_ghz: float = 1.0
    transmit_power_w: float = 0.1
    dags_per_device_per_slot: int = 1


@dataclass
class MemberUAVConfig:
    """Member-UAV mobility, computing, and energy parameters."""

    baseline_members_per_region: int = 4
    altitude_m: float = 50.0
    max_horizontal_speed_mps: float = 10.0
    min_safety_distance_m: float = 20.0
    coverage_radius_m: float = 180.0
    cpu_cores: int = 2
    cpu_frequency_per_core_ghz: float = 10.0
    transmit_power_w: float = 1.0
    expected_energy_budget_j_per_slot: float = 145.0
    capacitance_factor: float = 1e-28

    # Rotary-wing propulsion parameters used by the supplied JTPDTS code.
    blade_profile_power_coefficient: float = 85.0
    induced_power_coefficient: float = 0.131
    induced_power_parameter: float = 0.16
    parasite_power_coefficient: float = 0.0115
    base_power_coefficient: float = 76.0
    rotor_tip_speed_mps: float = 110.0


@dataclass
class BaseStationConfig:
    """Ground BS configuration.

    JTPDTS specifies a high-performance MEC platform but does not publish its
    CPU parameters, so those fields remain unset rather than inheriting the
    temporary 2-core/12-GHz value in the reference code.
    """

    stations_per_region: int = 1
    is_ground_station: bool = True
    cpu_cores: Optional[int] = None
    cpu_frequency_per_core_ghz: Optional[float] = None


@dataclass
class DAGConfig:
    """Default dependent-task generator parameters."""

    subtasks_per_dag: int = 10
    max_out_degree: int = 2
    shape_parameter: float = 1.0
    regularity_parameter: float = 0.5
    subtask_data_size_kb: Tuple[int, int] = (100, 500)
    subtask_cpu_cycles: Tuple[float, float] = (1e7, 1e8)
    intermediate_data_size_kb: Tuple[int, int] = (100, 300)

    # A DAG is discarded at the end of its arrival slot if it is incomplete.
    must_finish_within_current_slot: bool = True
    unfinished_dag_policy: str = "drop"


@dataclass
class CommunicationConfig:
    """JTPDTS air-ground and air-air link parameters."""

    gd_or_bs_to_uav_bandwidth_mhz: float = 4.0
    uav_to_uav_bandwidth_mhz: float = 6.0
    uav_to_gd_or_bs_bandwidth_mhz: float = 6.0
    reference_channel_gain_at_1m_db: float = -60.0
    noise_power_db: float = -100.0
    nlos_attenuation_factor: float = 0.2
    los_environment_alpha: float = 9.61
    los_environment_beta: float = 0.16


@dataclass
class DistributedScenarioConfig:
    """Parameters introduced by the multi-region distributed setting.

    ``None`` denotes a value that must be supplied when a concrete distributed
    scenario is created; it is not a numerical default.
    """

    num_regions: Optional[int] = None
    devices_per_region: Optional[Tuple[int, ...]] = None
    initial_members_per_region: Optional[Tuple[int, ...]] = None
    masters_per_region: int = 1
    members_can_migrate_across_regions: bool = True
    member_uavs_can_communicate_across_regions: bool = False
    master_uavs_can_communicate: bool = True
    master_update_interval_slots: Optional[int] = None


@dataclass
class EnvironmentConfig:
    """Top-level configuration passed to the future MRDTS environment."""

    time: TimeConfig = field(default_factory=TimeConfig)
    map: MapConfig = field(default_factory=MapConfig)
    ground_devices: GroundDeviceConfig = field(default_factory=GroundDeviceConfig)
    member_uavs: MemberUAVConfig = field(default_factory=MemberUAVConfig)
    base_stations: BaseStationConfig = field(default_factory=BaseStationConfig)
    dag: DAGConfig = field(default_factory=DAGConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    distributed: DistributedScenarioConfig = field(
        default_factory=DistributedScenarioConfig
    )


DEFAULT_CONFIG = EnvironmentConfig()
