"""Hand-computable centralized HEFT cases, independent of other methods."""
import unittest
from dataclasses import replace

import numpy as np

from env.communication.channel_model import ChannelModel
from env.contracts import DAGRequest
from env.runtime.event_runtime import SchedulingRuntime, ServerSpec
from env.settings import ChannelConfig
from env.types import EntityKind, EntityRef
from env.workload.dag_generator import DAG
from methods.heft.scheduler import plan_schedule, earliest_gap


GROUND = EntityRef(EntityKind.GROUND_DEVICE, 0)
MEMBER = EntityRef(EntityKind.MEMBER_UAV, 0)
BS = EntityRef(EntityKind.BS, 0)


class UnitChannel(ChannelModel):
    def calculate_link(self, *args, **kwargs):
        return replace(super().calculate_link(*args, **kwargs), average_channel_gain=1.)


def scenario(specs=None, regions=None, start=0.):
    specs = specs or {GROUND: (1.,), MEMBER: (2.,), BS: (4.,)}
    model = UnitChannel(replace(ChannelConfig.default(), total_bandwidth_mhz=.001,
                                noise_density_w_per_hz=.001))
    return SchedulingRuntime(model, {node: np.array([i * 10., 0., 10.])
                                    for i, node in enumerate(specs)},
                             {node: ServerSpec(cores, 0.) for node, cores in specs.items()},
                             {node: 1. for node in specs},
                             member_regions=regions or {MEMBER: 1}, slot_start=start)


def workload(cycles, inputs=None, edges=(), sizes=(), owner=MEMBER, ground=GROUND, dag_id=0):
    inputs = np.zeros(len(cycles)) if inputs is None else inputs
    return DAGRequest(dag_id, DAG(len(cycles), list(edges), np.column_stack((inputs, cycles)),
                                 dict(zip(edges, sizes))), owner, ground)


class TestHEFT(unittest.TestCase):
    def test_insertion_uses_an_earlier_gap_and_preserves_release_time(self):
        self.assertEqual(earliest_gap([(3., 5.), (8., 10.)], 1., 2.), (1., 0))
        self.assertEqual(earliest_gap([(3., 5.), (8., 10.)], 2., 3.), (5., 1))
        self.assertEqual(earliest_gap([(3., 5.), (8., 10.)], 7., 2.), (10., 2))

    def test_dependency_chain_uses_eft_and_does_not_advance_runtime(self):
        runtime = scenario(start=10.)
        request = workload([8., 4.], edges=((0, 1),), sizes=(0.,))
        runtime.register_dags([request])
        plan = plan_schedule(runtime, [request])
        first, second = request.task_keys
        self.assertEqual(list(plan.placements), [first, second])
        self.assertEqual(plan.placements[first].execution_node, BS)
        self.assertEqual(plan.tasks[first].finish, 12.)
        self.assertEqual(plan.tasks[second].finish, 13.)
        self.assertEqual(plan.ranks[first], 8 * (1 + .5 + .25) / 3 + plan.ranks[second])
        self.assertEqual(runtime.now, 10.)
        self.assertTrue(all(task.execution_node is None for task in runtime.tasks.values()))
        runtime.submit_placements(plan.placements)
        runtime.finish_slot()
        self.assertEqual(runtime.trace(second).compute_finish_at, 13.)

    def test_global_load_and_heterogeneous_cores_are_shared_across_dags(self):
        ground2 = EntityRef(EntityKind.GROUND_DEVICE, 1)
        runtime = scenario({GROUND: (.1,), ground2: (.1,), MEMBER: (.1,), BS: (4., 2.)})
        requests = [workload([8.], ground=ground2, dag_id=1), workload([8.]),
                    workload([8.], ground=ground2, dag_id=2)]
        runtime.register_dags(requests)
        plan = plan_schedule(runtime, requests)
        estimates = list(plan.tasks.values())
        self.assertEqual([p.execution_node for p in plan.placements.values()], [BS, BS, BS])
        self.assertEqual([(p.core, p.start, p.finish) for p in estimates],
                         [(0, 0., 2.), (1, 0., 4.), (0, 2., 4.)])

    def test_input_transfer_can_make_local_execution_best(self):
        runtime = scenario()
        request = workload([4.], inputs=[100.])
        runtime.register_dags([request])
        plan = plan_schedule(runtime, [request])
        key = request.task_keys[0]
        self.assertEqual(plan.placements[key].execution_node, GROUND)
        self.assertEqual(plan.tasks[key].finish, 4.)

    def test_nonroot_input_and_predecessor_transfer_are_both_counted(self):
        runtime = scenario()
        request = workload([4., 40.], inputs=[100., 1.], edges=((0, 1),), sizes=(2.,))
        runtime.register_dags([request])
        plan = plan_schedule(runtime, [request])
        parent, child = request.task_keys
        self.assertEqual(plan.placements[parent].execution_node, GROUND)
        self.assertEqual(plan.placements[child].execution_node, BS)
        # Ground -> owner -> BS costs 2s per Kbit at exclusive reference rates.
        self.assertEqual(plan.tasks[child].start, 8.)
        self.assertEqual(plan.tasks[child].finish, 18.)

    def test_region_restrictions_and_unassociated_local_tasks(self):
        remote = EntityRef(EntityKind.MEMBER_UAV, 1)
        ground2 = EntityRef(EntityKind.GROUND_DEVICE, 1)
        runtime = scenario({GROUND: (1.,), ground2: (1.,), MEMBER: (2.,),
                            remote: (1000.,), BS: (4.,)}, {MEMBER: 1, remote: 2})
        requests = [workload([8.]), workload([3.], owner=None, ground=ground2, dag_id=1)]
        runtime.register_dags(requests)
        plan = plan_schedule(runtime, requests)
        self.assertEqual(len(plan.placements), 1)
        self.assertNotEqual(next(iter(plan.placements.values())).execution_node, remote)
        runtime.submit_placements(plan.placements)
        result = runtime.finish_slot()
        self.assertTrue(all(dag.completion_time is not None for dag in result.dags))
        self.assertEqual(runtime.trace(requests[1].task_keys[0]).execution_node, ground2)

    def test_zero_rank_dependencies_and_request_order_are_deterministic(self):
        runtime = scenario()
        request = workload([0., 0., 0.], edges=((2, 0), (0, 1)), sizes=(0., 0.))
        runtime.register_dags([request])
        plan = plan_schedule(runtime, [request])
        self.assertEqual([key.node_id for key in plan.placements], [2, 0, 1])
        self.assertEqual(plan.placements, plan_schedule(runtime, [request]).placements)


if __name__ == '__main__':
    unittest.main()
