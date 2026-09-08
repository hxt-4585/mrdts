"""统一评估生命周期；学习更新仍由完整方法的 Trainer 负责。"""

import csv
from datetime import datetime, timezone

from experiments.artifacts import create_run, write_json
from experiments.metrics import slot_metrics, summarize
from experiments.randomness import RandomStreams
from experiments.scene import build_scene, resolved_settings, settings_snapshot
from methods.factory import create_method, validate_method


def evaluate(config):
    validate_method(config.method).validate_config(config, training=False)
    check_streams = RandomStreams.from_config(config)
    create_method(config.method, flight_rng=check_streams.flight,
                  scheduling_rng=check_streams.scheduling, checkpoint=config.checkpoint,
                  device=config.device)  # 创建结果目录之前验证组件名称。
    scenes = [resolved_settings(config) for _ in range(config.episodes)]
    output, metadata = create_run(config, "evaluate", [settings_snapshot(scene) for scene in scenes])
    if config.checkpoint is not None:
        import hashlib
        metadata.update(checkpoint=str(config.checkpoint.resolve()),
                        checkpoint_sha256=hashlib.sha256(config.checkpoint.read_bytes()).hexdigest())
    rows = []
    try:
        with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = None
            for episode, settings in enumerate(scenes):
                # 相同空间配置重建场景：用户位置固定，UAV 回到相同初始位置。
                streams = RandomStreams.from_config(config, config.episode_start + episode)
                scene = build_scene(settings, randomness=streams)
                method = create_method(config.method, flight_rng=streams.flight,
                                       scheduling_rng=streams.scheduling, checkpoint=config.checkpoint,
                                       device=config.device)
                for slot in range(config.slots):
                    row = slot_metrics(method.run_slot(scene, scene.workload(slot)), episode, slot)
                    rows.append(row)
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
        write_json(output / "summary.json", summarize(rows))
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)
    return output
