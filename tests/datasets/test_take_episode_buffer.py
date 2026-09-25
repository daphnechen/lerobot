"""take_episode_buffer must detach frames without losing or duplicating any."""
import numpy as np
import pytest


def test_detaching_a_buffer_preserves_frames_and_advances_the_index(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = {
        "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
        "action": {"dtype": "float32", "shape": (2,), "names": None},
    }
    ds = LeRobotDataset.create(repo_id="test/take_buffer", fps=10,
                               root=tmp_path / "ds", features=features, use_videos=False)
    for i in range(5):
        ds.add_frame({"observation.state": np.float32([i, i]),
                      "action": np.float32([i, i]), "task": "t"})

    detached = ds.take_episode_buffer()
    assert detached["size"] == 5, "frames were lost on detach"
    assert int(detached["episode_index"]) == 0

    live = ds.writer.episode_buffer
    assert live["size"] == 0, "the fresh buffer kept the old frames"
    assert int(live["episode_index"]) == 1, (
        "the next episode reused index 0; two episodes would claim the same "
        "frames directory"
    )

    # Recording continues into the new buffer while the old one is still held.
    for i in range(3):
        ds.add_frame({"observation.state": np.float32([9, 9]),
                      "action": np.float32([9, 9]), "task": "t"})
    assert detached["size"] == 5, "the detached buffer was mutated by later frames"
    assert ds.writer.episode_buffer["size"] == 3

    # Saving the detached one must not disturb the live buffer.
    ds.save_episode(episode_data=detached)
    assert ds.writer.episode_buffer["size"] == 3, "saving reset the live buffer"
    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == 5
