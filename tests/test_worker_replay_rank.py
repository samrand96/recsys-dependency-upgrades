import hashlib
import json

import numpy as np
import pytest

from silentrank.worker.replay_rank import main


def test_replay_loads_frozen_scores_and_seen_mask(tmp_path):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    scores = np.arange(25, dtype=np.float32).reshape(1, -1)
    seen = np.zeros(scores.shape, dtype=np.bool_)
    seen[0, -1] = True
    items = np.arange(101, 126, dtype=np.int64)
    np.save(frozen / "scores.npy", scores)
    np.save(frozen / "seen_mask.npy", seen)
    np.save(frozen / "item_ids.npy", items)
    score_hash = hashlib.sha256((frozen / "scores.npy").read_bytes()).hexdigest()

    output = tmp_path / "replay"
    args = ["--input-dir", str(frozen), "--output-dir", str(output), "--scores-sha256", score_hash]
    assert main(args) == 0
    np.testing.assert_array_equal(np.load(output / "topk_native10.npy"), items[-2:-12:-1][None, :])
    np.testing.assert_array_equal(np.load(output / "topk_native.npy"), items[-2:-22:-1][None, :])
    metadata = json.loads((output / "replay_meta.json").read_text(encoding="utf-8"))
    assert metadata["input_sha256"]["scores.npy"] == score_hash
    assert metadata["trace"] == {"numpy.argpartition": 2, "numpy.argsort": 2}
    assert metadata["wall_s"] >= 0
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(args)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main([*args, "--output-dir", str(tmp_path / "bad"), "--scores-sha256", "0" * 64])
