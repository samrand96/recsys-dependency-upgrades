import numpy as np
import pyarrow.parquet as pq
import pytest

from silentrank.orchestrator.evaluation import ArtifactContractFailure, evaluate


@pytest.fixture
def ranking_artifacts(tmp_path):
    catalogue = np.arange(101, 124, dtype=np.int64)
    dataset = tmp_path / "frozen.npz"
    np.savez(
        dataset, train=np.array([[0, 0, 5]], dtype=np.int64), validation=np.array([[0, 1, 2]], dtype=np.int64),
        test=np.array([[0, 2, 5]], dtype=np.int64), item_ids=catalogue, activity=np.array([0], dtype=np.int64),
    )
    np.save(tmp_path / "user_ids.npy", np.array([0], dtype=np.int64))
    np.save(tmp_path / "item_ids.npy", catalogue)
    mask = np.zeros((1, len(catalogue)), dtype=bool)
    mask[0, :2] = True
    np.save(tmp_path / "seen_mask.npy", mask)
    scores = -catalogue.astype(np.float32)[None, :]
    np.save(tmp_path / "scores.npy", scores)
    for k, suffix in ((10, "10"), (20, "")):
        np.save(tmp_path / f"topk_native{suffix}.npy", catalogue[None, 2:k + 2])
        np.save(tmp_path / f"scores_topk{suffix}.npy", scores[:, 2:k + 2])
    return tmp_path, dataset, tmp_path / "metrics.parquet"


def test_pilot_evaluates_native_lists_and_detects_reversed_sort(ranking_artifacts):
    directory, dataset, metrics = ranking_artifacts
    evaluated = evaluate(directory, dataset, "test", metrics)
    assert evaluated["violations"] == 0
    assert pq.read_table(metrics)["ndcg10"].to_pylist() == [1.0]

    for name in ("topk_native10.npy", "scores_topk10.npy"):
        file = directory / name
        np.save(file, np.load(file)[:, ::-1])
    evaluated = evaluate(directory, dataset, "test", metrics)
    assert evaluated["violations"] == 0
    assert pq.read_table(metrics)["ndcg10"].to_pylist() == pytest.approx([1 / np.log2(11)])


def test_pilot_detects_swapped_mapping(ranking_artifacts):
    directory, dataset, metrics = ranking_artifacts
    item_file = directory / "item_ids.npy"
    items = np.load(item_file)
    items[[1, 2]] = items[[2, 1]]
    np.save(item_file, items)
    with pytest.raises(ArtifactContractFailure, match="catalogue"):
        evaluate(directory, dataset, "test", metrics)


def test_pilot_detects_missing_seen_mask(ranking_artifacts):
    directory, dataset, metrics = ranking_artifacts
    mask_file = directory / "seen_mask.npy"
    np.save(mask_file, np.zeros_like(np.load(mask_file)))
    with pytest.raises(ArtifactContractFailure, match="observed history"):
        evaluate(directory, dataset, "test", metrics)
