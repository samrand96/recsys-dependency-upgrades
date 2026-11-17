import hashlib
import json
from zipfile import ZipFile

import numpy as np
import pytest

from silentrank.worker.__main__ import main


@pytest.fixture
def frozen(tmp_path):
    data = tmp_path / "ml1m.npz"
    np.savez(
        data,
        train=np.asarray([[0, 0, 5], [0, 2, 2], [1, 0, 5], [1, 1, 4], [1, 3, 4], [2, 1, 5], [2, 4, 5], [2, 5, 1]],
                         dtype=np.int64),
        validation=np.asarray([[0, 1, 2], [1, 2, 5], [2, 3, 5]], dtype=np.int64),
        test_users=np.asarray([0, 1], dtype=np.int64),
        user_ids=np.asarray([12, 35, 80], dtype=np.int64),
        item_ids=np.asarray([100, 101, 102, 103, 104, 105], dtype=np.int64),
    )
    return data


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_reference_model_round_trip_preserves_stages_and_seen_mask(tmp_path, frozen, dtype):
    trained = tmp_path / "fit"
    base = ["--pipeline", "ref_itemknn", "--data", str(frozen), "--dtype", dtype, "--trace", "numpy"]
    assert main([*base, "--mode", "retrain", "--output-dir", str(trained)]) == 0
    model = trained / "model.npz"
    model_hash = read_json(trained / "worker_meta.json")["model_output_sha256"]

    inference = tmp_path / "inference"
    assert main([*base, "--mode", "inference", "--model-in", str(model), "--model-sha256", model_hash,
                 "--output-dir", str(inference)]) == 0
    assert read_json(trained / "stage_hashes.json") == read_json(inference / "stage_hashes.json")
    assert read_json(inference / "worker_meta.json")["model_input_sha256"] == model_hash
    assert read_json(inference / "trace.json") == {"numpy.argpartition": 4, "numpy.argsort": 4}

    scores = np.load(inference / "scores.npy", allow_pickle=False)
    seen = np.load(inference / "seen_mask.npy", allow_pickle=False)
    top20 = np.load(inference / "topk_native.npy", allow_pickle=False)
    top10 = np.load(inference / "topk_native10.npy", allow_pickle=False)
    assert scores.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(seen[0], [True, True, True, False, False, False])
    assert top20.shape == (2, 20) and top10.shape == (2, 10)
    assert top10[0, 0] == 103 and set(top10[0, :3]) == {103, 104, 105}
    assert np.all(top10[0, 3:] == -1)
    assert scores[0, 3] == pytest.approx(1 / np.sqrt(2))


def test_inference_rejects_an_unverified_portable_model(tmp_path, frozen):
    fake_model = tmp_path / "wrong.npz"
    fake_model.write_bytes(b"not a model")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(["--pipeline", "ref_itemknn", "--mode", "inference", "--data", str(frozen),
              "--output-dir", str(tmp_path / "run"), "--model-in", str(fake_model), "--model-sha256", "0" * 64])


@pytest.mark.parametrize("pipeline", ["implicit_als", "implicit_cosine", "implicit_bm25"])
def test_implicit_round_trip_uses_portable_model_and_native_scores(tmp_path, frozen, pipeline):
    pytest.importorskip("implicit")
    rng = np.random.Generator(np.random.PCG64(17))
    initial = tmp_path / "initial.npz"
    np.savez(initial, user_factors=rng.random((3, 3), dtype=np.float32) * np.float32(0.01),
             item_factors=rng.random((6, 3), dtype=np.float32) * np.float32(0.01), seed=np.int64(17))
    config = ["--pipeline", pipeline, "--data", str(frozen), "--neighbors", "2", "--factors", "3", "--iterations", "2"]
    fit, replay = tmp_path / "fit", tmp_path / "replay"
    training_args = ["--initial-model", str(initial)] if pipeline == "implicit_als" else []
    assert main([*config, "--mode", "retrain", "--output-dir", str(fit), *training_args]) == 0
    assert main([*config, "--mode", "inference", "--output-dir", str(replay), "--model-in", str(fit / "model.npz")]) == 0
    assert read_json(fit / "stage_hashes.json") == read_json(replay / "stage_hashes.json")
    assert np.isfinite(np.load(replay / "scores.npy", allow_pickle=False)).all()
    top10 = np.load(replay / "topk_native10.npy", allow_pickle=False)
    mask = np.load(replay / "seen_mask.npy", allow_pickle=False)
    for row, seen in zip(top10, mask):
        assert set(row[row >= 0]) == set(np.arange(100, 106)[~seen])


def test_pandas_preparation_respects_frozen_cutoffs_and_training_catalogue(tmp_path):
    source = tmp_path / "ratings.zip"
    with ZipFile(source, "w") as archive:
        archive.writestr("ml-1m/ratings.dat", "12::100::1::10\n12::100::5::10\n35::101::5::11\n"
                         "35::100::4::12\n80::100::5::12\n12::102::4::13\n12::101::5::13\n")
    raw_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"source": {"sha256": raw_hash}, "selected_split": {
        "validation_start_unix": 12, "test_start_unix": 13,
    }}), encoding="utf-8")
    output = tmp_path / "run"
    assert main(["--pipeline", "prep_pandas", "--mode", "prep", "--data", str(tmp_path / "ml1m.npz"),
                 "--raw", str(source), "--output-dir", str(output), "--trace", "pandas"]) == 0
    trace = read_json(output / "trace.json")
    assert trace["pandas.read_csv"] == 1 and trace["pandas.Index.get_indexer"] == 6
    assert all(name.startswith("pandas.") for name in trace)
    stages = read_json(output / "stage_hashes.json")
    assert list(stages) == ["S0", "S1", "S2"]
    assert stages["S0"]["canonical_sha256"] == raw_hash
    with np.load(output / stages["S1"]["artifact"], allow_pickle=False) as maps:
        assert maps["user_ids"].tolist() == [12, 35]
        assert maps["item_ids"].tolist() == [100, 101]
    with np.load(output / stages["S2"]["artifact"], allow_pickle=False) as matrices:
        assert matrices["train_data"].tolist() == [1, 5]
        assert matrices["validation_data"].tolist() == [4]
        assert matrices["test_data"].tolist() == [5]
    counts = read_json(output / "prep_counts.json")
    assert counts["duplicate_rows"] == 1
    assert counts["splits"]["validation"]["cold_user_interactions"] == 1
    assert counts["splits"]["test"]["cold_item_interactions"] == 1
