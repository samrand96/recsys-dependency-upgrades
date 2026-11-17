import json
from subprocess import CompletedProcess

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from silentrank.orchestrator import neighbor_diagnosis as diagnosis
from silentrank.orchestrator import envs, ledger, pilot, runner
from silentrank.worker.pipelines.ref_itemknn import fit
from silentrank.worker.replay_neighbors import main
from silentrank.worker.stages import Stages, csr_arrays
from silentrank.worker.tracing import CallTrace


def test_capture_then_prune_matches_original_reference_model(tmp_path):
    interactions = csr_matrix(np.array([[1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1]], dtype=np.float32))
    stage2 = tmp_path / "S2.npz"
    np.savez(stage2, **csr_arrays(interactions))
    capture, replay = tmp_path / "capture", tmp_path / "replay"
    s2_hash = pilot._digest(stage2)
    assert main(["capture", "--input", str(stage2), "--input-sha256", s2_hash, "--output-dir", str(capture)]) == 0
    frozen = capture / "weights.npz"
    assert main(["prune", "--input", str(frozen), "--input-sha256", pilot._digest(frozen),
                 "--output-dir", str(replay), "--neighbors", "2", "--training-input-sha256", "0" * 64]) == 0
    expected = csr_arrays(fit(interactions, 2, CallTrace("numpy")))
    metadata = pilot._read_json(replay / "replay_meta.json")
    actual = diagnosis._arrays(replay / "model.npz", metadata["hashes"]["S3"], "S3")
    for key, values in expected.items():
        np.testing.assert_array_equal(actual[key], values)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        main(["capture", "--input", str(stage2), "--input-sha256", "0" * 64, "--output-dir", str(tmp_path / "bad")])


def test_changed_target_rows_distinguish_exact_boundary_ties():
    weights = {"shape": [4, 4], "indices": np.array([1, 3, 2, 0, 2, 3]),
               "indptr": [0, 3, 6, 6, 6], "data": np.array([0.8, 0.5, 0.5, 0.8, 0.5, 0.3])}
    old = {"indices": np.array([1, 2, 0, 2]), "indptr": [0, 2, 4, 4, 4], "data": np.array([0.8, 0.5, 0.8, 0.5])}
    new = {"indices": np.array([1, 3, 0, 3]), "indptr": [0, 2, 4, 4, 4], "data": np.array([0.8, 0.5, 0.8, 0.3])}
    changes = diagnosis._changed_rows(weights, old, new)
    assert [row["target_index"] for row in changes] == [0, 1]
    assert [row["all_swaps_at_pruning_boundary"] for row in changes] == [True, False]


@pytest.mark.parametrize("failure", [None, "encounter_order", "reapply"])
def test_neighbor_batch_replays_only_matching_encounter_order(tmp_path, tmp_ledger, monkeypatch, failure):
    stage2 = tmp_path / "S2.npz"
    interactions = csr_matrix(np.ones((1, 3), dtype=np.float32))
    np.savez(stage2, **csr_arrays(interactions))
    monkeypatch.setattr(diagnosis.paths, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(pilot, "_source_version", lambda pipeline: ("head", "worker-hash"))
    env = {"cpu": {"model": "synthetic"}, "kernel": "same"}
    old_model = {**csr_arrays(fit(interactions, 1, CallTrace("numpy"))),
                 "training_input_sha256": np.asarray("0" * 64), "neighbors": np.asarray(1, dtype=np.int64)}
    new_model = {**old_model, "indices": old_model["indices"].copy()}
    new_model["indices"][0] = 3 - old_model["indices"][0]
    originals = []
    for side, version, model in (("old", "1.26.4", old_model), ("new", "2.0.2", new_model)):
        stages = Stages(tmp_path)
        stages.record("S3", model, "model.npz")
        originals.append({"run_id": side, "env_id": side, "stage2": stage2, "host_id": envs.host_id_for(env),
                          "host": {**env, "numpy": {"version": version}}, "inputs": {
                              "image": "image", "neighbors": 1, "dataset_sha256": "0" * 64,
                          }, "stages": {"S3": stages.rows["S3"]["canonical_sha256"]}, "model": model})
    old, new = originals
    monkeypatch.setattr(diagnosis, "_original", lambda *args: ({"comparison_id": "ab"}, {"numpy": {}}, [old, new]))
    # Use a synthetic runner to separate orchestrator bookkeeping from native version effects.
    def simulate(command, *, mounts, **kwargs):
        (source, _, read_only), (output, _, _) = mounts
        assert read_only
        if failure == "reapply" and output.name == "reapply":
            return CompletedProcess(command, 124, "", "deadline")
        run = new if output.name in {"capture_new", "new", "reapply"} else old
        mode = "capture" if output.name.startswith("capture") else "prune"
        args = [mode, "--input", str(source), "--input-sha256", pilot._digest(source), "--output-dir", str(output)]
        if mode == "prune":
            args.extend(["--neighbors", "1", "--training-input-sha256", "0" * 64])
        main(args)
        meta_path = output / "replay_meta.json"
        metadata = pilot._read_json(meta_path)
        if mode == "capture" and failure == "encounter_order" and run is new:
            weights_path = output / "weights.npz"
            with np.load(weights_path, allow_pickle=False) as npz:
                arrays = dict(npz.items())
            arrays["indices"][:2] = arrays["indices"][1::-1]
            np.savez(weights_path, **arrays)
            stages = Stages(output)
            stages.record("encounter_order", arrays, weights_path.name)
            metadata["hashes"]["encounter_order"] = stages.rows["encounter_order"]["canonical_sha256"]
            metadata["output_sha256"] = pilot._digest(weights_path)
        if mode == "prune" and run is new:
            # Simulate the new library selecting another item at the same boundary score.
            model = output / "model.npz"
            np.savez(model, **new_model)
            metadata["hashes"]["S3"] = new["stages"]["S3"]
            metadata["output_sha256"] = pilot._digest(model)
        pilot._write_json(meta_path, metadata)
        pilot._write_json(output / "host.json", run["host"])
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", simulate)
    result = diagnosis.diagnose_neighbors("N1-P1-float32")
    with ledger.reader() as con:
        phases = dict(con.execute("SELECT phase, status FROM diagnostic_replays").fetchall())
    if failure == "encounter_order":
        assert result["status"] == "unresolved" and result["canonical_weights_equal"] and not result["encounter_order_equal"]
        assert set(phases) == {"S3:capture_old", "S3:capture_new"}
    else:
        assert set(phases) == {"S3:capture_old", "S3:capture_new", "S3:old", "S3:new", "S3:rollback", "S3:reapply"}
        if failure == "reapply":
            assert result["status"] == "unresolved" and phases["S3:reapply"] == "timeout"
        else:
            assert result["status"] == "reproduced" and result["changed_target_count"] == 1
            assert result["all_changes_at_exact_boundary_ties"] and result["cause_class"] == "intended_semantic_change"
