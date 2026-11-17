import json
from subprocess import CompletedProcess

import numpy as np
import pytest

from silentrank.orchestrator import diagnosis, envs, ledger, pilot, runner
from silentrank.worker.stages import Stages


@pytest.fixture
def s7_case(tmp_ledger, tmp_path, monkeypatch):
    case = "N1-P1-float32"
    old_env, new_env = "env-old", "env-new"
    monkeypatch.setattr(diagnosis.paths, "results_dir", lambda: tmp_path)
    monkeypatch.setattr(pilot, "_source_version", lambda pipeline: ("commit-1", "source-1"))
    monkeypatch.setattr(envs, "lock_sha256", lambda env: f"lock-{env}")
    diff = {"numpy": {"old": ["1.26.4"], "new": ["2.0.2"]}}
    monkeypatch.setattr(envs, "_resolved_diff", lambda *args: diff)

    with ledger.writer() as con:
        con.execute("INSERT INTO transitions (transition_id, family, resolved_diff_json) VALUES ('N1', 'numpy', ?)",
                    [json.dumps(diff)])
        con.execute("INSERT INTO cases (case_id, transition_id, pipeline, config_json) VALUES (?, 'N1', 'ref_itemknn', ?)",
                    [case, json.dumps({"old": old_env, "new": new_env})])
        for side, version in (("old", "1.26.4"), ("new", "2.0.2")):
            directory = tmp_path / side
            directory.mkdir()
            env_id = f"env-{side}"
            host = {"cpu": {"model": "test"}, "kernel": "test", "numpy": {"version": version}}
            pilot._write_json(directory / "host.json", host)
            pilot._write_json(directory / "meta.json", {"inputs": {
                "worker_sha256": "source-1", "lock_sha256": f"lock-{env_id}", "image": "runner-image",
            }})
            users = np.array([0], dtype=np.int64)
            items = np.arange(101, 125, dtype=np.int64)
            scores = np.ones((1, len(items)), dtype=np.float32)
            mask = np.zeros(scores.shape, dtype=bool)
            ranks = {k: items[None, :k].copy() for k in diagnosis.RANKS}
            if side == "new":
                for values in ranks.values():
                    values[0, :2] = values[0, 1::-1]
            arrays = {"user_ids.npy": users, "item_ids.npy": items, "scores.npy": scores, "seen_mask.npy": mask,
                      **{name: ranks[k] for k, name in diagnosis.RANKS.items()}}
            for name, values in arrays.items():
                np.save(directory / name, values, allow_pickle=False)
            stages = Stages(directory)
            stages.record("S4", {"user_ids": users, "item_ids": items}, "")
            stages.record("S5", {"scores": scores}, "")
            stages.record("S6", {"seen_mask": mask}, "")
            stages.record("S7", {f"items_{k}": rank for k, rank in ranks.items()}, "")
            stage_hashes = {key: stages.rows.get(key, {}).get("canonical_sha256", "unchanged")
                            for key in (*[f"S{i}" for i in range(9)], "S7c")}
            con.execute(
                "INSERT INTO runs (run_id, case_id, side, env_id, mode, status, host_id, artifact_dir) "
                "VALUES (?, ?, ?, ?, 'inference', 'ok', ?, ?)",
                [side, case, side, env_id, envs.host_id_for(host), str(directory)],
            )
            con.executemany("INSERT INTO stage_hashes (run_id, stage, canonical_sha256) VALUES (?, ?, ?)",
                            [[side, key, value] for key, value in stage_hashes.items()])
        con.execute("INSERT INTO comparisons (comparison_id, case_id, run_a, run_b, kind) VALUES ('ab', ?, 'old', 'new', 'AB')",
                    [case])
    comparison_dir = tmp_path / "compare"
    comparison_dir.mkdir()
    pilot._write_json(comparison_dir / f"{case}.json", {"case": case, "modes": {"inference": {"AB": {
        "comparison_id": "ab", "status": "ok", "contract_failure": False, "behavior": {"canonical_equal": True},
        "runs": [{"old": "old", "new": "new", "changed_stages": ["S7"]}],
    }}}})
    return case


@pytest.mark.parametrize("failure", [None, "reapply"])
def test_diagnose_records_every_attempt_and_requires_reappearance(s7_case, tmp_path, monkeypatch, failure):
    def replay(command, *, mounts, **kwargs):
        (source, _, read_only), (output, _, _) = mounts
        assert read_only and source == tmp_path / "old"
        if output.name == failure:
            return CompletedProcess(command, 124, "", "replay timeout")
        side = "new" if output.name in {"new", "reapply"} else "old"
        assert f"/venvs/env-{side}/bin/python" in command
        expected = tmp_path / side
        output_hashes = {}
        for name in diagnosis.RANKS.values():
            (output / name).write_bytes((expected / name).read_bytes())
            output_hashes[name] = pilot._digest(output / name)
        (output / "host.json").write_bytes((expected / "host.json").read_bytes())
        pilot._write_json(output / "replay_meta.json", {"output_sha256": output_hashes, "input_sha256": {
            name: command[command.index(option) + 1] for name, option in diagnosis.INPUTS.items()
        }})
        return CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(runner, "run", replay)
    result = diagnosis.diagnose_case(s7_case)
    assert [row["phase"] for row in result["replays"]] == ["old", "new", "rollback", "reapply"]
    assert result["ties_only"] and result["changed_packages"] == ["numpy"]
    with ledger.reader() as con:
        attempts = dict(con.execute("SELECT phase, status FROM diagnostic_replays").fetchall())
        adjudications = con.execute("SELECT cause_class FROM adjudications").fetchall()
    assert len(attempts) == 4
    if failure:
        assert result["status"] == "unresolved" and attempts[failure] == "timeout"
        assert adjudications == []
    else:
        assert result["status"] == "reproduced" and result["attribution"] == "numpy"
        assert adjudications == [("intended_semantic_change",)]


def test_diagnose_rejects_modified_original_artifact(s7_case, tmp_path):
    np.save(tmp_path / "old" / "scores.npy", np.zeros((1, 24), dtype=np.float32))
    with pytest.raises(ValueError, match="original artifacts no longer match"):
        diagnosis.diagnose_case(s7_case)
