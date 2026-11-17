# Cloud reruns with raw arrays retained

Declared on 15 November 2026, before any rerun. The researcher approved the reruns on 15 November; each launch still needs the researcher's go required by the project rules, with hardware, time limit and cost.

## Why

The evidence review of 15 November found that the first cloud Jobs exported comparison reports, metrics and run metadata, but not the model, score and ranking arrays. So the cloud results cannot be re-evaluated or re-hashed independently, and the GPU Job has no per-run `device.json`. The paper states both limits. These reruns remove them.

They are **additions**, as the G2 decision requires. The first Jobs' ledgers (`results/hosts/second-cpu-1`, `epyc-log1p-1`, `intel-cpu-1`, `gpu-t4-1`) are not modified, and the tables frozen at `results-g2-v1` are not replaced. Protocols v0 and v1 apply unchanged.

## Jobs

Same image, same harness and same case sets as the first Jobs. The two host-B Jobs become one.

| Job name | Flavor | Cases | First Job(s) |
|---|---|---|---|
| `second-cpu-2` | `cpu-upgrade` | N1-P1-float32, N1-P1-float64, N1-P2-als, N1-P2-bm25, N2-P2-als, N3-P2-als, N4-P6-ease, N5-P6-ease, N6-P6-ease, Q2-P1-float32, Q2-P2-bm25, Q3-P2-als, S5-P1-float32, T1–T6-P7-bpr (19) | `second-cpu-1`, `epyc-log1p-1` |
| `intel-cpu-2` | `t4-small` (CPU only) | the same 19 cases | `intel-cpu-1` |
| `gpu-t4-2` | `t4-medium` | T1–T6-P7-gpu (6) | `gpu-t4-1` |

Hugging Face does not document or guarantee the CPU behind a flavor. The probe records the CPU model in every run. If a Job lands on a different CPU model than its first Job, its results are reported as a further host, not as a replication of that host.

## What is exported

As before: `job.log`, `lscpu.txt`, `nvidia-smi.txt`, the ledger, comparison reports, metrics and run metadata. New, with `SILENTRANK_KEEP_ARRAYS=1`:

- `arrays/<case>.tar`: every run directory of the case (fitted model, full scores, seen mask, native and canonical lists, stage arrays, logs, `device.json`), written as soon as the case finishes.
- `arrays/<case>.json`: SHA-256 of the tar and of every file in it.
- `stages/S2.npz` is left out. It is the interaction matrix rebuilt from the frozen data in every run (about 12 MB each); its hash stays in `stage_hashes.json`, and the job verifies that the prepared data equal the frozen artifact.

The arrays stay in the private bucket `Samrand/silent-ranking-jobs` and are downloaded to `results/hosts/<job>/arrays/` (not in git).

## Checks, fixed in advance

1. **Integrity.** Every downloaded tar matches its recorded SHA-256, and every file in it matches its recorded hash.
2. **Reproduction of the first Jobs.** For every run that has a counterpart in the first Job on the same CPU model (same case, side, mode, seed), the stage hashes S1–S7 are compared. Identical hashes are reported as a bit-for-bit reproduction. Any difference is reported with its case and stage, not explained away.
3. **Independent re-evaluation.** The pinned local evaluator is run on the downloaded arrays of every completed test-split run. Its NDCG@10 per user must equal the metrics the Job produced.
4. **Outcomes.** The comparison outcome of every case (identical, changed, failure) and its verdicts are compared with the first Job's.

The paper then states what was reproduced and which cloud artifacts are retained. If a check fails, the paper keeps the present limitation text and reports the failure.

## Not done

Upstream test suites are still not run on the rented hosts; their status is reported from host A only.
