# silent-ranking

Code, protocols and manuscript sources for the study *Silent Ranking Changes Under Dependency Upgrades in MovieLens Recommender Pipelines* (Samrand M. Hassan, ORCID [0000-0001-6694-159X](https://orcid.org/0000-0001-6694-159X)).

The study upgrades NumPy, SciPy, pandas and PyTorch under fixed recommender pipelines and separates four outcomes: a changed checkpoint, a changed top-10 list, a contract violation and a loss of offline ranking quality. The manuscript is `paper/silent-ranking.pdf`.

## Requirements
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12
- Docker; every experiment run happens in the pinned Linux runner image (`docker/Dockerfile.runner`)

## Quick start
```
uv sync
uv run pytest
uv run silentrank init
uv run silentrank case list
```

Prepare MovieLens 1M (downloaded from GroupLens and checked against the pinned checksum), build and screen the environments, and run one case:
```
uv run --locked python data/prepare_ml1m.py
uv run --locked silentrank env screen-pilot all
uv run --locked silentrank run --case N1-P1-float32
uv run --locked silentrank compare --case N1-P1-float32
uv run --locked silentrank verdict --case N1-P1-float32
```

`run` tunes on baseline validation data, requires a bit-identical same-environment inference control, then runs A/B inference and paired-seed retraining. Results go to `results/` (not in git): a DuckDB ledger, run artifacts and per-user metrics.

Other commands:
```
uv run silentrank diagnose --case N1-P1-float32
uv run silentrank probes
uv run python -m silentrank.detect.synthetic_v2      # synthetic positive controls
uv run silentrank report pilot
```

The paper's tables, figures and numbers are generated from a results ledger by `paper/scripts/make_tables.py`, `make_figures.py` and `make_extras.py`. The manuscript builds with the TeX Live image (see the comment at the top of `paper/main.tex`).

## Layout
```
protocol/            frozen protocols (v0, v1), pipeline and host declarations, probe policy
protocol/cases/      case manifests: systematic transitions and issue-guided leads
protocol/reproducers/ minimal reproducers for the mechanisms
envs/<env_id>/       one locked environment under test per folder (pyproject.toml + uv.lock)
docker/              runner image (base pinned by digest) and cloud job scripts
data/                dataset download and preparation scripts, checksum manifests
src/silentrank/
  orchestrator/      environment builder, container runner, ledger, comparisons
  worker/            code that runs inside an environment under test
  evaluator/         metrics, canonical ranking, contracts
  stats/             paired intervals and verdicts
  detect/            probe panels and synthetic controls
  cli.py             `silentrank` command
tests/               pytest suite
paper/               manuscript sources, generated tables and figures, generator scripts
```

## Rules
The code and protocols follow these rules; docstrings refer to them by number.

**Evidence and claims**
1. Report only results produced by logged runs.
2. Record failures (resolve, build, import, crash, timeout) in the ledger. Never skip silently, never delete or overwrite a run; re-runs get a new `repeat` index.
3. Keep the outcome categories separate: install or runtime failure, behavioural change, quality regression, contract violation, uncertain.
4. "Silent" means execution completed and no configured failure signal fired. Always state the library's own test-suite status (passed, failed, unavailable, not runnable).
5. A different ranking is not a worse ranking. Harm needs independent relevance or contract evidence, with uncertainty.
6. Real cases, unresolved effects and synthetic faults are reported in separate tables. Systematic and issue-guided collections are never pooled.
7. The unit of independence is the upgrade case or family and the mechanism, not user rows.
8. Lack of statistical significance never produces a pass.

**Protocol**
9. Protocols are frozen by git tag before results are inspected. Changes after a freeze need a new version and a written reason.
10. Pilot transitions are development cases and are never reported as held-out evaluation.
11. Hyperparameters are tuned only in the baseline environment on validation data, then frozen.

**Engineering**
12. Experiment runs happen only inside the runner container, never in Windows-native Python.
13. `src/silentrank/worker/` runs inside environments under test: it may import only the standard library, NumPy, SciPy and the library under test, and exchanges `.npy`/`.npz` and JSON.
14. The evaluator and statistics run in the pinned orchestrator environment (root `pyproject.toml`), never inside an environment under test.
15. One host per comparison: runs with different `host_id` are never compared.
16. Ledger writers open, write and close quickly; DuckDB allows one writer or many readers.

**Operations**
17. Credentials and sign-ins (for example `hf auth login`) are entered by the researcher only.
18. Anything that spends money, such as a cloud job, needs the researcher's explicit go for that launch, with hardware, time limit and estimated cost.
19. Nothing is published or posted to upstream projects without the researcher's explicit go.
20. Datasets are not redistributed; only download and preparation scripts are.

## Data
MovieLens 1M and 20M are provided by GroupLens under their research-use terms and are not included. The preparation scripts download them and verify the published checksums.
