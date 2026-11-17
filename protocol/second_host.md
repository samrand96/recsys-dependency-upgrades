# Second CPU host: declaration

Declared on 14 November 2026, before any second-host run (PLAN.md WP3.3). The BLAS mechanisms found so far depend on which kernel the bundled OpenBLAS selects for the CPU. A second CPU type tests whether the findings hold, change or disappear on different hardware. Everything follows protocols v0 and v1 unchanged.

## Host

A Hugging Face Job, flavour `cpu-upgrade` (8 vCPU, 32 GB). The CPU model is not documented by Hugging Face. The worker's probe records it in every run, and the second host is identified by its own `host_id`. Every comparison stays within one host: a case is compared old vs new on host B, never against host A's runs.

The Job runs the same pinned base image as `docker/Dockerfile.runner` (`ghcr.io/astral-sh/uv@sha256:e5b65587…`), installs `g++` as the source-runner target does, and sets the same determinism environment variables. The harness runs in **local mode** (`SILENTRANK_RUNNER=local`): each worker call is a direct subprocess in the Job container instead of a fresh Docker container. The recorded runner image is `hfjob:<base digest>+g++`, so host-B runs can never be paired with host-A runs.

Code arrives as a git bundle of the current commit. Environment locks are the committed `uv.lock` files. MovieLens 1M is downloaded from GroupLens and prepared in the Job; preparation was checked to be byte-for-byte reproducible (same SHA-256 and manifest). Upstream test suites are not rerun on host B; their status is reported from host A. Results return through a private Hugging Face storage bucket and are kept in a separate ledger, `results/hosts/<host_id>/`.

## Cases (fixed)

| Purpose | Cases |
|---|---|
| NumPy 2.0 tie ordering | N1-P1-float32, N1-P1-float64, Q2-P1-float32 |
| OpenBLAS kernel class (ALS) | N1-P2-als, N2-P2-als, N3-P2-als, Q3-P2-als |
| OpenBLAS version (EASE) | N4-P6-ease, N5-P6-ease, N6-P6-ease |
| PyTorch | T1-P7-bpr … T6-P7-bpr |
| No-change control | S5-P1-float32 |

17 cases (the table above; an earlier draft of this sentence said 19 in error). Development cases (N1–N3) stay development evidence.

## Questions and reporting

For each case: is the host-B verdict the same as host A's, and does the same behavioural change (or no change) appear? Specifically, does the ALS kernel-class split (Cooperlake vs SkylakeX) and the NumPy 2.4.6 EASE difference reproduce on host B's CPU? Each answer is reported as same / different / not runnable, with both hosts' CPU models and BLAS kernels. Cross-host differences in absolute scores are expected and are not themselves findings.
