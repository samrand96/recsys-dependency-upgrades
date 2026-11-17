# P7 PyTorch BPR, GPU condition: declaration

Declared on 14 November 2026, before any GPU run. It reuses the P7 model, grid, frozen inputs, ranking and verdict rules of [pipelines_p7_torch.md](pipelines_p7_torch.md) unchanged. Only the device and the PyTorch build differ. GPU and CPU results are never pooled.

## Cases and environments

Case IDs `T1-P7-gpu` … `T6-P7-gpu`, on the same PyTorch transitions as the CPU condition.

No single CUDA build line of PyTorch covers 2.9.1 → 2.15.0 (checked 14 November on `download.pytorch.org`): `cu126` has 2.9.1–2.14.1 and `cu132` has 2.13.0–2.15.0. The CUDA line is therefore held fixed **within** each transition:
- T1–T5 use `cu126` on both sides;
- T6 (2.14.1 → 2.15.0) uses `cu132` on both sides.

Environment ids are the CPU ids with a `-cu126` or `-cu132` suffix. Other pins are unchanged (NumPy 2.2.6, SciPy 1.16.3, pandas 2.2.3), with the same upload cutoff.

## Device and determinism

- Training and scoring run on `cuda:0`. Frozen inputs are loaded on the CPU and moved to the device. Rankings and scores are copied back to the CPU for the evaluator.
- `torch.use_deterministic_algorithms(True)`, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` set as PyTorch requires for deterministic cuBLAS. A run that fails because an operation has no deterministic GPU implementation is recorded as a runtime failure, not retried non-deterministically.
- Each run records the GPU name, CUDA runtime and driver in `device.json`.

## Host

One Hugging Face Job on a single GPU flavour, chosen at launch and named in the launch approval. Both sides of every transition run in the same Job, so on the same GPU. If a CUDA line cannot initialise on that host's driver, its cases are recorded as not runnable.
