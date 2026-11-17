# Third CPU host (Intel) and the log1p check on host B: declaration

Declared on 14 November 2026, before any run on these hosts. It extends [second_host.md](second_host.md) under the same rules: local-mode harness in the pinned base image, frozen data reproduced byte for byte, one host per comparison, separate ledgers, no pooling.

## Job C: Intel CPU

- **Flavour:** `t4-small`, used for its CPU only; no case uses the GPU. Hugging Face does not document host CPUs. The `t4-medium` GPU Job ran on an Intel Xeon Platinum 8259CL (Cascade Lake: AVX-512 without AVX512_BF16), and the T4 flavours are expected to share that host type. The CPU actually used is recorded by the probe. If it turns out not to be Intel, that is reported, not hidden.
- **Cases (19):** the 17 cases of `second_host.md`, plus N1-P2-bm25 and Q2-P2-bm25 for mechanism M4 (NumPy 2.0 `log1p`).
- **Question:** which OpenBLAS kernel each NumPy wheel selects on this CPU, and whether M1 (ties), M2 (ALS kernel class), M3 (EASE) and M4 (`log1p`) appear.

## Job D: M4 on host B's flavour

- **Flavour:** `cpu-upgrade`, the flavour that gave the AMD EPYC 7R13 host. The CPU actually obtained is recorded and may differ.
- **Cases (2):** N1-P2-bm25, Q2-P2-bm25.
- **Question:** whether the `log1p` last-bit change also appears on an AVX2 host.

## Reporting

Same table format as [SECOND_HOST_RESULTS.md](SECOND_HOST_RESULTS.md), with each host's CPU model and kernels.
