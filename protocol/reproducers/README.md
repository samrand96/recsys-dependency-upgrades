# Synthetic NumPy top-K tie reproducer

The example uses 64 generated item IDs with identical `float32` scores. It runs the same NumPy partition and sort operations as the pilot's reference top-K stage. The JSON output shows the native item IDs and scores beside the canonical item IDs. This synthetic example contains no MovieLens data and provides no independent evidence about recommendation quality.

From the repository root, with the two N1 environments already built by `silentrank env screen`, run in PowerShell:

```powershell
$script = Join-Path (Get-Location) 'protocol/reproducers/numpy_unstable_topk.py'
foreach ($version in '1.26.4', '2.0.2') {
    $envId = "py312-np$version-sp1.13.1-pd2.2.3"
    docker run --rm -v "${script}:/repro.py:ro" -v silentrank-venvs:/venvs:ro silentrank-runner:py312 "/venvs/$envId/bin/python" /repro.py
}
```

The native ordering can vary with the NumPy version and hardware. All ten returned scores should be `1.0`, and the canonical IDs should be `0` through `9` in order.

## NumPy 2.0 `log1p` and implicit BM25 (`numpy_log1p_bm25_*.py`)

Run inside the runner image with the N1 implicit environments' Python, the prepared `ml1m.npz` mounted at `/d` and a scratch folder at `/w`.
- `numpy_log1p_bm25_steps.py` hashes each intermediate of `implicit.nearest_neighbours.bm25_weight`. Only `log1p` (and what follows from it) differs between NumPy 1.26.4 and 2.0.2: 1,073 of 5,186 values, by at most 8.9e-16.
- `numpy_log1p_bm25_swap.py` runs NumPy 2.0.2 with the 1.26.4 `log1p` values substituted and reproduces the 1.26.4 BM25 weights exactly (hash `e8d93a9d82ee`). Switching NumPy 2.0.2 back to legacy type promotion does not change the result, so NEP 50 is not the cause.
