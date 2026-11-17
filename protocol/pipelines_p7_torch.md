# P7 PyTorch BPR pipeline declaration

Declared on 14 November 2026, before any P7 ranking run, as [PROTOCOL_V1.md](PROTOCOL_V1.md) requires for new pipelines. Everything else follows protocols v0 and v1 unchanged.

## Why a reference model, not RecBole

The PyTorch transitions T1–T6 (2.9.1 → 2.15.0) need a pipeline whose only moving dependency is PyTorch. RecBole 1.2.1 crashes on NumPy 2 (`np.bool8`, issue-guided lead L17), and the held-fixed NumPy below is 2.2.6. P7 is therefore a small, transparent BPR matrix factorisation written directly against the PyTorch API, like the P1 reference item-kNN. P5 stays reserved for RecBole.

## Environments

Group `torch`, for the PyTorch transitions T1–T6 only. Each side pins `numpy==2.2.6`, `scipy==1.16.3`, `pandas==2.2.3`, `threadpoolctl==3.8.0` and the side's `torch` version. PyTorch comes from the CPU-only index `https://download.pytorch.org/whl/cpu`, with the upload cutoff `exclude-newer = 2026-11-13T00:00:00Z`. These are the "held at versions the resolver accepts" pins that `cases/systematic_single_heldout.yaml` leaves to screening. T1–T6 are held-out cases.

**Conditions.** This declaration covers the **CPU condition** on the local host. A later GPU condition uses the same model and inputs, is declared separately, and is never pooled with CPU results.

## Model

- **Data:** training events with rating ≥ 4 as positive (user, item) pairs, as the protocol requires for implicit-feedback adapters.
- **Parameters:** user and item embeddings of dimension d, float32, no biases. Score = dot product.
- **Loss:** BPR, −mean log σ(s(u, i⁺) − s(u, i⁻)), plus L2 penalty 1e-4 × (‖u‖² + ‖i⁺‖² + ‖i⁻‖²) averaged over the batch.
- **Optimiser:** `torch.optim.Adam(lr=0.01)`. 10 epochs, batch size 4,096 positive pairs.
- **Grid:** d ∈ (32, 64), in that order; chosen on baseline validation NDCG@10.
- **Determinism:** `torch.manual_seed(seed)`, `torch.use_deterministic_algorithms(True)`, `torch.set_num_threads(1)`.

## Frozen inputs (generated once in the orchestrator, checksummed, loaded on both sides)

For each seed in 17, 29, 43, 59, 71 and each d, generated with `numpy.random.Generator(PCG64(seed))`:
- initial user and item embeddings, N(0, 0.1²), float32;
- for each epoch, a permutation of the positive pairs;
- for each epoch and positive pair, one negative item drawn uniformly from the catalogue, redrawn until it is not one of that user's training positives.

So both sides see identical initial weights, minibatch order and negatives. Only PyTorch's arithmetic and ranking can differ.

## Scoring and ranking

- **Scores:** `user_embeddings @ item_embeddings.T` in PyTorch (float32), over the full catalogue.
- **Native top-N:** seen items are set to −inf in a torch tensor, then `torch.topk(scores, N, sorted=True)` for N = 10 and 20. This is PyTorch's own selection routine, so a change in its tie handling would show up here.
- **Inference mode:** the old side's embeddings are loaded and only scoring and ranking run.

## Cases

`T1-P7-bpr` to `T6-P7-bpr`.
