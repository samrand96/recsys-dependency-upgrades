"""P7: BPR matrix factorisation written against the PyTorch API (protocol/pipelines_p7_torch.md).

Initial embeddings, per-epoch minibatch order and negatives come from a frozen input file
generated once in the orchestrator, so both sides of a transition see identical inputs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from ..io import FrozenData
from ..ranking import save_rankings, seen_mask
from ..stages import Stages
from ..tracing import CallTrace

EPOCHS = 10
BATCH = 4096
LEARNING_RATE = 0.01
L2 = 1e-4


def _native_call(trace: CallTrace, operation: str, function, *args, **kwargs):
    """Count Python-visible calls into torch made by this pipeline while it runs."""
    if trace.dependency not in (None, "torch"):
        return trace.call(operation, function, *args, **kwargs)

    def observe(frame, event, argument):
        if event == "c_call":
            module = getattr(argument, "__module__", None) or type(getattr(argument, "__self__", None)).__module__ or ""
            if module.split(".", 1)[0] == "torch":
                trace.counts[f"{module}.{getattr(argument, '__name__', '')}"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        return trace.call(operation, function, *args, **kwargs)
    finally:
        sys.setprofile(previous)


def _inputs(path: Path, n_users: int, n_items: int, dim: int, seed: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        inputs = {name: arrays[name] for name in arrays.files}
    if inputs["seed"].item() != seed or inputs["dim"].item() != dim:
        raise ValueError("frozen P7 inputs do not match the requested seed or dimension")
    if inputs["user_init"].shape != (n_users, dim) or inputs["item_init"].shape != (n_items, dim):
        raise ValueError("frozen P7 embeddings do not match the frozen catalogue")
    if inputs["order"].shape != inputs["negatives"].shape or inputs["order"].shape[0] != EPOCHS:
        raise ValueError("frozen P7 schedule has the wrong number of epochs")
    return inputs


def _train(torch, positives: np.ndarray, inputs: dict, seed: int, device: str = "cpu"):
    torch.manual_seed(seed)
    users_emb = torch.nn.Parameter(torch.from_numpy(inputs["user_init"].astype(np.float32)).to(device))
    items_emb = torch.nn.Parameter(torch.from_numpy(inputs["item_init"].astype(np.float32)).to(device))
    optimiser = torch.optim.Adam([users_emb, items_emb], lr=LEARNING_RATE)
    pos_users = torch.from_numpy(positives[:, 0].astype(np.int64)).to(device)
    pos_items = torch.from_numpy(positives[:, 1].astype(np.int64)).to(device)
    for epoch in range(EPOCHS):
        order = torch.from_numpy(inputs["order"][epoch].astype(np.int64)).to(device)
        negatives = torch.from_numpy(inputs["negatives"][epoch].astype(np.int64)).to(device)
        for start in range(0, len(order), BATCH):
            index = order[start:start + BATCH]
            u = users_emb[pos_users[index]]
            i = items_emb[pos_items[index]]
            j = items_emb[negatives[index]]
            difference = (u * i).sum(dim=1) - (u * j).sum(dim=1)
            penalty = L2 * ((u * u).sum(dim=1) + (i * i).sum(dim=1) + (j * j).sum(dim=1)).mean()
            loss = -torch.nn.functional.logsigmoid(difference).mean() + penalty
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
    return users_emb.detach(), items_emb.detach()


def _rank(torch, users_emb, items_emb, data: FrozenData, k: int):
    mask = seen_mask(data)
    device = users_emb.device
    with torch.no_grad():
        scores = users_emb[torch.from_numpy(data.evaluation_users.astype(np.int64)).to(device)] @ items_emb.T
        masked = scores.clone()
        masked[torch.from_numpy(mask).to(device)] = float("-inf")
        native = {}
        for n in ((10, 20) if k == 20 else (k,)):
            values, indices = torch.topk(masked, n, dim=1, largest=True, sorted=True)
            values, indices = values.cpu().numpy(), indices.cpu().numpy().astype(np.int64)
            valid = np.isfinite(values)
            items = np.full(indices.shape, -1, dtype=np.int64)
            items[valid] = data.item_ids[indices[valid]]
            native[n] = (items, np.where(valid, values, np.nan).astype(np.float32))
    return scores.cpu().numpy().astype(np.float32), native


def _device_report(torch, device: str) -> dict:
    report = {"device": device, "torch": torch.__version__, "cuda_build": torch.version.cuda}
    if device.startswith("cuda"):
        report.update(name=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)),
                      driver=getattr(torch.cuda, "driver_version", lambda: None)() if hasattr(torch.cuda, "driver_version") else None,
                      cudnn=torch.backends.cudnn.version())
    return report


def run(data: FrozenData, stages: Stages, trace: CallTrace, *, dim: int, seed: int, k: int,
        input_sha256: str, model_in: Path | None, initial_model: Path | None, device: str = "cpu") -> None:
    import json

    import torch

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("the GPU condition was requested but CUDA is not available")
    (stages.output / "device.json").write_text(json.dumps(_device_report(torch, device), indent=2) + "\n", encoding="utf-8")
    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    positives = data.train[data.train[:, 2] >= 4][:, :2].astype(np.int64)
    stages.bundle("S2", {"positives": positives}, shape=positives.shape, dtype="int64")
    n_users, n_items = data.user_ids.size, data.item_ids.size
    if model_in is None:
        if initial_model is None:
            raise ValueError("P7 retraining requires frozen inputs")
        inputs = _inputs(initial_model, n_users, n_items, dim, seed)
        if inputs["order"].shape[1] != len(positives):
            raise ValueError("frozen P7 schedule does not cover the training positives")
        users_emb, items_emb = _native_call(trace, "torch.bpr.train", _train, torch, positives, inputs, seed, device)
    else:
        with np.load(model_in, allow_pickle=False) as arrays:
            if arrays["training_input_sha256"].item() != input_sha256 or arrays["dim"].item() != dim:
                raise ValueError("portable P7 model does not match the requested configuration")
            users_emb = torch.from_numpy(arrays["user_embeddings"]).to(device)
            items_emb = torch.from_numpy(arrays["item_embeddings"]).to(device)
    arrays = {"user_embeddings": users_emb.cpu().numpy().astype(np.float32),
              "item_embeddings": items_emb.cpu().numpy().astype(np.float32),
              "training_input_sha256": np.asarray(input_sha256), "dim": np.asarray(dim, dtype=np.int64)}
    np.savez(stages.output / "model.npz", **arrays)
    stages.record("S3", arrays, "model.npz", dtype="float32")
    scores, native = _native_call(trace, "torch.bpr.rank", _rank, torch, users_emb, items_emb, data, k)
    save_rankings(stages.output, stages, data, scores, k, trace, native=native)
