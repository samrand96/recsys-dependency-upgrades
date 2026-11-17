"""P3: Surprise SVD on explicit training ratings, ranked the way Surprise documents.

Declared in protocol/pipelines_p3_p4.md before any P3 ranking run:

* The trainset is built exactly like ``Dataset.construct_trainset``: inner ids are
  assigned in order of first appearance in the frozen training rows, ratings are
  all observed training ratings (1-5) as floats, and the rating scale is (1, 5).
* ``SVD(n_factors=<grid>, n_epochs=20, random_state=<seed>)`` with every other
  setting at the library default. Surprise does not accept initial factors, so the
  seed drives its legacy ``RandomState`` stream.
* Scores come from ``SVD.predict`` (clipped to the rating scale, as by default).
  Native top-N follows the Surprise FAQ ``get_top_n`` recipe: for each user, the
  unseen catalogue items in catalogue order are sorted by estimate with
  ``list.sort(reverse=True)``, so ties keep catalogue order.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..io import FrozenData
from ..ranking import save_rankings, seen_mask
from ..stages import Stages
from ..tracing import CallTrace

N_EPOCHS = 20
RATING_SCALE = (1, 5)
PARAMETERS = ("pu", "qi", "bu", "bi")


def _native_call(trace: CallTrace, operation: str, function, *args, **kwargs):
    """Count Python-visible NumPy and SciPy calls made directly by Surprise code."""
    if trace.dependency not in (None, "numpy", "scipy"):
        return trace.call(operation, function, *args, **kwargs)

    def observe(frame, event, argument):
        if event == "call":
            caller, module, name = frame.f_back, frame.f_globals.get("__name__", ""), frame.f_code.co_name
        elif event == "c_call":
            caller = frame
            module = getattr(argument, "__module__", None) or type(getattr(argument, "__self__", None)).__module__
            name = getattr(argument, "__name__", "")
        else:
            return
        package = (module or "").split(".", 1)[0]
        if (package in ("numpy", "scipy") and trace.dependency in (None, package) and caller is not None
                and caller.f_globals.get("__name__", "").startswith("surprise")):
            trace.counts[f"{module}.{name}"] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        return trace.call(operation, function, *args, **kwargs)
    finally:
        sys.setprofile(previous)


def build_trainset(data: FrozenData):
    """Mirror Dataset.construct_trainset on the frozen rows, keeping raw ids as catalogue indices."""
    from surprise import Trainset

    raw2inner_users: dict[int, int] = {}
    raw2inner_items: dict[int, int] = {}
    ur: defaultdict[int, list] = defaultdict(list)
    ir: defaultdict[int, list] = defaultdict(list)
    for raw_user, raw_item, rating in data.train[:, :3].tolist():
        user = raw2inner_users.setdefault(raw_user, len(raw2inner_users))
        item = raw2inner_items.setdefault(raw_item, len(raw2inner_items))
        ur[user].append((item, float(rating)))
        ir[item].append((user, float(rating)))
    return Trainset(ur, ir, len(ur), len(ir), len(data.train), RATING_SCALE, raw2inner_users, raw2inner_items)


def _check_coverage(trainset, data: FrozenData) -> None:
    cold_users = [int(u) for u in data.evaluation_users if int(u) not in trainset._raw2inner_id_users]
    cold_items = [int(i) for i in range(data.item_ids.size) if i not in trainset._raw2inner_id_items]
    if cold_users or cold_items:
        raise ValueError(f"P3 refuses cold users ({len(cold_users)}) or items ({len(cold_items)}); "
                         "the declared splits contain none")


def _load(path: Path, trainset, factors: int, input_sha256: str):
    from surprise import SVD
    from surprise.prediction_algorithms.algo_base import AlgoBase

    with np.load(path, allow_pickle=False) as arrays:
        if (arrays["training_input_sha256"].item() != input_sha256 or arrays["n_factors"].item() != factors
                or arrays["n_epochs"].item() != N_EPOCHS):
            raise ValueError("portable SVD model does not match the requested configuration")
        weights = {name: arrays[name] for name in PARAMETERS}
    expected = {"pu": (trainset.n_users, factors), "qi": (trainset.n_items, factors),
                "bu": (trainset.n_users,), "bi": (trainset.n_items,)}
    for name, shape in expected.items():
        if weights[name].shape != shape or weights[name].dtype != np.float64 or not np.isfinite(weights[name]).all():
            raise ValueError(f"portable SVD parameter {name} has an invalid shape, dtype or value")
    model = SVD(n_factors=factors, n_epochs=N_EPOCHS)
    AlgoBase.fit(model, trainset)  # attach the trainset without running SGD
    for name, value in weights.items():
        setattr(model, name, value)
    return model


def _rank(model, data: FrozenData, k: int, trace: CallTrace):
    mask = seen_mask(data)
    n_items = data.item_ids.size
    scores = np.empty(mask.shape, dtype=np.float64)
    cutoffs = (10, 20) if k == 20 else (k,)
    native = {n: (np.full((len(mask), n), -1, dtype=np.int64), np.full((len(mask), n), np.nan, dtype=np.float64))
              for n in cutoffs}

    def score_all():
        for row, user in enumerate(data.evaluation_users.tolist()):
            for item in range(n_items):
                scores[row, item] = model.predict(user, item).est

    _native_call(trace, "surprise.SVD.predict", score_all)
    for row in range(len(mask)):
        candidates = [(item, scores[row, item]) for item in np.flatnonzero(~mask[row]).tolist()]
        candidates.sort(key=lambda pair: pair[1], reverse=True)  # Surprise FAQ get_top_n
        for n, (top_items, top_scores) in native.items():
            chosen = candidates[:n]
            top_items[row, :len(chosen)] = data.item_ids[np.asarray([item for item, _ in chosen], dtype=np.int64)]
            top_scores[row, :len(chosen)] = [value for _, value in chosen]
    return scores, native


def run(data: FrozenData, stages: Stages, trace: CallTrace, *, factors: int, seed: int, k: int,
        input_sha256: str, model_in: Path | None) -> None:
    from surprise import SVD

    stages.bundle("S1", {"user_ids": data.user_ids, "item_ids": data.item_ids}, dtype="int64")
    trainset = build_trainset(data)
    _check_coverage(trainset, data)
    inner_users = np.asarray(list(trainset._raw2inner_id_users), dtype=np.int64)
    inner_items = np.asarray(list(trainset._raw2inner_id_items), dtype=np.int64)
    ratings = np.asarray([r for _, _, r in trainset.all_ratings()], dtype=np.float64)
    stages.bundle("S2", {"inner_user_order": inner_users, "inner_item_order": inner_items, "ratings": ratings},
                  shape=(trainset.n_users, trainset.n_items), dtype="trainset:float64")

    if model_in is None:
        model = SVD(n_factors=factors, n_epochs=N_EPOCHS, random_state=seed)
        _native_call(trace, "surprise.SVD.fit", model.fit, trainset)
    else:
        model = _load(model_in, trainset, factors, input_sha256)
    arrays = {name: np.asarray(getattr(model, name), dtype=np.float64) for name in PARAMETERS}
    metadata = {"training_input_sha256": np.asarray(input_sha256), "n_factors": np.asarray(factors, dtype=np.int64),
                "n_epochs": np.asarray(N_EPOCHS, dtype=np.int64), "seed": np.asarray(seed, dtype=np.int64)}
    np.savez(stages.output / "model.npz", **arrays, **metadata)
    stages.record("S3", {**arrays, **{key: value for key, value in metadata.items() if key != "seed"}},
                  "model.npz", dtype="float64")
    scores, native = _rank(model, data, k, trace)
    save_rankings(stages.output, stages, data, scores, k, trace, native=native)
