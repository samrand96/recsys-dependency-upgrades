"""Replay only the reference native top-K stage with its upstream inputs frozen."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .io import sha256_file, write_json
from .ranking import _reference_ranking
from .tracing import CallTrace


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scores-sha256")
    parser.add_argument("--seen-mask-sha256")
    parser.add_argument("--item-ids-sha256")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k must be positive")

    cutoffs = {10: "topk_native10.npy", 20: "topk_native.npy"} if args.k == 20 else {args.k: "topk_native.npy"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in (*cutoffs.values(), "replay_meta.json"):
        if (args.output_dir / name).exists():
            raise FileExistsError(f"refusing to overwrite replay artifact {args.output_dir / name}")

    expected = {"scores.npy": args.scores_sha256, "seen_mask.npy": args.seen_mask_sha256,
                "item_ids.npy": args.item_ids_sha256}
    input_hashes = {name: sha256_file(args.input_dir / name, digest) for name, digest in expected.items()}
    scores, mask, item_ids = (
        np.load(args.input_dir / name, mmap_mode="r", allow_pickle=False) for name in expected
    )
    if scores.ndim != 2 or scores.dtype not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("scores.npy must contain a matrix of float32 or float64 scores")
    if mask.shape != scores.shape or mask.dtype != np.bool_:
        raise ValueError("seen_mask.npy must be boolean and match the score matrix")
    if item_ids.shape != (scores.shape[1],) or item_ids.dtype != np.int64:
        raise ValueError("item_ids.npy must contain one int64 external ID per score column")

    trace = CallTrace("numpy")
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    rankings = {name: _reference_ranking(scores, mask, item_ids, n, trace)[0] for n, name in cutoffs.items()}
    wall_s, cpu_s = time.perf_counter() - start_wall, time.process_time() - start_cpu
    output_hashes = {}
    for name, ranking in rankings.items():
        path = args.output_dir / name
        np.save(path, ranking, allow_pickle=False)
        output_hashes[name] = sha256_file(path)
    write_json(args.output_dir / "replay_meta.json", {
        "stage": "S7", "k": args.k, "input_sha256": input_hashes, "output_sha256": output_hashes,
        "wall_s": wall_s, "cpu_s": cpu_s, "trace": dict(sorted(trace.counts.items())),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
