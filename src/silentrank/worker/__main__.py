"""Execute one pipeline inside the dependency environment under study."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .io import load_data, sha256_file, write_json
from .stages import Stages
from .tracing import CallTrace


def _peak_rss_mb() -> float | None:
    if sys.platform != "linux":
        return None
    from resource import RUSAGE_SELF, getrusage

    return getrusage(RUSAGE_SELF).ru_maxrss / 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True,
                        choices=("ref_itemknn", "prep_pandas", "implicit_als", "implicit_cosine", "implicit_bm25",
                                 "surprise_svd", "sar", "cornac_ease", "cornac_knn", "torch_bpr"))
    parser.add_argument("--mode", required=True, choices=("inference", "retrain", "prep"))
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--data-sha256")
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--model-in", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--initial-model", type=Path)
    parser.add_argument("--initial-model-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--factors", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=20.0)
    parser.add_argument("--lamb", type=float, default=500.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--similarity", choices=("jaccard", "cosine", "lift"), default="jaccard")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--trace", choices=("numpy", "scipy", "pandas", "implicit", "torch"))
    args = parser.parse_args(argv)

    if args.k < 1 or args.neighbors < 1:
        parser.error("--k and --neighbors must be positive")
    if (args.pipeline == "prep_pandas") != (args.mode == "prep"):
        parser.error("prep_pandas requires prep mode, and prep mode requires prep_pandas")
    if args.mode == "prep" and args.raw is None:
        parser.error("prep mode requires --raw")
    if (args.mode == "inference") != (args.model_in is not None):
        parser.error("inference mode requires --model-in and only inference mode accepts it")
    if args.model_sha256 is not None and args.model_in is None:
        parser.error("--model-sha256 requires --model-in")
    als_training = args.pipeline in ("implicit_als", "torch_bpr") and args.mode == "retrain"
    if als_training != (args.initial_model is not None):
        parser.error("--initial-model is required for and applies only to ALS and P7 retraining")
    if args.initial_model_sha256 is not None and args.initial_model is None:
        parser.error("--initial-model-sha256 requires --initial-model")
    if min(args.factors, args.iterations, args.regularization, args.alpha) <= 0 or not np.isfinite(
        [args.regularization, args.alpha]
    ).all():
        parser.error("ALS hyperparameters must be finite and positive")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "stages").exists() or (output / "worker_meta.json").exists():
        raise FileExistsError(f"refusing to overwrite a worker run in {output}")
    stages = Stages(output)
    trace = CallTrace(args.trace)
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    model_input_hash = model_output_hash = initial_model_hash = None
    if args.mode == "prep":
        from .pipelines import prep_pandas

        manifest = json.loads((args.manifest or args.data.with_name("manifest.json")).read_text(encoding="utf-8"))
        input_sha256 = sha256_file(args.raw, manifest["source"]["sha256"])
        stages.input(input_sha256)
        prep_pandas.run(args.raw, manifest["selected_split"], stages, trace)
    else:
        input_sha256 = sha256_file(args.data, args.data_sha256)
        if args.model_in is not None:
            model_input_hash = sha256_file(args.model_in, args.model_sha256)
        if args.initial_model is not None:
            initial_model_hash = sha256_file(args.initial_model, args.initial_model_sha256)
        stages.input(input_sha256)
        data = load_data(args.data, args.split)
        settings = {"dtype": np.dtype(args.dtype), "neighbors": args.neighbors, "k": args.k,
                    "input_sha256": input_sha256, "model_in": args.model_in}
        if args.pipeline == "ref_itemknn":
            from .pipelines import ref_itemknn

            ref_itemknn.run(data, stages, trace, **settings)
        elif args.pipeline == "surprise_svd":
            from .pipelines import surprise_svd

            surprise_svd.run(data, stages, trace, factors=args.factors, seed=args.seed, k=args.k,
                             input_sha256=input_sha256, model_in=args.model_in)
        elif args.pipeline == "torch_bpr":
            from .pipelines import torch_bpr

            torch_bpr.run(data, stages, trace, dim=args.factors, seed=args.seed, k=args.k, input_sha256=input_sha256,
                          model_in=args.model_in, initial_model=args.initial_model, device=args.device)
        elif args.pipeline in ("cornac_ease", "cornac_knn"):
            from .pipelines import cornac_models

            cornac_models.run(data, stages, trace, pipeline=args.pipeline, lamb=args.lamb, neighbors=args.neighbors,
                              k=args.k, input_sha256=input_sha256, model_in=args.model_in)
        elif args.pipeline == "sar":
            from .pipelines import sar

            sar.run(data, stages, trace, similarity=args.similarity, k=args.k,
                    input_sha256=input_sha256, model_in=args.model_in)
        else:
            from .pipelines import implicit_models

            implicit_models.run(data, stages, trace, **settings, pipeline=args.pipeline, initial_model=args.initial_model,
                                factors=args.factors, iterations=args.iterations, regularization=args.regularization,
                                alpha=args.alpha, seed=args.seed)
        model_output_hash = sha256_file(output / "model.npz")
    stages.save()
    write_json(output / "trace.json", dict(sorted(trace.counts.items())))
    write_json(output / "worker_meta.json", {
        "pipeline": args.pipeline, "mode": args.mode, "split": args.split,
        "seed": args.seed, "dtype": args.dtype, "k": args.k, "neighbors": args.neighbors,
        "input_sha256": input_sha256, "model_input_sha256": model_input_hash, "model_output_sha256": model_output_hash,
        "initial_model_sha256": initial_model_hash, "factors": args.factors, "iterations": args.iterations,
        "regularization": args.regularization, "alpha": args.alpha, "similarity": args.similarity, "lamb": args.lamb,
        "wall_s": time.perf_counter() - start_wall, "cpu_s": time.process_time() - start_cpu,
        "peak_rss_mb": _peak_rss_mb(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
