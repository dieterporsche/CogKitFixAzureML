import subprocess
import yaml
import logging
import time
from pathlib import Path
from typing import Dict
import torch
import os

# We want the repository root so metrics and output paths resolve correctly.
# __file__ = quickstart/scripts/i2v/hyperparam_search.py
# parents[0] -> quickstart/scripts/i2v
# parents[1] -> quickstart/scripts
# parents[2] -> quickstart
# parents[3] -> repo root
ROOT = Path(__file__).resolve().parents[3]
CONFIG_TEMPLATE = Path(__file__).parent / "config.yaml"
TRAIN_SCRIPT = Path(__file__).parent.parent / "train.py"
METRICS_SCRIPT = ROOT / "metrics" / "compute_metrics.py"
OUTPUT_ROOT = ROOT / "output"

OUTPUT_ROOT.mkdir(exist_ok=True)
LOG_FILE = OUTPUT_ROOT / "iterative_training.log"
GPU_COUNT = torch.cuda.device_count()

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("iterative_training")
logger.addHandler(logging.StreamHandler())
logger.info("GPUs available: %s", GPU_COUNT)


def _count_training_data(data_root: Path) -> int:
    meta = data_root / "train" / "metadata.jsonl"
    if meta.exists():
        with open(meta, "r") as f:
            return sum(1 for _ in f)
    return 0


def _write_config(lr: float, batch_size: int, epochs: int) -> Path:
    with open(CONFIG_TEMPLATE, "r") as f:
        cfg = yaml.safe_load(f)
    # ------------------------------------------------------------------
    # Resolve paths from environment variables.  When running on Azure
    # the ``MODEL_PATH``, ``DATA_ROOT`` and ``OUTPUT_DIR`` variables are
    # provided via the job definition.  Without replacing them here the
    # configuration would contain the literal strings ``$MODEL_PATH`` etc.
    # which leads to failures when the trainer tries to load the model.
    # ------------------------------------------------------------------
    cfg["model_path"] = os.environ["MODEL_PATH"]
    cfg["data_root"] = os.environ["DATA_ROOT"]
    cfg["output_dir"] = os.environ["OUTPUT_DIR"]
    cfg["learning_rate"] = lr
    cfg["batch_size"] = batch_size
    cfg["train_epochs"] = epochs
    #tmp = CONFIG_TEMPLATE.parent / f"tmp_{lr}_{batch_size}.yaml"
    tmp = CONFIG_TEMPLATE.parent / f"tmp_{lr}_{batch_size}_{epochs}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    logger.info(
        "Created config %s (lr=%s, batch_size=%s, epochs=%s)",
        tmp.name,
        lr,
        batch_size,
        epochs,
    )
    data_count = _count_training_data(Path(cfg.get("data_root", ".")))
    logger.info("Training data count: %s", data_count)
    return tmp


def _run_training(config: Path) -> float:
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)
    nproc = max(GPU_COUNT, 1)
    cmd = [
        "torchrun",
        f"--nproc_per_node={nproc}",
        "--master_port=29501",
        TRAIN_SCRIPT.as_posix(),
        "--yaml",
        str(config),
    ]
    logger.info(
        "Start training %s with nproc_per_node=%s", config.name, nproc
    )
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    duration = time.perf_counter() - start
    logger.info(
        "Finished training %s in %.2fs", config.name, duration
    )
    return duration


def _run_metrics() -> None:
    logger.info("Computing metrics...")
    start = time.perf_counter()
    subprocess.run(["python", METRICS_SCRIPT.as_posix()], check=True)
    duration = time.perf_counter() - start
    logger.info("Metrics computed in %.2fs", duration)
    for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
        metrics = _parse_metrics_file(p)
        for epoch, mse in metrics.items():
                logger.info("%s %s -> MSE=%.6f", p.name, epoch, mse)

#         for epoch, vals in metrics.items():
#             logger.info(
#                 "%s %s -> MSE=%.6f SSIM=%.6f INTR=%.6f",
#                 p.name,
#                 epoch,
#                 vals[0],
#                 vals[1],
#                 vals[2],
#             )

# def _parse_metrics_file(path: Path) -> Dict[str, Tuple[float, float, float]]:
#     metrics: Dict[str, Tuple[float, float, float]] = {}

def _parse_metrics_file(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    epoch = None
    with open(path, "r") as f:
        lines = [line.strip() for line in f]
    try:
        idx = lines.index("#### Average ALL: ####") + 1
    except ValueError:
        return metrics
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("Epoch_"):
            epoch = line.rstrip(":")
            idx += 1
            if idx < len(lines) and lines[idx].lower() == "mse":
                idx += 1
            if idx < len(lines):
                try:
                    metrics[epoch] = float(lines[idx].split()[0])
                except ValueError:
                    pass
            idx += 1
            # if idx + 2 < len(lines):
            #     vals = lines[idx + 2].split()
            #     if len(vals) >= 3:
            #         metrics[epoch] = (
            #             float(vals[0]),
            #             float(vals[1]),
            #             float(vals[2]),
            #         )
            # idx += 3
        else:
            idx += 1
    return metrics


# def _select_best(metrics: Dict[str, Tuple[float, float, float]]) -> Tuple[str, Tuple[float, float, float]]:
#     if not metrics:
#         return "", (float("inf"), float("inf"), float("inf"))
#     mins = [
#         min(v[i] for v in metrics.values())
#         for i in range(3)
#     ]
#     best_epoch = None
#     best_vals = None
#     best_count = -1
#     best_sum = float("inf")
#     for epoch, vals in metrics.items():
#         count = sum(vals[i] == mins[i] for i in range(3))
#         total = sum(vals)
#         if count > best_count or (count == best_count and total < best_sum):
#             best_epoch = epoch
#             best_vals = vals
#             best_count = count
#             best_sum = total
#     return best_epoch or "", best_vals or (float("inf"), float("inf"), float("inf"))


# def _extract_lr_bs(name: str) -> Tuple[float, int]:
#     m = re.search(r"LR_([\d.eE-]+)__BS_(\d+)", name)
#     if not m:
#         return 0.0, 0
#     return float(m.group(1)), int(m.group(2))


# def _choose_best_lr() -> float:
#     results = {}
#     for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
#         lr, bs = _extract_lr_bs(p.stem)
#         if bs != 4:
#             continue
#         metrics = _parse_metrics_file(p)
#         _, vals = _select_best(metrics)
#         results[lr] = vals
#     if not results:
#         return 0.0
#     mins = [min(v[i] for v in results.values()) for i in range(3)]
#     best_lr = None
#     best_count = -1
#     best_sum = float("inf")
#     for lr, vals in results.items():
#         count = sum(vals[i] == mins[i] for i in range(3))
#         total = sum(vals)
#         if count > best_count or (count == best_count and total < best_sum):
#             best_lr = lr
#             best_count = count
#             best_sum = total
#     return best_lr or 0.0


# def _choose_best_overall() -> Tuple[float, int, str]:
#     results = {}
#     epochs: Dict[Tuple[float, int], str] = {}
#     for p in OUTPUT_ROOT.glob("metrics_Output*.txt"):
#         lr, bs = _extract_lr_bs(p.stem)
#         metrics = _parse_metrics_file(p)
#         epoch, vals = _select_best(metrics)
#         results[(lr, bs)] = vals
#         epochs[(lr, bs)] = epoch
#     if not results:
#         return 0.0, 0, ""
#     mins = [min(v[i] for v in results.values()) for i in range(3)]
#     best = None
#     best_count = -1
#     best_sum = float("inf")
#     for key, vals in results.items():
#         count = sum(vals[i] == mins[i] for i in range(3))
#         total = sum(vals)
#         if count > best_count or (count == best_count and total < best_sum):
#             best = key
#             best_count = count
#             best_sum = total
#     if not best:
#         return 0.0, 0, ""
#     lr, bs = best
#     return lr, bs, epochs.get(best, "")


# def main() -> None:
#     first_lrs = [5.0e-6, 2.0e-5, 5.0e-5, 1.0e-4]
#     for lr in first_lrs:
#         cfg = _write_config(lr, 4, 7)
#         _run_training(cfg)
#     _run_metrics()
#     best_lr = _choose_best_lr()
#     logger.info("Best learning rate: %s", best_lr)
#     for bs in [2, 8]:
#         cfg = _write_config(best_lr, bs, 7)
#         _run_training(cfg)
    # cfg = _write_config(args.learning_rate, args.batch_size, args.epochs)
    # _run_training(cfg)
    # _run_metrics()
    # lr, bs, epoch = _choose_best_overall()
    # logger.info(
    #     "Best configuration: lr=%s, batch_size=%s (epoch %s)", lr, bs, epoch
    # )

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()
    cfg = _write_config(args.learning_rate, args.batch_size, args.epochs)
    _run_training(cfg)
    _run_metrics()


if __name__ == "__main__":
    main()