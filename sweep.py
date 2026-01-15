"""Run experiment sweeps from YAML."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _has_completed(results_root: str | Path, experiment: str, dataset: str, model: str, seed: int) -> bool:
    run_dir = Path(results_root) / experiment / dataset / model / str(seed)
    return (run_dir / "best.ckpt").exists() or (run_dir / "metrics.json").exists()


def _run_cmd(cmd: List[str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.check_call(cmd)


def _build_cmd(mode: str, cfg_path: str, dataset: str, model: str, seed: int, overrides: Dict[str, Any]) -> List[str]:
    base = ["python", f"scripts/{mode}.py", "--config", cfg_path, "--dataset", dataset, "--model", model, "--seed", str(seed)]
    if overrides:
        opts = []
        for k, v in overrides.items():
            opts.append(f"{k}={v}")
        base += ["--opts", *opts]
    return base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", required=True, help="Sweep YAML file.")
    args = parser.parse_args()

    sweep_cfg = _load_yaml(args.sweep)
    base_cfg = sweep_cfg.get("base_config")
    experiment = sweep_cfg.get("experiment", "sweep")
    results_root = sweep_cfg.get("results_root", "outputs/runs")
    mode = sweep_cfg.get("mode", "both")
    overrides = sweep_cfg.get("overrides", {})

    if not base_cfg:
        raise ValueError("Sweep config must define base_config.")

    def run_one(dataset: str, model: str, seed: int, run_overrides: Dict[str, Any]) -> None:
        if _has_completed(results_root, experiment, dataset, model, seed):
            print(f"[skip] {dataset} {model} seed{seed} already done")
            return
        merged = dict(overrides)
        merged.update(run_overrides or {})
        merged["exp.name"] = experiment
        if mode in {"train", "both"}:
            _run_cmd(_build_cmd("train", base_cfg, dataset, model, seed, merged))
        if mode in {"adapt", "both"}:
            _run_cmd(_build_cmd("adapt", base_cfg, dataset, model, seed, merged))

    if "runs" in sweep_cfg:
        for run in sweep_cfg["runs"]:
            dataset = run["dataset"]
            model = run["model"]
            seeds = run.get("seeds", sweep_cfg.get("seeds", [0]))
            run_overrides = run.get("overrides", {})
            for seed in seeds:
                run_one(dataset, model, seed, run_overrides)
    else:
        datasets = sweep_cfg.get("datasets", [])
        models = sweep_cfg.get("models", [])
        seeds = sweep_cfg.get("seeds", [0])
        for dataset in datasets:
            for model in models:
                for seed in seeds:
                    run_one(dataset, model, seed, {})


if __name__ == "__main__":
    main()
