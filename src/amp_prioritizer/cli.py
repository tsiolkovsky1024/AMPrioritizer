"""Command line interface.

    amp-prioritizer data verify            verify the released dataset checksums
    amp-prioritizer train --family ...     launch a training run
    amp-prioritizer score --config ...     score a FASTA file with a checkpoint
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .data_versions import verify
from .provenance import ROOT


def _run(module: str, argv: list[str]) -> None:
    """Run a package module, or a script file when `module` ends in .py."""
    if module.endswith(".py"):
        command = [sys.executable, str(ROOT / module), *argv]
    else:
        command = [sys.executable, "-m", module, *argv]
    raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)


def main() -> None:
    parser = argparse.ArgumentParser(prog="amp-prioritizer")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="dataset utilities")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    verify_parser = data_sub.add_parser("verify", help="check files against manifest.json")
    verify_parser.add_argument("--version", default="v2_censor_safe")

    train = commands.add_parser("train", help="train a predictor")
    train.add_argument("family", choices=["benchmark", "full"])
    train.add_argument("--config")
    train.add_argument("--data-dir", default="data/current")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--warm-start", help="full family only")

    score = commands.add_parser("score", help="score sequences with a trained checkpoint")
    score.add_argument("--config", required=True)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--fasta", required=True, help="input FASTA file")
    score.add_argument("--out", default="screen_out.csv")
    score.add_argument("--mode", choices=["classic", "narrow"], default="classic")
    score.add_argument("--target-species", nargs="+")
    score.add_argument("--offtarget-species", nargs="+")
    score.add_argument("--top-k", type=int, default=100)

    args = parser.parse_args()

    if args.command == "data":
        print(json.dumps(verify(args.version), indent=2))
        return

    if args.command == "train":
        config = args.config or f"configs/{args.family}.yaml"
        module = "amp_prioritizer.train_stripped" if args.family == "benchmark" else "amp_prioritizer.train"
        argv = ["--config", config, "--data_dir", args.data_dir]
        if args.family == "full":
            argv += ["--stage", "2b"]
        if args.max_steps is not None:
            argv += ["--max_steps", str(args.max_steps)]
        if args.warm_start:
            if args.family != "full":
                raise SystemExit("--warm-start is only supported for the full family")
            argv += ["--warm_start", args.warm_start]
        _run(module, argv)

    argv = ["--config", args.config, "--ckpt", args.checkpoint,
            "--fasta", args.fasta, "--mode", args.mode,
            "--top_k", str(args.top_k), "--out", args.out]
    if args.target_species:
        argv += ["--target_species", *args.target_species]
    if args.offtarget_species:
        argv += ["--offtarget_species", *args.offtarget_species]
    _run("scripts/screen.py", argv)


if __name__ == "__main__":
    main()
