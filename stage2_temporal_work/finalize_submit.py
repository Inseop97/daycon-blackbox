from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model" / "stage2" / "temporal_best.pt"
SUBMIT_PATH = ROOT / "submit.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Install Stage2 temporal checkpoint and build a complete submit.zip"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("/data/daycon-stage2/artifacts"),
    )
    return parser.parse_args()


def validate_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "arch",
        "backbone_name",
        "backbone",
        "temporal_model",
        "feature_dim",
        "target_fps",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"checkpoint keys missing: {sorted(missing)}")
    if checkpoint["arch"] != "dinov2_tcn":
        raise ValueError(f"unexpected checkpoint arch: {checkpoint['arch']}")


def main():
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    validate_checkpoint(checkpoint)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.is_file():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = MODEL_PATH.with_name(f"temporal_best_before_{timestamp}.pt")
        shutil.copy2(MODEL_PATH, backup)
        print(f"existing temporal checkpoint backed up: {backup}")
    shutil.copy2(checkpoint, MODEL_PATH)
    print(f"installed: {MODEL_PATH}")

    from build_submit_zip import main as build_submit

    build_submit()
    if not SUBMIT_PATH.is_file():
        raise RuntimeError("build_submit_zip.py did not create submit.zip")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output_submit = args.artifact_dir / "submit.zip"
    output_checkpoint = args.artifact_dir / "temporal_best.pt"
    shutil.copy2(SUBMIT_PATH, output_submit)
    shutil.copy2(MODEL_PATH, output_checkpoint)

    manifest = args.artifact_dir / "SHA256SUMS.txt"
    manifest.write_text(
        f"{sha256(output_submit)}  submit.zip\n"
        f"{sha256(output_checkpoint)}  temporal_best.pt\n",
        encoding="utf-8",
    )
    print(f"artifacts ready: {args.artifact_dir}")
    print(f"submit.zip: {output_submit.stat().st_size / 1024**2:.1f} MB")
    print(f"checkpoint: {output_checkpoint.stat().st_size / 1024**2:.1f} MB")
    print(f"checksums: {manifest}")


if __name__ == "__main__":
    main()
