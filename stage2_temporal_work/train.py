from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .data import (
    CachedFeatureDataset,
    cache_key,
    collate_sequences,
    extract_sampled_frames,
    load_labels,
    make_split,
)
from .model import BACKBONE_NAME, FRAME_TRANSFORM, IMAGE_SIZE, Stage2TemporalModel, build_backbone


def parse_args():
    p = argparse.ArgumentParser(description="Train Stage2 DINOv2 + TCN model")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--nexar-master", type=Path)
    p.add_argument("--nexar-dir", type=Path, required=True)
    p.add_argument("--ccd-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("runs/stage2_temporal"))
    p.add_argument("--cache-dir", type=Path, default=Path("cache/stage2_dinov2"))
    p.add_argument("--target-fps", type=float, default=5.0)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--feature-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sigma-sec", type=float, default=0.2)
    p.add_argument("--smoke", action="store_true", help="use first 10 rows and train one epoch")
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_cache(manifest, cache_dir, target_fps, batch_size, dev):
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing = [row for _, row in manifest.iterrows() if not (cache_dir / f"{cache_key(row, target_fps)}.pt").is_file()]
    if not missing:
        print("feature cache already complete")
        sample = torch.load(next(cache_dir.glob("*.pt")), map_location="cpu", weights_only=False)
        return int(sample["features"].shape[1]), None

    backbone = build_backbone(pretrained=True).to(dev).eval()
    for number, row in enumerate(missing, 1):
        frames, frame_indices = [], []
        feature_chunks = []
        for frame_index, image in extract_sampled_frames(Path(row.video_path), float(row.fps), target_fps):
            frames.append(FRAME_TRANSFORM(image))
            frame_indices.append(frame_index)
            if len(frames) == batch_size:
                with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                    feature_chunks.append(backbone(torch.stack(frames).to(dev)).float().cpu())
                frames = []
        if frames:
            with torch.inference_mode(), torch.autocast(device_type=dev.type, enabled=dev.type == "cuda", dtype=torch.float16):
                feature_chunks.append(backbone(torch.stack(frames).to(dev)).float().cpu())
        if not feature_chunks:
            raise RuntimeError(f"no frames decoded: {row.video_path}")
        payload = {
            "features": torch.cat(feature_chunks),
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "times": torch.tensor(frame_indices, dtype=torch.float32) / float(row.fps),
        }
        torch.save(payload, cache_dir / f"{cache_key(row, target_fps)}.pt")
        print(f"cache {number}/{len(missing)} {row.source}/{row.video_id} frames={len(frame_indices)}")
    feature_dim = int(payload["features"].shape[1])
    backbone_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
    del backbone
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return feature_dim, backbone_state


def temporal_nll(logits, target, mask):
    masked_logits = logits.masked_fill(~mask, -1e9)
    distribution = target * mask
    distribution = distribution / distribution.sum(1, keepdim=True).clamp_min(1e-8)
    return -(distribution * nn.functional.log_softmax(masked_logits, dim=1)).sum(1).mean()


def run_epoch(model, loader, dev, optimizer, evasion_weights):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    records = []
    for batch in loader:
        tensor_keys = [k for k, v in batch.items() if torch.is_tensor(v)]
        for key in tensor_keys:
            batch[key] = batch[key].to(dev)
        with torch.set_grad_enabled(training):
            out = model(batch["features"], batch["entry_index"], batch["collision_index"])
            collision_loss = temporal_nll(out["collision_logits"], batch["collision_target"], batch["mask"])
            entry_loss = temporal_nll(out["entry_logits"], batch["entry_target"], batch["mask"])
            side_loss = nn.functional.cross_entropy(out["side_logits"], batch["side"])
            evasion_loss = nn.functional.cross_entropy(out["evasion_logits"], batch["evasion"], weight=evasion_weights)
            loss = collision_loss + entry_loss + 0.5 * side_loss + 0.5 * evasion_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.item()) * len(batch["video_id"])
        masked_collision = out["collision_logits"].masked_fill(~batch["mask"], -1e9)
        masked_entry = out["entry_logits"].masked_fill(~batch["mask"], -1e9)
        collision_pred = masked_collision.argmax(1)
        # Entry must not occur after collision.
        positions = torch.arange(masked_entry.shape[1], device=dev)[None]
        entry_pred = masked_entry.masked_fill(positions > collision_pred[:, None], -1e9).argmax(1)
        if not training:
            out = model(batch["features"], entry_pred, collision_pred)
        for i, video_id in enumerate(batch["video_id"]):
            records.append({
                "video_id": video_id,
                "collision_true": float(batch["collision_time"][i].item()),
                "collision_pred": float(batch["times"][i, collision_pred[i]].item()),
                "entry_true": float(batch["entry_time"][i].item()),
                "entry_pred": float(batch["times"][i, entry_pred[i]].item()),
                "side_true": int(batch["side"][i].item()),
                "side_pred": int(out["side_logits"][i].argmax().item()),
                "evasion_true": int(batch["evasion"][i].item()),
                "evasion_pred": int(out["evasion_logits"][i].argmax().item()),
            })
    return total_loss / len(loader.dataset), pd.DataFrame(records)


def metrics(records: pd.DataFrame) -> dict:
    result = {}
    for name in ("collision", "entry"):
        error = np.abs(records[f"{name}_pred"] - records[f"{name}_true"])
        result[f"{name}_acc_03"] = float((error <= 0.3).mean())
        result[f"{name}_acc_05"] = float((error <= 0.5).mean())
        result[f"{name}_mae"] = float(error.mean())
        result[f"{name}_median_ae"] = float(np.median(error))
    for name in ("side", "evasion"):
        result[f"{name}_macro_f1"] = float(f1_score(records[f"{name}_true"], records[f"{name}_pred"], average="macro", zero_division=0))
        result[f"{name}_balanced_accuracy"] = float(balanced_accuracy_score(records[f"{name}_true"], records[f"{name}_pred"]))
        result[f"{name}_confusion"] = confusion_matrix(records[f"{name}_true"], records[f"{name}_pred"], labels=[0, 1]).tolist()
    return result


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels = load_labels(args.labels, args.nexar_master, args.nexar_dir, args.ccd_dir)
    if args.smoke:
        labels = labels.groupby("source", group_keys=False).head(5).reset_index(drop=True)
        args.epochs = 1
        args.val_ratio = 0.2
    manifest_path = args.output_dir / "split_manifest.csv"
    if manifest_path.is_file() and not args.smoke:
        manifest = pd.read_csv(manifest_path)
        if set(manifest.video_id) != set(labels.video_id):
            raise ValueError("existing split_manifest.csv does not match current labels")
    else:
        manifest = make_split(labels, args.val_ratio, args.seed)
        manifest.to_csv(manifest_path, index=False)
    print(pd.crosstab([manifest.split, manifest.source], manifest.entry_side))
    print(pd.crosstab([manifest.split, manifest.source], manifest.evasion_space))

    dev = device()
    print(f"device={dev}")
    feature_dim, backbone_state = build_cache(manifest, args.cache_dir, args.target_fps, args.feature_batch_size, dev)
    if backbone_state is None:
        # A complete cache may come from an earlier run. Recreate the frozen weights for the final offline checkpoint.
        backbone = build_backbone(pretrained=True)
        backbone_state = {k: v.cpu() for k, v in backbone.state_dict().items()}
        del backbone

    train_manifest = manifest[manifest.split == "train"]
    val_manifest = manifest[manifest.split == "val"]
    train_ds = CachedFeatureDataset(train_manifest, args.cache_dir, args.target_fps, args.sigma_sec)
    val_ds = CachedFeatureDataset(val_manifest, args.cache_dir, args.target_fps, args.sigma_sec)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_sequences)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_sequences)

    model = Stage2TemporalModel(feature_dim).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = train_manifest.evasion_space.value_counts().reindex([0, 1], fill_value=0).to_numpy(dtype=float)
    evasion_weights = torch.tensor(len(train_manifest) / np.maximum(2 * counts, 1), dtype=torch.float32, device=dev)
    best_score, stale = -1.0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, dev, optimizer, evasion_weights)
        val_loss, records = run_epoch(model, val_loader, dev, None, evasion_weights)
        report = metrics(records)
        score = (report["collision_acc_03"] + report["entry_acc_03"] + report["side_macro_f1"] + report["evasion_macro_f1"]) / 4
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "selection_score": score, **report}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if score > best_score:
            best_score, stale = score, 0
            checkpoint = {
                "arch": "dinov2_tcn",
                "backbone_name": BACKBONE_NAME,
                "backbone": backbone_state,
                "temporal_model": {k: v.cpu() for k, v in model.state_dict().items()},
                "feature_dim": feature_dim,
                "hidden_dim": 256,
                "target_fps": args.target_fps,
                "image_size": IMAGE_SIZE,
                "side_labels": ["LEFT", "RIGHT"],
                "evasion_labels": [0, 1],
                "seed": args.seed,
                "metrics": report,
            }
            torch.save(checkpoint, args.output_dir / "best.pt")
            records.to_csv(args.output_dir / "best_val_predictions.csv", index=False)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}")
                break
    pd.DataFrame(history).to_json(args.output_dir / "history.json", orient="records", indent=2)
    print(f"best checkpoint: {args.output_dir / 'best.pt'} score={best_score:.4f}")


if __name__ == "__main__":
    main()
