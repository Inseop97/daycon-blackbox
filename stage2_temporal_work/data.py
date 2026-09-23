from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


SIDE_TO_INDEX = {"LEFT": 0, "RIGHT": 1}


def resolve_video(row: pd.Series, nexar_dir: Path, ccd_dir: Path) -> Path:
    video_id = str(row.video_id)
    if str(row.source).lower() == "nexar":
        stem = Path(video_id).stem
        padded = f"{int(stem):05d}" if stem.isdigit() else stem
        candidates = [nexar_dir / video_id, nexar_dir / f"{stem}.mp4", nexar_dir / f"{padded}.mp4"]
    else:
        stem = Path(video_id).stem
        padded = f"{int(stem):06d}" if stem.isdigit() else stem
        candidates = [ccd_dir / video_id, ccd_dir / f"{stem}.mp4", ccd_dir / f"{padded}.mp4"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"video not found for {row.source}/{video_id}: {candidates}")


def video_info(path: Path) -> tuple[float, int, float]:
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if not np.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise ValueError(f"invalid video metadata: {path} fps={fps}, frames={frame_count}")
    return fps, frame_count, frame_count / fps


def load_labels(
    labels_path: Path,
    nexar_master_path: Path | None,
    nexar_dir: Path,
    ccd_dir: Path,
) -> pd.DataFrame:
    df = pd.read_csv(labels_path, dtype={"video_id": str, "source": str})
    canonical = "relative_path" in df.columns and "filename" in df.columns
    nexar_master = None
    if not canonical:
        if nexar_master_path is None:
            raise ValueError(
                "--nexar-master is required for the original manual label CSV; "
                "use stage2_gpu_data/labels/train_labels.csv to run without it"
            )
        nexar_master = pd.read_csv(nexar_master_path, dtype={"video_id": str})
    required = {"video_id", "source", "collision_time", "entry_time", "entry_side", "evasion_space"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing label columns: {sorted(missing)}")
    df = df.copy()
    df["entry_side"] = df.entry_side.astype(str).str.upper()
    df["evasion_space"] = pd.to_numeric(df.evasion_space, errors="raise").astype(int)
    if df.video_id.duplicated().any():
        raise ValueError(f"duplicate video_id: {df.loc[df.video_id.duplicated(), 'video_id'].tolist()}")
    rows = []
    for _, original_row in df.iterrows():
        row = original_row.copy()
        if str(row.source).lower() == "nexar" and not canonical:
            label_index = int(str(row.video_id))
            if label_index < 0 or label_index >= len(nexar_master):
                raise ValueError(f"Nexar label index out of range: {row.video_id}")
            row["label_index"] = str(row.video_id)
            row["video_id"] = str(nexar_master.iloc[label_index].video_id)
        if row.entry_side not in SIDE_TO_INDEX or row.evasion_space not in (0, 1):
            raise ValueError(f"invalid class label: {row.to_dict()}")
        path = resolve_video(row, nexar_dir, ccd_dir)
        fps, frame_count, duration = video_info(path)
        if not (0 <= float(row.entry_time) <= float(row.collision_time) < duration + 1 / fps):
            raise ValueError(f"invalid event times for {row.video_id}: entry={row.entry_time}, collision={row.collision_time}, duration={duration}")
        item = row.to_dict()
        item.update(video_path=str(path), fps=fps, frame_count=frame_count, duration=duration)
        rows.append(item)
    return pd.DataFrame(rows)


def make_split(df: pd.DataFrame, val_ratio: float, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import train_test_split

    strata = df.source.astype(str) + "_" + df.entry_side.astype(str)
    indices = np.arange(len(df))
    val_size = max(1, int(round(len(df) * val_ratio)))
    can_stratify = strata.value_counts().min() >= 2 and val_size >= strata.nunique()
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_ratio,
        random_state=seed,
        stratify=strata if can_stratify else None,
    )
    result = df.copy()
    result["split"] = "train"
    result.loc[val_idx, "split"] = "val"
    return result


def cache_key(row: pd.Series, target_fps: float) -> str:
    raw = f"{row.source}|{row.video_id}|{target_fps:.4f}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def extract_sampled_frames(path: Path, source_fps: float, target_fps: float):
    step = max(source_fps / target_fps, 1.0)
    next_position = 0.0
    cap = cv2.VideoCapture(str(path))
    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index + 1e-6 >= next_position:
            yield frame_index, Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            next_position += step
        frame_index += 1
    cap.release()


class CachedFeatureDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: Path, target_fps: float, sigma_sec: float):
        self.rows = [row for _, row in manifest.reset_index(drop=True).iterrows()]
        self.cache_dir = cache_dir
        self.target_fps = target_fps
        self.sigma_sec = sigma_sec

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = torch.load(self.cache_dir / f"{cache_key(row, self.target_fps)}.pt", map_location="cpu", weights_only=False)
        times = item["times"].float()
        sigma = max(self.sigma_sec, 1 / self.target_fps)
        collision_target = torch.exp(-0.5 * ((times - float(row.collision_time)) / sigma) ** 2)
        entry_target = torch.exp(-0.5 * ((times - float(row.entry_time)) / sigma) ** 2)
        return {
            "features": item["features"].float(),
            "times": times,
            "collision_target": collision_target,
            "entry_target": entry_target,
            "collision_index": torch.tensor(int(torch.argmin(torch.abs(times - float(row.collision_time))))),
            "entry_index": torch.tensor(int(torch.argmin(torch.abs(times - float(row.entry_time))))),
            "collision_time": torch.tensor(float(row.collision_time)),
            "entry_time": torch.tensor(float(row.entry_time)),
            "side": torch.tensor(SIDE_TO_INDEX[row.entry_side]),
            "evasion": torch.tensor(int(row.evasion_space)),
            "video_id": str(row.video_id),
        }


def collate_sequences(batch):
    max_len = max(x["features"].shape[0] for x in batch)
    feature_dim = batch[0]["features"].shape[1]
    b = len(batch)
    features = torch.zeros(b, max_len, feature_dim)
    times = torch.zeros(b, max_len)
    mask = torch.zeros(b, max_len, dtype=torch.bool)
    collision_target = torch.zeros(b, max_len)
    entry_target = torch.zeros(b, max_len)
    for i, item in enumerate(batch):
        n = item["features"].shape[0]
        features[i, :n] = item["features"]
        times[i, :n] = item["times"]
        mask[i, :n] = True
        collision_target[i, :n] = item["collision_target"]
        entry_target[i, :n] = item["entry_target"]
    return {
        "features": features,
        "times": times,
        "mask": mask,
        "collision_target": collision_target,
        "entry_target": entry_target,
        "collision_index": torch.stack([x["collision_index"] for x in batch]),
        "entry_index": torch.stack([x["entry_index"] for x in batch]),
        "collision_time": torch.stack([x["collision_time"] for x in batch]),
        "entry_time": torch.stack([x["entry_time"] for x in batch]),
        "side": torch.stack([x["side"] for x in batch]),
        "evasion": torch.stack([x["evasion"] for x in batch]),
        "video_id": [x["video_id"] for x in batch],
    }
