"""Stage 1 학습용 프레임 데이터셋 생성.

data/stage1/original, data/stage2/videos, data/stage3/videos 의 진짜 촬영된
원본 영상 15개를 소스로 삼아:
  - RERECORDED: recapture_augment로 재녹화 특성을 합성한 여러 변형
  - ORIGINAL  : 순한 증강만 적용한 여러 변형 + 무증강 원본
을 생성하고, 소스 영상 단위(source_id)로 train/val을 나눠 씬(장면) 누출을
방지한다. 대회가 실제로 제공한 data/stage1/original·rerecorded 5쌍은
학습에 쓰지 않고 별도의 "공식 샘플 점검"용으로만 둔다.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from recapture_augment import (
    MildOriginalParams,
    RecaptureParams,
    render_mild_original_frame,
    render_recapture_frame,
)

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent
OUT_DIR = ROOT / "data"
OUT_SIZE = (256, 256)  # 학습 시 RandomCrop(224)로 한 번 더 다양화
FRAMES_PER_CLIP = 8
N_RERECORD_VARIANTS = 6
N_MILD_ORIGINAL_VARIANTS = 4
VAL_SOURCE_IDS = {"S3_OPEN_004", "S2_005"}  # 씬 누출 방지용 완전 홀드아웃
SEED = 20260907


def _read_all_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"cannot decode: {path}")
    return frames


def _sample_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total)) + [total - 1] * (n - total)
    return list(np.linspace(0, total - 1, n).round().astype(int))


def _gather_source_videos() -> list[tuple[str, Path]]:
    sources = []
    for i, p in enumerate(sorted((BASELINE / "data/stage1/original").glob("*.mp4")), 1):
        sources.append((f"S1_{i:03d}", p))
    for i, p in enumerate(sorted((BASELINE / "data/stage2/videos").glob("*.mp4")), 1):
        sources.append((f"S2_{i:03d}", p))
    for i, p in enumerate(sorted((BASELINE / "data/stage3/videos").glob("*.mp4")), 1):
        sources.append((f"S3_OPEN_{i:03d}", p))
    return sources


def _save(frame_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def build() -> None:
    rng = random.Random(SEED)
    rows = []

    sources = _gather_source_videos()
    print(f"소스 영상 {len(sources)}개 발견: {[s for s, _ in sources]}")

    for source_id, path in sources:
        split = "val" if source_id in VAL_SOURCE_IDS else "train"
        frames = _read_all_frames(path)
        total = len(frames)
        idx = _sample_indices(total, FRAMES_PER_CLIP)

        # --- RERECORDED 변형들 ---
        for v in range(N_RERECORD_VARIANTS):
            clip_rng = random.Random(rng.randint(0, 2**31 - 1))
            params = RecaptureParams.sample(clip_rng)
            clip_id = f"{source_id}_RR{v}"
            for t, fi in enumerate(idx):
                out = render_recapture_frame(frames[fi], params, t, OUT_SIZE)
                rel = f"{split}/RERECORDED/{clip_id}_{t}.jpg"
                _save(out, OUT_DIR / rel)
                rows.append([rel, "RERECORDED", source_id, clip_id, split])

        # --- ORIGINAL: 무증강 1개 + 순한 증강 N개 ---
        clip_id = f"{source_id}_ORIG_raw"
        for t, fi in enumerate(idx):
            out = cv2.resize(frames[fi], OUT_SIZE, interpolation=cv2.INTER_AREA)
            rel = f"{split}/ORIGINAL/{clip_id}_{t}.jpg"
            _save(out, OUT_DIR / rel)
            rows.append([rel, "ORIGINAL", source_id, clip_id, split])

        for v in range(N_MILD_ORIGINAL_VARIANTS):
            clip_rng = random.Random(rng.randint(0, 2**31 - 1))
            params = MildOriginalParams.sample(clip_rng)
            clip_id = f"{source_id}_ORIG{v}"
            for t, fi in enumerate(idx):
                out = render_mild_original_frame(frames[fi], params, OUT_SIZE)
                rel = f"{split}/ORIGINAL/{clip_id}_{t}.jpg"
                _save(out, OUT_DIR / rel)
                rows.append([rel, "ORIGINAL", source_id, clip_id, split])

        print(f"  {source_id} ({split}): {total} frames decoded, variants written")

    # --- 공식 제공 5+5 샘플: 학습에 쓰지 않고 점검용으로만 저장 ---
    for label, folder in [("ORIGINAL", "original"), ("RERECORDED", "rerecorded")]:
        for i, path in enumerate(sorted((BASELINE / "data/stage1" / folder).glob("*.mp4")), 1):
            frames = _read_all_frames(path)
            idx = _sample_indices(len(frames), FRAMES_PER_CLIP)
            clip_id = f"OFFICIAL_{label}_{i:03d}"
            for t, fi in enumerate(idx):
                out = cv2.resize(frames[fi], OUT_SIZE, interpolation=cv2.INTER_AREA)
                rel = f"official_check/{label}/{clip_id}_{t}.jpg"
                _save(out, OUT_DIR / rel)
                rows.append([rel, label, "OFFICIAL", clip_id, "official_check"])

    with open(OUT_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "source_id", "clip_id", "split"])
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r[4] == "train")
    n_val = sum(1 for r in rows if r[4] == "val")
    n_official = sum(1 for r in rows if r[4] == "official_check")
    print(f"\n총 {len(rows)}개 프레임 생성: train={n_train}, val={n_val}, official_check={n_official}")
    print(f"labels.csv -> {OUT_DIR / 'labels.csv'}")


if __name__ == "__main__":
    build()
