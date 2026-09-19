"""Stage 1 1차 실험용 데이터셋 생성 — CCD(Car Crash Dataset)를 ORIGINAL 소스로 사용.

CCD에는 재녹화 여부 라벨이 없다. Crash-1500 + Normal 영상을 순수하게
"진짜 블랙박스가 찍은 원본 영상 풀"로만 쓰고, RERECORDED는 recapture_augment로
합성한다. train/val은 서로 다른 합성 레시피(recapture_augment.RecaptureParams
의 recipe="train"/"val")를 써서 "합성 함수 하나를 암기"하는 지름길을 막는다.

기존 15개 소스 영상 기반 build_dataset.py는 그대로 두고, 이 스크립트는
같은 data/labels.csv 스키마(path,label,source_id,clip_id,split)로 출력하여
train.py를 코드 수정 없이 재사용할 수 있게 한다.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2
import numpy as np

from recapture_augment import RecaptureParams, render_recapture_frame

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent
CCD = BASELINE.parent / "외부데이터" / "CCD(CarClashDataset)" / "videos"
OUT_DIR = ROOT / "data"
OUT_SIZE = (256, 256)
FRAMES_PER_CLIP = 8
N_TRAIN_VIDEOS = 400
N_VAL_VIDEOS = 100
N_RERECORD_VARIANTS = 3  # 클립(원본 영상)당 서로 다른 서브스타일 조합의 RERECORDED 변형 개수
SEED = 20260910


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


def _save(frame_rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def _gather_ccd_videos() -> list[tuple[str, Path]]:
    videos = []
    for p in sorted((CCD / "Crash-1500").glob("*.mp4")):
        videos.append((f"CCD_CRASH_{p.stem}", p))
    for p in sorted((CCD / "Normal").glob("*.mp4")):
        videos.append((f"CCD_NORMAL_{p.stem}", p))
    return videos


def build() -> None:
    rng = random.Random(SEED)

    all_videos = _gather_ccd_videos()
    print(f"CCD 전체 영상 {len(all_videos)}개 발견 (Crash-1500 + Normal)")

    n_needed = N_TRAIN_VIDEOS + N_VAL_VIDEOS
    sampled = rng.sample(all_videos, n_needed)  # 비복원추출 -> train/val 영상 ID 겹칠 수 없음
    train_videos = sampled[:N_TRAIN_VIDEOS]
    val_videos = sampled[N_TRAIN_VIDEOS:]

    n_crash_train = sum(1 for sid, _ in train_videos if "CRASH" in sid)
    n_crash_val = sum(1 for sid, _ in val_videos if "CRASH" in sid)
    print(f"train: {len(train_videos)}개 (Crash {n_crash_train} / Normal {len(train_videos)-n_crash_train})")
    print(f"val  : {len(val_videos)}개 (Crash {n_crash_val} / Normal {len(val_videos)-n_crash_val})")

    rows = []
    for split, videos, recipe in [("train", train_videos, "train"), ("val", val_videos, "val")]:
        for i, (source_id, path) in enumerate(videos, 1):
            frames = _read_all_frames(path)
            idx = _sample_indices(len(frames), FRAMES_PER_CLIP)

            # ORIGINAL: CCD 원본 프레임 그대로 (리사이즈만, 증강 없음)
            clip_id = f"{source_id}_ORIG"
            for t, fi in enumerate(idx):
                out = cv2.resize(frames[fi], OUT_SIZE, interpolation=cv2.INTER_AREA)
                rel = f"{split}/ORIGINAL/{clip_id}_{t}.jpg"
                _save(out, OUT_DIR / rel)
                rows.append([rel, "ORIGINAL", source_id, clip_id, split])

            # RERECORDED: 같은 클립에서 split별 recipe로 서로 다른 서브스타일 변형을 N개 생성
            # (같은 원본 소스라도 변형마다 다른 스타일이 뽑히므로 "블러 하나만" 암기하지 않게 됨)
            for v in range(N_RERECORD_VARIANTS):
                clip_rng = random.Random(rng.randint(0, 2**31 - 1))
                params = RecaptureParams.sample(clip_rng, recipe=recipe)
                clip_id = f"{source_id}_RR{v}"
                for t, fi in enumerate(idx):
                    out = render_recapture_frame(frames[fi], params, t, OUT_SIZE)
                    rel = f"{split}/RERECORDED/{clip_id}_{t}.jpg"
                    _save(out, OUT_DIR / rel)
                    rows.append([rel, "RERECORDED", source_id, clip_id, split])

            if i % 100 == 0:
                print(f"  {split}: {i}/{len(videos)} 완료")

    # --- 공식 제공 5+5 샘플: 학습에 쓰지 않고 점검용으로만 저장 (기존과 동일) ---
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
