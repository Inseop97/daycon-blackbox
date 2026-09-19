"""build_dataset.py가 원본 비율을 무시하고 256x256으로 찌그러뜨려 저장한 기존
evasion_space 학습 이미지를, inference.py와 동일한 "짧은 변 256 기준 비율 보존
리사이즈"로 다시 만든다.

원본 라벨링 시 썼던 (vid, frame_idx) 매핑 CSV(evasion_labels_combined.csv)가
스크래치패드에서 이미 지워져 더는 없으므로, 대신 이미 저장된 (찌그러진) 학습
이미지와 CCD 원본 영상의 각 프레임을 256x256로 찌그러뜨려 비교해 가장 가까운
프레임을 역으로 찾아낸다 — 찌그러뜨리는 리사이즈는 결정적(deterministic)이고
JPEG quality 95라 압축 잡음 외엔 오차가 거의 없어, 정확한 프레임 인덱스를
높은 신뢰도로 복원할 수 있다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

CCD = Path("/Users/supper/Desktop/daycon/블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회/외부데이터/CCD(CarClashDataset)")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
LABELS_CSV = DATA / "labels.csv"
TARGET_SHORT_SIDE = 256


def load_video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames


def find_matching_frame(saved_squashed: np.ndarray, frames: list[np.ndarray]) -> int:
    best_idx, best_diff = 0, float("inf")
    target = saved_squashed.astype(np.float32)
    for i, f in enumerate(frames):
        candidate = cv2.resize(f, (saved_squashed.shape[1], saved_squashed.shape[0]), interpolation=cv2.INTER_AREA)
        diff = float(np.abs(candidate.astype(np.float32) - target).mean())
        if diff < best_diff:
            best_diff, best_idx = diff, i
    return best_idx, best_diff


def aspect_preserving_resize(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = TARGET_SHORT_SIDE / min(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    rows = list(csv.DictReader(open(LABELS_CSV, encoding="utf-8")))
    print(f"대상: {len(rows)}개")

    video_cache: dict[str, list[np.ndarray]] = {}
    n_fixed, n_fail, high_diff = 0, 0, []

    for i, r in enumerate(rows, 1):
        vid, path = r["vid"], r["path"]
        img_path = DATA / path
        saved = cv2.imread(str(img_path))
        if saved is None:
            n_fail += 1
            continue

        if vid not in video_cache:
            video_cache[vid] = load_video_frames(CCD / "videos" / "Crash-1500" / f"{vid}.mp4")
        frames = video_cache[vid]
        if not frames:
            n_fail += 1
            continue

        frame_idx, diff = find_matching_frame(saved, frames)
        if diff > 15.0:  # 정상 매칭은 보통 1~5 수준 — 너무 크면 매칭 실패로 보고 원본 유지
            high_diff.append((vid, path, diff))
            continue

        fixed = aspect_preserving_resize(frames[frame_idx])
        cv2.imwrite(str(img_path), fixed, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n_fixed += 1

        if i % 50 == 0:
            print(f"  {i}/{len(rows)} 처리 중...")

    print(f"\n완료: 수정 {n_fixed}개 / 실패(디코딩) {n_fail}개 / 매칭불확실 {len(high_diff)}개")
    if high_diff:
        print("매칭 불확실(원본 유지, 찌그러진 채로 남음):")
        for vid, path, diff in high_diff[:20]:
            print(f"  {vid} {path} diff={diff:.1f}")


if __name__ == "__main__":
    main()
