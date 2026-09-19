"""evasion_space 1차 실험용 데이터셋 생성.

수작업으로 라벨링한 evasion_labels.csv(vid, frame, evasion_space, batch)를
읽어서, 각 (vid, frame)의 실제 이미지를 CCD 원본 영상에서 다시 디코딩해
저장한다. 영상 1개 = 샘플 1개(수작업 라벨링 특성상 프레임 하나만 봤으므로)
라서, train/val은 영상 단위로 랜덤 분할하면 곧 샘플 단위 분할과 같다 —
다만 명시적으로 video id 리스트를 셔플해서 나눠 실수를 방지한다.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import cv2

CCD = Path("/Users/supper/Desktop/daycon/블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회/외부데이터/CCD(CarClashDataset)")
LABELS_CSV = Path(
    "/private/tmp/claude-501/-Users-supper-Desktop-daycon---------------------------AI-----/f7b3750b-fafc-4e9b-ab44-48198fafc199/scratchpad/evasion_grids2/evasion_labels_combined.csv"
)
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data"
# inference.py의 predict_stage2 전처리(짧은 변 256 기준 비율 보존 리사이즈)와 반드시
# 동일해야 한다 — 예전에 여기서 정사각형(256,256)으로 찌그러뜨려 저장했다가 학습/추론
# 전처리가 어긋나는 버그가 있었다(fix_aspect_squash.py로 사후 복구).
TARGET_SHORT_SIDE = 256
VAL_RATIO = 0.2
SEED = 20260912


def get_frame(path: Path, idx: int):
    cap = cv2.VideoCapture(str(path))
    frame = None
    for i in range(idx + 1):
        ok, f = cap.read()
        if not ok:
            break
        frame = f
    cap.release()
    return frame


def main():
    rows = [r for r in csv.DictReader(open(LABELS_CSV, encoding="utf-8")) if r["evasion_space"] != ""]
    print(f"라벨 있는 샘플: {len(rows)}개")

    rng = random.Random(SEED)
    vids = list({r["vid"] for r in rows})
    rng.shuffle(vids)
    n_val = int(len(vids) * VAL_RATIO)
    val_vids = set(vids[:n_val])

    out_rows = []
    n_fail = 0
    for r in rows:
        vid, frame_idx, label = r["vid"], int(r["frame"]), r["evasion_space"]
        split = "val" if vid in val_vids else "train"
        video_path = CCD / "videos" / "Crash-1500" / f"{vid}.mp4"
        img = get_frame(video_path, frame_idx)
        if img is None:
            n_fail += 1
            continue
        h, w = img.shape[:2]
        scale = TARGET_SHORT_SIDE / min(h, w)
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        rel = f"{split}/{label}/{vid}.jpg"
        (OUT_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT_DIR / rel), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        out_rows.append([rel, label, vid, split])

    with open(OUT_DIR / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "vid", "split"])
        w.writerows(out_rows)

    n_train = sum(1 for r in out_rows if r[3] == "train")
    n_val_actual = sum(1 for r in out_rows if r[3] == "val")
    print(f"저장 완료: train={n_train} val={n_val_actual} (디코딩 실패 {n_fail}개 제외)")
    for split in ["train", "val"]:
        n0 = sum(1 for r in out_rows if r[3] == split and r[1] == "0")
        n1 = sum(1 for r in out_rows if r[3] == split and r[1] == "1")
        print(f"  {split}: 0={n0} 1={n1}")


if __name__ == "__main__":
    main()
