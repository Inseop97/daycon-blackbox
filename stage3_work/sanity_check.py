"""라벨 파이프라인 결과를 몇 개 영상에서 눈으로 검증하기 위한 스크립트.

각 영상에서 accel_label/steer_label이 바뀌는 지점 근처 프레임을 뽑아
3x3 그리드 이미지로 저장한다 — evasion_space 라벨링 때 쓴 방식과 동일.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT_DIR = Path(
    "/private/tmp/claude-501/-Users-supper-Desktop-daycon---------------------------AI-----/f7b3750b-fafc-4e9b-ab44-48198fafc199/scratchpad/stage3_sanity"
)


def load_rows():
    by_id = defaultdict(list)
    with open(DATA / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_id[r["ID"]].append(r)
    for vid in by_id:
        by_id[vid].sort(key=lambda r: int(r["sample_index"]))
    return by_id


def transitions(rows, key):
    idxs = []
    prev = None
    for i, r in enumerate(rows):
        if r[key] != prev:
            idxs.append(i)
            prev = r[key]
    return idxs


def main(n_videos=6, seed=1):
    by_id = load_rows()
    rng = random.Random(seed)
    vids = rng.sample(sorted(by_id.keys()), min(n_videos, len(by_id)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for vid in vids:
        rows = by_id[vid]
        steer_trans = transitions(rows, "steer_label")
        accel_trans = transitions(rows, "accel_label")
        # steer 전환점 위주로 최대 9개 샘플 프레임 선택(앞뒤 몇 프레임 곁들여서)
        picks = sorted(set(steer_trans + accel_trans))[:9]
        if len(picks) < 9:
            picks += [rows_i for rows_i in range(0, len(rows), max(1, len(rows) // 9)) if rows_i not in picks]
            picks = sorted(set(picks))[:9]

        cap = cv2.VideoCapture(str(DATA / "videos" / f"{vid}.mp4"))
        cells = []
        for idx in picks[:9]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.resize(frame, (360, 270))
            r = rows[idx]
            label = f"{vid} f{idx} {r['accel_label'][:4]}/{r['steer_label']}"
            cv2.putText(frame, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cells.append(frame)
        cap.release()
        while len(cells) < 9:
            cells.append(np.zeros((270, 360, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(cells[i : i + 3]) for i in range(0, 9, 3)])
        cv2.imwrite(str(OUT_DIR / f"{vid}.jpg"), grid)
        print(f"{vid}: 저장 완료 (steer 전환 {len(steer_trans)}회, accel 전환 {len(accel_trans)}회)")


if __name__ == "__main__":
    main()
