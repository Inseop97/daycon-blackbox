"""evasion_space의 "세그멘테이션/기하 정보 활용" 아이디어를 구현.

처음엔 YOLOP 사전학습 가중치를 구하기 까다로울 것으로 보고, YOLO 차량 탐지로
"박스 기반 기하 특징"만 근사로 뽑았었다. 이후 YOLOP(hustvl/YOLOP, MIT License)을
로컬에 받아 벤더링해서(`yolop_vendor/`, `model/stage2/yolop_end2end.pth`)
실제 주행가능영역/차선 세그멘테이션을 쓸 수 있게 됐으므로, 두 종류의 특징을
합친다: YOLO 박스 기하특징(상대차량 위치/면적) + YOLOP 세그멘테이션 특징
(주행가능영역이 실제로 얼마나/어디에 있는지).

data/labels.csv(build_dataset.py 산출물)의 각 이미지에 대해 YOLO+YOLOP를 돌려
data/box_features.csv를 만든다.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import yolop_utils

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
VEHICLE_CLS = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO 클래스 인덱스)

BOX_FEATURE_NAMES = [
    "n_vehicles", "largest_area_frac", "largest_cx", "largest_cy",
    "coverage_frac", "gap_left_frac", "gap_right_frac",
]
SEG_FEATURE_NAMES = [
    "drivable_frac_total", "drivable_frac_left", "drivable_frac_right", "drivable_frac_near",
]
FEATURE_NAMES = BOX_FEATURE_NAMES + SEG_FEATURE_NAMES


def extract_features(image, model) -> list[float]:
    h, w = image.shape[:2]
    res = model.predict(image, conf=0.15, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        return [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0]

    xyxy, cls = boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
    vehicle_boxes = xyxy[np.isin(cls, list(VEHICLE_CLS))]
    if len(vehicle_boxes) == 0:
        return [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0]

    areas = (vehicle_boxes[:, 2] - vehicle_boxes[:, 0]) * (vehicle_boxes[:, 3] - vehicle_boxes[:, 1])
    largest = vehicle_boxes[np.argmax(areas)]
    largest_area_frac = float(areas.max() / (w * h))
    largest_cx = float((largest[0] + largest[2]) / 2 / w)
    largest_cy = float((largest[1] + largest[3]) / 2 / h)

    # 화면을 좌우로 나눠 각 차량 박스가 덮는 x범위의 합집합 -> "가려진 폭 비율"
    intervals = sorted((float(b[0] / w), float(b[2] / w)) for b in vehicle_boxes)
    covered, cur_s, cur_e = 0.0, None, None
    for s, e in intervals:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            covered += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        covered += cur_e - cur_s

    leftmost = min(b[0] / w for b in vehicle_boxes)
    rightmost = max(b[2] / w for b in vehicle_boxes)

    return [
        float(len(vehicle_boxes)), largest_area_frac, largest_cx, largest_cy,
        covered, leftmost, 1.0 - rightmost,
    ]


def extract_seg_features(image, yolop_model, device) -> list[float]:
    h, w = image.shape[:2]
    da_mask, _ = yolop_utils.run_yolop(yolop_model, image, device)
    total = float(da_mask.mean())
    left = float(da_mask[:, : w // 2].mean())
    right = float(da_mask[:, w // 2 :].mean())
    near = float(da_mask[int(h * 0.6) :, :].mean())  # 화면 하단(자차와 가까운 쪽)
    return [total, left, right, near]


def main():
    rows = list(csv.DictReader(open(DATA / "labels.csv", encoding="utf-8")))
    model = YOLO("yolo11n.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolop_model = yolop_utils.load_yolop(ROOT.parent / "model" / "stage2" / "yolop_end2end.pth", device)

    out_rows = []
    for i, r in enumerate(rows, 1):
        image = cv2.imread(str(DATA / r["path"]))
        if image is None:
            box_feats, seg_feats = [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]
        else:
            box_feats = extract_features(image, model)
            seg_feats = extract_seg_features(image, yolop_model, device)
        out_rows.append({"vid": r["vid"], **dict(zip(FEATURE_NAMES, box_feats + seg_feats))})
        if i % 50 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] 완료")

    with open(DATA / "box_features.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid"] + FEATURE_NAMES)
        w.writeheader()
        w.writerows(out_rows)
    print(f"저장 완료: {DATA / 'box_features.csv'}")


if __name__ == "__main__":
    main()
