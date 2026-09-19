"""곡선 구간(find_curve_segments.py 산출물)에서 entry_frame 알고리즘이
오탐(가짜 진입 검출)을 내는지 정량 검증 — negative control.

comma2k19는 사고가 없는 일반 주행이라 "진짜 진입 이벤트"는 없어야 한다.
그런데도 entry_frame이 0이 아닌 값으로 나오면(=진입을 검출했다고 주장하면)
고정 사다리꼴이 곡선을 직선으로 착각해 선행차량을 오판한 것 — 이게 known
limitation이었던 그 문제다.

inference.py의 _s2_entry를 그대로 재사용한다(수정 전/후 비교의 기준선).
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent
sys.path.insert(0, str(BASELINE))
import inference as inf  # noqa: E402

STAGE3_VIDEOS = ROOT.parent / "stage3_work" / "data" / "videos"
CURVE_CSV = Path(
    "/private/tmp/claude-501/-Users-supper-Desktop-daycon---------------------------AI-----/f7b3750b-fafc-4e9b-ab44-48198fafc199/scratchpad/curve_segments.csv"
)
MARGIN = 50  # 곡선 구간 앞뒤로 이만큼 더 디코딩(트랙 궤적 분석에 문맥 필요)


def main():
    import os

    import torch
    from ultralytics import YOLO

    yolo_model = YOLO(str(BASELINE / "model" / "stage2" / "yolo11n.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_lane = os.environ.get("USE_LANE_FIT", "1") != "0"
    yolop_model = inf._load_yolop(BASELINE / "model" / "stage2", device) if use_lane else None
    print(f"차선 기반 경계 사용: {use_lane}")

    rows = list(csv.DictReader(open(CURVE_CSV, encoding="utf-8")))
    print(f"검증 대상 곡선 구간: {len(rows)}개")

    results = []
    for i, r in enumerate(rows, 1):
        vid = r["vid"]
        mid = int(r["mid_frame"])
        start = max(0, int(r["start_frame"]) - MARGIN)
        end = int(r["end_frame"]) + MARGIN

        cap = cv2.VideoCapture(str(STAGE3_VIDEOS / f"{vid}.mp4"))
        frames = []
        idx = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if start <= idx <= end:
                frames.append(f)
            idx += 1
        cap.release()
        if len(frames) < 10:
            continue

        local_mid = mid - start
        entry_frame, entry_side = inf._s2_entry(
            frames, local_mid, yolo_model, fps=10.0, yolop_model=yolop_model, device=device
        )
        false_positive = entry_frame != 0
        results.append({
            "vid": vid, "direction": r["direction"], "duration_frames": r["duration_frames"],
            "entry_frame_local": entry_frame, "entry_side": entry_side, "false_positive": false_positive,
        })
        print(f"[{i}/{len(rows)}] {vid} dir={r['direction']} -> entry_frame={entry_frame} "
              f"{'*** 오탐 ***' if false_positive else '(정상: 미검출)'}")

    n_fp = sum(1 for x in results if x["false_positive"])
    print(f"\n=== 결과: {len(results)}개 중 오탐 {n_fp}개 ({n_fp/len(results)*100:.1f}%) ===")
    for d in ("LEFT", "RIGHT"):
        sub = [x for x in results if x["direction"] == d]
        n_fp_d = sum(1 for x in sub if x["false_positive"])
        print(f"  {d}: {len(sub)}개 중 오탐 {n_fp_d}개 ({n_fp_d/len(sub)*100:.1f}%)" if sub else f"  {d}: 0개")

    import os
    tag = os.environ.get("VALIDATION_TAG", "latest")
    out_csv = CURVE_CSV.parent / f"curve_validation_{tag}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "direction", "duration_frames", "entry_frame_local", "entry_side", "false_positive"])
        w.writeheader()
        w.writerows(results)
    print(f"상세 결과 저장 -> {out_csv}")


if __name__ == "__main__":
    main()
