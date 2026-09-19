"""comma2k19 조향각 라벨(stage3_work/data/labels.csv)에서 "진짜 곡선" 구간을
자동으로 골라낸다.

entry_frame/entry_side 알고리즘이 고정 사다리꼴(직선도로 가정) 때문에 곡선
도로에서 오탐(선행차량을 진입으로 오판)한다는 게 알려진 한계였다. 이걸 고치기
전에, comma2k19의 조향각(steer_label)이 sustained LEFT/RIGHT인 구간을 찾아
"실제로는 진입 이벤트가 없는데 알고리즘이 진입을 검출하는지"를 검증할 negative
control 세트로 쓴다.

comma2k19는 사고가 없는 일반 주행 데이터라 진짜 진입/충돌 정답은 없다 —
"오탐이 안 나는지"만 검증 가능하다(곡선에서 실제 진입이 일어나는 케이스는
CCD에서 별도로 눈으로 골라야 함, 이 스크립트의 범위 밖).
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGE3_LABELS = ROOT.parent / "stage3_work" / "data" / "labels.csv"
STAGE3_VIDEOS = ROOT.parent / "stage3_work" / "data" / "videos"

MIN_RUN_LEN = 15  # 10Hz 기준 1.5초 이상 지속돼야 "진짜 곡선"으로 인정(짧은 노이즈성 흔들림 제외)
N_SAMPLE_PER_DIRECTION = 25


def find_runs(rows: list[dict]) -> list[tuple[int, int, str]]:
    """(start_idx, end_idx, direction) 리스트. steer_label 이 연속 동일한 구간만."""
    runs = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i]["steer_label"] != rows[start]["steer_label"]:
            label = rows[start]["steer_label"]
            if label in ("LEFT", "RIGHT") and (i - start) >= MIN_RUN_LEN:
                runs.append((start, i - 1, label))
            start = i
    return runs


def main():
    by_id = defaultdict(list)
    with open(STAGE3_LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_id[r["ID"]].append(r)
    for vid in by_id:
        by_id[vid].sort(key=lambda r: int(r["sample_index"]))

    candidates = {"LEFT": [], "RIGHT": []}
    for vid, rows in by_id.items():
        for start, end, direction in find_runs(rows):
            candidates[direction].append({
                "vid": vid,
                "start_frame": start,
                "end_frame": end,
                "mid_frame": (start + end) // 2,
                "duration_frames": end - start + 1,
                "direction": direction,
            })

    print(f"LEFT 곡선 구간 후보: {len(candidates['LEFT'])}개")
    print(f"RIGHT 곡선 구간 후보: {len(candidates['RIGHT'])}개")

    rng = random.Random(20260918)
    selected = []
    for direction in ("LEFT", "RIGHT"):
        pool = sorted(candidates[direction], key=lambda c: -c["duration_frames"])[:200]  # 긴 구간 위주로 후보 축소
        rng.shuffle(pool)
        selected.extend(pool[:N_SAMPLE_PER_DIRECTION])

    out_dir = Path(
        "/private/tmp/claude-501/-Users-supper-Desktop-daycon---------------------------AI-----/f7b3750b-fafc-4e9b-ab44-48198fafc199/scratchpad"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "curve_segments.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["vid", "start_frame", "end_frame", "mid_frame", "duration_frames", "direction"])
        w.writeheader()
        w.writerows(selected)

    print(f"\n선정된 곡선 구간 {len(selected)}개 저장 -> {out_csv}")
    durations = [c["duration_frames"] for c in selected]
    print(f"지속 길이(프레임, 10Hz): 최소 {min(durations)} 최대 {max(durations)} 평균 {sum(durations)/len(durations):.1f}")


if __name__ == "__main__":
    main()
