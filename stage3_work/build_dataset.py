"""comma2k19 -> Stage3(가감속/조향) 학습용 데이터셋 생성.

comma2k19은 CAN 신호(speed, steering_angle)가 영상과 동기화되어 있어,
위키 Q&A에서 밝힌 정답 생성 방식(차속+종가속도 -> accel_label,
조향각 -> steer_label)과 같은 종류의 신호로 pseudo-label을 만들 수 있다.

- speed 단위 확인: global_pose/frame_velocities(GPS, m/s 확실)와 CAN speed 값 범위가
  거의 일치 -> CAN speed는 m/s.
- steer_label 부호: 실제 영상 프레임으로 시각 검증함 — steering_angle이 음수(-160.9)인
  구간에서 좌회전하는 장면을 확인함. 즉 **음수=LEFT, 양수=RIGHT**
  (표준 ISO8855 차량축 관례와 반대이므로 특히 주의).
- 비공개 평가 영상은 10Hz(위키 Q&A 확인, sample_index=디코딩 프레임수 1:1).
  comma2k19 원본은 20fps이므로 홀수 프레임을 버려 10Hz로 맞춘다.
- 청크 하나(9~10GB)씩 압축 해제 -> 라벨/축소 영상 생성 -> 원본 삭제 순서로 처리해
  디스크(여유 ~50GB)를 넘지 않게 한다.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

COMMA = Path(
    "/Users/supper/Desktop/daycon/블랙박스 영상 기반 지능형 고의사고 분석 모델 AI 경진대회/외부데이터/comma2k19/comma2k19"
)
EXTRACT_ROOT = Path(
    "/private/tmp/claude-501/-Users-supper-Desktop-daycon---------------------------AI-----/f7b3750b-fafc-4e9b-ab44-48198fafc199/scratchpad/comma2k19_extract"
)
ROOT = Path(__file__).resolve().parent
OUT_VIDEOS = ROOT / "data" / "videos"
OUT_LABELS = ROOT / "data" / "labels.csv"

STOP_SPEED_MS = 0.5  # 이하면 STOPPED
ACCEL_TH = 0.3  # m/s^2, 이 이상이면 ACCELERATING/DECELERATING
ACCEL_WINDOW_S = 0.5  # 종가속도 계산용 중심차분 반경(초)
STEER_TH_DEG = 6.0  # 이 각도 이하는 STRAIGHT
STEER_SMOOTH_WIN = 3  # 10Hz 기준 0.3초 이동평균

OUT_WIDTH = 480  # 재인코딩 시 가로 해상도(용량 절감, 세부 화질은 학습 crop에서 커버)


def valid_chunks() -> list[int]:
    chunks = []
    for n in range(1, 11):
        p = COMMA / f"Chunk_{n}.zip"
        if p.exists():
            chunks.append(n)
    return chunks


def list_segments(zf: zipfile.ZipFile, chunk: int) -> list[str]:
    seg_dirs = set()
    prefix = f"Chunk_{chunk}/"
    for n in zf.namelist():
        if n.startswith(prefix) and n.endswith("/video.hevc"):
            seg_dirs.add(n[: -len("/video.hevc")])
    return sorted(seg_dirs)


def make_labels(speed_t, speed_v, steer_t, steer_v, frame_times):
    # 20fps -> 10Hz: 짝수 인덱스만 사용
    sel_times = frame_times[0::2]
    n = len(sel_times)

    speed_i = np.interp(sel_times, speed_t, speed_v)
    steer_i = np.interp(sel_times, steer_t, steer_v)

    # 조향각 노이즈 완화용 이동평균(대칭 윈도우)
    if n >= STEER_SMOOTH_WIN:
        kernel = np.ones(STEER_SMOOTH_WIN) / STEER_SMOOTH_WIN
        steer_smooth = np.convolve(steer_i, kernel, mode="same")
    else:
        steer_smooth = steer_i

    # 종가속도: t[k+w] - t[k-w] 사이의 speed 변화를 중심차분 (경계는 편측차분)
    accel = np.zeros(n)
    for k in range(n):
        t0 = sel_times[k] - ACCEL_WINDOW_S
        t1 = sel_times[k] + ACCEL_WINDOW_S
        v0 = np.interp(t0, sel_times, speed_i)
        v1 = np.interp(t1, sel_times, speed_i)
        dt = t1 - t0
        accel[k] = (v1 - v0) / dt if dt > 0 else 0.0

    accel_labels, steer_labels = [], []
    for k in range(n):
        if speed_i[k] < STOP_SPEED_MS:
            accel_labels.append("STOPPED")
        elif accel[k] > ACCEL_TH:
            accel_labels.append("ACCELERATING")
        elif accel[k] < -ACCEL_TH:
            accel_labels.append("DECELERATING")
        else:
            accel_labels.append("CONSTANT")

        angle = steer_smooth[k]
        if abs(angle) <= STEER_TH_DEG:
            steer_labels.append("STRAIGHT")
        elif angle < 0:
            steer_labels.append("LEFT")  # 음수 = LEFT (영상으로 검증된 부호)
        else:
            steer_labels.append("RIGHT")

    return accel_labels, steer_labels, n


def encode_10hz_video(src_hevc: Path, dst_mp4: Path, n_frames_20fps: int):
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    # 짝수 프레임만 선택(20fps->10Hz), 가로 OUT_WIDTH로 축소, 10fps로 저장
    vf = f"select='not(mod(n\\,2))',scale={OUT_WIDTH}:-2,setpts=N/(10*TB)"
    cmd = [
        "ffmpeg", "-y", "-i", str(src_hevc),
        "-vf", vf,
        "-r", "10",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-an", "-loglevel", "error",
        str(dst_mp4),
    ]
    subprocess.run(cmd, check=True)


def process_chunk(chunk: int, rows: list, limit: int | None = None):
    zip_path = COMMA / f"Chunk_{chunk}.zip"
    extract_dir = EXTRACT_ROOT / f"Chunk_{chunk}"
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        segments = list_segments(zf, chunk)
        if limit is not None:
            segments = segments[:limit]
        print(f"Chunk_{chunk}: {len(segments)}개 세그먼트 처리 시작")

        for i, seg in enumerate(segments, 1):
            members = [
                f"{seg}/video.hevc",
                f"{seg}/global_pose/frame_times",
                f"{seg}/processed_log/CAN/speed/t",
                f"{seg}/processed_log/CAN/speed/value",
                f"{seg}/processed_log/CAN/steering_angle/t",
                f"{seg}/processed_log/CAN/steering_angle/value",
            ]
            zf.extractall(extract_dir, members=members)

            seg_path = extract_dir / seg
            seg_id = f"c2k19_{chunk}_" + seg.split("/", 1)[1].replace("/", "_").replace("|", "-")

            try:
                speed_t = np.load(seg_path / "processed_log/CAN/speed/t").ravel()
                speed_v = np.load(seg_path / "processed_log/CAN/speed/value").ravel()
                steer_t = np.load(seg_path / "processed_log/CAN/steering_angle/t").ravel()
                steer_v = np.load(seg_path / "processed_log/CAN/steering_angle/value").ravel()
                frame_times = np.load(seg_path / "global_pose/frame_times").ravel()
            except Exception as e:
                print(f"  [{i}/{len(segments)}] {seg_id}: 신호 로드 실패 ({e}) -> 스킵")
                shutil.rmtree(seg_path, ignore_errors=True)
                continue

            if len(frame_times) < 20 or len(speed_v) < 20:
                shutil.rmtree(seg_path, ignore_errors=True)
                continue

            accel_labels, steer_labels, n = make_labels(speed_t, speed_v, steer_t, steer_v, frame_times)

            dst_mp4 = OUT_VIDEOS / f"{seg_id}.mp4"
            try:
                encode_10hz_video(seg_path / "video.hevc", dst_mp4, len(frame_times))
            except subprocess.CalledProcessError as e:
                print(f"  [{i}/{len(segments)}] {seg_id}: 인코딩 실패 ({e}) -> 스킵")
                shutil.rmtree(seg_path, ignore_errors=True)
                continue

            for k in range(n):
                rows.append({
                    "ID": seg_id,
                    "sample_index": k,
                    "frame_index": k,
                    "time_seconds": round(k * 0.1, 2),
                    "accel_label": accel_labels[k],
                    "steer_label": steer_labels[k],
                })

            shutil.rmtree(seg_path, ignore_errors=True)
            if i % 20 == 0 or i == len(segments):
                print(f"  [{i}/{len(segments)}] 완료 (누적 라벨 {len(rows)}행)")

    shutil.rmtree(extract_dir, ignore_errors=True)


FIELDNAMES = ["ID", "sample_index", "frame_index", "time_seconds", "accel_label", "steer_label"]


def _save(rows: list):
    with open(OUT_LABELS, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT_VIDEOS.mkdir(parents=True, exist_ok=True)

    rows: list = []
    done_chunks = set()
    if OUT_LABELS.exists():
        with open(OUT_LABELS, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            # ID 형식: c2k19_{chunk}_...
            done_chunks.add(int(r["ID"].split("_")[1]))
        print(f"기존 라벨 {len(rows)}행 로드 (이미 처리된 청크: {sorted(done_chunks)})")

    chunks = valid_chunks()
    print(f"사용 가능한 청크: {chunks}")

    for chunk in chunks:
        if chunk in done_chunks:
            print(f"Chunk_{chunk}: 이미 처리됨 -> 스킵")
            continue
        process_chunk(chunk, rows)
        _save(rows)  # 청크 하나 끝날 때마다 저장(중간에 끊겨도 안전)
        print(f"Chunk_{chunk} 처리 후 누적 {len(rows)}행 저장 완료")

    print(f"\n총 {len(rows)}행 저장 완료 -> {OUT_LABELS}")
    from collections import Counter
    print("accel_label 분포:", Counter(r["accel_label"] for r in rows))
    print("steer_label 분포:", Counter(r["steer_label"] for r in rows))


if __name__ == "__main__":
    main()
