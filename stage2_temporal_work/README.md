# Stage 2 DINOv2 + TCN GPU pilot

수작업으로 확정한 Nexar/CCD 영상에서 네 Stage 2 출력을 함께 학습한다.

- frame encoder: frozen DINOv2 ViT-S/14
- temporal model: 4-block TCN
- temporal outputs: collision_frame, entry_frame
- classification outputs: entry_side, evasion_space

DINOv2 특징은 한 번 추출해 cache에 저장한다. 최종 체크포인트에는 DINOv2
가중치도 포함되므로 제출 추론은 인터넷 연결 없이 동작한다.

## GPU 서버 준비

```bash
git clone https://github.com/Inseop97/daycon-blackbox.git
cd daycon-blackbox
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

데이터는 Git과 분리해 배치한다.

```text
/data/daycon-stage2/
├── labels/
│   ├── train_labels.csv
│   ├── rejected_rows.csv
│   └── SHA256SUMS.txt
└── videos/
    ├── nexar/*.mp4
    └── ccd/*.mp4
```

## 10개 스모크 테스트

```bash
python -m stage2_temporal_work.train \
  --labels /data/daycon-stage2/labels/train_labels.csv \
  --nexar-dir /data/daycon-stage2/videos/nexar \
  --ccd-dir /data/daycon-stage2/videos/ccd \
  --cache-dir /data/daycon-stage2/cache/dinov2_5fps_smoke \
  --output-dir /data/daycon-stage2/runs/smoke \
  --feature-batch-size 32 \
  --batch-size 2 \
  --num-workers 2 \
  --smoke
```

## 전체 학습

```bash
python -m stage2_temporal_work.train \
  --labels /data/daycon-stage2/labels/train_labels.csv \
  --nexar-dir /data/daycon-stage2/videos/nexar \
  --ccd-dir /data/daycon-stage2/videos/ccd \
  --cache-dir /data/daycon-stage2/cache/dinov2_5fps \
  --output-dir /data/daycon-stage2/runs/dinov2_tcn_v1 \
  --target-fps 5 \
  --feature-batch-size 64 \
  --batch-size 4 \
  --num-workers 2 \
  --epochs 20 \
  --patience 4
```

OOM이면 feature-batch-size를 32, 이어서 16으로만 낮춘다. 캐시는 영상별로
즉시 저장되므로 같은 명령을 재실행하면 완료된 영상은 건너뛴다.

결과는 best.pt, best_val_predictions.csv, history.json,
split_manifest.csv로 저장된다.

## 제출 모델 설치

검증 결과를 확인한 다음 한 명령으로 체크포인트를 설치하고 전체 제출 파일을 만든다.

```bash
python -m stage2_temporal_work.finalize_submit \
  --checkpoint /data/daycon-stage2/runs/dinov2_tcn_v1/best.pt \
  --artifact-dir /data/daycon-stage2/artifacts
```

이 명령은 기존 Stage 1/2/3 가중치와 YOLOP 코드에 temporal_best.pt를 추가한 뒤,
저장소 루트의 build_submit_zip.py로 전체 submit.zip을 다시 만든다. 회수할 파일은
다음 폴더에 모인다.

```text
/data/daycon-stage2/artifacts/
├── submit.zip
├── temporal_best.pt
└── SHA256SUMS.txt
```

inference.py는 temporal_best.pt가 있으면 새 모델을 사용하고, 없으면 기존 Stage 2
방식으로 폴백한다. temporal_best.pt는 Git에서 제외된다.

```bash
rsync -avhP USER@GPU_SERVER:/data/daycon-stage2/artifacts/ ./gpu_stage2_artifacts/
```

validation split은 최초 실행 때 저장되고 이후 재사용된다. test 데이터는 이
학습 코드에서 사용하지 않는다.
