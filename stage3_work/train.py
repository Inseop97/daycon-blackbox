"""Stage3(가감속/조향) 학습 스크립트 — 로컬(Mac/MPS)과 GPU 서버(CUDA) 겸용.

처음엔 베이스라인처럼 mvit_v2_s류 3D-CNN/비디오 트랜스포머(r2plus1d_18)로
시도했으나, MPS(CUDA 없는 Mac)에서 벤치마크해보니 3D conv 자체가 근본적으로
느려서(batch=8, 16프레임 학습 스텝 하나에 ~4.7초) 현실적이지 않았다. 대신
EfficientNet-B0(2D CNN)을 프레임별로 적용하고 클립 내 프레임 특징을 평균
풀링(TSN 스타일)해서 시간 축을 합치는 구조로 교체했다.

GPU 서버(CUDA)에서 돌릴 때는 아래 값들을 환경변수로 조정할 수 있다(모두
선택사항 — 안 주면 CUDA 여부에 따라 적당한 기본값을 자동으로 씀):
  STAGE3_DATA_DIR   데이터 폴더 경로 (기본: 이 스크립트 옆 data/)
  STAGE3_BATCH_SIZE, STAGE3_EPOCHS, STAGE3_LR, STAGE3_NUM_WORKERS
  STAGE3_CLIP_LEN, STAGE3_CLIP_STRIDE, STAGE3_MAX_TRAIN_VIDEOS, STAGE3_MAX_CACHE_VIDEOS

예) GPU 서버에서: STAGE3_BATCH_SIZE=48 STAGE3_NUM_WORKERS=8 python3 train.py
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

cv2.setNumThreads(1)  # DataLoader worker 여러 개일 때 cv2 내부 스레드와 경합 방지

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STAGE3_DATA_DIR", ROOT / "data"))
MODEL_OUT = Path(os.environ.get("STAGE3_MODEL_OUT", ROOT.parent / "model" / "stage3"))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


_DEVICE = _device()
_ON_CUDA = _DEVICE.type == "cuda"
if _ON_CUDA:
    torch.backends.cudnn.benchmark = True

CLIP_LEN = int(os.environ.get("STAGE3_CLIP_LEN", 16 if _ON_CUDA else 8))
# 클립 중심 간 간격(윈도우 개수 제어, 초 단위 아님 — 10Hz 프레임 인덱스 기준)
CLIP_STRIDE = int(os.environ.get("STAGE3_CLIP_STRIDE", 8 if _ON_CUDA else 15))
IMG_SIZE = 224  # EfficientNet-B0 사전학습 기준
EPOCHS = int(os.environ.get("STAGE3_EPOCHS", 5))
BATCH_SIZE = int(os.environ.get("STAGE3_BATCH_SIZE", 48 if _ON_CUDA else 8))
NUM_WORKERS = int(os.environ.get("STAGE3_NUM_WORKERS", 8 if _ON_CUDA else 0))
LR = float(os.environ.get("STAGE3_LR", 1e-4))
SEED = 20260913
VAL_RATIO = 0.15
_max_train_videos = os.environ.get("STAGE3_MAX_TRAIN_VIDEOS", "")
MAX_TRAIN_VIDEOS = int(_max_train_videos) if _max_train_videos else None  # None=전체 사용
# 영상 캐시 상한(224px 기준 영상당 ~115MB). DataLoader worker마다 Dataset 인스턴스가
# 복제되어 캐시도 worker별로 따로 쌓이므로(총 메모리 = 이 값 x worker 수), worker 수로
# 나눠서 "전체" 캐시 예산을 맞춘다. GPU 서버는 RAM이 넉넉해 총 예산을 크게 잡는다.
_TOTAL_CACHE_BUDGET = int(os.environ.get("STAGE3_MAX_CACHE_VIDEOS", 300 if _ON_CUDA else 80))
MAX_CACHE_VIDEOS = max(5, _TOTAL_CACHE_BUDGET // max(1, NUM_WORKERS))

# 시간축을 어떻게 합칠지 선택:
#   tsn  - EfficientNet-B0 프레임 특징을 평균 풀링(순서 정보 버림, 기존 방식)
#   gru  - 위와 동일한 프레임 특징을 GRU로 순서까지 반영해 합침 (연산량 거의 그대로)
#   x3d  - pytorchvideo의 X3D(모션 특화 3D-CNN). CUDA 없이는 비현실적으로 느리므로
#          GPU 서버 전용 — 로컬(MPS/CPU)에서 고르면 에러로 막는다.
STAGE3_ARCH = os.environ.get("STAGE3_ARCH", "tsn").lower()
if STAGE3_ARCH not in ("tsn", "gru", "x3d"):
    raise ValueError(f"알 수 없는 STAGE3_ARCH: {STAGE3_ARCH} (tsn/gru/x3d 중 하나)")
if STAGE3_ARCH == "x3d" and not _ON_CUDA:
    raise RuntimeError(
        "STAGE3_ARCH=x3d 는 3D conv라 CUDA 없이는 비현실적으로 느립니다"
        "(mvit_v2_s/r2plus1d_18과 같은 이유로 이 저장소에서 이미 확인함). "
        "GPU 서버(CUDA)에서만 사용하세요."
    )

ACCEL_LABELS = ["CONSTANT", "ACCELERATING", "DECELERATING", "STOPPED"]
STEER_LABELS = ["STRAIGHT", "LEFT", "RIGHT"]
ACCEL2IDX = {l: i for i, l in enumerate(ACCEL_LABELS)}
STEER2IDX = {l: i for i, l in enumerate(STEER_LABELS)}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_label_rows() -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = defaultdict(list)
    with open(DATA / "labels.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_id[r["ID"]].append(r)
    for vid in by_id:
        by_id[vid].sort(key=lambda r: int(r["sample_index"]))
    return by_id


class ClipDataset(Dataset):
    """영상 하나당 CLIP_LEN 프레임 윈도우 여러 개를 (center, 라벨) 형태로 미리 뽑아둔다.

    영상마다 매 클립마다 VideoCapture를 새로 열고 seek하면 x264 GOP 구조 때문에
    느려서, 영상 전체를 한 번만 순차 디코딩해 캐싱해둔다(프로세스 생존 동안 유지).
    """

    def __init__(self, video_ids: list[str], by_id: dict[str, list[dict]], train: bool):
        self.train = train
        self.items = []  # (video_id, center_sample_index, rows)
        for vid in video_ids:
            rows = by_id[vid]
            n = len(rows)
            for center in range(0, n, CLIP_STRIDE):
                self.items.append((vid, center, rows))
        # LRU 캐시: 청크 확장 후 영상 수가 많아지면(수백~천 개) 전부 메모리에 못 올리므로
        # 최근 사용한 MAX_CACHE_VIDEOS개만 유지하고 오래된 것부터 내보낸다.
        self._frame_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()

    def __len__(self):
        return len(self.items)

    def sample_weights(self) -> list[float]:
        """(accel_label, steer_label) 조합의 빈도 역수로 가중치를 매긴다.

        위키 Q&A로 macro-F1이 "정답에 등장한 클래스만"이 아니라 "정의된 전체
        클래스"를 항상 기준으로 계산된다는 걸 확인했다 — 즉 소수 클래스
        (ACCELERATING/DECELERATING/STOPPED, LEFT/RIGHT) F1이 0에 가까우면
        macro-F1이 크게 무너진다. Stage1/2는 이미 balanced sampler를 썼는데
        Stage3만 안 썼던 게 실제 제출 점수(0.188)가 낮았던 주요 원인으로
        의심됨 — 이 가중치로 train 시 소수 클래스를 더 자주 보게 한다.
        """
        counts: dict[tuple[str, str], int] = defaultdict(int)
        keys = []
        for _, center, rows in self.items:
            key = (rows[center]["accel_label"], rows[center]["steer_label"])
            keys.append(key)
            counts[key] += 1
        return [1.0 / counts[k] for k in keys]

    def _frames_for(self, vid: str) -> np.ndarray:
        cached = self._frame_cache.get(vid)
        if cached is not None:
            self._frame_cache.move_to_end(vid)
            return cached
        cap = cv2.VideoCapture(str(DATA / "videos" / f"{vid}.mp4"))
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        h, w = frames[0].shape[:2]
        scale = IMG_SIZE / min(h, w)
        nh, nw = round(h * scale), round(w * scale)
        arr = np.stack([cv2.resize(f, (nw, nh)) for f in frames])
        self._frame_cache[vid] = arr
        if len(self._frame_cache) > MAX_CACHE_VIDEOS:
            self._frame_cache.popitem(last=False)
        return arr

    def __getitem__(self, index):
        vid, center, rows = self.items[index]
        n = len(rows)
        idx = np.clip(center - CLIP_LEN // 2 + np.arange(CLIP_LEN), 0, n - 1)
        row = rows[center]
        accel_y = ACCEL2IDX[row["accel_label"]]
        steer_y = STEER2IDX[row["steer_label"]]

        full = self._frames_for(vid)  # (N, nh, nw, 3)
        idx = np.clip(idx, 0, len(full) - 1)
        frames = full[idx]

        nh, nw = frames.shape[1:3]
        top = random.randint(0, max(0, nh - IMG_SIZE)) if self.train else (nh - IMG_SIZE) // 2
        left = random.randint(0, max(0, nw - IMG_SIZE)) if self.train else (nw - IMG_SIZE) // 2
        flip = self.train and random.random() < 0.5

        clip = frames[:, top: top + IMG_SIZE, left: left + IMG_SIZE, :]
        if flip:
            clip = clip[:, :, ::-1, :]

        x = torch.from_numpy(clip.copy()).permute(0, 3, 1, 2).float() / 255.0  # (T,3,H,W)
        x = (x - MEAN) / STD
        return x, accel_y, steer_y


def _effnet_frame_features(x: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    """(B, T, 3, H, W) -> 프레임마다 backbone 적용 -> (B, T, dim).

    MPS 백엔드에서 (B*T)로 합친 배치를 그대로 backbone에 넣으면 역전파 중
    "view size is not compatible..." 에러가 남(torchvision 내부 view() 이슈로
    추정) -> contiguous()로 명시적 메모리 복사를 강제해 우회한다.
    """
    b, t = x.shape[:2]
    feat = backbone(x.reshape(b * t, *x.shape[2:]).contiguous())
    return feat.reshape(b, t, -1).contiguous()


class Stage3TSNModel(nn.Module):
    """프레임별 EfficientNet-B0 특징 -> 시간축 평균 풀링(TSN 스타일) -> 분류 헤드 2개.

    구현이 가장 단순하고 빠르지만, 평균은 프레임 "순서" 정보를 버린다 —
    STOPPED/CONSTANT 구분(변화량)이나 LEFT/RIGHT(진행 방향)처럼 순서가 중요한
    이 과제엔 구조적으로 불리하다(Stage3GRUModel 참고).
    """

    def __init__(self):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.accel = nn.Linear(dim, len(ACCEL_LABELS))
        self.steer = nn.Linear(dim, len(STEER_LABELS))

    def forward(self, x):
        feat = _effnet_frame_features(x, self.backbone).mean(dim=1)
        return self.accel(feat), self.steer(feat)


class Stage3GRUModel(nn.Module):
    """프레임별 EfficientNet-B0 특징을 GRU에 순서대로 흘려보내 시간 정보를 보존.

    TSN(평균 풀링) 대비 연산량 증가는 미미하다(GRU 자체는 가벼움) — 백본이
    프레임마다 도는 비용은 동일하고, 그 위에 작은 시퀀스 모델만 얹는 구조.
    """

    def __init__(self, hidden=256):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.gru = nn.GRU(dim, hidden, batch_first=True, bidirectional=True)
        self.accel = nn.Linear(hidden * 2, len(ACCEL_LABELS))
        self.steer = nn.Linear(hidden * 2, len(STEER_LABELS))

    def forward(self, x):
        feat_seq = _effnet_frame_features(x, self.backbone)  # (B, T, dim)
        out, _ = self.gru(feat_seq)
        # 양방향 GRU의 마지막 시점 forward-hidden + 첫 시점 backward-hidden을
        # 이어붙여 "클립 전체를 훑은 요약"으로 사용 (표준적인 BiGRU 풀링 방식)
        h = out.shape[-1] // 2
        pooled = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)
        return self.accel(pooled), self.steer(pooled)


class Stage3X3DModel(nn.Module):
    """pytorchvideo의 X3D-S — 모션 특화 3D-CNN, SlowFast/I3D보다 훨씬 가볍게
    설계되었지만 그래도 3D conv라 CUDA 없이는 비현실적으로 느리다(모듈 로드
    시점에 STAGE3_ARCH=x3d + CUDA 없음 조합을 막아둠). GPU 서버 전용.
    """

    def __init__(self, clip_len: int, img_size: int):
        super().__init__()
        from pytorchvideo.models.x3d import create_x3d

        backbone = create_x3d(input_clip_length=clip_len, input_crop_size=img_size, model_num_class=400)
        head = backbone.blocks[-1]
        dim = head.proj.in_features
        head.proj = nn.Identity()
        head.activation = nn.Identity()  # softmax 제거, raw pooled feature만 사용
        self.backbone = backbone
        self.accel = nn.Linear(dim, len(ACCEL_LABELS))
        self.steer = nn.Linear(dim, len(STEER_LABELS))

    def forward(self, x):
        # x: (B, T, 3, H, W) -> X3D는 (B, 3, T, H, W) 채널-우선 순서를 기대
        feat = self.backbone(x.permute(0, 2, 1, 3, 4).contiguous()).flatten(1)
        return self.accel(feat), self.steer(feat)


def build_stage3_model(arch: str, clip_len: int = CLIP_LEN, img_size: int = IMG_SIZE) -> nn.Module:
    if arch == "tsn":
        return Stage3TSNModel()
    if arch == "gru":
        return Stage3GRUModel()
    if arch == "x3d":
        return Stage3X3DModel(clip_len, img_size)
    raise ValueError(f"알 수 없는 arch: {arch}")


def _run_epoch(model, loader, device, optimizer=None, scaler=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    a_preds, a_labels, s_preds, s_labels = [], [], [], []
    total_loss = 0.0
    n_total = 0
    context = torch.enable_grad() if train_mode else torch.inference_mode()
    with context:
        for x, ay, sy in loader:
            x = x.to(device, non_blocking=_ON_CUDA)
            ay = ay.to(device, non_blocking=_ON_CUDA)
            sy = sy.to(device, non_blocking=_ON_CUDA)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=_ON_CUDA):
                a_logits, s_logits = model(x)
                loss = nn.functional.cross_entropy(a_logits, ay) + nn.functional.cross_entropy(s_logits, sy)
            if train_mode:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.detach()) * len(ay)
            n_total += len(ay)
            a_preds.extend(a_logits.argmax(1).cpu().tolist())
            a_labels.extend(ay.cpu().tolist())
            s_preds.extend(s_logits.argmax(1).cpu().tolist())
            s_labels.extend(sy.cpu().tolist())
    a_f1 = f1_score(a_labels, a_preds, average="macro")
    s_f1 = f1_score(s_labels, s_preds, average="macro")
    return total_loss / n_total, a_f1, s_f1, a_labels, a_preds, s_labels, s_preds


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = _DEVICE
    print("device:", device, "| arch:", STAGE3_ARCH, "batch_size:", BATCH_SIZE, "clip_len:", CLIP_LEN,
          "num_workers:", NUM_WORKERS, "cache_cap:", MAX_CACHE_VIDEOS)

    by_id = _load_label_rows()
    video_ids = sorted(by_id.keys())
    rng = random.Random(SEED)
    rng.shuffle(video_ids)
    video_ids = video_ids[:MAX_TRAIN_VIDEOS]
    n_val = max(1, int(len(video_ids) * VAL_RATIO))
    val_ids, train_ids = video_ids[:n_val], video_ids[n_val:]
    print(f"영상 수: train={len(train_ids)} val={len(val_ids)}")

    train_ds = ClipDataset(train_ids, by_id, train=True)
    val_ds = ClipDataset(val_ids, by_id, train=False)
    print(f"클립 수: train={len(train_ds)} val={len(val_ds)}")

    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_ds.sample_weights(), num_samples=len(train_ds), replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=NUM_WORKERS,
        pin_memory=_ON_CUDA, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=_ON_CUDA, persistent_workers=NUM_WORKERS > 0,
    )

    model = build_stage3_model(STAGE3_ARCH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=_ON_CUDA)

    best_score = -1.0
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_a_f1, train_s_f1, *_ = _run_epoch(model, train_loader, device, optimizer, scaler)
        val_loss, val_a_f1, val_s_f1, val_a_y, val_a_p, val_s_y, val_s_p = _run_epoch(model, val_loader, device)
        scheduler.step()
        score = (val_a_f1 + val_s_f1) / 2
        print(
            f"epoch {epoch:02d} train loss {train_loss:.4f} (accel_f1 {train_a_f1:.3f} steer_f1 {train_s_f1:.3f})"
            f"  |  val loss {val_loss:.4f} accel_f1 {val_a_f1:.3f} steer_f1 {val_s_f1:.3f}"
        )
        # 클래스별 F1 — balanced sampler가 소수 클래스(ACCELERATING/DECELERATING/
        # STOPPED, LEFT/RIGHT)를 실제로 끌어올리는지 확인하기 위한 진단용 출력.
        a_f1_per_class = f1_score(val_a_y, val_a_p, average=None, labels=range(len(ACCEL_LABELS)), zero_division=0)
        s_f1_per_class = f1_score(val_s_y, val_s_p, average=None, labels=range(len(STEER_LABELS)), zero_division=0)
        print("  accel 클래스별 F1:", dict(zip(ACCEL_LABELS, [round(v, 3) for v in a_f1_per_class])))
        print("  steer 클래스별 F1:", dict(zip(STEER_LABELS, [round(v, 3) for v in s_f1_per_class])))
        if score >= best_score:
            best_score = score
            torch.save(
                {"model": model.state_dict(), "arch": STAGE3_ARCH, "clip_len": CLIP_LEN, "img_size": IMG_SIZE,
                 "accel_labels": ACCEL_LABELS, "steer_labels": STEER_LABELS},
                MODEL_OUT / "best.pt",
            )
            print(f"  -> saved best.pt (val avg macro-F1 {score:.4f})")


if __name__ == "__main__":
    main()
