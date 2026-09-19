"""Stage 1 프레임 분류기 학습 (EfficientNet-B0, ImageNet 사전학습).

build_dataset.py가 만든 data/labels.csv 를 읽어 train/val로 학습하고,
official_check(대회 공식 제공 5+5 샘플)로 최종 점검한다.
평가 서버는 인터넷이 차단되므로, 여기서 받은 ImageNet 가중치는 학습에만
쓰고 inference.py에는 최종 state_dict 체크포인트만 담아 weights=None +
load_state_dict 로 불러오게 한다.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_OUT = Path(os.environ.get("STAGE1_MODEL_OUT", ROOT.parent / "model" / "stage1"))  # 실제 제출 구조와 동일한 경로
IMG_SIZE = 224
CROP_FROM = 256
LABELS = ["ORIGINAL", "RERECORDED"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
EPOCHS = int(os.environ.get("STAGE1_EPOCHS", 2))  # 2차 실험: 1차 실험에서 epoch1 이후 val이 계속 나빠졌으므로 1~2로 제한
BATCH_SIZE = int(os.environ.get("STAGE1_BATCH_SIZE", 32))
LR = float(os.environ.get("STAGE1_LR", 3e-4))
SEED = 20260907

# 모델 아키텍처 선택:
#   efficientnet - 기존 방식(EfficientNet-B0, ImageNet 사전학습)
#   xception     - timm의 Xception. 딥페이크/recapture 탐지에서 널리 쓰이는 백본으로,
#                  depthwise-separable conv가 모아레·압축흔적 같은 미세한 주파수 패턴을
#                  EfficientNet보다 잘 잡는다는 보고가 많음
#   srm_dual     - RGB 스트림(EfficientNet-B0) + 고정 고역통과 필터(SRM 계열)로 뽑은
#                  노이즈 잔차 스트림(경량 CNN)을 합치는 이중 스트림. "이미지 내용"이
#                  아니라 "촬영/재녹화 과정의 흔적"을 직접 겨냥하는 구조
STAGE1_ARCH = os.environ.get("STAGE1_ARCH", "efficientnet").lower()
if STAGE1_ARCH not in ("efficientnet", "xception", "srm_dual"):
    raise ValueError(f"알 수 없는 STAGE1_ARCH: {STAGE1_ARCH} (efficientnet/xception/srm_dual 중 하나)")

# 3차 실험(class-ratio ablation): 클립당 RERECORDED 변형이 3개라 데이터셋 자체가
# ORIGINAL:RERECORDED = 1:3으로 치우쳐 있다(build_dataset_ccd.py는 그대로 둠).
# 데이터셋을 다시 만들지 않고, train 배치 샘플링만 WeightedRandomSampler로
# 보정해 "class prior"만 분리해서 본다. val/official은 실제 분포를 그대로
# 반영해야 하므로 건드리지 않는다.
USE_BALANCED_SAMPLER = True
TARGET_ORIGINAL_RATIO = 1 / 3  # 4차 실험: 1:2 (3차의 1:1이 과했을 가능성을 확인하는 중간값)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool):
        self.rows = rows
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(DATA / row["path"]).convert("RGB")
        image = np.asarray(image)
        h, w = image.shape[:2]
        if self.train:
            # RandomCrop(224) from 256 + 좌우 flip
            top = random.randint(0, max(0, h - IMG_SIZE))
            left = random.randint(0, max(0, w - IMG_SIZE))
            image = image[top : top + IMG_SIZE, left : left + IMG_SIZE]
            if random.random() < 0.5:
                image = image[:, ::-1]
        else:
            top, left = (h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2
            image = image[top : top + IMG_SIZE, left : left + IMG_SIZE]
        x = torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255.0
        x = (x - MEAN) / STD
        y = LABEL2IDX[row["label"]]
        return x, y


def _load_rows(split: str) -> list[dict]:
    with open(DATA / "labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    return rows


class _SRMResidual(nn.Module):
    """고정(학습되지 않는) 고역통과 필터 3종으로 촬영/재녹화 과정의 노이즈 잔차를 추출.

    표준 SRM(Spatial Rich Model, 이미지 포렌식에서 흔히 쓰는 30여개 필터 뱅크)을
    단순화해 대표적인 1차/2차 고역통과 커널 3개만 사용한다. RGB 각 채널에 독립적으로
    적용(depthwise)해 3채널 -> 9채널 잔차 맵을 만든다.
    """

    def __init__(self):
        super().__init__()
        k1 = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
        k2 = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        k3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        kernels = torch.tensor([k1, k2, k3], dtype=torch.float32) / 4.0  # (3,3,3)
        weight = kernels.repeat(3, 1, 1).unsqueeze(1)  # (9,1,3,3): 입력채널 3개 x 필터 3개
        self.conv = nn.Conv2d(3, 9, kernel_size=3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class Stage1SRMDualStreamModel(nn.Module):
    """RGB 스트림(EfficientNet-B0, 사전학습) + 노이즈 잔차 스트림(경량 CNN, 처음부터 학습)."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.srm = _SRMResidual()
        rgb_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        rgb_dim = rgb_backbone.classifier[1].in_features
        rgb_backbone.classifier = nn.Identity()
        self.rgb_backbone = rgb_backbone
        self.noise_backbone = nn.Sequential(
            nn.Conv2d(9, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.classifier = nn.Linear(rgb_dim + 128, num_labels)

    def forward(self, x):
        rgb_feat = self.rgb_backbone(x)
        noise_feat = self.noise_backbone(self.srm(x))
        return self.classifier(torch.cat([rgb_feat, noise_feat], dim=1))


def build_stage1_model(arch: str) -> nn.Module:
    if arch == "efficientnet":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
        return model
    if arch == "xception":
        import timm
        return timm.create_model("legacy_xception", pretrained=True, num_classes=len(LABELS))
    if arch == "srm_dual":
        return Stage1SRMDualStreamModel(len(LABELS))
    raise ValueError(f"알 수 없는 arch: {arch}")


def _run_epoch(model, loader, device, optimizer=None) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_labels = [], []
    total_loss = 0.0
    context = torch.enable_grad() if train_mode else torch.inference_mode()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * len(y)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(all_labels), macro_f1


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device()
    print("device:", device, "| arch:", STAGE1_ARCH)

    train_rows, val_rows, official_rows = _load_rows("train"), _load_rows("val"), _load_rows("official_check")
    print(f"train={len(train_rows)} val={len(val_rows)} official_check={len(official_rows)}")

    if USE_BALANCED_SAMPLER:
        n_orig = sum(1 for r in train_rows if r["label"] == "ORIGINAL")
        n_rerec = len(train_rows) - n_orig
        w_orig = TARGET_ORIGINAL_RATIO / n_orig
        w_rerec = (1 - TARGET_ORIGINAL_RATIO) / n_rerec
        sample_weights = [w_orig if r["label"] == "ORIGINAL" else w_rerec for r in train_rows]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_rows), replacement=True)
        print(f"balanced sampler: target ORIGINAL:RERECORDED = {TARGET_ORIGINAL_RATIO:.2f}:{1-TARGET_ORIGINAL_RATIO:.2f} "
              f"(원본 카운트 ORIGINAL={n_orig} RERECORDED={n_rerec})")
        train_loader = DataLoader(FrameDataset(train_rows, True), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(FrameDataset(train_rows, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(FrameDataset(val_rows, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    official_loader = DataLoader(FrameDataset(official_rows, False), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_stage1_model(STAGE1_ARCH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1 = _run_epoch(model, train_loader, device, optimizer)
        val_loss, val_f1 = _run_epoch(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:02d}  train loss {train_loss:.4f} f1 {train_f1:.4f}  |  val loss {val_loss:.4f} f1 {val_f1:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"model": model.state_dict(), "arch": STAGE1_ARCH, "img_size": IMG_SIZE, "labels": LABELS},
                MODEL_OUT / "best.pt",
            )
            print(f"  -> saved best.pt (val f1 {val_f1:.4f})")

    # 최종 점검: 공식 제공 5+5 샘플
    checkpoint = torch.load(MODEL_OUT / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, official_f1 = _run_epoch(model, official_loader, device)
    print(f"\n공식 5+5 샘플 점검 macro-F1: {official_f1:.4f} (씬 일부 중복 가능 — 참고용 스모크테스트)")


if __name__ == "__main__":
    main()
