"""evasion_space 학습 (EfficientNet-B0, ImageNet 사전학습).

Stage1 train.py와 동일한 구조를 재사용한다.

STAGE2_ARCH=box_fusion 을 주면 "세그멘테이션/기하 정보 활용" 아이디어를 실용적으로
구현한 이중 분기 모델을 쓴다 — 진짜 주행가능영역 세그멘테이션(YOLOP류)은 사전학습
가중치 구하기가 까다로워서, 대신 entry_frame 작업에서 쓰던 YOLO 차량 탐지로 뽑은
기하 특징(박스 면적/위치/좌우 여백, extract_box_features.py 산출물)을 이미지 특징과
합친다. 먼저 `python3 extract_box_features.py`로 data/box_features.csv를 만들어야 함.
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_OUT = Path(os.environ.get("STAGE2_MODEL_OUT", ROOT.parent / "model" / "stage2" / "evasion_space"))
IMG_SIZE = 224
LABELS = ["0", "1"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}
EPOCHS = int(os.environ.get("STAGE2_EPOCHS", 8))
BATCH_SIZE = int(os.environ.get("STAGE2_BATCH_SIZE", 16))
LR = float(os.environ.get("STAGE2_LR", 3e-4))
SEED = 20260912

# val에서 "공간없음(0)" 재현율이 낮게 나온 것을 보정 — train 샘플링을 1:1로 균형
USE_BALANCED_SAMPLER = True

# efficientnet(기존) / box_fusion(이미지 + YOLO 박스 기하특징 이중 분기)
STAGE2_ARCH = os.environ.get("STAGE2_ARCH", "efficientnet").lower()
if STAGE2_ARCH not in ("efficientnet", "box_fusion"):
    raise ValueError(f"알 수 없는 STAGE2_ARCH: {STAGE2_ARCH} (efficientnet/box_fusion 중 하나)")

BOX_FEATURE_NAMES = [
    "n_vehicles", "largest_area_frac", "largest_cx", "largest_cy",
    "coverage_frac", "gap_left_frac", "gap_right_frac",
    "drivable_frac_total", "drivable_frac_left", "drivable_frac_right", "drivable_frac_near",
]

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_box_features() -> dict[str, list[float]]:
    path = DATA / "box_features.csv"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["vid"]: [float(r[k]) for k in BOX_FEATURE_NAMES] for r in csv.DictReader(f)}


class FrameDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, box_features: dict[str, list[float]]):
        self.rows = rows
        self.train = train
        self.box_features = box_features

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = np.asarray(Image.open(DATA / row["path"]).convert("RGB"))
        h, w = image.shape[:2]
        if self.train:
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
        box = torch.tensor(self.box_features.get(row["vid"], [0.0] * len(BOX_FEATURE_NAMES)), dtype=torch.float32)
        return x, box, y


class Stage2BoxFusionModel(nn.Module):
    """이미지 분기(EfficientNet-B0) + YOLO 박스 기하특징 분기(작은 MLP)를 합쳐 분류."""

    def __init__(self, box_dim: int = len(BOX_FEATURE_NAMES)):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        img_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.box_mlp = nn.Sequential(
            nn.Linear(box_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(img_dim + 32, len(LABELS))

    def forward(self, x, box):
        img_feat = self.backbone(x)
        box_feat = self.box_mlp(box)
        return self.classifier(torch.cat([img_feat, box_feat], dim=1))


def build_stage2_model(arch: str) -> nn.Module:
    if arch == "box_fusion":
        return Stage2BoxFusionModel()
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(LABELS))
    return model


def _load_rows(split: str) -> list[dict]:
    with open(DATA / "labels.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def _run_epoch(model, loader, device, arch, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    all_preds, all_labels = [], []
    total_loss = 0.0
    context = torch.enable_grad() if train_mode else torch.inference_mode()
    with context:
        for x, box, y in loader:
            x, box, y = x.to(device), box.to(device), y.to(device)
            logits = model(x, box) if arch == "box_fusion" else model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += float(loss) * len(y)
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / len(all_labels), macro_f1, all_labels, all_preds


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = _device()
    print("device:", device, "| arch:", STAGE2_ARCH)

    box_features = _load_box_features()
    if STAGE2_ARCH == "box_fusion" and not box_features:
        raise FileNotFoundError(
            "STAGE2_ARCH=box_fusion 이려면 먼저 `python3 extract_box_features.py`로 "
            "data/box_features.csv를 만들어야 합니다."
        )

    train_rows, val_rows = _load_rows("train"), _load_rows("val")
    print(f"train={len(train_rows)} val={len(val_rows)}")

    if USE_BALANCED_SAMPLER:
        n0 = sum(1 for r in train_rows if r["label"] == "0")
        n1 = len(train_rows) - n0
        w0, w1 = 0.5 / n0, 0.5 / n1
        weights = [w0 if r["label"] == "0" else w1 for r in train_rows]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True)
        print(f"balanced sampler: 0={n0} 1={n1} -> 1:1로 샘플링")
        train_loader = DataLoader(FrameDataset(train_rows, True, box_features), batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(FrameDataset(train_rows, True, box_features), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(FrameDataset(val_rows, False, box_features), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_stage2_model(STAGE2_ARCH).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_f1 = -1.0
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_f1, _, _ = _run_epoch(model, train_loader, device, STAGE2_ARCH, optimizer)
        val_loss, val_f1, val_y, val_pred = _run_epoch(model, val_loader, device, STAGE2_ARCH)
        scheduler.step()
        print(f"epoch {epoch:02d}  train loss {train_loss:.4f} f1 {train_f1:.4f}  |  val loss {val_loss:.4f} f1 {val_f1:.4f}")
        if val_f1 >= best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"model": model.state_dict(), "arch": STAGE2_ARCH, "img_size": IMG_SIZE, "labels": LABELS},
                MODEL_OUT / "best.pt",
            )
            cm = confusion_matrix(val_y, val_pred, labels=[0, 1])
            print(f"  -> saved best.pt (val f1 {val_f1:.4f})  confusion matrix (row=true,col=pred)[0,1]:\n{cm}")


if __name__ == "__main__":
    main()
