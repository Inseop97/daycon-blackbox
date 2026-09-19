"""2차(1:3)/3차(1:1)/4차(1:2) Stage1 체크포인트를 동일 기준(같은 val/official_check)으로 재평가해 한 표로 비교한다.

각 체크포인트는 학습 시 서로 다른 class-ratio sampler만 썼을 뿐, 데이터셋
(data/labels.csv)과 합성 레시피는 모두 동일하므로 공정하게 비교 가능하다.
공식 5+5 결과는 진단용으로만 표시하며, 최종 추천은 val/balanced val 기준으로
매긴다.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BACKUP_DIR = ROOT.parent / "model_backup"
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

CHECKPOINTS = [
    ("2차 (1:3, 비보정)", BACKUP_DIR / "stage1_best_v2_ccd_ratio1to3_20260910.pt"),
    ("3차 (1:1 balanced)", BACKUP_DIR / "stage1_best_v3_ccd_sampler1to1_20260910.pt"),
    ("4차 (1:2 balanced)", ROOT.parent / "model" / "stage1" / "best.pt"),  # 이번 실험 직후 아직 백업 전 상태로 호출
]


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(path: Path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    labels = ckpt["labels"]
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, labels


def _predict_prob(model, device, labels, rel_path: str) -> float:
    img = np.asarray(Image.open(DATA / rel_path).convert("RGB"))
    h, w = img.shape[:2]
    top, left = (h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2
    img = img[top : top + IMG_SIZE, left : left + IMG_SIZE]
    x = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
    x = ((x - MEAN) / STD).unsqueeze(0).to(device)
    with torch.inference_mode():
        prob = torch.softmax(model(x), 1)[0].cpu().tolist()
    return prob[labels.index("RERECORDED")]


def _macro_f1_at(y_true, probs, threshold) -> float:
    y_pred = ["RERECORDED" if p >= threshold else "ORIGINAL" for p in probs]
    return f1_score(y_true, y_pred, average="macro", labels=["ORIGINAL", "RERECORDED"])


def evaluate(name: str, ckpt_path: Path, val_rows, official_rows, bal_seed=20260910) -> dict:
    device = _device()
    model, labels = _load_model(ckpt_path, device)

    # val: 클립 단위
    clip_probs, clip_true = defaultdict(list), {}
    for r in val_rows:
        p = _predict_prob(model, device, labels, r["path"])
        clip_probs[r["clip_id"]].append(p)
        clip_true[r["clip_id"]] = r["label"]
    clip_ids = list(clip_probs.keys())
    clip_mean = {c: float(np.mean(clip_probs[c])) for c in clip_ids}
    y_true_c = [clip_true[c] for c in clip_ids]
    probs_c = [clip_mean[c] for c in clip_ids]
    y_pred_c = ["RERECORDED" if p >= 0.5 else "ORIGINAL" for p in probs_c]
    f1_c = f1_score(y_true_c, y_pred_c, average="macro", labels=["ORIGINAL", "RERECORDED"])
    cm_c = confusion_matrix(y_true_c, y_pred_c, labels=["ORIGINAL", "RERECORDED"])
    orig_recall = cm_c[0, 0] / cm_c[0].sum() if cm_c[0].sum() else float("nan")
    rerec_recall = cm_c[1, 1] / cm_c[1].sum() if cm_c[1].sum() else float("nan")

    # balanced val (1:1 서브샘플, 고정 시드로 세 실험 모두 동일 샘플 사용)
    orig_clip_ids = [c for c in clip_ids if clip_true[c] == "ORIGINAL"]
    rerec_clip_ids = [c for c in clip_ids if clip_true[c] == "RERECORDED"]
    bal_rng = random.Random(bal_seed)
    rerec_sample = bal_rng.sample(rerec_clip_ids, min(len(orig_clip_ids), len(rerec_clip_ids)))
    bal_clip_ids = orig_clip_ids + rerec_sample
    y_true_bal = [clip_true[c] for c in bal_clip_ids]
    y_pred_bal = [("RERECORDED" if clip_mean[c] >= 0.5 else "ORIGINAL") for c in bal_clip_ids]
    f1_bal = f1_score(y_true_bal, y_pred_bal, average="macro", labels=["ORIGINAL", "RERECORDED"])

    # val threshold sweep -> best threshold/F1 (val 기준으로만 사용)
    sweep = {t: _macro_f1_at(y_true_c, probs_c, t) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    best_t = max(sweep, key=sweep.get)

    # official 5+5 (진단용)
    off_clip_probs, off_clip_true = defaultdict(list), {}
    for r in official_rows:
        off_clip_probs[r["clip_id"]].append(_predict_prob(model, device, labels, r["path"]))
        off_clip_true[r["clip_id"]] = r["label"]
    off_clip_ids = sorted(off_clip_probs.keys())
    off_mean = {c: float(np.mean(off_clip_probs[c])) for c in off_clip_ids}
    y_true_off = [off_clip_true[c] for c in off_clip_ids]
    probs_off = [off_mean[c] for c in off_clip_ids]
    y_pred_off = ["RERECORDED" if p >= 0.5 else "ORIGINAL" for p in probs_off]
    f1_off = f1_score(y_true_off, y_pred_off, average="macro", labels=["ORIGINAL", "RERECORDED"])
    off_fp = sum(1 for c in off_clip_ids if off_clip_true[c] == "ORIGINAL" and off_mean[c] >= 0.5)
    off_fn = sum(1 for c in off_clip_ids if off_clip_true[c] == "RERECORDED" and off_mean[c] < 0.5)
    off_rerec_mean = float(np.mean([off_mean[c] for c in off_clip_ids if off_clip_true[c] == "RERECORDED"]))

    del model
    return {
        "name": name,
        "val_macro_f1": f1_c,
        "val_cm": cm_c,
        "val_orig_recall": orig_recall,
        "val_rerec_recall": rerec_recall,
        "balanced_val_macro_f1": f1_bal,
        "val_best_threshold": best_t,
        "val_best_threshold_f1": sweep[best_t],
        "official_macro_f1_at_0.5": f1_off,
        "official_fp": off_fp,
        "official_fn": off_fn,
        "official_rerec_mean_prob": off_rerec_mean,
    }


def main():
    all_rows = list(csv.DictReader(open(DATA / "labels.csv", encoding="utf-8")))
    val_rows = [r for r in all_rows if r["split"] == "val"]
    official_rows = [r for r in all_rows if r["split"] == "official_check"]

    results = []
    for name, path in CHECKPOINTS:
        if not path.is_file():
            print(f"[스킵] {name}: 체크포인트 없음 ({path})")
            continue
        print(f"평가 중: {name} ({path.name}) ...")
        results.append(evaluate(name, path, val_rows, official_rows))

    print("\n" + "=" * 100)
    print("2차 / 3차 / 4차 종합 비교 (동일 val/official_check, 동일 코드로 재평가)")
    print("=" * 100)
    header = f"{'':30s}" + "".join(f"{r['name']:>20s}" for r in results)
    print(header)
    rows_def = [
        ("val macro-F1", lambda r: f"{r['val_macro_f1']:.4f}"),
        ("balanced val macro-F1", lambda r: f"{r['balanced_val_macro_f1']:.4f}"),
        ("val ORIGINAL recall", lambda r: f"{r['val_orig_recall']:.4f}"),
        ("val RERECORDED recall", lambda r: f"{r['val_rerec_recall']:.4f}"),
        ("val 최적 threshold(F1)", lambda r: f"{r['val_best_threshold']:.1f} ({r['val_best_threshold_f1']:.4f})"),
        ("official macro-F1@0.5", lambda r: f"{r['official_macro_f1_at_0.5']:.4f}"),
        ("official ORIGINAL FP", lambda r: f"{r['official_fp']}/5"),
        ("official RERECORDED FN", lambda r: f"{r['official_fn']}/5"),
        ("official RERECORDED 평균확률", lambda r: f"{r['official_rerec_mean_prob']:.4f}"),
    ]
    for label, fn in rows_def:
        print(f"{label:30s}" + "".join(f"{fn(r):>20s}" for r in results))

    print("\nval confusion matrix (row=true, col=pred) [ORIGINAL, RERECORDED]:")
    for r in results:
        print(f"  {r['name']}: {r['val_cm'].tolist()}")

    print("\n주의: official 지표는 n=5/5짜리 진단 신호일 뿐, 최종 추천 기준이 아님.")
    print("추천은 val macro-F1 / balanced val macro-F1 / 두 recall의 균형으로 판단할 것.")


if __name__ == "__main__":
    main()
