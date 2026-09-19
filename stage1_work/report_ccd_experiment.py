"""CCD 기반 Stage1 1차 실험 결과 리포트.

- val 프레임/클립(=영상 단위 8프레임) 단위 accuracy, macro-F1, confusion matrix
- RERECORDED 확률 분포(ORIGINAL vs RERECORDED)
- 공식 Stage1 5+5 샘플의 영상별 확률·예측 (threshold 0.5)
- threshold sweep (val, official 각각) — 공식 10개에 과적합해서 threshold를
  고정하지 않기 위해 val 결과를 기준으로 삼고 official은 참고용으로만 표기
"""
from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MODEL_PATH = ROOT.parent / "model" / "stage1" / "best.pt"
IMG_SIZE = 224
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
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


def main():
    device = _device()
    print("device:", device)
    model, labels = _load_model(device)

    all_rows = list(csv.DictReader(open(DATA / "labels.csv", encoding="utf-8")))
    train_rows = [r for r in all_rows if r["split"] == "train"]
    val_rows = [r for r in all_rows if r["split"] == "val"]
    official_rows = [r for r in all_rows if r["split"] == "official_check"]

    print("\n=== 1. 데이터 구성 ===")
    for name, rows in [("train", train_rows), ("val", val_rows)]:
        n_orig = sum(1 for r in rows if r["label"] == "ORIGINAL")
        n_rerec = sum(1 for r in rows if r["label"] == "RERECORDED")
        n_clips = len({r["clip_id"] for r in rows})
        print(f"{name}: 프레임 {len(rows)}개 (ORIGINAL {n_orig} / RERECORDED {n_rerec}), 클립(영상) {n_clips}개")
    print("사용한 합성(2차 실험, recapture_augment.RecaptureParams.sample 서브스타일):")
    print("  train: mild_blur_compress(약한 블러+압축, 확률적) / sharp_clean(디노이즈+샤프닝) /")
    print("         resize_sharpen(다운업스케일+샤프닝) / pixel_grid_mild(미세 픽셀그리드) /")
    print("         classic_structural_rare(베젤·반사광·모아레, 낮은 확률로만)")
    print("  val  : gamma_contrast_shift(감마·대비·채도만) / sharp_clean_v2(디노이즈+샤프닝, 다른 강도) /")
    print("         scan_band_mild(약한 스캔라인) / white_balance_perspective(화이트밸런스+미세원근) /")
    print("         classic_structural_rare2(베젤·반사광·모아레, 낮은 확률로만, train과 다른 값 범위)")
    print("  공통(모든 스타일에 약하게): contrast/saturation(대비·채도는 오히려 증가 쪽으로 치우침),")
    print("         노이즈(낮은 확률), 손떨림, 화면비 크롭(35% 확률)")
    print("  클립당 RERECORDED 변형 수: 3개 (서로 다른 서브스타일 조합)")

    # === 2. val 프레임 단위 평가 ===
    print("\n=== 2. val 프레임 단위 평가 ===")
    y_true, probs = [], []
    for r in val_rows:
        probs.append(_predict_prob(model, device, labels, r["path"]))
        y_true.append(r["label"])
    y_pred = ["RERECORDED" if p >= 0.5 else "ORIGINAL" for p in probs]
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=["ORIGINAL", "RERECORDED"])
    cm = confusion_matrix(y_true, y_pred, labels=["ORIGINAL", "RERECORDED"])
    print(f"accuracy={acc:.4f}  macro-F1={macro_f1:.4f}")
    print("confusion matrix (row=true, col=pred) [ORIGINAL, RERECORDED]:")
    print(cm)
    orig_recall = cm[0, 0] / cm[0].sum() if cm[0].sum() else float("nan")
    rerec_recall = cm[1, 1] / cm[1].sum() if cm[1].sum() else float("nan")
    print(f"ORIGINAL recall={orig_recall:.4f}  RERECORDED recall={rerec_recall:.4f}")

    orig_probs = [p for t, p in zip(y_true, probs) if t == "ORIGINAL"]
    rerec_probs = [p for t, p in zip(y_true, probs) if t == "RERECORDED"]
    print("\nRERECORDED 확률 분포 (val, 프레임 단위):")
    print(f"  true=ORIGINAL   : mean={np.mean(orig_probs):.3f} median={np.median(orig_probs):.3f} "
          f"p10={np.percentile(orig_probs,10):.3f} p90={np.percentile(orig_probs,90):.3f}")
    print(f"  true=RERECORDED : mean={np.mean(rerec_probs):.3f} median={np.median(rerec_probs):.3f} "
          f"p10={np.percentile(rerec_probs,10):.3f} p90={np.percentile(rerec_probs,90):.3f}")

    # === 3. val 클립(영상) 단위 평가 (8프레임 평균) ===
    print("\n=== 3. val 클립(영상) 단위 평가 (8프레임 평균) ===")
    clip_probs, clip_true = defaultdict(list), {}
    for r, p in zip(val_rows, probs):
        clip_probs[r["clip_id"]].append(p)
        clip_true[r["clip_id"]] = r["label"]
    clip_ids = list(clip_probs.keys())
    clip_mean = {c: float(np.mean(clip_probs[c])) for c in clip_ids}
    clip_pred = {c: ("RERECORDED" if clip_mean[c] >= 0.5 else "ORIGINAL") for c in clip_ids}
    y_true_c = [clip_true[c] for c in clip_ids]
    y_pred_c = [clip_pred[c] for c in clip_ids]
    acc_c = sum(t == p for t, p in zip(y_true_c, y_pred_c)) / len(y_true_c)
    f1_c = f1_score(y_true_c, y_pred_c, average="macro", labels=["ORIGINAL", "RERECORDED"])
    cm_c = confusion_matrix(y_true_c, y_pred_c, labels=["ORIGINAL", "RERECORDED"])
    print(f"video-level accuracy={acc_c:.4f}  macro-F1={f1_c:.4f}")
    print("confusion matrix (row=true, col=pred) [ORIGINAL, RERECORDED]:")
    print(cm_c)
    orig_recall_c = cm_c[0, 0] / cm_c[0].sum() if cm_c[0].sum() else float("nan")
    rerec_recall_c = cm_c[1, 1] / cm_c[1].sum() if cm_c[1].sum() else float("nan")
    print(f"ORIGINAL recall={orig_recall_c:.4f}  RERECORDED recall={rerec_recall_c:.4f}")

    # === 3b. balanced val (ORIGINAL 전체 + RERECORDED를 같은 개수로 서브샘플) ===
    print("\n=== 3b. balanced val (ORIGINAL:RERECORDED 1:1로 서브샘플, 클립 단위) ===")
    orig_clip_ids = [c for c in clip_ids if clip_true[c] == "ORIGINAL"]
    rerec_clip_ids = [c for c in clip_ids if clip_true[c] == "RERECORDED"]
    bal_rng = random.Random(20260910)
    rerec_sample = bal_rng.sample(rerec_clip_ids, min(len(orig_clip_ids), len(rerec_clip_ids)))
    bal_clip_ids = orig_clip_ids + rerec_sample
    y_true_bal = [clip_true[c] for c in bal_clip_ids]
    y_pred_bal = [clip_pred[c] for c in bal_clip_ids]
    cm_bal = confusion_matrix(y_true_bal, y_pred_bal, labels=["ORIGINAL", "RERECORDED"])
    f1_bal = f1_score(y_true_bal, y_pred_bal, average="macro", labels=["ORIGINAL", "RERECORDED"])
    print(f"n={len(bal_clip_ids)} (ORIGINAL {len(orig_clip_ids)} / RERECORDED {len(rerec_sample)})  macro-F1={f1_bal:.4f}")
    print("confusion matrix (row=true, col=pred) [ORIGINAL, RERECORDED]:")
    print(cm_bal)
    orig_recall_bal = cm_bal[0, 0] / cm_bal[0].sum() if cm_bal[0].sum() else float("nan")
    rerec_recall_bal = cm_bal[1, 1] / cm_bal[1].sum() if cm_bal[1].sum() else float("nan")
    print(f"ORIGINAL recall={orig_recall_bal:.4f}  RERECORDED recall={rerec_recall_bal:.4f}")

    # === 4. threshold sweep on val (클립 단위) ===
    print("\n=== 4. threshold sweep (val, 클립 단위) — 참고용, 이 값으로 official을 맞추지 않음 ===")
    val_probs_list = [clip_mean[c] for c in clip_ids]
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        f1 = _macro_f1_at(y_true_c, val_probs_list, t)
        print(f"  threshold={t:.1f}  macro-F1={f1:.4f}")

    # === 5. 공식 Stage1 5+5 샘플 ===
    # 3차 실험(class-ratio 보정)의 목표: 2차 실험의 RERECORDED 확률 상승 효과는
    # 유지하면서, 2차 실험에서 새로 생긴 OFFICIAL_ORIGINAL_002 오판(FP)을
    # 줄이는 것. 공식 10개에 맞춰 threshold를 고정하지 않고, 확률 자체의
    # 변화(2차 대비)만 진단한다.
    PREV_V2_PROBS = {  # 2차 실험(class 비율 1:3, 보정 전) 결과 — 비교 기준
        "OFFICIAL_ORIGINAL_001": 0.301,
        "OFFICIAL_ORIGINAL_002": 0.741,  # 2차에서 유일한 FP
        "OFFICIAL_ORIGINAL_003": 0.000,
        "OFFICIAL_ORIGINAL_004": 0.012,
        "OFFICIAL_ORIGINAL_005": 0.150,
        "OFFICIAL_RERECORDED_001": 0.806,
        "OFFICIAL_RERECORDED_002": 0.864,
        "OFFICIAL_RERECORDED_003": 0.003,  # 2차에서도 FN
        "OFFICIAL_RERECORDED_004": 0.135,  # 2차에서도 FN
        "OFFICIAL_RERECORDED_005": 0.539,
    }
    print("\n=== 5. 공식 Stage1 5+5 샘플 (영상 단위, 8프레임 평균) — 2차 실험 대비 확률 변화 ===")
    off_clip_probs, off_clip_true = defaultdict(list), {}
    for r in official_rows:
        off_clip_probs[r["clip_id"]].append(_predict_prob(model, device, labels, r["path"]))
        off_clip_true[r["clip_id"]] = r["label"]
    off_clip_ids = sorted(off_clip_probs.keys())
    off_mean = {c: float(np.mean(off_clip_probs[c])) for c in off_clip_ids}
    for c in off_clip_ids:
        pred = "RERECORDED" if off_mean[c] >= 0.5 else "ORIGINAL"
        mark = "OK" if pred == off_clip_true[c] else "WRONG"
        prev = PREV_V2_PROBS.get(c)
        delta = f"  (2차 실험 {prev:.3f} -> 이번 {off_mean[c]:.3f})" if prev is not None else ""
        print(f"  {c:25s} true={off_clip_true[c]:10s} p_rerec={off_mean[c]:.3f} pred={pred:10s} {mark}{delta}")

    y_true_off = [off_clip_true[c] for c in off_clip_ids]
    probs_off = [off_mean[c] for c in off_clip_ids]
    y_pred_off = ["RERECORDED" if p >= 0.5 else "ORIGINAL" for p in probs_off]
    acc_off = sum(t == p for t, p in zip(y_true_off, y_pred_off)) / len(y_true_off)
    f1_off = f1_score(y_true_off, y_pred_off, average="macro", labels=["ORIGINAL", "RERECORDED"])
    rerec_only = [off_mean[c] for c in off_clip_ids if off_clip_true[c] == "RERECORDED"]
    orig_false_positive = sum(
        1 for c in off_clip_ids if off_clip_true[c] == "ORIGINAL" and off_mean[c] >= 0.5
    )
    rerec_false_negative = sum(
        1 for c in off_clip_ids if off_clip_true[c] == "RERECORDED" and off_mean[c] < 0.5
    )
    print(f"\nofficial 5+5 @ threshold=0.5 (참고용, 이 값으로 최종 threshold를 고정하지 않음): "
          f"accuracy={acc_off:.4f}  macro-F1={f1_off:.4f}")
    print("예측 분포:", Counter(y_pred_off))
    print(f"공식 ORIGINAL false positive 개수: {orig_false_positive}/5  (2차 실험: 1/5)")
    print(f"공식 RERECORDED false negative 개수: {rerec_false_negative}/5  (2차 실험: 2/5)")
    print(f"RERECORDED 5개 확률: mean={np.mean(rerec_only):.4f} max={np.max(rerec_only):.4f} "
          f"(2차 실험 대비: mean {np.mean([PREV_V2_PROBS[c] for c in off_clip_ids if off_clip_true[c]=='RERECORDED']):.4f} -> {np.mean(rerec_only):.4f})")

    print("\n=== 6. threshold sweep (official 5+5) — 순수 참고/진단용, 이 10개로 최종 threshold를 고정하지 않음 ===")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        f1 = _macro_f1_at(y_true_off, probs_off, t)
        print(f"  threshold={t:.1f}  macro-F1={f1:.4f}")
    print("\n최종 배포 threshold는 위 4번(val 기준) sweep으로만 정한다 — 공식 10개는 진단 신호로만 사용.")


if __name__ == "__main__":
    main()
