"""로컬(맥, GPU 없음)에서 predict_stage1 파이프라인 자체를 점검하는 스모크테스트.

infer_stage1.py는 제출 규격상 CUDA를 강제하므로, 여기서는 device 선택만
mps/cpu로 바꿔치기해서 실제 제출 코드와 동일한 전처리·샘플링·모델 로딩
경로를 그대로 검증한다. 평가 산식(Macro-F1)도 build_dataset이 만든
official_check 라벨 기준으로 재확인해 train.py의 내부 점검과 교차검증한다.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from sklearn.metrics import f1_score

import infer_stage1

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT.parent
MODEL_DIR = BASELINE / "model" / "stage1"


def _patched_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    infer_stage1._device = _patched_device  # 제출 코드는 그대로 두고 이 스크립트에서만 완화

    smoke_dir = ROOT / "_smoke_data"
    if smoke_dir.exists():
        shutil.rmtree(smoke_dir)
    (smoke_dir / "videos").mkdir(parents=True)

    ground_truth = {}
    for label, folder in [("ORIGINAL", "original"), ("RERECORDED", "rerecorded")]:
        for path in sorted((BASELINE / "data/stage1" / folder).glob("*.mp4")):
            new_name = f"{label}_{path.stem}"
            shutil.copy2(path, smoke_dir / "videos" / f"{new_name}.mp4")
            ground_truth[new_name] = label

    prediction = infer_stage1.predict_stage1(smoke_dir, MODEL_DIR)
    print(prediction)

    y_true = [ground_truth[row.ID] for row in prediction.itertuples()]
    y_pred = list(prediction["answer"])
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=["ORIGINAL", "RERECORDED"])
    print(f"\n공식 5+5 샘플(영상 단위, 실제 제출 경로 그대로): {correct}/{len(y_true)} 정답, macro-F1={macro_f1:.4f}")

    shutil.rmtree(smoke_dir)


if __name__ == "__main__":
    main()
