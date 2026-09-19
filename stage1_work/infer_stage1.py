"""제출용 predict_stage1 — EfficientNet-B0 프레임 분류 + 영상 단위 평균 투표.

inference.py에 그대로 옮겨 넣을 수 있도록, baseline과 동일한 시그니처
predict_stage1(data_dir, model_dir) -> pandas.DataFrame(columns=[ID, answer])
을 따른다. 학습 때(build_dataset.py)와 동일하게 정사각 리사이즈(256) 후
중앙크롭(224)을 적용해 전처리를 일치시킨다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
RESIZE_SIZE = 256
CROP_SIZE = 224
N_SAMPLE_FRAMES = 16
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _video_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def _sample_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    return list(np.linspace(0, total - 1, n).round().astype(int))


def _preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE_SIZE, RESIZE_SIZE), interpolation=cv2.INTER_AREA)
    off = (RESIZE_SIZE - CROP_SIZE) // 2
    rgb = rgb[off : off + CROP_SIZE, off : off + CROP_SIZE]
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD


class _VideoFrames(Dataset):
    """영상 하나에서 샘플링한 프레임들을 반환 (영상 단위 배치 처리를 위해 flatten).

    주의: cap.get(CAP_PROP_FRAME_COUNT)나 CAP_PROP_POS_FRAMES seek에 의존하지
    않는다. 컨테이너 메타데이터(프레임 수·fps)가 실제 디코딩 결과와 다른
    영상이 이 대회 데이터에 실제로 존재함이 확인되었으므로(Stage3 Q&A),
    항상 전체 프레임을 순차 디코딩한 뒤 실제 개수 기준으로 균등 샘플링한다.
    """

    def __init__(self, videos: list[Path]):
        self.items: list[tuple[int, torch.Tensor]] = []
        self.video_count = len(videos)
        for video_index, path in enumerate(videos):
            cap = cv2.VideoCapture(str(path))
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
            cap.release()
            if not frames:
                # 디코딩 실패 시 검정 프레임으로 대체(영상 자체는 반드시 결과를 내야 함)
                self.items.append((torch.zeros(3, CROP_SIZE, CROP_SIZE), video_index))
                continue
            for fi in _sample_indices(len(frames), N_SAMPLE_FRAMES):
                self.items.append((_preprocess(frames[fi]), video_index))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("이 제출물은 CUDA GPU 평가환경을 필요로 합니다.")
    return torch.device("cuda")


def predict_stage1(data_dir, model_dir) -> pd.DataFrame:
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    labels = checkpoint["labels"]  # ["ORIGINAL", "RERECORDED"]
    rerecorded_index = labels.index("RERECORDED")

    root = Path(data_dir) / "videos"
    videos = _video_paths(root)
    dataset = _VideoFrames(videos)
    loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

    scores = [[] for _ in videos]
    with torch.inference_mode():
        for frames, video_indices in loader:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probs = torch.softmax(model(frames.to(device, non_blocking=True)).float(), 1)
            for vi, p in zip(video_indices.tolist(), probs[:, rerecorded_index].cpu().tolist()):
                scores[vi].append(p)

    rows = []
    for path, values in zip(videos, scores):
        prob = float(np.mean(values)) if values else 0.0
        rows.append({"ID": path.stem, "answer": "RERECORDED" if prob >= 0.5 else "ORIGINAL"})
    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "answer"])
