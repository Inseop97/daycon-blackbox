"""3-Stage 영상 분석 추론 코드.

Stage 1: EfficientNet-B0 프레임 분류 + 영상 단위 평균 투표.
Stage 2: collision_frame(모션 급변) + entry_frame/entry_side(YOLO 추적 + 기하규칙)
         + evasion_space(EfficientNet-B0 분류).
Stage 3: 프레임별 EfficientNet-B0 특징을 시간축(tsn/gru/x3d)으로 합치는 다중헤드 모델.

각 Stage의 평가 데이터를 예측하여 정해진 형식의 DataFrame을 반환한다.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0
from torchvision.transforms import v2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".3gp", ".3gpp", ".wmv"}
S3_IMG_SIZE = 224  # EfficientNet-B0 사전학습 기준 (stage3_work/train.py와 동일)
# 클립 길이(clip_len)는 체크포인트에서 읽어옴 (로컬/GPU 서버 학습 설정이 다를 수 있음)
S3_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
S3_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
cv2.setNumThreads(1)


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("이 제출물은 CUDA GPU 평가환경을 필요로 합니다.")
    return torch.device("cuda")


def _video_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def _sample_indices(total: int, n: int) -> list[int]:
    if total <= n:
        return list(range(total))
    return list(np.linspace(0, total - 1, n).round().astype(int))


# ---------------------------------------------------------------------------
# Stage 1: EfficientNet-B0 프레임 분류 + 영상 단위 평균 투표
# ---------------------------------------------------------------------------
S1_RESIZE_SIZE = 256
S1_CROP_SIZE = 224
S1_N_SAMPLE_FRAMES = 16
S1_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
S1_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _s1_preprocess(frame_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (S1_RESIZE_SIZE, S1_RESIZE_SIZE), interpolation=cv2.INTER_AREA)
    off = (S1_RESIZE_SIZE - S1_CROP_SIZE) // 2
    rgb = rgb[off : off + S1_CROP_SIZE, off : off + S1_CROP_SIZE]
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    return (x - S1_MEAN) / S1_STD


class _Stage1Frames(Dataset):
    """영상 전체를 순차 디코딩한 뒤 균등 샘플링한다.

    cap.get(CAP_PROP_FRAME_COUNT)/POS_FRAMES seek에 의존하지 않는다 — 이
    대회 데이터 중 일부는 컨테이너 메타데이터(프레임수·fps)가 실제 디코딩
    결과와 다름이 Q&A로 확인되었다.
    """

    def __init__(self, videos: list[Path]):
        self.items: list[tuple[int, torch.Tensor]] = []
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
                self.items.append((torch.zeros(3, S1_CROP_SIZE, S1_CROP_SIZE), video_index))
                continue
            for fi in _sample_indices(len(frames), S1_N_SAMPLE_FRAMES):
                self.items.append((_s1_preprocess(frames[fi]), video_index))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class _Stage1SRMResidual(nn.Module):
    """stage1_work/train.py의 _SRMResidual과 동일 — 고정 고역통과 필터로 노이즈 잔차 추출."""

    def __init__(self):
        super().__init__()
        k1 = [[0, 0, 0], [0, -1, 1], [0, 0, 0]]
        k2 = [[0, 1, 0], [0, -2, 0], [0, 1, 0]]
        k3 = [[-1, 2, -1], [2, -4, 2], [-1, 2, -1]]
        kernels = torch.tensor([k1, k2, k3], dtype=torch.float32) / 4.0
        weight = kernels.repeat(3, 1, 1).unsqueeze(1)
        self.conv = nn.Conv2d(3, 9, kernel_size=3, padding=1, groups=3, bias=False)
        with torch.no_grad():
            self.conv.weight.copy_(weight)
        self.conv.weight.requires_grad_(False)

    def forward(self, x):
        return self.conv(x)


class _Stage1SRMDualStream(nn.Module):
    def __init__(self, num_labels: int):
        super().__init__()
        self.srm = _Stage1SRMResidual()
        rgb_backbone = efficientnet_b0(weights=None)
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


def _build_stage1_model(arch: str, num_labels: int) -> nn.Module:
    if arch == "xception":
        import timm
        return timm.create_model("legacy_xception", pretrained=False, num_classes=num_labels)
    if arch == "srm_dual":
        return _Stage1SRMDualStream(num_labels)
    model = efficientnet_b0(weights=None)  # "efficientnet" 및 이전 체크포인트 기본값
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
    return model


def predict_stage1(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]  # ["ORIGINAL", "RERECORDED"]
    arch = checkpoint.get("arch", "efficientnet")
    model = _build_stage1_model(arch, len(labels))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    rerecorded_index = labels.index("RERECORDED")

    # "videos/" 하위 폴더 유무가 위키에서 표기가 엇갈렸던 적이 있어(Stage2와 같은
    # 문서 문제), data_dir 전체를 재귀 탐색해 두 경우 모두 대응한다.
    videos = _video_paths(Path(data_dir))
    dataset = _Stage1Frames(videos)
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


# ---------------------------------------------------------------------------
# Stage 2: collision_frame(모션 급변 지점) + entry_frame/entry_side(YOLO 추적 +
# 원근보정 기하규칙) + evasion_space(EfficientNet-B0 분류, 충돌 근처 가장 선명한
# 프레임) — 세 방식 모두 stage2_evasion_work/entry_extract.py 실험에서 검증한
# 로직을 그대로 옮긴 것. collision_frame은 실험 중엔 CCD의 binlabels 전환점+
# 보정치를 썼지만, 그건 CCD 전용 라벨이라 평가 영상엔 없다 — 대신 그 라벨을
# 프레임간 모션 급변(motion-spike) 신호로 교차검증했을 때 ±0.3초 허용오차 내
# 81~82% 일치했으므로, 여기서는 motion-spike 피크를 그대로 collision_frame
# 추정치로 쓴다.
# ---------------------------------------------------------------------------
S2_VEHICLE_CLS = {2, 5, 7}  # car, bus, truck (COCO 클래스 인덱스) — 위키 Q&A로 피의차량/
# 블랙박스차량이 오토바이·자전거인 경우는 없음을 확인해 motorcycle(3)은 제외
S2_DEFAULT_FPS = 10.0  # fps를 못 구하거나 비정상일 때 폴백(CCD/comma2k19 실험 기준)
S2_IMG_SIZE = 224
S2_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
S2_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
S2_TEMPORAL_TRANSFORM = v2.Compose([
    v2.Resize(256, antialias=True),
    v2.CenterCrop(224),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
S2_BOX_FEATURE_NAMES = [
    "n_vehicles", "largest_area_frac", "largest_cx", "largest_cy",
    "coverage_frac", "gap_left_frac", "gap_right_frac",
    "drivable_frac_total", "drivable_frac_left", "drivable_frac_right", "drivable_frac_near",
]
S2_YOLOP_IMG_SIZE = 640
S2_YOLOP_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
S2_YOLOP_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_yolop(model_dir: Path, device: torch.device):
    """YOLOP(hustvl/YOLOP, MIT License) 로컬 벤더 코드 로더 — 인터넷 없이 model_dir 안의
    yolop_code/lib 와 yolop_end2end.pth 만으로 구성한다(stage2_evasion_work/yolop_utils.py와
    동일 로직, submit.zip이 model_dir/inference.py/requirements.txt만 허용해 이 파일 안에 인라인)."""
    import sys

    vendor_dir = str(model_dir / "yolop_code")
    if vendor_dir not in sys.path:
        sys.path.insert(0, vendor_dir)
    from lib.config import cfg
    from lib.models import get_net

    model = get_net(cfg)
    checkpoint = torch.load(str(model_dir / "yolop_end2end.pth"), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def _run_yolop(model, frame_bgr: np.ndarray, device: torch.device):
    """frame_bgr -> (drivable_mask, lane_mask), 원본 해상도의 bool 배열. 종횡비를 유지하는
    letterbox 리사이즈 필요(정사각형으로 그냥 squish하면 도로 형태가 왜곡돼 거의 항상 빈
    마스크가 나옴을 확인함)."""
    from lib.utils.augmentations import letterbox_for_img

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, ratio, (dw, dh) = letterbox_for_img(rgb, new_shape=S2_YOLOP_IMG_SIZE, auto=True)
    x = (letterboxed.astype(np.float32) / 255.0 - S2_YOLOP_MEAN) / S2_YOLOP_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        _, da_seg_out, ll_seg_out = model(x)

    lh, lw = letterboxed.shape[:2]
    dw, dh = int(round(dw)), int(round(dh))
    da_full = da_seg_out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    ll_full = ll_seg_out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    da_crop = da_full[dh : lh - dh, dw : lw - dw]
    ll_crop = ll_full[dh : lh - dh, dw : lw - dw]
    da_mask = cv2.resize(da_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_mask = cv2.resize(ll_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_mask, ll_mask


def _s2_seg_features(frame, yolop_model, device) -> list[float]:
    h, w = frame.shape[:2]
    da_mask, _ = _run_yolop(yolop_model, frame, device)
    total = float(da_mask.mean())
    left = float(da_mask[:, : w // 2].mean())
    right = float(da_mask[:, w // 2 :].mean())
    near = float(da_mask[int(h * 0.6) :, :].mean())
    return [total, left, right, near]


def _s2_ego_lane_trapezoid(w, h):
    """화면 하단은 넓고 소실점(화면 중앙 상단)으로 갈수록 좁아지는 자차 차선 영역."""
    bottom_half_width, top_half_width = 0.30 * w, 0.06 * w
    top_y, bottom_y = 0.45 * h, h
    return np.array(
        [
            [w / 2 - bottom_half_width, bottom_y],
            [w / 2 + bottom_half_width, bottom_y],
            [w / 2 + top_half_width, top_y],
            [w / 2 - top_half_width, top_y],
        ],
        dtype=np.int32,
    )


def _s2_lane_boundary_fit(frame, yolop_model, device):
    """YOLOP 차선 세그멘테이션으로 화면 중앙 좌우에서 "가장 가까운" 차선을 각각
    y에 대한 1차함수 x=f(y)로 피팅한다 — 실제 차선 형태를 따라가므로 곡선도로에서도
    유효(고정 사다리꼴과 달리). 각 행(row)마다 중앙에서 가장 가까운 좌/우 픽셀만
    골라 다른 차선(옆 차로 경계 등)이 섞여 들어가는 걸 줄인다.
    실패(차선을 충분히 못 찾음)하면 (None, None)을 반환하고, 호출부는 고정
    사다리꼴 기반 normalized_lateral로 폴백한다.
    """
    h, w = frame.shape[:2]
    _, ll_mask = _run_yolop(yolop_model, frame, device)
    left_pts, right_pts = [], []
    for y in range(int(h * 0.35), h, 2):  # 화면 하단 65%(멀리 있는 노이즈 배제), 2행 간격 샘플링
        xs = np.where(ll_mask[y])[0]
        if len(xs) == 0:
            continue
        left_xs = xs[xs < w / 2]
        right_xs = xs[xs >= w / 2]
        if len(left_xs) > 0:
            left_pts.append((y, float(left_xs.max())))  # 중앙에 가장 가까운(가장 큰 x) 좌측 픽셀
        if len(right_xs) > 0:
            right_pts.append((y, float(right_xs.min())))  # 중앙에 가장 가까운(가장 작은 x) 우측 픽셀

    def fit(pts):
        if len(pts) < 8:
            return None
        ys, xs = zip(*pts)
        a, b = np.polyfit(ys, xs, 1)
        return lambda yy, a=a, b=b: a * yy + b

    return fit(left_pts), fit(right_pts)


def _s2_normalized_lateral_lane(x, y, left_fn, right_fn):
    """실제 검출된 좌/우 차선 경계 기준으로 정규화한 좌우 위치. 두 경계 폭이
    비정상적으로 좁거나 넓으면(오검출 가능성) None을 반환해 폴백을 유도한다."""
    left_x, right_x = left_fn(y), right_fn(y)
    half_width = (right_x - left_x) / 2
    if half_width < 10:  # 좌우 경계가 역전되었거나 너무 붙어있음 — 신뢰 불가
        return None
    center = (left_x + right_x) / 2
    return (x - center) / half_width


def _s2_normalized_lateral(x, y, w, h, top_ratio=0.06, bottom_ratio=0.30, top_y_ratio=0.45):
    """화면 중앙선 대비 좌우 위치를, 그 높이에서의 사다리꼴 반폭으로 정규화(원근 착시 제거)."""
    top_y, bottom_y = top_y_ratio * h, h
    t = np.clip((y - top_y) / (bottom_y - top_y), 0, 1)
    half_width = (top_ratio + t * (bottom_ratio - top_ratio)) * w
    return (x - w / 2) / max(half_width, 1e-6)


def _s2_collision_frame(frames) -> int:
    if len(frames) < 2:
        return 0
    diffs = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for f in frames[1:]:
        cur_gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diffs.append(np.abs(cur_gray - prev_gray).mean())
        prev_gray = cur_gray
    return int(np.argmax(diffs)) + 1


def _s2_video_fps(path) -> float:
    """영상마다 fps가 다르고(위키 Q&A로 확인, Stage3와 달리 고정 아님) 컨테이너
    메타데이터가 실제 디코딩과 다른 경우도 있었으므로(Stage1 Q&A), 범위를 벗어나면
    CCD/comma2k19 실험 기준값(10fps)으로 폴백한다."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if not fps or fps < 3 or fps > 60:
        return S2_DEFAULT_FPS
    return float(fps)


def _s2_sharpest_frame_near(frames, center: int, fps: float, window_sec: float = 1.0) -> int:
    window = max(1, round(window_sec * fps))
    lo, hi = max(0, center - window), min(len(frames), center + window + 1)
    best_idx, best_val = lo, -1.0
    for i in range(lo, hi):
        val = cv2.Laplacian(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if val > best_val:
            best_val, best_idx = val, i
    return best_idx


def _s2_entry(frames, collision_frame_est: int, yolo_model, fps: float = S2_DEFAULT_FPS, yolop_model=None, device=None):
    """시간 기준 윈도우(초)를 fps로 환산해 프레임 수로 바꾼다 — Stage2 평가 영상은
    영상마다 fps가 다를 수 있다고 위키 Q&A로 확인됐으므로, 원래 CCD/comma2k19(10fps)
    기준으로 튜닝된 프레임 수 상수들을 그대로 쓰면 fps가 다른 영상에서 의도한
    시간 길이와 어긋난다."""
    h, w = frames[0].shape[:2]
    candidate_window = max(1, round(0.6 * fps))  # CCD/comma2k19(10fps) 기준 6프레임 튜닝값
    baseline_window = max(1, round(2.0 * fps))  # 위와 동일 기준 20프레임
    max_transition = max(1, round(2.0 * fps))  # 위와 동일 기준 20프레임
    fallback_offset = max(1, round(1.0 * fps))  # 위와 동일 기준 10프레임

    # YOLOP 차선 세그멘테이션으로 실제 차선 경계를 한 번 피팅(collision_frame_est 근처
    # 프레임 기준) — 곡선도로에서도 유효한 기준선을 얻는다. 프레임마다 다시 돌리면
    # 정확도는 더 좋아지겠지만 계산량이 커서, 짧은 분석 구간 내에서는 도로 형태가
    # 크게 안 변한다고 가정하고 한 번만 계산해 재사용한다. 실패하면 None,None이라
    # 이후 로직에서 자동으로 고정 사다리꼴로 폴백한다.
    lane_left_fn = lane_right_fn = None
    if yolop_model is not None:
        anchor_idx = min(max(0, collision_frame_est), len(frames) - 1)
        lane_left_fn, lane_right_fn = _s2_lane_boundary_fit(frames[anchor_idx], yolop_model, device)
    track_boxes: dict[int, dict[int, tuple]] = {}
    for i, f in enumerate(frames):
        res = yolo_model.track(f, conf=0.1, persist=(i > 0), verbose=False, tracker="bytetrack.yaml")[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            continue
        for b, tid in zip(boxes, boxes.id.tolist()):
            if int(b.cls) not in S2_VEHICLE_CLS:
                continue
            track_boxes.setdefault(int(tid), {})[i] = tuple(map(float, b.xyxy[0].tolist()))

    def box_area(b):
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    def dist_to_center(b):
        return abs((b[0] + b[2]) / 2 - w / 2)

    best_tid, best_score = None, -1.0
    for tid, fbox in track_boxes.items():
        near = [i for i in fbox if abs(i - collision_frame_est) <= candidate_window]
        if not near:
            continue
        score = max(box_area(fbox[i]) for i in near) / (min(dist_to_center(fbox[i]) for i in near) + 50)
        if score > best_score:
            best_score, best_tid = score, tid

    if best_tid is None:
        # 상대차량 후보 트랙을 아예 못 찾음 — 위키 규칙(영상 시작 전 이미 진입)으로 대체
        return 0, "LEFT"

    fbox = track_boxes[best_tid]
    frame_idxs = sorted(fbox.keys())

    def _norm_for(i):
        cx, cy = (fbox[i][0] + fbox[i][2]) / 2, fbox[i][3]
        if lane_left_fn is not None and lane_right_fn is not None:
            lane_norm = _s2_normalized_lateral_lane(cx, cy, lane_left_fn, lane_right_fn)
            if lane_norm is not None:
                return lane_norm
        return _s2_normalized_lateral(cx, cy, w, h)

    norm = {i: _norm_for(i) for i in frame_idxs}

    # 곡선도로 대응: normalized_lateral()은 사다리꼴이 직선을 가정하므로, 곡선에서는
    # 선행차량도 화면상 좌우로 서서히 "드리프트"하는 것처럼 보여 절대 임계값 방식은
    # 오탐이 잦다(comma2k19 곡선 구간 50개로 실측한 오탐률 54%). 절대 위치 대신
    # "최근 자기 궤적(rolling median) 대비 상대적 이탈"을 보면 — 곡선을 따라가는
    # 완만한 드리프트는 베이스라인이 같이 따라가서 이탈이 작게 유지되고, 실제
    # 차선변경처럼 빠른 변화만 커다란 이탈로 남는다(고전적인 디트렌딩/고역통과 방식).
    baseline: dict[int, float] = {}
    for idx, i in enumerate(frame_idxs):
        hist = [norm[frame_idxs[j]] for j in range(max(0, idx - baseline_window), idx)]
        baseline[i] = float(np.median(hist)) if hist else norm[i]
    deviation = {i: norm[i] - baseline[i] for i in frame_idxs}

    OUTSIDE_TH, INSIDE_TH = 0.5, 0.25
    entry_frame, saw_clearly_outside, last_outside_frame = None, False, None
    for idx, i in enumerate(frame_idxs):
        if abs(deviation[i]) > OUTSIDE_TH:
            saw_clearly_outside, last_outside_frame = True, i
            continue
        if (
            saw_clearly_outside
            and abs(deviation[i]) <= INSIDE_TH
            and last_outside_frame is not None
            and (i - last_outside_frame) <= max_transition
        ):
            next_i = frame_idxs[idx + 1] if idx + 1 < len(frame_idxs) else None
            if next_i is None or abs(deviation[next_i]) <= INSIDE_TH * 1.25:
                entry_frame = i
                break

    if entry_frame is None:
        # 처음부터 계속 자기 차선 근처였으면(선행차량 등) 위키 규칙대로 0,
        # 바깥이었던 적은 있는데 빠른 전환을 못 찾은 애매한 경우는 충돌 직전으로 대체
        entry_frame = 0 if not saw_clearly_outside else max(0, collision_frame_est - fallback_offset)

    pre = [i for i in frame_idxs if i < entry_frame][-5:] or frame_idxs[:3] or [frame_idxs[0]]
    entry_side = "LEFT" if np.mean([norm[i] for i in pre]) < 0 else "RIGHT"
    return entry_frame, entry_side


def _s2_box_features(frame, yolo_model, yolop_model, device) -> list[float]:
    """extract_box_features.py와 동일 로직(box_fusion 아키텍처용) — YOLO 박스 기하특징
    7개 + YOLOP 주행가능영역 세그멘테이션 특징 4개, 총 11개."""
    h, w = frame.shape[:2]
    res = yolo_model.predict(frame, conf=0.15, verbose=False)[0]
    boxes = res.boxes
    if boxes is None or len(boxes) == 0:
        box_feats = [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0]
    else:
        xyxy, cls = boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy().astype(int)
        vb = xyxy[np.isin(cls, list(S2_VEHICLE_CLS))]
        if len(vb) == 0:
            box_feats = [0.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0]
        else:
            areas = (vb[:, 2] - vb[:, 0]) * (vb[:, 3] - vb[:, 1])
            largest = vb[np.argmax(areas)]
            intervals = sorted((float(b[0] / w), float(b[2] / w)) for b in vb)
            covered, cur_s, cur_e = 0.0, None, None
            for s, e in intervals:
                if cur_s is None:
                    cur_s, cur_e = s, e
                elif s <= cur_e:
                    cur_e = max(cur_e, e)
                else:
                    covered += cur_e - cur_s
                    cur_s, cur_e = s, e
            if cur_s is not None:
                covered += cur_e - cur_s
            box_feats = [
                float(len(vb)), float(areas.max() / (w * h)),
                float((largest[0] + largest[2]) / 2 / w), float((largest[1] + largest[3]) / 2 / h),
                covered, min(b[0] / w for b in vb), 1.0 - max(b[2] / w for b in vb),
            ]
    return box_feats + _s2_seg_features(frame, yolop_model, device)


class _Stage2BoxFusion(nn.Module):
    """stage2_evasion_work/train.py의 Stage2BoxFusionModel과 동일 구조 —
    평가서버는 인터넷이 없으므로 weights=None으로 구조만 만들고 체크포인트로 채운다."""

    def __init__(self, num_labels: int, box_dim: int = len(S2_BOX_FEATURE_NAMES)):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        img_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.box_mlp = nn.Sequential(
            nn.Linear(box_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(img_dim + 32, num_labels)

    def forward(self, x, box):
        return self.classifier(torch.cat([self.backbone(x), self.box_mlp(box)], dim=1))


class _Stage2ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.net(x)


class _Stage2TemporalModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()
        )
        self.tcn = nn.Sequential(
            _Stage2ResidualTCNBlock(hidden_dim, 1, dropout),
            _Stage2ResidualTCNBlock(hidden_dim, 2, dropout),
            _Stage2ResidualTCNBlock(hidden_dim, 4, dropout),
            _Stage2ResidualTCNBlock(hidden_dim, 8, dropout),
        )
        self.collision_head = nn.Conv1d(hidden_dim, 1, 1)
        self.entry_head = nn.Conv1d(hidden_dim, 1, 1)
        self.side_head = nn.Linear(hidden_dim, 2)
        self.evasion_head = nn.Linear(hidden_dim, 2)

    def forward(self, features):
        hidden = self.tcn(self.input_projection(features).transpose(1, 2)).transpose(1, 2)
        collision_logits = self.collision_head(hidden.transpose(1, 2)).squeeze(1)
        entry_logits = self.entry_head(hidden.transpose(1, 2)).squeeze(1)
        collision_idx = collision_logits.argmax(1)
        positions = torch.arange(entry_logits.shape[1], device=entry_logits.device)[None]
        entry_idx = entry_logits.masked_fill(positions > collision_idx[:, None], -1e9).argmax(1)
        batch = torch.arange(features.shape[0], device=features.device)
        return (
            collision_idx,
            entry_idx,
            self.side_head(hidden[batch, entry_idx]),
            self.evasion_head(hidden[batch, collision_idx]),
        )


def _s2_temporal_preprocess(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return S2_TEMPORAL_TRANSFORM(Image.fromarray(rgb))


def _load_stage2_temporal(model_dir, device):
    import timm

    checkpoint = torch.load(model_dir / "temporal_best.pt", map_location="cpu", weights_only=False)
    backbone = timm.create_model(
        checkpoint["backbone_name"],
        pretrained=False,
        num_classes=0,
        img_size=int(checkpoint.get("image_size", 224)),
    )
    backbone.load_state_dict(checkpoint["backbone"])
    temporal = _Stage2TemporalModel(
        int(checkpoint["feature_dim"]), int(checkpoint.get("hidden_dim", 256))
    )
    temporal.load_state_dict(checkpoint["temporal_model"])
    return backbone.to(device).eval(), temporal.to(device).eval(), checkpoint


def _predict_stage2_temporal(frames, frame_numbers, fps, backbone, temporal, checkpoint, device):
    target_fps = float(checkpoint.get("target_fps", 5.0))
    stride = max(1, int(round(fps / target_fps)))
    sampled_indices = list(range(0, len(frames), stride))
    feature_chunks = []
    batch_size = 64
    for start in range(0, len(sampled_indices), batch_size):
        indices = sampled_indices[start : start + batch_size]
        x = torch.stack([_s2_temporal_preprocess(frames[i]) for i in indices]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            feature_chunks.append(backbone(x).float())
    features = torch.cat(feature_chunks).unsqueeze(0)
    collision_pos, entry_pos, side_logits, evasion_logits = temporal(features)
    collision_idx = sampled_indices[int(collision_pos.item())]
    entry_idx = sampled_indices[int(entry_pos.item())]
    side_labels = checkpoint.get("side_labels", ["LEFT", "RIGHT"])
    return {
        "collision_frame": frame_numbers[collision_idx],
        "entry_frame": frame_numbers[entry_idx],
        "entry_side": side_labels[int(side_logits.argmax(1).item())],
        "evasion_space": int(evasion_logits.argmax(1).item()),
    }


def _build_stage2_evasion_model(arch: str, num_labels: int) -> nn.Module:
    if arch == "box_fusion":
        return _Stage2BoxFusion(num_labels)
    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_labels)
    return model


def _s2_frame_number(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else 0


def _s2_sample_dirs(data_dir) -> list[Path]:
    """Stage2 평가 데이터는 영상 파일이 아니라 샘플별로 이미 추출된 프레임
    이미지(frame_*.jpg)로 제공된다 — 위키 Q&A(평가 데이터 디렉토리 구조 문의)로
    확인. 문서마다 `images/` 중간 폴더가 있다/없다로 표기가 엇갈렸는데(운영진
    답변도 "수정반영"이라고만 해서 최종 형태가 명확하지 않음), frame_*.jpg가
    있는 폴더를 깊이에 상관없이 찾아 그 폴더 자체를 샘플로 취급하면 두 경우
    모두 대응된다."""
    root = Path(data_dir)
    return sorted({p.parent for p in root.rglob("frame_*.jpg")})


def _s2_samples(data_dir):
    """(ID, frames, frame_numbers, fps)를 순서대로 내놓는다.

    위키 Q&A로 Stage2는 프레임 이미지(frame_*.jpg)로 제공된다고 확인했지만,
    문서 표기가 한 번 엇갈렸던 적이 있어(운영진이 "수정반영"이라고만 답해 최종
    형태를 100% 장담할 수 없음) 혹시 영상 파일로 오는 경우까지 대비해 폴백을
    둔다 — 이미지 폴더를 먼저 찾고, 하나도 없으면 영상 파일로 시도한다.
    이미지는 컨테이너 fps 정보가 없어 CCD/comma2k19 튜닝 기준(10fps)을 가정하고,
    영상 폴백에서는 실제 fps를 읽는다(영상마다 fps가 다르다고 확인됐으므로).
    """
    sample_dirs = _s2_sample_dirs(data_dir)
    if sample_dirs:
        for sample_dir in sample_dirs:
            frame_paths = sorted(sample_dir.glob("frame_*.jpg"), key=_s2_frame_number)
            frame_numbers = [_s2_frame_number(p) for p in frame_paths]
            frames = [f for f in (cv2.imread(str(p)) for p in frame_paths) if f is not None]
            yield sample_dir.name, frames, frame_numbers, S2_DEFAULT_FPS
        return

    for path in _video_paths(Path(data_dir)):
        cap = cv2.VideoCapture(str(path))
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        yield path.stem, frames, list(range(len(frames))), _s2_video_fps(path)


def predict_stage2(data_dir, model_dir):
    device = _device()
    model_dir = Path(model_dir)

    temporal_path = model_dir / "temporal_best.pt"
    if temporal_path.is_file():
        backbone, temporal, checkpoint = _load_stage2_temporal(model_dir, device)
        rows = []
        with torch.inference_mode():
            for sample_id, frames, frame_numbers, fps in _s2_samples(data_dir):
                if not frames:
                    rows.append({"ID": sample_id, "collision_frame": 0, "entry_frame": 0, "evasion_space": 0, "entry_side": "LEFT"})
                    continue
                prediction = _predict_stage2_temporal(
                    frames, frame_numbers, fps, backbone, temporal, checkpoint, device
                )
                rows.append({"ID": sample_id, **prediction})
        del backbone, temporal
        torch.cuda.empty_cache()
        return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])

    checkpoint = torch.load(model_dir / "evasion_space" / "best.pt", map_location="cpu", weights_only=False)
    labels = checkpoint["labels"]
    arch = checkpoint.get("arch", "efficientnet")
    evasion_model = _build_stage2_evasion_model(arch, len(labels))
    evasion_model.load_state_dict(checkpoint["model"])
    evasion_model.to(device).eval()

    from ultralytics import YOLO

    yolo_model = YOLO(str(model_dir / "yolo11n.pt"))
    # entry_frame 차선 기반 경계 시도는 디트렌딩 단독과 동률(comma2k19 곡선 50개, 38%)로
    # 끝나 계산량만 아끼려고 껐다 — box_fusion(evasion_space)일 때만 필요.
    yolop_model = _load_yolop(model_dir, device) if arch == "box_fusion" else None

    rows = []
    with torch.inference_mode():
        for sample_id, frames, frame_numbers, fps in _s2_samples(data_dir):
            if not frames:
                rows.append({"ID": sample_id, "collision_frame": 0, "entry_frame": 0, "evasion_space": 0, "entry_side": "LEFT"})
                continue

            collision_idx = _s2_collision_frame(frames)
            # YOLOP 차선 기반 경계 시도 결과 디트렌딩 단독과 동률(38%)이라 계산량만 아끼고
            # 끔 — _s2_entry는 yolop_model=None이면 자동으로 고정 사다리꼴+디트렌딩만 씀.
            entry_idx, entry_side = _s2_entry(frames, collision_idx, yolo_model, fps)

            sharp_idx = _s2_sharpest_frame_near(frames, collision_idx, fps)
            frame = frames[sharp_idx]
            h, w = frame.shape[:2]
            scale = 256 / min(h, w)
            resized = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (round(w * scale), round(h * scale)))
            rh, rw = resized.shape[:2]
            top, left = (rh - S2_IMG_SIZE) // 2, (rw - S2_IMG_SIZE) // 2
            crop = resized[top : top + S2_IMG_SIZE, left : left + S2_IMG_SIZE]
            x = torch.from_numpy(crop.copy()).permute(2, 0, 1).float() / 255.0
            x = ((x - S2_MEAN) / S2_STD).unsqueeze(0).to(device)
            if arch == "box_fusion":
                box = torch.tensor([_s2_box_features(frame, yolo_model, yolop_model, device)], dtype=torch.float32, device=device)
                logits = evasion_model(x, box)
            else:
                logits = evasion_model(x)
            evasion_space = int(logits.argmax(1).item())

            rows.append(
                {
                    "ID": sample_id,
                    "collision_frame": frame_numbers[collision_idx],
                    "entry_frame": frame_numbers[entry_idx],
                    "evasion_space": evasion_space,
                    "entry_side": entry_side,
                }
            )
    del evasion_model, yolo_model, yolop_model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "collision_frame", "entry_frame", "evasion_space", "entry_side"])


# ---------------------------------------------------------------------------
# Stage 3: 프레임별 EfficientNet-B0 특징을 시간축으로 합치는 다중헤드 모델.
# mvit_v2_s/r2plus1d_18 같은 3D-CNN은 이 하드웨어(MPS, CUDA 없음)에서 벤치마크해보니
# 배치 하나 학습 스텝에 몇 초씩 걸려 현실적이지 않아, Stage1/2에서 이미 빠른 것으로
# 확인된 2D CNN(EfficientNet-B0) 기반으로 교체했다 (stage3_work/train.py 참고).
# 체크포인트의 "arch" 값(tsn/gru/x3d)에 맞는 클래스를 골라 씀 — 학습 시 고른
# 아키텍처와 추론이 항상 일치하도록.
# ---------------------------------------------------------------------------
def _s3_frame_features(x: torch.Tensor, backbone: nn.Module) -> torch.Tensor:
    b, t = x.shape[:2]
    feat = backbone(x.reshape(b * t, *x.shape[2:]))
    return feat.reshape(b, t, -1)


class _Stage3TSN(nn.Module):
    def __init__(self, n_accel: int, n_steer: int):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        dimension = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.accel = nn.Linear(dimension, n_accel)
        self.steer = nn.Linear(dimension, n_steer)

    def forward(self, x):
        feat = _s3_frame_features(x, self.backbone).mean(dim=1)
        return self.accel(feat), self.steer(feat)


class _Stage3GRU(nn.Module):
    def __init__(self, n_accel: int, n_steer: int, hidden: int = 256):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        dimension = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.gru = nn.GRU(dimension, hidden, batch_first=True, bidirectional=True)
        self.accel = nn.Linear(hidden * 2, n_accel)
        self.steer = nn.Linear(hidden * 2, n_steer)

    def forward(self, x):
        feat_seq = _s3_frame_features(x, self.backbone)
        out, _ = self.gru(feat_seq)
        h = out.shape[-1] // 2
        pooled = torch.cat([out[:, -1, :h], out[:, 0, h:]], dim=-1)
        return self.accel(pooled), self.steer(pooled)


class _Stage3X3D(nn.Module):
    def __init__(self, n_accel: int, n_steer: int, clip_len: int, img_size: int):
        super().__init__()
        from pytorchvideo.models.x3d import create_x3d

        backbone = create_x3d(input_clip_length=clip_len, input_crop_size=img_size, model_num_class=400)
        head = backbone.blocks[-1]
        dimension = head.proj.in_features
        head.proj = nn.Identity()
        head.activation = nn.Identity()
        self.backbone = backbone
        self.accel = nn.Linear(dimension, n_accel)
        self.steer = nn.Linear(dimension, n_steer)

    def forward(self, x):
        # x: (B, T, 3, H, W) -> X3D는 (B, 3, T, H, W)를 기대
        feat = self.backbone(x.permute(0, 2, 1, 3, 4)).flatten(1)
        return self.accel(feat), self.steer(feat)


def _build_stage3_model(arch: str, n_accel: int, n_steer: int, clip_len: int, img_size: int) -> nn.Module:
    if arch == "gru":
        return _Stage3GRU(n_accel, n_steer)
    if arch == "x3d":
        return _Stage3X3D(n_accel, n_steer, clip_len, img_size)
    return _Stage3TSN(n_accel, n_steer)  # "tsn" 및 이전 체크포인트(arch 필드 없음) 기본값


def _stage3_frames(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width, height = image.size
        scale = S3_IMG_SIZE / min(width, height)
        image = image.resize((round(width * scale), round(height * scale)))
        width, height = image.size
        x, y = (width - S3_IMG_SIZE) // 2, (height - S3_IMG_SIZE) // 2
        image = image.crop((x, y, x + S3_IMG_SIZE, y + S3_IMG_SIZE))
        frames.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).to(torch.uint8))
    capture.release()
    if not frames:
        raise ValueError(f"cannot decode video: {path.name}")
    return torch.stack(frames)


def predict_stage3(data_dir, model_dir):
    device = _device()
    checkpoint = torch.load(Path(model_dir) / "best.pt", map_location="cpu", weights_only=False)
    accel_labels = checkpoint["accel_labels"]
    steer_labels = checkpoint["steer_labels"]
    # 클립 길이/아키텍처는 학습 환경(로컬 Mac은 tsn+8프레임, GPU 서버는 gru/x3d+16프레임 등)에
    # 따라 다를 수 있으므로 체크포인트에 저장된 값을 그대로 써서 학습/추론 불일치를 방지한다.
    clip_len = checkpoint.get("clip_len", 8)
    arch = checkpoint.get("arch", "tsn")
    model = _build_stage3_model(arch, len(accel_labels), len(steer_labels), clip_len, S3_IMG_SIZE)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    videos = _video_paths(Path(data_dir))  # "videos/" 유무 상관없이 재귀 탐색 (위키 문서 표기 불일치 대응)
    rows = []
    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    half = clip_len // 2
    with torch.inference_mode():
        for path in videos:
            frames = _stage3_frames(path)
            count = len(frames)
            centers = np.arange(count)
            accel_predictions, steer_predictions = [], []
            for start in range(0, count, clip_len):
                center = centers[start : start + clip_len]
                indices = np.clip(center[:, None] - half + np.arange(clip_len)[None, :], 0, count - 1)
                clips = frames[torch.from_numpy(indices)].float() / 255.0  # (B, T, 3, H, W)
                clips = (clips - S3_MEAN[None, None, :, :, :]) / S3_STD[None, None, :, :, :]
                with torch.autocast(device_type=autocast_device, dtype=torch.float16, enabled=device.type == "cuda"):
                    accel_logits, steer_logits = model(clips.to(device, non_blocking=True))
                accel_predictions.extend(accel_logits.argmax(1).cpu().tolist())
                steer_predictions.extend(steer_logits.argmax(1).cpu().tolist())
            for sample_index, (accel, steer) in enumerate(zip(accel_predictions, steer_predictions)):
                rows.append(
                    {
                        "ID": path.stem,
                        "sample_index": sample_index,
                        "accel_label": accel_labels[accel],
                        "steer_label": steer_labels[steer],
                    }
                )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(rows, columns=["ID", "sample_index", "accel_label", "steer_label"])
