"""YOLOP(hustvl/YOLOP, MIT License) 로컬 벤더 코드 로더.

evasion_space의 "세그멘테이션(주행가능영역) 활용" 아이디어와 entry_frame의
"차선 기반 곡선도로 대응"을 위해 사전학습된 YOLOP(BDD100K로 학습, 객체탐지+
주행가능영역 세그멘테이션+차선 세그멘테이션을 한 모델로 동시 수행)를 쓴다.

원본 repo(github.com/hustvl/YOLOP)를 pip으로 설치할 수 없어서 필요한 코드(lib/)만
그대로 복사해뒀다 — 평가서버는 인터넷이 없어 torch.hub로 받을 수 없기 때문.
제출용 submit.zip 구조가 `model/`, `inference.py`, `requirements.txt`만 허용하므로,
벤더 코드와 체크포인트 둘 다 `model/stage2/`(yolop_code/lib/, yolop_end2end.pth) 안에
둔다 — inference.py도 이 경로를 그대로 참조한다(dev 스크립트와 제출 코드가 같은
자산을 공유).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

_VENDOR_DIR = Path(__file__).resolve().parent.parent / "model" / "stage2" / "yolop_code"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

YOLOP_IMG_SIZE = 640
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_yolop(weights_path, device: torch.device):
    from lib.config import cfg
    from lib.models import get_net

    model = get_net(cfg)
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model


def run_yolop(model, frame_bgr: np.ndarray, device: torch.device):
    """frame_bgr(원본 해상도) -> (drivable_mask, lane_mask), 둘 다 원본 해상도의 bool 배열.

    처음에 정사각형으로 단순 리사이즈했더니 드리브러블 영역이 거의 0으로 나와서
    (BDD100K 학습 시 종횡비를 유지하는 letterbox 640 리사이즈를 썼는데, 정사각형
    squish는 도로 형태를 심하게 왜곡해 모델을 크게 out-of-distribution으로 만듦)
    원본 repo와 동일하게 종횡비 유지 letterbox(32의 배수로 패딩)를 그대로 재사용한다.
    """
    from lib.utils.augmentations import letterbox_for_img

    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    letterboxed, ratio, (dw, dh) = letterbox_for_img(rgb, new_shape=YOLOP_IMG_SIZE, auto=True)
    x = (letterboxed.astype(np.float32) / 255.0 - _MEAN) / _STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)

    with torch.inference_mode():
        _, da_seg_out, ll_seg_out = model(x)

    lh, lw = letterboxed.shape[:2]
    dw, dh = int(round(dw)), int(round(dh))
    da_full = da_seg_out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    ll_full = ll_seg_out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    # 패딩 영역 제거 후 원본 해상도로 복원(letterbox의 역변환)
    da_crop = da_full[dh : lh - dh, dw : lw - dw]
    ll_crop = ll_full[dh : lh - dh, dw : lw - dw]
    da_mask = cv2.resize(da_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ll_mask = cv2.resize(ll_crop, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return da_mask, ll_mask
