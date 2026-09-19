"""Stage 1 재녹화(RERECORDED) 합성 증강.

블랙박스 원본 영상을 입력으로 받아, 위키의 "확인할 수 있는 영상 특성 예시"
(화면 테두리, 주사 패턴/모아레, 반사광, 밝기·색상·명암 변화, 추가 압축,
재촬영 기기의 움직임/원근, 해상도·화면비 차이)를 모사하는 프레임 시퀀스를
생성한다. 클립 단위로 파라미터를 고정하고 프레임마다 약간씩 흔들어,
같은 영상을 손에 든 카메라로 재촬영한 것처럼 보이게 한다.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class RecaptureParams:
    """클립 하나에 고정으로 적용할 재녹화 파라미터 (프레임마다 재사용)."""

    bezel: bool = False
    bezel_top: float = 0.0
    bezel_bottom: float = 0.0
    bezel_left: float = 0.0
    bezel_right: float = 0.0

    moire: bool = False
    moire_freq: float = 0.0
    moire_angle: float = 0.0
    moire_amp: float = 0.0
    moire_speed: float = 0.0

    glare: bool = False
    glare_cx: float = 0.5
    glare_cy: float = 0.3
    glare_r: float = 0.35
    glare_strength: float = 0.25
    glare_drift: float = 0.0

    band: bool = False
    band_freq: float = 0.0
    band_amp: float = 0.0
    band_speed: float = 0.0

    color_gamma: float = 1.0
    color_sat: float = 1.0
    color_temp: float = 0.0  # -1(차갑게) ~ +1(따뜻하게)
    contrast: float = 1.0
    brightness: float = 0.0

    downscale: float = 1.0  # 재촬영 기기 해상도 체감을 위한 다운스케일 배수
    crop_ratio: float = 1.0  # 화면비 다름(가로/세로 크롭)
    crop_axis: str = "w"  # 'w' or 'h'

    shake_amp: float = 0.0  # 손떨림 진폭(px)
    shake_freq: float = 0.0
    perspective: float = 0.0  # 원근 왜곡 강도(0~1)
    perspective_dir: tuple = (1.0, 1.0)

    noise_sigma: float = 0.0
    jpeg_quality: int | None = None  # None이면 미적용
    sharpen_amount: float = 0.0  # 화면 재촬영 시 오히려 선명해지는 경우를 위한 언샤프마스크
    denoise_sigma: float = 0.0  # sharpen 전에 넣는 완만한 디노이즈(양방향 필터) — "디노이즈+샤프닝" 조합용

    seed: int = 0
    style: str = ""  # 디버깅/리포트용 — 어떤 서브스타일이 뽑혔는지 기록

    @staticmethod
    def sample(rng: random.Random, recipe: str = "both") -> "RecaptureParams":
        """recipe: "train" | "val" | "both".

        2차 실험 개정판. 1차 실험(공식 5+5 RERECORDED 확률이 전부 0.000~0.008)의
        문제는 "흐릿해지는 재녹화"만 잔뜩 만들고, 공식 샘플처럼 오히려
        선명해지거나 대비가 강해지는 재녹화 스타일이 부족했던 것으로 보인다.
        또한 베젤/강한 모아레/반사광처럼 눈에 띄는 구조적 단서에 기대는
        지름길 학습 위험도 있었다.

        그래서 이번에는 "서브스타일" 여러 개를 정의해 train/val에 서로 다른
        조합으로 배분한다. 같은 계열(예: 샤프닝)이라도 train과 val에서
        파라미터 범위가 달라 정확히 같은 함수를 암기할 수 없게 한다.
        블러/압축은 특정 서브스타일에서만, 그것도 약하게·확률적으로만 쓰고,
        베젤/반사광/강한 모아레는 전체 확률을 크게 낮췄다.
        """
        p = RecaptureParams(seed=rng.randint(0, 2**31 - 1))

        # --- 공통 베이스 (양쪽 recipe, 모든 서브스타일에 항상 약하게 적용) ---
        # "재녹화는 곧 저화질"이라는 지름길을 막기 위해 대부분의 서브스타일이
        # 오히려 대비·채도를 올리는 쪽으로 치우치게 했다.
        p.contrast = rng.uniform(0.95, 1.35)
        p.brightness = rng.uniform(-0.05, 0.08)
        p.color_sat = rng.uniform(0.95, 1.35)
        p.noise_sigma = rng.uniform(0.0, 4.0) if rng.random() < 0.4 else 0.0
        if rng.random() < 0.4:
            p.shake_amp = rng.uniform(0.2, 1.5)
            p.shake_freq = rng.uniform(0.3, 1.2)

        if recipe == "train":
            styles = [
                ("mild_blur_compress", 0.15),
                ("sharp_clean", 0.25),
                ("resize_sharpen", 0.20),
                ("pixel_grid_mild", 0.20),
                ("classic_structural_rare", 0.20),
            ]
        elif recipe == "val":
            styles = [
                ("gamma_contrast_shift", 0.20),
                ("sharp_clean_v2", 0.25),
                ("scan_band_mild", 0.20),
                ("white_balance_perspective", 0.20),
                ("classic_structural_rare2", 0.15),
            ]
        else:  # "both" — 하위호환
            styles = [
                ("mild_blur_compress", 0.14),
                ("sharp_clean", 0.14),
                ("resize_sharpen", 0.14),
                ("pixel_grid_mild", 0.14),
                ("gamma_contrast_shift", 0.14),
                ("scan_band_mild", 0.14),
                ("classic_structural_rare", 0.16),
            ]

        style = rng.choices([s for s, _ in styles], weights=[w for _, w in styles], k=1)[0]
        p.style = style

        if style == "mild_blur_compress":
            # 완만한 블러 + 약한 압축. 확률적으로만, 그리고 예전보다 훨씬 약하게.
            p.downscale = rng.uniform(0.7, 0.95)
            if rng.random() < 0.5:
                p.jpeg_quality = rng.randint(60, 95)
            p.color_gamma = rng.uniform(0.85, 1.15)

        elif style in ("sharp_clean", "sharp_clean_v2"):
            # 디노이즈(양방향 필터)로 미세 텍스처를 지운 뒤 언샤프마스크로
            # 엣지를 다시 세움 — "화질이 오히려 좋아 보이는" 재녹화.
            if style == "sharp_clean":
                p.denoise_sigma = rng.uniform(15.0, 45.0)
                p.sharpen_amount = rng.uniform(0.6, 1.6)
                p.contrast = rng.uniform(1.05, 1.35)
            else:
                p.denoise_sigma = rng.uniform(25.0, 60.0)
                p.sharpen_amount = rng.uniform(0.4, 1.2)
                p.color_sat = rng.uniform(1.05, 1.4)
            p.color_gamma = rng.uniform(0.85, 1.25)

        elif style == "resize_sharpen":
            # 다운스케일 후 업스케일(블러) + 강한 샤프닝으로 블러를 상쇄
            # → 순 효과가 오히려 또렷해 보이는 경우를 포함한다.
            p.downscale = rng.uniform(0.55, 0.85)
            p.sharpen_amount = rng.uniform(0.7, 1.8)
            p.color_gamma = rng.uniform(0.85, 1.2)

        elif style == "gamma_contrast_shift":
            # 구조적 단서 없이 감마·대비·채도 변화만으로 만든 재녹화.
            p.color_gamma = rng.uniform(0.7, 1.4)
            p.contrast = rng.uniform(1.0, 1.4)
            p.color_sat = rng.uniform(1.0, 1.4)

        elif style == "pixel_grid_mild":
            # 아주 미세한 픽셀 격자/모아레 (예전의 "강한 모아레"보다 훨씬 약함)
            p.moire = True
            p.moire_freq = rng.uniform(0.3, 0.6)
            p.moire_angle = rng.uniform(0, np.pi)
            p.moire_amp = rng.uniform(0.015, 0.045)
            p.moire_speed = rng.uniform(0.05, 0.2)

        elif style == "scan_band_mild":
            # 아주 약한 스캔라인/롤링밴드
            p.band = True
            p.band_freq = rng.uniform(2.0, 5.0)
            p.band_amp = rng.uniform(0.02, 0.06)
            p.band_speed = rng.uniform(0.2, 0.6)

        elif style == "white_balance_perspective":
            p.color_temp = rng.uniform(-0.35, 0.35)
            if rng.random() < 0.6:
                p.perspective = rng.uniform(0.005, 0.02)
                p.perspective_dir = (rng.choice([-1, 1]), rng.choice([-1, 1]))

        elif style in ("classic_structural_rare", "classic_structural_rare2"):
            # 베젤/반사광/모아레 등 눈에 띄는 구조적 단서 — 예전보다 훨씬
            # 낮은 확률·강도로만 등장시켜 "이것만 보면 RERECORDED"라는
            # 지름길이 생기지 않게 한다.
            if rng.random() < 0.35:
                p.bezel = True
                p.bezel_top = rng.uniform(0.0, 0.03)
                p.bezel_bottom = rng.uniform(0.0, 0.05)
                p.bezel_left = rng.uniform(0.0, 0.02)
                p.bezel_right = rng.uniform(0.0, 0.02)
            if rng.random() < 0.3:
                p.glare = True
                p.glare_cx = rng.uniform(0.15, 0.85)
                p.glare_cy = rng.uniform(0.1, 0.6)
                p.glare_r = rng.uniform(0.3, 0.6)
                p.glare_strength = rng.uniform(0.08, 0.2)
                p.glare_drift = rng.uniform(0.0, 0.01)
            if rng.random() < 0.3:
                p.moire = True
                p.moire_freq = rng.uniform(0.15, 0.4)
                p.moire_angle = rng.uniform(0, np.pi)
                p.moire_amp = rng.uniform(0.03, 0.07)
                p.moire_speed = rng.uniform(0.05, 0.3)
            if rng.random() < 0.4:
                p.perspective = rng.uniform(0.005, 0.025)
                p.perspective_dir = (rng.choice([-1, 1]), rng.choice([-1, 1]))

        # 항상: 화면비 살짝 다름(크롭) — 약한 확률로, 어느 서브스타일이든 겹칠 수 있게
        if rng.random() < 0.35:
            p.crop_ratio = rng.uniform(0.88, 0.98)
            p.crop_axis = rng.choice(["w", "h"])

        return p


def _apply_color(frame: np.ndarray, p: RecaptureParams) -> np.ndarray:
    x = frame.astype(np.float32) / 255.0
    x = np.clip(x * p.contrast + p.brightness, 0, 1)
    x = np.clip(x ** (1.0 / max(p.color_gamma, 1e-3)), 0, 1)
    hsv = cv2.cvtColor((x * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * p.color_sat, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0
    if abs(p.color_temp) > 1e-6:
        x[..., 0] = np.clip(x[..., 0] + 0.06 * p.color_temp, 0, 1)  # R
        x[..., 2] = np.clip(x[..., 2] - 0.06 * p.color_temp, 0, 1)  # B
    return (x * 255).astype(np.uint8)


def _apply_moire(frame: np.ndarray, p: RecaptureParams, t: int) -> np.ndarray:
    h, w = frame.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    phase = p.moire_speed * t
    carrier = (
        np.cos(xx * p.moire_freq * np.cos(p.moire_angle) + yy * p.moire_freq * np.sin(p.moire_angle) + phase)
    )
    mod = 1.0 + p.moire_amp * carrier
    out = frame.astype(np.float32) * mod[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_band(frame: np.ndarray, p: RecaptureParams, t: int) -> np.ndarray:
    h = frame.shape[0]
    yy = np.arange(h, dtype=np.float32)
    phase = p.band_speed * t
    band = 1.0 + p.band_amp * np.sin(2 * np.pi * (yy / h * p.band_freq + phase))
    out = frame.astype(np.float32) * band[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_glare(frame: np.ndarray, p: RecaptureParams, t: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cx = (p.glare_cx + p.glare_drift * t) * w
    cy = p.glare_cy * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((xx - cx) / (p.glare_r * w)) ** 2 + ((yy - cy) / (p.glare_r * h)) ** 2
    mask = np.exp(-r2) * p.glare_strength
    out = frame.astype(np.float32) + mask[..., None] * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_bezel(frame: np.ndarray, p: RecaptureParams) -> np.ndarray:
    h, w = frame.shape[:2]
    top, bottom = int(h * p.bezel_top), int(h * p.bezel_bottom)
    left, right = int(w * p.bezel_left), int(w * p.bezel_right)
    if top + bottom + left + right == 0:
        return frame
    out = frame.copy()
    color = (np.random.randint(5, 25),) * 3
    if top:
        out[:top] = color
    if bottom:
        out[h - bottom :] = color
    if left:
        out[:, :left] = color
    if right:
        out[:, w - right :] = color
    return out


def _apply_shake_perspective(frame: np.ndarray, p: RecaptureParams, t: int) -> np.ndarray:
    h, w = frame.shape[:2]
    dx = p.shake_amp * np.sin(2 * np.pi * p.shake_freq * t / 10.0 + p.seed % 7)
    dy = p.shake_amp * np.cos(2 * np.pi * p.shake_freq * t / 9.0 + p.seed % 5)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst = src.copy()
    dst[:, 0] += dx
    dst[:, 1] += dy
    if p.perspective > 0:
        dxp = p.perspective * w * p.perspective_dir[0]
        dyp = p.perspective * h * p.perspective_dir[1]
        dst[0] += [dxp, dyp]
        dst[3] -= [dxp, dyp]
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _apply_crop_resize(frame: np.ndarray, p: RecaptureParams, out_size: tuple[int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    if p.crop_ratio < 1.0:
        if p.crop_axis == "w":
            new_w = int(w * p.crop_ratio)
            x0 = (w - new_w) // 2
            frame = frame[:, x0 : x0 + new_w]
        else:
            new_h = int(h * p.crop_ratio)
            y0 = (h - new_h) // 2
            frame = frame[y0 : y0 + new_h]
    ow, oh = out_size
    dh, dw = max(1, int(oh * p.downscale)), max(1, int(ow * p.downscale))
    small = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (ow, oh), interpolation=cv2.INTER_LINEAR)


def _apply_denoise(frame: np.ndarray, p: RecaptureParams) -> np.ndarray:
    """양방향 필터로 미세 텍스처를 지운다 — "디노이즈 후 샤프닝" 조합의 앞단계.

    가우시안 블러와 달리 엣지는 어느 정도 보존하면서 잔털 같은 고주파
    텍스처만 뭉개, 이후 샤프닝을 거치면 원본과는 다른 "매끈하게 재처리된"
    느낌의 질감이 남는다.
    """
    if p.denoise_sigma <= 0:
        return frame
    return cv2.bilateralFilter(frame, d=7, sigmaColor=p.denoise_sigma, sigmaSpace=p.denoise_sigma)


def _apply_sharpen(frame: np.ndarray, p: RecaptureParams) -> np.ndarray:
    if p.sharpen_amount <= 0:
        return frame
    blurred = cv2.GaussianBlur(frame, (0, 0), sigmaX=1.2)
    out = frame.astype(np.float32) + p.sharpen_amount * (frame.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_noise_jpeg(frame: np.ndarray, p: RecaptureParams) -> np.ndarray:
    if p.noise_sigma > 0:
        noise = np.random.normal(0, p.noise_sigma, frame.shape).astype(np.float32)
        frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if p.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        if ok:
            frame = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return frame


def render_recapture_frame(frame_rgb: np.ndarray, p: RecaptureParams, t: int, out_size: tuple[int, int]) -> np.ndarray:
    """RGB uint8 프레임 하나에 재녹화 파라미터 p를 순서대로 적용한다."""
    x = frame_rgb
    x = _apply_crop_resize(x, p, out_size)
    if p.moire:
        x = _apply_moire(x, p, t)
    if p.band:
        x = _apply_band(x, p, t)
    x = _apply_color(x, p)
    if p.glare:
        x = _apply_glare(x, p, t)
    if p.shake_amp > 0 or p.perspective > 0:
        x = _apply_shake_perspective(x, p, t)
    if p.bezel:
        x = _apply_bezel(x, p)
    x = _apply_denoise(x, p)
    x = _apply_sharpen(x, p)
    x = _apply_noise_jpeg(x, p)
    return x


@dataclass
class MildOriginalParams:
    """ORIGINAL 클래스 다양성을 위한 증강.

    재녹화 전용 구조적 특징(베젤·모아레·반사광·주사밴드·손떨림/원근)은
    절대 넣지 않되, "화질 저하" 자체(노이즈·JPEG 재압축·해상도 손실·색감
    변화)는 RecaptureParams와 겹치는 분포로 확률적으로 넣는다. 그래야
    모델이 화질 저하 유무가 아니라 진짜 구조적 재녹화 흔적으로 판별을
    학습한다.
    """

    brightness: float = 0.0
    contrast: float = 1.0
    sat: float = 1.0
    crop_ratio: float = 1.0
    noise_sigma: float = 0.0
    downscale: float = 1.0
    jpeg_quality: int | None = None

    @staticmethod
    def sample(rng: random.Random) -> "MildOriginalParams":
        return MildOriginalParams(
            brightness=rng.uniform(-0.08, 0.08),
            contrast=rng.uniform(0.88, 1.15),
            sat=rng.uniform(0.85, 1.15),
            crop_ratio=rng.uniform(0.9, 1.0),
            noise_sigma=rng.uniform(0.0, 6.0) if rng.random() < 0.5 else 0.0,
            downscale=rng.uniform(0.6, 1.0) if rng.random() < 0.4 else 1.0,
            jpeg_quality=rng.randint(50, 95) if rng.random() < 0.45 else None,
        )


def render_mild_original_frame(frame_rgb: np.ndarray, p: MildOriginalParams, out_size: tuple[int, int]) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    if p.crop_ratio < 1.0:
        new_w, new_h = int(w * p.crop_ratio), int(h * p.crop_ratio)
        x0, y0 = (w - new_w) // 2, (h - new_h) // 2
        frame_rgb = frame_rgb[y0 : y0 + new_h, x0 : x0 + new_w]
    ow, oh = out_size
    if p.downscale < 1.0:
        dh, dw = max(1, int(oh * p.downscale)), max(1, int(ow * p.downscale))
        small = cv2.resize(frame_rgb, (dw, dh), interpolation=cv2.INTER_AREA)
        x = cv2.resize(small, out_size, interpolation=cv2.INTER_LINEAR)
    else:
        x = cv2.resize(frame_rgb, out_size, interpolation=cv2.INTER_AREA)
    x = np.clip(x.astype(np.float32) * p.contrast + p.brightness * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * p.sat, 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if p.noise_sigma > 0:
        noise = np.random.normal(0, p.noise_sigma, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if p.jpeg_quality is not None:
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(x, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, p.jpeg_quality])
        if ok:
            x = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return x
