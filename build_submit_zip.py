"""submit.zip 생성 — 데이콘 코드 제출 규격에 맞춰 model/, inference.py,
requirements.txt만 최상위에 두고, model/ 안은 재귀적으로 전부 포함한다.

베이스라인 노트북([Baseline_Inference]...)의 검증 로직을 그대로 참고하되,
우리가 실제로 predict_stage2를 통째로 교체했으므로 필수 파일 목록은 베이스라인
placeholder 기준이 아니라 우리 inference.py가 실제로 여는 경로 기준으로 다시
정의했다.
"""
from __future__ import annotations

import ast
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
INFERENCE_PATH = ROOT / "inference.py"
REQUIREMENTS_PATH = ROOT / "requirements.txt"
MODEL_DIR = ROOT / "model"
SUBMIT_PATH = ROOT / "submit.zip"

# inference.py가 실제로 torch.load/YOLO()/open()으로 여는 경로들 (efficientnet
# arch 기준 — box_fusion으로 바꾸면 yolop_end2end.pth/yolop_code도 필요해지는데,
# 지금 model/stage2/evasion_space/best.pt의 arch가 efficientnet이라 필수는 아님).
REQUIRED_FILES = [
    "model/stage1/best.pt",
    "model/stage2/evasion_space/best.pt",
    "model/stage2/yolo11n.pt",
    "model/stage3/best.pt",
]
IGNORE_NAMES = {".DS_Store"}


def main():
    if not INFERENCE_PATH.is_file():
        raise FileNotFoundError(f"inference.py가 없습니다: {INFERENCE_PATH}")
    if not REQUIREMENTS_PATH.is_file():
        raise FileNotFoundError(f"requirements.txt가 없습니다: {REQUIREMENTS_PATH}")

    inference_source = INFERENCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(inference_source, filename="inference.py")
    defined = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required_functions = {"predict_stage1", "predict_stage2", "predict_stage3"}
    missing_functions = sorted(required_functions - defined)
    if missing_functions:
        raise RuntimeError(f"필수 추론 함수 누락: {missing_functions}")
    print("inference.py 문법/필수 함수 확인 완료:", sorted(required_functions))

    missing_assets = [f for f in REQUIRED_FILES if not (ROOT / f).is_file()]
    if missing_assets:
        raise RuntimeError(f"필수 모델 자산 누락: {missing_assets}")
    print("필수 모델 자산 확인 완료:", REQUIRED_FILES)

    if SUBMIT_PATH.exists():
        SUBMIT_PATH.unlink()

    with zipfile.ZipFile(SUBMIT_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.write(INFERENCE_PATH, "inference.py")
        archive.write(REQUIREMENTS_PATH, "requirements.txt")
        for path in sorted(MODEL_DIR.rglob("*")):
            if path.is_file() and path.name not in IGNORE_NAMES:
                archive.write(path, path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(SUBMIT_PATH) as archive:
        names = archive.namelist()

    missing_in_zip = sorted(set(REQUIRED_FILES) - set(names))
    if missing_in_zip:
        raise RuntimeError(f"ZIP 내부 필수 자산 누락: {missing_in_zip}")

    size_gb = SUBMIT_PATH.stat().st_size / 1024**3
    print(f"\n생성 완료: {SUBMIT_PATH} ({size_gb:.3f} GB, 제한 10GB)")
    print(f"파일 수: {len(names)}개")
    print("\nZIP 내부 파일 목록:")
    for name in names:
        print(" -", name)


if __name__ == "__main__":
    main()
