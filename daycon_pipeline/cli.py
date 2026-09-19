"""Stage1/2/3 학습 통합 CLI.

`pip install -e .` 로 설치하면 `daycon-train` 명령으로 쓸 수 있다.

  daycon-train --stage1            # Stage1(재녹화 판별)만 학습
  daycon-train --stage3            # Stage3(가감속/조향)만 학습
  daycon-train --stage1 --stage2   # 여러 개 지정하면 순서대로 실행

설치 없이 바로 쓰려면 Baseline/ 안에서:
  python -m daycon_pipeline.cli --stage3
"""
from __future__ import annotations

import argparse

STAGES = ["stage1", "stage2", "stage3"]


def _run_stage1():
    from stage1_work.train import main as stage1_main
    stage1_main()


def _run_stage2():
    from stage2_evasion_work.train import main as stage2_main
    stage2_main()


def _run_stage3():
    from stage3_work.train import main as stage3_main
    stage3_main()


_RUNNERS = {"stage1": _run_stage1, "stage2": _run_stage2, "stage3": _run_stage3}


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="DACON 블랙박스 경진대회 Stage 학습 CLI")
    for stage in STAGES:
        parser.add_argument(f"--{stage}", action="store_true", help=f"{stage} 학습 실행")
    args = parser.parse_args(argv)

    selected = [s for s in STAGES if getattr(args, s)]
    if not selected:
        parser.error("--stage1 / --stage2 / --stage3 중 최소 하나는 지정해야 합니다")

    for stage in selected:
        print(f"\n{'=' * 20} {stage} 학습 시작 {'=' * 20}")
        _RUNNERS[stage]()


if __name__ == "__main__":
    main()
