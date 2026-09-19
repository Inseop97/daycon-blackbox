"""[Baseline_Train] 노트북에서 Stage2/3 학습 코드만 그대로 추출해 실행한다.

목적: submit.zip 구조를 끝까지 검증하기 위해 model/stage2, model/stage3
체크포인트를 만든다 (전사 오류를 피하려고 노트북 코드를 직접 재입력하지
않고 JSON에서 그대로 꺼내온다). model/stage1은 이미 개선된 EfficientNet
체크포인트가 있으므로 fit_stage1()은 호출하지 않는다 — 원본 노트북의
placeholder MViT로 덮어쓰면 안 된다.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
notebook = json.loads((ROOT / "[Baseline_Train]_3Stage_학습.ipynb").read_text(encoding="utf-8"))
code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]

# cell 0 = %pip install (skip, 이미 설치됨)
# cell 1~5 = imports / 설정 / 헬퍼 / 모델 클래스 / fit_stage1,2,3 정의 (그대로 실행)
# cell 6 = 실제 fit_stage1/2/3 호출 (여기만 stage2/3로 교체)
source = "\n\n".join("".join(c["source"]) for c in code_cells[1:6])

# 베이스라인 노트북 자체의 버그 우회: fit_stage2가 torch.inference_mode() 안에서
# ResNet 특징을 뽑아 sequences에 저장한 뒤, 나중에 그 텐서로 temporal 모델을
# 학습(backward)하려고 한다. inference_mode 텐서는 이후 autograd에 쓸 수 없어
# "Inference tensors cannot be saved for backward" 에러가 난다. 원본 노트북
# 파일은 그대로 두고, 여기서 실행할 사본에서만 no_grad로 바꿔 우회한다.
source = source.replace(
    "with torch.inference_mode():\n        for r in df.itertuples():\n            frames=_video_frames(DATA/'stage2'/r.path)",
    "with torch.no_grad():\n        for r in df.itertuples():\n            frames=_video_frames(DATA/'stage2'/r.path)",
)

namespace = {"__name__": "__main__"}
exec(compile(source, "baseline_train_notebook_cells_1_5", "exec"), namespace)

print("device:", namespace["DEVICE"])
namespace["fit_stage2"]()
print("Stage 2 완료 (placeholder)")
namespace["fit_stage3"]()
print("Stage 3 완료 (placeholder)")

model_dir = ROOT / "model"
for p in sorted(model_dir.rglob("*")):
    if p.is_file():
        print(p.relative_to(ROOT), f"{p.stat().st_size / 1024**2:.1f} MB")
