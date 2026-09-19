# 블랙박스 영상 기반 지능형 고의사고 분석 — Stage1/2/3 학습 파이프라인

각 Stage는 정답 라벨이 없거나 부족해서, 라벨링 방식이 서로 다릅니다. 모델은 전부
EfficientNet-B0(2D CNN, ImageNet 사전학습)로 통일해 실험 속도를 확보했습니다.

## Stage1 — 재녹화 여부 판별 (`stage1_work/`)

- **라벨링**: 실제 재녹화 데이터가 없어 CCD 원본 영상을 `ORIGINAL`, 자체 제작
  합성 augmentation(`recapture_augment.py`)을 `RERECORDED`로 라벨링(둘 다 자체 정의).
  train/val마다 다른 `recipe`로 블러+압축/선명화/리사이즈+샤프닝/베젤·무아레(드물게)
  등 서로 다른 스타일 조합을 무작위 적용 — 노이즈·jpeg를 상시 넣으면 모델이
  "압축 흔적=재녹화"로 지름길 학습하는 걸 발견해서 recipe를 분리함.
- **특징추출/모델**: EfficientNet-B0 프레임 단위 분류 → 영상 단위는 여러 프레임
  평균(soft voting). `WeightedRandomSampler`로 클래스 비율(1:2) 보정.
- **아키텍처 옵션** (`STAGE1_ARCH` 환경변수, 기본 `efficientnet`):
  - `xception` — timm의 Xception. 딥페이크/recapture 탐지에서 널리 쓰이는 백본
  - `srm_dual` — RGB 스트림(EfficientNet-B0) + 고정 고역통과 필터(SRM 계열)로
    뽑은 노이즈 잔차 스트림(경량 CNN)을 합치는 이중 스트림

## Stage2 — 사고 주요시점·상황 분석 (`stage2_evasion_work/` 등)

세부 항목 4개, 방식이 각각 다릅니다.

- **collision_frame**: 모델 없이 규칙 기반. CCD `binlabels`(50프레임 이진 라벨)
  전환 시점을 라벨 proxy로 쓰고, 프레임간 픽셀 차이(motion-spike)로 교차검증해
  +3~4프레임 보정.
- **entry_frame/entry_side**: YOLO11n `.track()`(ByteTrack)으로 전 프레임 차량
  박스+트랙ID 수집 → collision_frame 근처에서 크고 중앙에 가까운 박스를 상대차량
  후보로 선정 → `normalized_lateral()`(사다리꼴 반폭 정규화, 원근 착시 제거)로
  좌우 위치 계산 → 최근 궤적 대비 상대 이탈(rolling median 디트렌딩)에 히스테리시스
  적용해 "명확히 바깥→안쪽" 전환만 진입으로 인정. 곡선도로에서 선행차량을
  진입으로 오판하는 문제(comma2k19 곡선 50개로 측정한 오탐률 54%)를 절대
  위치 대신 자기 궤적 대비 상대 이탈로 바꿔 38%로 낮춤(관련 연구의 요레이트
  디트렌딩 방식과 같은 원리). YOLOP 차선 세그멘테이션으로 실제 차선 경계를
  추정해 절대 위치 기준으로 바꿔보는 것도 시도했으나(`_s2_lane_boundary_fit`,
  `_s2_normalized_lateral_lane`, 코드는 남아있지만 기본 비활성) 동일 세트에서
  38%로 동률(LEFT는 개선 28%, RIGHT는 악화 48%로 상쇄)이라 추가 이득이 없어
  꺼둠 — 계산량만 아끼는 방향으로 정리. 더 줄이려면 결국 프레임마다 차선을
  다시 추정하는 등 더 무거운 개선이 필요해 보임.
  - **fps 대응**: 위키 Q&A로 Stage2 평가 영상은 Stage3와 달리 영상마다 fps가
    다르다고 확인됨(범위 비공개). 원래 CCD/comma2k19(10fps) 기준으로 튜닝했던
    프레임 수 상수(후보 탐색 반경, 디트렌딩 윈도우 등)를 전부 초 단위로 바꾸고
    영상별 실제 fps(`cv2.CAP_PROP_FPS`, 비정상값이면 10fps로 폴백)로 환산하도록
    수정함 — 안 하면 fps가 다른 영상에서 윈도우가 의도한 시간 길이와 어긋남.
  - **차량 클래스**: 위키 Q&A로 피의차량·블랙박스차량이 오토바이/자전거인 경우는
    없다고 확인돼, YOLO 탐지 클래스에서 motorcycle을 제외(car/bus/truck만).
- **evasion_space**: 대응 데이터셋이 없어 직접 라벨링. collision_frame 근처에서
  Laplacian 분산이 최대(가장 선명)인 프레임을 자동 선택 → 9장씩 그리드 이미지로
  묶어 일괄 육안 라벨링(410개, 사용가능 354개). EfficientNet-B0 이진분류 +
  balanced sampler.
  - **아키텍처 옵션** (`STAGE2_ARCH=box_fusion`): 처음엔 세그멘테이션 사전학습
    가중치를 구하기 까다로울 것으로 보고 YOLO 차량 탐지 박스 기하특징(면적/위치/
    좌우 여백)만으로 근사했으나, 이후 **YOLOP**(hustvl/YOLOP, MIT License, BDD100K로
    학습된 객체탐지+주행가능영역+차선 세그멘테이션 동시 수행 모델)를 로컬에 받아
    벤더링해서 실제 주행가능영역 비율(전체/좌/우/근거리 4개)을 추가 특징으로 합침
    (`extract_box_features.py`, 총 11개 특징). YOLOP repo는 pip 설치가 안 돼서
    필요한 코드(`lib/`)만 `model/stage2/yolop_code/`에 복사해뒀고, 체크포인트도
    `model/stage2/yolop_end2end.pth`로 로컬 보관(제출 시 인터넷 불필요).

## Stage3 — 가감속/조향 (`stage3_work/`)

- **라벨링**: comma2k19의 CAN 신호(`speed`, `steering_angle`)를 규칙으로 변환.
  - speed<0.5m/s → `STOPPED`, 종가속도(±0.5초 중심차분) 임계값(±0.3)으로
    `ACCELERATING`/`DECELERATING`/`CONSTANT`
  - 조향각 절댓값 6도 임계값 + 부호로 `STRAIGHT`/`LEFT`/`RIGHT`
    (부호는 실제 좌회전 영상으로 직접 검증 — CAN 신호 부호가 표준 관례와 반대였음)
- **전처리**: comma2k19 20fps 원본을 10Hz로 다운샘플(평가 영상 규격과 동일),
  CAN 신호를 프레임 타임스탬프에 선형보간, 조향각은 0.3초 이동평균.
- **특징추출/모델**: 클립(기본 8~16프레임)의 프레임마다 EfficientNet-B0로 특징
  추출 → 시간축을 합치는 방식은 `STAGE3_ARCH` 환경변수로 선택:
  - `tsn` (기본) — 클립 내 평균 풀링. 구현이 가장 단순/빠르지만 프레임 "순서"
    정보를 버림(STOPPED/CONSTANT 구분·LEFT/RIGHT 판단엔 구조적으로 불리)
  - `gru` — 같은 프레임 특징을 양방향 GRU에 순서대로 흘려보내 시간 정보 보존.
    연산량 증가는 미미(GRU 자체가 가벼움), GPU 없이도 바로 시도 가능
  - `x3d` — pytorchvideo의 X3D-S(모션 특화 3D-CNN). SlowFast/I3D보다 가볍지만
    그래도 3D conv라 **GPU 서버 전용**(CUDA 없이 고르면 에러로 막아둠)
  - 베이스라인 제안(`mvit_v2_s`, 3D-CNN)은 CUDA 없는 환경에서 배치 하나에
    4.7초씩 걸려 비현실적이라 `tsn`/`gru`로 교체함.

## 실행 방법

각 Stage는 **따로 실행해도 되고**, 아래처럼 통합 CLI로도 실행할 수 있습니다(둘 다
같은 코드를 그대로 호출하므로 결과는 동일합니다).

```bash
# 개별 실행 (기존 방식, 그대로 동작)
python3 stage1_work/train.py
python3 stage3_work/train.py

# 통합 CLI (pip install -e . 필요)
cd Baseline && pip install -e .
daycon-train --stage1
daycon-train --stage1 --stage2 --stage3   # 여러 개 순서대로

# GPU 서버에서 stage3 설정 조정(환경변수, 선택사항)
STAGE3_BATCH_SIZE=64 STAGE3_NUM_WORKERS=8 daycon-train --stage3

# 아키텍처 옵션 선택 (모두 선택사항, 기본값은 기존 방식)
STAGE1_ARCH=xception daycon-train --stage1
STAGE2_ARCH=box_fusion daycon-train --stage2   # 먼저 stage2_evasion_work/extract_box_features.py 실행 필요
STAGE3_ARCH=gru daycon-train --stage3          # GPU 없이도 시도 가능
STAGE3_ARCH=x3d daycon-train --stage3          # GPU 서버 전용
```

`inference.py`(제출용 `submit.zip`에 들어가는 코드)는 각 체크포인트에 저장된
`arch` 값을 보고 알맞은 모델 클래스를 자동으로 골라 쓰므로, 위 옵션 중 어떤
걸로 학습해도 추론 코드를 따로 손댈 필요는 없습니다.

`predict_stage2`는 베이스라인 원본 placeholder(ResNet18+BiGRU)를 걷어내고
실제 파이프라인(collision_frame은 프레임간 모션 급변 지점, entry_frame/entry_side는
YOLO 추적+기하규칙, evasion_space는 EfficientNet-B0 분류)으로 교체 완료했습니다.
`model/stage2/yolo11n.pt`(제출용 필수 자산, 인터넷 없이 로컬 파일로 로드)가
있어야 동작하며, `evasion_space/best.pt`의 `arch` 값(`efficientnet`/`box_fusion`)에
맞춰 자동 분기합니다.

**주의**: collision_frame은 CCD `binlabels`(대회 평가 영상엔 없는 라벨)가 아니라
모션 급변(motion-spike) 피크를 직접 쓰므로, 실험 때 확인한 "±0.3초 허용오차 내
81~82% 일치"보다 정확도가 낮을 수 있습니다 — 실제 데이콘 샘플 5개로 테스트해보니
4개는 근접(오차 1~2프레임), 1개는 크게 빗나감(모션 스파이크가 진짜 충돌이 아닌
다른 지점에서 잡힘). entry_frame도 기존에 확인된 대로 곡선도로/저시인성에서는
못 찾는 경우가 많아 0(위키 규칙상 "이미 진입")으로 대체됩니다.

## 제출 직전 발견한 중요 버그 — Stage2 입력 형식

submit.zip 패키징 전 위키 Q&A("평가 데이터의 디렉토리 구조 및 촬영 방식 확인")를
재확인하다가 발견: **Stage2 평가 데이터는 영상 파일이 아니라 샘플별로 이미
추출된 프레임 이미지(`{ID}/frame_000000.jpg`, ...)로 제공됩니다.** 처음엔
Stage1/3처럼 `data_dir/videos/*.mp4`를 직접 디코딩하도록 짰었는데, 그대로
제출했으면 Stage2가 영상을 하나도 못 찾아 통째로 실패했을 것 — 잡아서 다행.

- `_s2_samples()`가 `data_dir` 안에서 `frame_*.jpg`가 있는 폴더를 재귀적으로
  찾아 샘플로 취급(경로에 `images/` 중간 폴더가 있는지는 문서마다 표기가
  엇갈렸고 운영진 답변도 "수정반영"이라고만 해서 최종 형태를 100% 장담할 수
  없어, 깊이에 상관없이 찾도록 함). 이미지 폴더를 하나도 못 찾으면 영상 파일
  방식으로 자동 폴백.
- Stage1/3의 영상 탐색도 `data_dir/videos/` 하위만 보지 않고 `data_dir` 전체를
  재귀 탐색하도록 같이 방어적으로 바꿔둠(같은 문서 불일치 이슈).
- 이미지 프레임 방식이라 컨테이너 fps 정보가 없어(위키 Q&A로 영상마다 fps가
  다르다고 확인은 됐지만 알아낼 방법이 없음), 이 경로에서는 CCD/comma2k19
  튜닝 기준값(10fps)을 그대로 가정 — 영상 폴백 경로에서만 실제 fps를 읽음.
