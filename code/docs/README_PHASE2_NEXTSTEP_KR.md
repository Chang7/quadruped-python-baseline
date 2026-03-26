# MuJoCo Phase-2 정리

이번 단계는 `phase-1`에서 확인된 큰 병목 두 가지를 직접 고치는 패치입니다.

1. **actual contact**를 calf subtree 전체가 아니라 **foot geom 하나** 기준으로 계산
2. MPC에서 만든 발 힘을 **실제 foot geom 위치**에서 적용

## 왜 이게 중요하나

phase-1 결과에서는 swing amplitude를 키울수록 mismatch가 `0.67 -> 0.62 -> 0.54`로 내려갔습니다.
즉 swing lift 자체는 효과가 있었지만, 여전히 actual contact 판단이 너무 넓거나 force 적용점이 부정확할 가능성이 컸습니다.

Unitree A1 Menagerie 계열 모델은 calf body 아래에 여러 collision geom이 있고, 끝단에는 별도 foot geom이 있는 구조가 흔합니다.
따라서 이번 단계에서는 **가장 distal한 collision-enabled geom**을 자동으로 foot geom 후보로 선택합니다.

## 바꿀 파일

- `mujoco_phase2_helpers.py`
- `runner_mujoco_phase2.py`
- `inspect_leg_geoms.py`

## 추천 실행 순서

```bash
cd /mnt/c/quadruped-python-baseline/code
source .venv/bin/activate
python inspect_leg_geoms.py
```

출력에서 각 calf body의 geom 중 가장 아래(local z가 가장 작음)에 있는 geom이 foot 후보인지 먼저 눈으로 확인합니다.

그 다음:

```bash
python phases/runner_mujoco_phase2.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase2/straight_amp034 \
  --swing-amp 0.34
```

필요하면 amplitude를 더 올려서 비교:

```bash
python phases/runner_mujoco_phase2.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase2/straight_amp040 \
  --swing-amp 0.40
```

## 기대 효과

- 앞다리 actual contact ratio가 더 낮아질 수 있음
- mismatch summary가 더 현실적인 값이 됨
- `qfrc_applied`가 실제 foot point 기준으로 들어가므로 stance wrench 전달이 덜 거칠어짐

## 여전히 남는 것

이 패치는 어디까지나 phase-2입니다. 아직 아래는 남아 있습니다.

- stance leg의 보다 정교한 `J^T f` 분해와 saturation 처리
- swing foot placement / touchdown logic
- dual-rate low-level loop
- turning case 재검증

그래도 이번 패치가 먹히면, 다음 단계는 훨씬 명확해집니다.
