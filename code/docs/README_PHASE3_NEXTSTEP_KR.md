
# MuJoCo Phase-3 (KR)

이번 단계의 목적은 phase-2에서 확인된 두 가지 사실을 바로 반영하는 것입니다.

1. A1 모델의 실제 foot collision geom은 각 calf body 아래의 가장 아래 sphere geom입니다.
   - FR=19, FL=30, RR=40, RL=50
2. actuator는 direct torque motor가 아니라 position-servo 계열이므로,
   stance에서 `d.ctrl`로 계속 자세를 강하게 밀면 MPC가 내는 GRF와 충돌합니다.

따라서 phase-3는 다음을 합니다.

- stance: `d.ctrl` target을 현재 joint q로 두어 servo fighting을 줄임
- swing: swing 시작 순간의 joint 상태를 anchor로 저장
- swing: Jacobian pseudo-inverse를 사용해
  - 위로 드는(clearance) foot displacement
  - 앞으로 보내는(step length) foot displacement
  를 joint-space target으로 변환

## 실행

```bash
source .venv/bin/activate
python phases/runner_mujoco_phase3.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase3/straight_default
```

파라미터 스윕 예시:

```bash
python phases/runner_mujoco_phase3.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase3/c050_s060 \
  --clearance 0.05 \
  --step-len 0.06

python phases/runner_mujoco_phase3.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase3/c060_s080 \
  --clearance 0.06 \
  --step-len 0.08
```

## 이번 단계 성공 기준

- mean mismatch가 phase-2의 0.499보다 더 내려감
- `v_x` 평균이 phase-2보다 올라감
- front 뿐 아니라 rear mismatch도 같이 내려가기 시작함

## 다음 단계

phase-3가 먹히면 그 다음은
- touchdown heuristic(간단한 foothold update)
- MPC 50 Hz vs low-level faster inner loop 정리
- turning scenario 재실행
순서로 가면 됩니다.
