# MuJoCo Phase-9: support-geom audit + foot-only collision toggle

이번 단계의 핵심은 **foot sphere만 보고 있던 actual GRF logger가 실제 하중 지지(support)를 놓치고 있는지 확인하는 것**입니다.

A1 lower-leg에는 foot sphere 외에도 floor와 접촉 가능한 capsule geoms가 들어 있습니다.
따라서 foot-only actual GRF가 아주 작게 나왔다면, 실제 지지는 **shin / lower-leg capsule**에서 일어나고 있을 가능성이 큽니다.

## 새 기능

- foot-only actual GRF와 all-support actual GRF를 **동시에 기록**
- `phase9_summary.json`에 두 값 모두 저장
- 새 그림
  - `fig_total_commanded_vs_foot_vs_support_force.png`
  - `fig_per_leg_foot_vs_support_fz.png`
- `--disable-nonfoot-collision`
  - foot sphere 외의 lower-leg collision geoms를 런타임에서 꺼서
    **오직 foot만 floor에 닿도록 강제**

## 권장 실행 순서

### 1) 현재 contact support audit
```bash
python phases/runner_mujoco_phase9.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase9/support_audit \
  --realization external \
  --force-frame body
```

### 2) foot-only collision 강제
```bash
python phases/runner_mujoco_phase9.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase9/foot_only_collision \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision
```

## 이번에 제일 중요하게 볼 값

- `mean_vx_after_1s`
- `mean_actual_sum_fz_after_1s` (foot-only)
- `mean_actual_support_sum_fz_after_1s` (all-support)
- 각 다리의
  - `foot_act_fz`
  - `support_act_fz`
  - `support_minus_foot_fz`

## 해석 가이드

### support Fz >> foot-only Fz 이면
현재 로봇은 **발바닥이 아니라 shin/calf contact로 지지**되고 있다는 뜻입니다.
즉 MPC 문제보다 **contact geometry / realization** 문제가 더 큽니다.

### `--disable-nonfoot-collision` 후 foot-only Fz가 크게 올라가면
원인이 거의 확정입니다.
다음 단계는 **foot-only collision 상태에서 swing / stance를 다시 조정**하면 됩니다.

### `--disable-nonfoot-collision` 후 바로 쓰러지면
지금까지는 shin contact가 로봇을 버티게 해주고 있었던 겁니다.
이 경우에도 정보는 매우 유용합니다. 이후에는 **true foot-touchdown / foothold update**를 더 넣어야 합니다.
