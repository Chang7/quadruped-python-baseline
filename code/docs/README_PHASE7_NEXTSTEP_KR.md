# Phase-7: external foot wrench vs internal J^T f torque realization

이번 단계의 목적은 아주 단순합니다.

지금까지의 실험으로,
- x축 방향 자체가 뒤집힌 건 아니고
- fy를 줄여도 근본 해결이 안 되며
- body-frame 해석이 약간 더 낫다는 것
까지 확인했습니다.

이제 남은 가장 큰 구조적 의심은 **"GRF를 외력으로 넣고 있는 방식 자체"** 입니다.

원래 논문의 stance realization은 개념적으로 `tau = J^T f` 입니다.
반면 지금 phase-6까지는 `qfrc_applied += J_full^T f` 형태라, floating base까지 직접 밀어버리는 외력 realization에 더 가깝습니다.

그래서 phase-7은 두 가지 realization을 비교합니다.

- `--realization external` : 기존 방식 (external foot wrench 스타일)
- `--realization joint`    : 새로운 방식 (leg dof에만 `tau = J_leg^T f` 적용)

## 바로 돌릴 추천 2개

```bash
python phases/runner_mujoco_phase7.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase7/external_ref \
  --realization external \
  --force-frame body
```

```bash
python phases/runner_mujoco_phase7.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase7/joint_realization \
  --realization joint \
  --force-frame body
```

## 해석 기준

다음 4개만 비교하면 됩니다.

- `Mean vx after 1.0 s`
- `Mean mismatch ratio`
- `Mean sum Fx after 1.0 s`
- RL/RR `stance_success_ratio`

### joint realization이 좋아지면
그러면 지금까지의 핵심 병목은 **force sign/frame**이 아니라,
**GRF를 MuJoCo에 넣는 방식**이었다는 뜻입니다.

### 둘 다 비슷하게 나쁘면
그 다음은 foothold update / swing touchdown 쪽으로 다시 돌아가야 합니다.
