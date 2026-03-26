# MuJoCo Phase-5 Next Step (KR)

이번 단계의 핵심은 두 가지입니다.

1. **rear-biased touchdown**
   - rear leg는 step을 조금 짧게 하고
   - touchdown depth/search window를 더 크게 줍니다.
   - 지금까지 결과에서 RL, RR의 actual contact ratio와 force-enabled ratio가 특히 낮았기 때문입니다.

2. **stance latch / hysteresis**
   - actual contact가 한 번 생기면 짧은 시간(`stance_hold_time`) 동안은
     contact가 순간적으로 끊겨도 stance force를 바로 끄지 않습니다.
   - 이렇게 해야 `force_enabled_ratio`가 너무 낮아지는 문제를 줄일 수 있습니다.

## 추천 첫 실행

```bash
source .venv/bin/activate
python phases/runner_mujoco_phase5.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase5/default
```

## 조금 더 rear-biased 하게

```bash
python phases/runner_mujoco_phase5.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase5/rear_more \
  --rear-step-scale 0.55 \
  --touchdown-depth-rear 0.055 \
  --touchdown-search-window-rear 0.12 \
  --stance-hold-time 0.05
```

## 이번에 가장 먼저 볼 값

- `mean_mismatch_ratio`
- `mean_vx_after_1s`
- `stance_success_ratio` (특히 RL, RR)
- `force_enabled_ratio`
- `force_latched_only_ratio`

## 해석 기준

- mismatch만 낮고 `mean_vx_after_1s`가 0 근처면 아직 실패입니다.
- RL/RR의 `stance_success_ratio`, `force_enabled_ratio`가 올라가야 진짜 개선입니다.
- `force_latched_only_ratio`가 너무 크면 latch가 과하다는 뜻입니다.
