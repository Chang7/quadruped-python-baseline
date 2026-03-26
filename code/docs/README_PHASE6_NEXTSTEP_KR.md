
# MuJoCo Phase-6: stance-force direction / frame audit

이번 단계의 목적은 더 이상 touchdown을 계속 만지는 것이 아니라,
`vx`가 계속 음수로 나오는 원인이 **수평 힘의 방향/프레임/스케일**인지 확인하는 것입니다.

## 추천 실행 순서

### 1) 기준점: no-latch + 기존 힘 그대로
```bash
python phases/runner_mujoco_phase6.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase6/baseline
```

### 2) 수평 힘 제거 테스트
```bash
python phases/runner_mujoco_phase6.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase6/zero_tangent \
  --zero-tangential
```

### 3) x-force 부호 뒤집기 테스트
```bash
python phases/runner_mujoco_phase6.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase6/flip_fx \
  --fx-scale -1.0
```

### 4) body-frame -> world-frame 회전 테스트
```bash
python phases/runner_mujoco_phase6.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase6/body_frame \
  --force-frame body
```

## 해석 기준

- `zero_tangential`에서 `vx`가 덜 음수면: 수평 힘 적용이 문제
- `flip_fx`에서 `vx`가 양수로 바뀌면: x-force 부호가 뒤집혀 있을 가능성 높음
- `body_frame`에서 `vx`가 개선되면: 힘 프레임 해석이 어긋났을 가능성 있음

## 꼭 봐야 할 숫자
- `mean_vx_after_1s`
- `mean_sum_fx_after_1s`
- 각 다리의 `mean_fx_when_enabled`
- `mean_mismatch_ratio`

결론적으로 phase-6은 성능 튜닝 단계가 아니라 **원인 규명 단계**입니다.
