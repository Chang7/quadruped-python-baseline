# Phase 14: Forward-Supported MuJoCo Demo

이 버전은 phase-13 support demo에서 **앞으로 기울이며 머뭇거리다 멈추는 현상**을 줄이기 위해,
trunk support wrench에 **x-target tracking + 작은 forward bias**를 추가한 버전입니다.

핵심:
- high-level Python MPC는 계속 solve
- 저층은 보수적 crawl realization 유지
- trunk support는
  - 높이 유지
  - yaw 유지
  - x-target 추종
  - 작은 forward bias
  를 함께 사용

## 추천 실행

실시간 보기:
```bash
python phases/runner_mujoco_phase14_forward_supported_visual.py   --model ./mujoco_menagerie/unitree_a1/scene.xml   --scenario straight_trot   --output-dir local_outputs/outputs_mujoco_phase14/live_view   --disable-nonfoot-collision   --support-enabled
```

조금 더 앞으로 끌기:
```bash
python phases/runner_mujoco_phase14_forward_supported_visual.py   --model ./mujoco_menagerie/unitree_a1/scene.xml   --scenario straight_trot   --output-dir local_outputs/outputs_mujoco_phase14/live_view_push   --disable-nonfoot-collision   --support-enabled   --support-weight-frac 0.90   --support-target-height 0.31   --support-x-k 45   --support-forward-bias 5.0   --step-len-front 0.020   --rear-step-scale 0.60
```

만약 코가 아래로 숙여지면:
```bash
  --support-target-pitch -0.03
```
또는
```bash
  --support-target-pitch 0.03
```
중 더 나은 쪽을 선택하세요. 모델 부호에 따라 다를 수 있습니다.

## 출력
- phase14_summary.json
- GIF/MP4 저장 가능
