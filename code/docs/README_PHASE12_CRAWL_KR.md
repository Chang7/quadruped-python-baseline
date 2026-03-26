# Phase-12: MPC-informed conservative crawl gait for MuJoCo

이 버전은 기존의 동적 trot을 바로 살리려는 시도 대신,
**같은 high-level MPC를 유지한 채 MuJoCo에서 시각적으로 더 안정적인 보행**을 먼저 얻기 위한 보수적 crawl(한 번에 한 다리 swing) 래퍼입니다.

## 핵심 아이디어
- high-level MPC는 계속 풉니다.
- 다만 MuJoCo low-level realization은 trot 대신 **3발 지지 crawl gait**로 바꿉니다.
- MPC 결과는 직접 외력으로 크게 넣지 않고,
  **stance-leg load ratio를 stance press bias로 반영**합니다.
- foot-only collision을 유지하여 shin/calf 숨은 지지를 피합니다.

## 권장 실행
### 실시간 보기
```bash
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase12/live_view \
  --disable-nonfoot-collision
```

### GIF 저장
```bash
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase12/gif_run \
  --disable-nonfoot-collision \
  --save-gif local_outputs/outputs_mujoco_phase12/gif_run/phase12.gif
```

### MP4 저장
```bash
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase12/mp4_run \
  --disable-nonfoot-collision \
  --save-mp4 local_outputs/outputs_mujoco_phase12/mp4_run/phase12.mp4
```

## 더 보수적인 설정
```bash
python phases/runner_mujoco_phase12_crawl_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase12/live_view_safe \
  --disable-nonfoot-collision \
  --settle-time 1.0 \
  --gait-ramp-time 1.4 \
  --crawl-phase-duration 0.36 \
  --swing-duration 0.18 \
  --step-len-front 0.03 \
  --rear-step-scale 0.8 \
  --stance-drive-front 0.003 \
  --stance-drive-rear 0.004 \
  --height-k 1.1 \
  --pitch-k 0.04 \
  --roll-k 0.03 \
  --dq-limit 0.08
```

## 해석 포인트
- phase-11처럼 바로 일어서서 뒤집히면 안 됩니다.
- ideal은:
  - settle 동안 서 있는 자세 유지
  - 1발씩 천천히 swing
  - trunk 높이를 유지한 채 아주 느리더라도 forward drift
- `phase12_summary.json`에서 특히 볼 값:
  - `mean_vx_after_1s`
  - `mean_trunk_height_after_1s`
  - `min_trunk_height_after_1s`
  - `collapse_time`

## 주의
이 버전은 **final dynamic trot**이 아니라, MuJoCo articulated simulator에서
**MPC-informed locomotion prototype을 눈으로 확인하기 위한 보수적 visual integration step**입니다.
