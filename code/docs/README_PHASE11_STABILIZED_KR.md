# Phase-11 Stabilized Visual Runner

이 버전은 **지금까지의 direct GRF realization 실험보다, 먼저 MuJoCo에서 몸통 높이를 유지하면서 더 덜 무너지는 보행 모양을 확인**하기 위한 안정화(wrapper) 러너입니다.

## 중요한 점
- 이 러너는 `phase10/phase9`처럼 commanded force를 바로 external wrench로 쓰는 것이 목적이 아닙니다.
- 기본값에서는 `mpc_force_gain=0.0`이므로, **MPC는 주로 고수준 스케줄/부하 분배 힌트로만 사용**되고,
  실제 시각적 안정화는 foot-space / position-servo 기반 low-level stabilization이 담당합니다.
- 즉 **시각적으로 더 덜 무너지는지, trunk height를 더 오래 유지하는지**를 먼저 확인하는 버전입니다.

## 주요 변화
1. `settle_time` 동안 네 다리를 모두 지지 모드로 두고 먼저 자세를 가라앉힘
2. `gait_ramp_time` 동안 스윙/스텝을 서서히 키움
3. stance leg는
   - trunk height error
   - roll/pitch error
   - rear/back bias
   를 반영해 foot-space에서 목표를 생성
4. swing leg는 liftoff -> touchdown target으로 부드러운 foot trajectory를 따라감
5. 기본값에서는 `disable_nonfoot_collision=True`로 foot-only contact를 유지하는 것이 좋음

## 추천 실행
실시간 보기:
```bash
python runner_mujoco_phase11_stabilized_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir outputs_mujoco_phase11/live_view \
  --disable-nonfoot-collision
```

GIF 저장:
```bash
python runner_mujoco_phase11_stabilized_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase11/gif_run \
  --disable-nonfoot-collision \
  --save-gif outputs_mujoco_phase11/gif_run/phase11.gif
```

MP4 저장:
```bash
python runner_mujoco_phase11_stabilized_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase11/mp4_run \
  --disable-nonfoot-collision \
  --save-mp4 outputs_mujoco_phase11/mp4_run/phase11.mp4
```

## 처음 만져볼 파라미터
- `--settle-time 0.8`
- `--gait-ramp-time 1.2`
- `--stance-press-rear 0.020`
- `--rear-back-bias 0.012`
- `--height-k 1.0`
- `--dq-limit 0.14`

## 성공 기준
- trunk가 바로 바닥에 주저앉지 않음
- 1~2초 정도라도 몸통 높이를 유지하며 발을 교대로 내딛음
- `phase11_summary.json`에서 `collapse_time`이 늦어지거나 없어짐
- `mean_trunk_height_after_1s`가 기존보다 높아짐
- 영상상으로 “즉시 주저앉기 + 발버둥”이 줄어듦
