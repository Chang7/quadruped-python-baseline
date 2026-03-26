# MuJoCo Adaptive Clean Runner

이 패치는 기존 `run_mujoco_clean.py` 위에 **adaptive recovery supervisor**를 얹은 버전입니다.

핵심 아이디어:
- 시작 직후에는 `STARTUP` 모드로 천천히 서 있게 함
- trunk height / pitch / roll 이 불안정해지면 `RECOVERY` 모드로 전환
- recovery 동안 gait clock을 멈추고 4발 stance + 강한 support로 자세를 다시 잡음
- 안정되면 다시 `WALK` 모드로 돌아가 crawl gait를 계속 진행
- health 기반으로
  - desired speed
  - mpc force gain
  - step amplitude
  - stance drive
  를 자동으로 줄이거나 늘림

## 필요한 기존 파일
이 runner는 기존 `code/` 폴더에 아래 파일들이 있다고 가정합니다.
- `config.py`
- `reference.py`
- `model.py`
- `qp_builder.py`
- `controller_osqp.py`
- `plotting.py`
- `low_level_realizer.py`
- `mujoco_clean_visual.py`

## 기본 실행
```bash
python experiments/run_mujoco_adaptive_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_adaptive/live \
  --disable-nonfoot-collision \
  --support-enabled
```

## 조금 더 보수적인 추천 실행
```bash
python experiments/run_mujoco_adaptive_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_adaptive/live_safe \
  --disable-nonfoot-collision \
  --support-enabled \
  --desired-speed-cap 0.12 \
  --step-len-front 0.060 \
  --rear-step-scale 0.90 \
  --startup-time 0.55 \
  --recovery-height-enter-frac 0.75 \
  --recovery-height-exit-frac 0.84 \
  --recovery-pitch-enter 0.34 \
  --recovery-roll-enter 0.24
```

## MP4 저장
```bash
python experiments/run_mujoco_adaptive_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --headless \
  --output-dir local_outputs/outputs_mujoco_adaptive/mp4 \
  --disable-nonfoot-collision \
  --support-enabled \
  --save-mp4 local_outputs/outputs_mujoco_adaptive/mp4/adaptive.mp4
```

## 새로 기록되는 핵심 지표
`adaptive_summary.json`에 아래가 저장됩니다.
- `mean_vx_after_1s`
- `mean_trunk_height_after_1s`
- `min_trunk_height_after_1s`
- `mean_abs_pitch_after_1s`
- `mean_abs_roll_after_1s`
- `collapse_time`
- `mode_counts`
- `recovery_count`
- `mean_health_after_1s`

이 버전의 목적은 **무작정 patch를 더 붙이는 것보다, 걷다가 몸통이 무너질 때 recovery를 통해 다시 버티고 계속 진행할 수 있게 하는 것**입니다.
