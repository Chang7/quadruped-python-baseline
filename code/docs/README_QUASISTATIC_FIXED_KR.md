# MuJoCo quasi-static fixed branch

이 버전은 기존 `run_mujoco_quasistatic_confirmed.py`가 너무 일찍 `RECOVERY`에 들어가서
실제로 step phase로 거의 못 넘어가는 문제를 직접 고친 수정본입니다.

## 핵심 수정
- recovery 너무 일찍 들어가던 문제 수정 (`--recovery-enable-after` 추가)
- recovery에 갇히던 문제 수정 (`--recovery-force-exit-after`, `--recovery-required-contacts` 추가)
- 기본 crawl order를 rear-first로 변경
- target height, support gain, step length를 더 보수적으로 변경
- shift target의 fore/aft bias를 줄여 rear-sitting을 완화
- recovery 중 body target을 약간 forward로 잡아 뒤로 주저앉는 경향을 줄임

## 필요한 파일
- `run_mujoco_quasistatic_fixed.py`
- `quasistatic_confirmed_helper_fixed.py`

기존 code 폴더에 아래 파일들이 이미 있어야 합니다.
- `config.py`
- `reference.py`
- `model.py`
- `qp_builder.py`
- `controller_osqp.py`
- `low_level_realizer.py`
- `quasistatic_visual_helpers.py`
- `plotting.py`

## 권장 실행
기본:
```bash
python run_mujoco_quasistatic_fixed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir outputs_mujoco_quasistatic_fixed/live \
  --disable-nonfoot-collision \
  --support-enabled
```

발이 더 보이게 움직이는 버전:
```bash
python run_mujoco_quasistatic_fixed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir outputs_mujoco_quasistatic_fixed/visible \
  --disable-nonfoot-collision \
  --support-enabled \
  --visual-step-boost 1.25 \
  --clearance 0.070 \
  --step-len-front 0.050 \
  --rear-step-scale 0.95 \
  --recovery-enable-after 1.40 \
  --recovery-force-exit-after 0.70
```

headless mp4:
```bash
python run_mujoco_quasistatic_fixed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --headless \
  --output-dir outputs_mujoco_quasistatic_fixed/mp4 \
  --disable-nonfoot-collision \
  --support-enabled \
  --save-mp4 outputs_mujoco_quasistatic_fixed/mp4/fixed.mp4
```

## 이번 버전에서 봐야 할 것
- 이제 `mode_counts`에 `SHIFT`, `SWING`, `TOUCHDOWN`, `HOLD`가 실제로 생기는지
- `collapse_time`이 늘어나는지
- `mean_actual_contact_ratio`에서 rear foot 접촉 비율이 이전보다 올라가는지
