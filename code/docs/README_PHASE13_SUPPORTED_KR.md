# Phase-13: Support-assisted MuJoCo demo

이 버전은 **free locomotion을 바로 완성하려는 코드가 아니라**, 금요일 미팅 전에
**MuJoCo에서 로봇이 즉시 주저앉지 않도록 trunk support를 추가한 지원형(assisted) 데모**입니다.

핵심:
- high-level Python MPC loop는 계속 돌립니다.
- MuJoCo 저층은 보수적인 crawl gait를 사용합니다.
- trunk COM에 가상 support wrench를 추가해 즉시 collapse를 줄입니다.
- 목적은 **"MPC-informed MuJoCo integration demo"**를 안정적으로 보여주는 것입니다.

## 파일
- runner_mujoco_phase13_supported_visual.py
- mujoco_phase13_support_helpers.py
- mujoco_visual_helpers.py
- requirements_mujoco_phase13_visual.txt

## 설치
```bash
cd /mnt/c/quadruped-python-baseline/code
source .venv/bin/activate
pip install -r requirements_mujoco_phase13_visual.txt
```

## 가장 먼저 권장하는 live 보기
```bash
python phases/runner_mujoco_phase13_supported_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase13/live_view \
  --disable-nonfoot-collision \
  --support-enabled
```

## 더 보수적인 시작
```bash
python phases/runner_mujoco_phase13_supported_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir local_outputs/outputs_mujoco_phase13/live_view_safe \
  --disable-nonfoot-collision \
  --support-enabled \
  --settle-time 1.2 \
  --gait-ramp-time 2.4 \
  --support-weight-frac 0.90 \
  --support-target-height 0.30 \
  --pitch-sign -1.0 \
  --step-len-front 0.025 \
  --rear-step-scale 0.75
```

## MP4 저장
```bash
python phases/runner_mujoco_phase13_supported_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase13/mp4_run \
  --disable-nonfoot-collision \
  --support-enabled \
  --save-mp4 local_outputs/outputs_mujoco_phase13/mp4_run/phase13.mp4
```

## 요점
이건 **정직하게 말하면 assisted demo**입니다.
즉, 금요일 미팅에서는 다음처럼 설명하면 됩니다.
- "The high-level Python MPC loop is integrated and running in MuJoCo."
- "For bring-up, I added a virtual trunk-support wrench so that I can observe contact, gait timing, and articulated behavior without immediate collapse."
- "Free locomotion without assistance is the next step."
