
# MuJoCo clean rewrite patch

이 패치는 이전 phase-11~15처럼 wrapper를 계속 얹는 방식이 아니라,
**high-level MPC core는 유지하고 low-level realization만 깨끗하게 다시 쓰는 버전**이다.

## 들어있는 파일
- `experiments/run_mujoco_clean.py`
- `low_level_realizer.py`
- `mujoco_clean_visual.py`
- `requirements_mujoco_clean.txt`

## 설계 철학
- `fsm.py`, `reference.py`, `model.py`, `qp_builder.py`, `controller_osqp.py`는 그대로 사용
- MuJoCo 쪽은 한 파일(`run_mujoco_clean.py`)과 한 모듈(`low_level_realizer.py`)로 다시 정리
- 기본 스케줄은 `crawl`로 시작해서 **발이 실제로 움직이고 보이는 것**을 우선
- startup support는 **짧게만 fade**
- MPC force는 **시간 지나며 점점 키움**
- swing은 **lift -> advance -> touchdown** 3단계
- stance는 **actual contact가 있을 때만** force 적용

## 설치
```bash
source .venv/bin/activate
pip install -r requirements_mujoco_clean.txt
```

## 기본 실행
```bash
python experiments/run_mujoco_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_clean/live \
  --disable-nonfoot-collision \
  --support-enabled
```

## 발을 더 보이게 움직이는 버전
```bash
python experiments/run_mujoco_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_clean/push \
  --disable-nonfoot-collision \
  --support-enabled \
  --settle-time 0.5 \
  --gait-ramp-time 1.0 \
  --clearance 0.08 \
  --step-len-front 0.07 \
  --rear-step-scale 1.00 \
  --dq-limit 0.18 \
  --mpc-force-gain-end 0.40
```

## MP4 저장
```bash
python experiments/run_mujoco_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --headless \
  --output-dir local_outputs/outputs_mujoco_clean/mp4 \
  --disable-nonfoot-collision \
  --support-enabled \
  --save-mp4 local_outputs/outputs_mujoco_clean/mp4/clean.mp4
```

## 추천 시작점
### 안정성 우선
```bash
python experiments/run_mujoco_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_clean/stable \
  --disable-nonfoot-collision \
  --support-enabled \
  --settle-time 0.7 \
  --gait-ramp-time 1.4 \
  --clearance 0.065 \
  --step-len-front 0.05 \
  --rear-step-scale 0.85 \
  --dq-limit 0.14 \
  --mpc-force-gain-end 0.24
```

### 발이 보이게 움직이는 쪽
```bash
python experiments/run_mujoco_clean.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_clean/visible_steps \
  --disable-nonfoot-collision \
  --support-enabled \
  --settle-time 0.45 \
  --gait-ramp-time 0.9 \
  --clearance 0.085 \
  --step-len-front 0.08 \
  --rear-step-scale 1.0 \
  --dq-limit 0.20 \
  --mpc-force-gain-end 0.42
```

## 중요한 해석
이 코드는 **논문 수준의 완전한 free trot 재현**이 아니라,
지금 네 코드베이스에서 **MPC가 들어간 articulated MuJoCo demo를 cleaner way로 다시 세우는 시작점**이다.

즉, 목적은:
- support wrapper가 발을 얼리는 문제 줄이기
- phase patch를 정리하기
- 금요일 미팅 이후 개발을 계속할 수 있는 **메인 branch**로 삼기
