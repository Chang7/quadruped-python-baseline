# MuJoCo Phase-1 정리

이번 묶음은 "바로 잘 걷게 만드는 최종본"이 아니라, 아래 3가지를 한 번에 정리한 **다음 단계용 코드 정리본**입니다.

1. `fig_xy_path.png`가 실제 기준으로 그려지도록 수정
2. `scheduled contact`와 `actual contact`를 같이 기록하고 비교 플롯 추가
3. `runner_mujoco_phase1.py`에서 **stance / swing 분리 뼈대**를 넣어, 
   - stance leg는 MPC 힘을 generalized force로 반영
   - swing leg는 간단한 joint-space swing lift를 사용

## 추가/교체 파일

### 교체
- `plotting.py`

### 추가
- `mujoco_phase1_helpers.py`
- `runner_mujoco_phase1.py`
- `README_PHASE1_NEXTSTEP_KR.md`

## 실행 순서

```bash
cd /mnt/c/Users/<YOUR_NAME>/quadruped-python-baseline/code
source .venv/bin/activate
python runner_mujoco_phase1.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot
```

headless:

```bash
python runner_mujoco_phase1.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase1/straight_trot
```

## 새로 생기는 그림

- `fig_velocity_tracking.png`
- `fig_yaw_tracking.png`
- `fig_leg_fz_subplots.png`
- `fig_xy_path.png`  ← 이제 reference velocity 적분 기반
- `fig_contact_schedule_vs_actual.png`
- `fig_contact_mismatch_summary.png`

## 해석 포인트

### 1. mismatch ratio가 높다
아직 실제 MuJoCo 접촉이 MPC의 가정과 잘 안 맞는 상태입니다.

### 2. swing lift를 넣었는데도 전진 속도가 안 나온다
이 경우는 다음 우선순위가 `J^T f` 기반 stance torque 정교화 / foot placement 개선입니다.

### 3. turning은 아직 안 좋아도 괜찮다
먼저 `straight_trot`에서 scheduled vs actual contact mismatch를 줄이는 것이 우선입니다.

## 권장 다음 단계

1. `straight_trot`만 반복 실행
2. mismatch ratio부터 줄이기
3. 그 다음에 turning 케이스 재실행
4. 이후 `stance: J^T f`, `swing: simple foot placement`로 확장
