
# MuJoCo phase-10: rear stance load-bias experiment

## 목적
phase-9 결과로 숨은 shin/calf 지지보다는 **rear feet가 actual load를 거의 못 받는 문제**가 더 크다는 점이 드러났습니다.

A1의 actuator는 position-servo 계열이라, scheduled stance + actual contact 이후 `target=current_q`만 주면
rear leg가 **실제로 땅을 누르지 못한 채** 가볍게 스치고 지나갈 수 있습니다.

phase-10은 이 가설을 직접 실험하기 위한 patch입니다.

## 바뀐 점
- **rear actual contact 후**: 발을 약간 **더 아래(-dq_up)** / **더 뒤(-dq_fwd)** 로 누르는 stance target 추가
- **front actual contact 후**: 발을 약간 **위(+dq_up)** 로 풀어주는 unload target 추가
- touchdown-search / foot-only / support-force logging 은 phase-9와 동일

## 권장 실행
### 1) foot-only collision + external realization 기준점
```bash
python phases/runner_mujoco_phase10.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase10/foot_only_bias_default \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision
```

### 2) 더 강한 rear press/back
```bash
python phases/runner_mujoco_phase10.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco_phase10/foot_only_bias_stronger \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision \
  --rear-stance-press 0.025 \
  --rear-stance-back 0.015 \
  --front-stance-unload 0.006
```

## 해석 포인트
좋아지는 패턴:
- RL/RR `foot_act_fz` 또는 `support_act_fz` 증가
- `mean_vx_after_1s`가 덜 음수이거나 양수로 이동
- `support_minus_foot_fz`는 여전히 거의 0 유지 (foot-only contact 유지)

만약 이 patch에서 rear actual Fz가 올라가고 vx가 좋아지면,
다음 단계는 **rear stance bias를 정식 low-level stance controller로 정리**하는 것입니다.

반대로 rear actual Fz가 여전히 거의 0이면,
다음 우선순위는 **rear foothold placement/geometry** 쪽입니다.
