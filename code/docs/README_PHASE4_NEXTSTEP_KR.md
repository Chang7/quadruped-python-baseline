
# MuJoCo Phase-4 (KR)

이번 단계의 목적은 phase-3에서 드러난 핵심 문제를 직접 해결하는 것입니다.

phase-3의 낮은 mismatch는 일부 구간에서 **발이 거의 닿지 않아서 생긴 착시**일 수 있습니다.
따라서 phase-4는 단순히 mismatch만 줄이는 것이 아니라, 아래 세 가지를 동시에 보도록 바꿉니다.

1. mean mismatch
2. mean `v_x` after 1.0 s
3. scheduled stance 동안 실제 contact가 살아 있는지 (`stance_success_ratio`)

## 핵심 로직

- scheduled swing:
  - 위로 들고(clearance)
  - 앞으로 보내고(step length)
  - 후반부에는 다시 아래로 내립니다(late-swing descent)
- scheduled stance + actual no contact:
  - 아직 touchdown이 안 되었으므로 **MPC stance force를 넣지 않습니다**
  - 대신 작은 전진 bias와 downward search를 넣어 **touchdown search**를 수행합니다
- scheduled stance + actual contact:
  - 그때만 MPC의 GRF를 `qfrc_applied`로 넣습니다

즉, phase-4는 **actual contact가 생길 때만 stance force를 쓰는 contact-gated stance** 구조입니다.

## 실행 예시

기본값:

```bash
source .venv/bin/activate
python runner_mujoco_phase4.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase4/default
```

touchdown search를 조금 더 강하게:

```bash
python runner_mujoco_phase4.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase4/td_stronger \
  --touchdown-depth 0.04 \
  --touchdown-forward 0.025
```

## 저장 결과

기존 figure들 외에 `phase4_summary.json`도 같이 저장됩니다.

여기서 꼭 볼 값:
- `mean_mismatch_ratio`
- `mean_vx_after_1s`
- `stance_success_ratio`
- `force_enabled_ratio`
- `touchdown_delay_mean_s`

## 해석 기준

- mismatch만 낮고 `mean_vx_after_1s`가 거의 0이면 아직 착시일 수 있습니다.
- `stance_success_ratio`와 `force_enabled_ratio`가 같이 올라가야 진짜 개선입니다.
- phase-4가 먹히면 그 다음은
  - foothold / touchdown heuristic refinement
  - turning scenario
  - low-level inner loop tuning
  순서로 가면 됩니다.
