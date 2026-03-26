# MuJoCo Phase-8 Actual GRF Audit (KR)

이번 단계는 **명령한 힘(commanded force)** 과 **실제로 바닥이 로봇에 준 힘(actual GRF)** 를 분리해서 보는 단계입니다.

지금까지 결과에서 보인 핵심은:
- sign / frame / fy 스윕을 해도 `vx < 0`가 크게 안 바뀜
- 즉, **명령한 힘을 어떻게 넣었느냐** 보다 **실제로 contact solver가 어떤 힘을 만들었느냐** 를 봐야 함

그래서 phase-8은 다음을 추가합니다.

1. `mj_contactForce`로 **foot-floor actual contact force** 를 기록
2. contact sign ambiguity를 피하기 위해 **두 candidate sign convention** 을 모두 계산
3. 평균 total Fz가 더 양수인 convention을 자동 선택
4. `phase8_summary.json`에 commanded vs actual force를 함께 저장
5. 아래 그림을 추가 저장
   - `fig_total_commanded_vs_actual_force.png`
   - `fig_per_leg_commanded_vs_actual_fx.png`

## 설치 / 복사

이 patch의 파일 3개를 기존 `code/` 폴더에 복사하면 됩니다.

- `mujoco_phase8_helpers.py`
- `runner_mujoco_phase8.py`
- `README_PHASE8_NEXTSTEP_KR.md`

## 가장 먼저 돌릴 추천 2개

### 1) 기준점: external + body frame

```bash
source .venv/bin/activate
python runner_mujoco_phase8.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase8/external_ref \
  --realization external \
  --force-frame body
```

### 2) 비교점: joint realization + body frame

```bash
python runner_mujoco_phase8.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase8/joint_realization \
  --realization joint \
  --force-frame body
```

## 가장 먼저 볼 값

`phase8_summary.json`에서 아래 값 8개가 핵심입니다.

- `mean_vx_after_1s`
- `mean_sum_fx_after_1s`
- `mean_actual_sum_fx_after_1s`
- `mean_sum_fy_after_1s`
- `mean_actual_sum_fy_after_1s`
- `mean_sum_fz_after_1s`
- `mean_actual_sum_fz_after_1s`
- `actual_grf_sign_convention`

그리고 각 다리별로는:
- `mean_fx_when_enabled` vs `mean_actual_fx_when_enabled`
- `mean_fy_when_enabled` vs `mean_actual_fy_when_enabled`
- `mean_fz_when_enabled` vs `mean_actual_fz_when_enabled`

## 해석 가이드

### A. actual sum Fx도 음수면
실제로 ground reaction이 backward 방향이라는 뜻입니다.
이 경우는 servo / contact solver / stance realization 쪽이 더 의심됩니다.

### B. commanded sum Fx는 양수인데 actual sum Fx가 0 근처 또는 음수면
명령한 추진력이 actual contact force로 잘 전달되지 않는 것입니다.
즉, **realization mismatch** 입니다.

### C. actual sum Fx는 양수인데 `vx`는 음수면
그때는 foothold placement / touchdown geometry / body attitude 쪽으로 넘어가야 합니다.

## 메모

이 단계는 “성능 향상”이 목적이 아니라 **원인 규명**이 목적입니다.
그래서 `mismatch` 하나보다 **actual GRF와 commanded force의 차이**를 더 중요하게 봐야 합니다.
