# Quadruped LTV-MPC Clean Repo

이 압축파일은 **기존에 흩어져 있던 Python baseline + MuJoCo 실험 코드를 한 번에 정리한 버전**이다.

## 폴더 구조

```text
quadruped_mpc_mujoco_clean_repo/
├── core/                  # high-level Python MPC core
├── mujoco/                # MuJoCo low-level helpers / quasi-static branch
├── runners/               # 실제 실행 파일
├── scripts/               # setup script
├── docs/                  # attribution / current status
├── requirements.txt
└── README.md
```

## 이 패키지에서 유지한 것

### 1) Python baseline core
- `config.py`
- `fsm.py`
- `reference.py`
- `footholds.py`
- `model.py`
- `qp_builder.py`
- `controller_osqp.py`
- `plant.py`
- `plotting.py`

### 2) MuJoCo branch
- `low_level_realizer.py`
- `quasistatic_confirmed_helper_heightfix.py`
- `quasistatic_visual_helpers.py`
- `run_mujoco_quasistatic.py`

## 왜 이렇게 정리했나

기존에는 `phase1`, `phase2`, ..., `phase16`처럼 실험용 runner가 너무 많아졌고,
무엇이 현재 기준 branch인지 헷갈리기 쉬웠다.

그래서 이번 정리본은 다음 원칙으로 만들었다.

- **실제로 의미 있었던 branch만 남김**
- **baseline / MuJoCo 두 경로만 유지**
- **실험용 누적 phase 파일은 제거**
- **앞으로는 이 구조 위에서 계속 수정**

## 권장 사용 순서

### 0) 새 폴더에 압축 해제
기존 복잡한 폴더를 그대로 쓰지 말고, 이 압축을 새 폴더에 풀어서 시작하는 걸 권장한다.

### 1) 환경 설치
```bash
bash scripts/setup_mujoco_env.sh
source .venv/bin/activate
```

### 2) Python baseline 확인
```bash
python runners/run_python_baseline.py --scenario straight_trot
python runners/run_python_baseline.py --scenario turn_pi_over_4
```

### 3) MuJoCo 확인
```bash
python runners/run_mujoco_quasistatic.py   --model ./mujoco_menagerie/unitree_a1/scene.xml   --support-enabled   --disable-nonfoot-collision
```

### 4) 영상 저장
```bash
python runners/run_mujoco_quasistatic.py   --model ./mujoco_menagerie/unitree_a1/scene.xml   --headless   --support-enabled   --disable-nonfoot-collision   --save-mp4 outputs/mujoco/quasistatic.mp4
```

## 현재 코드 상태를 어떻게 이해하면 되나

- **Python baseline**: high-level MPC loop 검증 완료
- **MuJoCo**: stable dynamic trot은 아직 아님
- **MuJoCo current branch**: immediate collapse 없이 quasi-static crawl prototype 수준
- 즉, 현재 위치는 **“working MuJoCo integration prototype with stable support and slow forward crawl”**에 가깝다.

## 앞으로 이 코드베이스에서 수정할 곳

앞으로는 새 phase 파일을 계속 추가하지 말고, 아래 3개 위주로 수정하는 걸 권장한다.

1. `mujoco/quasistatic_confirmed_helper_heightfix.py`
   - stance / swing / touchdown / recovery 로직
2. `mujoco/low_level_realizer.py`
   - 바인딩, Jacobian, contact, support utility
3. `runners/run_mujoco_quasistatic.py`
   - 실험 파라미터, 로깅, 영상 저장

## 삭제해도 되는 것

기존 폴더에 남아 있는 아래 류의 파일은 더 이상 기준 branch가 아니므로, 이 정리본을 쓰기 시작하면 사용하지 않아도 된다.

- `runner_mujoco_phase*.py`
- `mujoco_phase*_patch*`
- 예전 `outputs_mujoco_phase*`

## 메모

이 repo는 **논문 수준 free trot 완성본**이 아니라,
**Python MPC core + MuJoCo prototype을 한 코드베이스로 계속 발전시키기 위한 clean starting point**다.
