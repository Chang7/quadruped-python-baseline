# MuJoCo phase-10 visual runner

이 파일은 `phase-10` 기본 로직을 그대로 쓰면서,

- **실시간 MuJoCo GUI 재생**
- **GIF 저장**
- **MP4 저장**

을 추가한 버전입니다.

## 파일
- `runner_mujoco_phase10_visual.py`
- `mujoco_visual_helpers.py`
- `requirements_mujoco_visual.txt`

## 필요한 준비
기존 phase-10 관련 파일들이 이미 `code/` 폴더에 있어야 합니다.
즉 아래 모듈이 이미 있어야 합니다.

- `mujoco_phase5_helpers.py`
- `mujoco_phase7_helpers.py`
- `mujoco_phase9_helpers.py`
- `mujoco_phase10_helpers.py`
- `config.py`, `fsm.py`, `reference.py`, `model.py`, `qp_builder.py`, `controller_osqp.py`, `plotting.py`

추가 패키지:
```bash
source .venv/bin/activate
pip install -r requirements_mujoco_visual.txt
```

## 가장 추천하는 실행 1: 실제 MuJoCo 창으로 보기
`--headless`를 빼면 실시간 창이 뜹니다.

```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir outputs_mujoco_phase10_visual/live_view \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision
```

## 가장 추천하는 실행 2: GIF 저장
```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase10_visual/gif_run \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision \
  --save-gif outputs_mujoco_phase10_visual/gif_run/straight_trot.gif
```

## MP4 저장
```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase10_visual/mp4_run \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision \
  --save-mp4 outputs_mujoco_phase10_visual/mp4_run/straight_trot.mp4
```

## 카메라 조정 예시
```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase10_visual/cam_test \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision \
  --save-gif outputs_mujoco_phase10_visual/cam_test/straight_trot.gif \
  --camera-distance 1.6 \
  --camera-azimuth 120 \
  --camera-elevation -18
```

## 옵션 설명
- `--headless`: GUI 창 없이 실행
- `--save-gif PATH`: GIF 저장
- `--save-mp4 PATH`: MP4 저장
- `--render-width`, `--render-height`: 렌더 해상도
- `--render-fps`: 저장 fps
- `--camera-distance`, `--camera-azimuth`, `--camera-elevation`: 카메라 각도
- `--camera-lookat X Y Z`: lookat 고정
- `--no-follow-trunk`: 기본은 로봇 trunk를 따라가는데, 이 옵션을 주면 고정 카메라

## 지금 네 상황에서 가장 추천하는 명령
### 1. 먼저 실제 창으로 보기
```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --output-dir outputs_mujoco_phase10_visual/live_view \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision
```

### 2. 그다음 GIF 저장
```bash
python runner_mujoco_phase10_visual.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir outputs_mujoco_phase10_visual/record \
  --realization external \
  --force-frame body \
  --disable-nonfoot-collision \
  --save-gif outputs_mujoco_phase10_visual/record/straight_trot.gif
```

## 출력 파일
- 기존 phase-10 plot들
- `phase9_summary.json` (현재 summary writer 재사용)
- `straight_trot.gif` 또는 `straight_trot.mp4`

## 팁
- WSL GUI가 되면 `--headless` 없이 실제 MuJoCo 창을 바로 볼 수 있습니다.
- 속도가 느리면 `--render-width 640 --render-height 360 --render-fps 20`로 낮추세요.
- 먼저 GIF부터 성공시키고, 그 다음 MP4를 시도하는 것이 가장 안전합니다.


추가 변경사항(v2)
- offscreen 렌더링 기본 크기를 640x480으로 낮췄습니다.
- 모델 framebuffer가 더 작으면 자동으로 가능한 크기로 줄여 다시 시도합니다.
- 요약 JSON에 requested/actual render size를 같이 남깁니다.
