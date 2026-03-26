# Step 3: MuJoCo smoke test from your current baseline repo

Put the three files below into your repo's `code/` folder:
- `phases/runner_mujoco_smoke.py`
- `requirements/requirements_mujoco.txt`
- `scripts/setup_mujoco_env.sh`

Then inside WSL Ubuntu:

```bash
cd /mnt/c/Users/<YOUR_WINDOWS_NAME>/quadruped-python-baseline/code
chmod +x scripts/setup_mujoco_env.sh
./scripts/setup_mujoco_env.sh
source .venv/bin/activate
python -m mujoco.viewer --mjcf=./mujoco_menagerie/unitree_a1/scene.xml
```

If the viewer opens, run the first integration test:

```bash
source .venv/bin/activate
python phases/runner_mujoco_smoke.py --model ./mujoco_menagerie/unitree_a1/scene.xml --scenario straight_trot
```

For the turning case:

```bash
source .venv/bin/activate
python phases/runner_mujoco_smoke.py --model ./mujoco_menagerie/unitree_a1/scene.xml --scenario turn_pi_over_4
```

For headless plots only:

```bash
source .venv/bin/activate
python phases/runner_mujoco_smoke.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --scenario straight_trot \
  --headless \
  --output-dir local_outputs/outputs_mujoco/straight_trot
```

What success looks like:
- MuJoCo window opens.
- The robot model loads.
- The script runs without import/path errors.
- At the end, figures are saved under `local_outputs/outputs_mujoco/...`.

If the robot immediately collapses, that is still acceptable for a first smoke test.
This stage is only meant to show that your **current high-level MPC loop is connected to an articulated MuJoCo backend**.
