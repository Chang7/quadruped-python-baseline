
MuJoCo confirmed quasi-static crawl (phase-16)

What this is
------------
This is a clean, event-driven crawl runner meant to replace endless patch stacking.

Core idea
---------
Instead of trying to free-trot immediately, this runner uses:

1. STARTUP
   - short settling phase
   - all 4 feet down

2. SHIFT
   - move trunk COM target over the support polygon of the 3 stance feet

3. SWING
   - lift / advance one leg only

4. TOUCHDOWN
   - drive the swing foot down until actual foot contact is detected

5. HOLD
   - keep all 4 feet down briefly to transfer load

6. RECOVERY
   - if height / pitch / roll become unhealthy, stop gait progression and recover before continuing

MPC usage
---------
The original Python MPC core is still used:
- config.py
- reference.py
- model.py
- qp_builder.py
- controller_osqp.py

The MuJoCo side is rewritten so that:
- stance legs hold anchored world contact points
- swing legs follow an explicit trajectory
- actual contact gates stance force application
- support force is applied at the trunk
- MPC force is ramped in gradually

Recommended first run
---------------------
python experiments/run_mujoco_quasistatic_confirmed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_quasistatic/live \
  --disable-nonfoot-collision \
  --support-enabled

More visible stepping
---------------------
python experiments/run_mujoco_quasistatic_confirmed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --output-dir local_outputs/outputs_mujoco_quasistatic/visible \
  --disable-nonfoot-collision \
  --support-enabled \
  --clearance 0.095 \
  --step-len-front 0.075 \
  --rear-step-scale 1.0 \
  --dq-limit 0.20 \
  --mpc-force-gain-end 0.28

Video export
------------
python experiments/run_mujoco_quasistatic_confirmed.py \
  --model ./mujoco_menagerie/unitree_a1/scene.xml \
  --headless \
  --output-dir local_outputs/outputs_mujoco_quasistatic/mp4 \
  --disable-nonfoot-collision \
  --support-enabled \
  --save-mp4 local_outputs/outputs_mujoco_quasistatic/mp4/quasistatic.mp4

Main outputs
------------
- quasistatic_summary.json
- fig_velocity_tracking.png
- fig_trunk_height_phase.png
- fig_support_force_z.png
- fig_xy_path.png
- fig_leg_fz_subplots.png
- fig_contact_schedule_vs_actual.png
