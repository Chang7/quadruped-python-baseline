import mujoco
import numpy as np

m = mujoco.MjModel.from_xml_path("./mujoco_menagerie/unitree_a1/scene.xml")

legs = ["FR", "FL", "RR", "RL"]
for leg in legs:
    calf_name = f"{leg}_calf"
    calf_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, calf_name)
    print(f"\n[{leg}] calf body id = {calf_id}")
    for gid in range(m.ngeom):
        if int(m.geom_bodyid[gid]) != calf_id:
            continue
        pos = np.asarray(m.geom_pos[gid], dtype=float)
        size = np.asarray(m.geom_size[gid], dtype=float)
        print(
            f" geom {gid}: type={int(m.geom_type[gid])}, pos={np.round(pos,4)}, size={np.round(size,4)}, "
            f"contype={int(m.geom_contype[gid])}, conaffinity={int(m.geom_conaffinity[gid])}"
        )
