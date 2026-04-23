import argparse
import time

import rclpy
from rclpy.node import Node
from dls2_interface.msg import BaseState, BlindState, ControlSignal, TrajectoryGenerator, TimeDebug

import numpy as np
np.set_printoptions(precision=3, suppress=True)

import mujoco

# Gym and Simulation related imports
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr


# Config imports
from quadruped_pympc import config as cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS2 Quadruped-PyMPC simulator node.",
    )
    parser.add_argument(
        "--scheduler-freq",
        type=float,
        default=500.0,
        help="Simulation update frequency in Hz.",
    )
    parser.add_argument(
        "--render-freq",
        type=float,
        default=30.0,
        help="Viewer render frequency in Hz.",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run the simulator headless.",
    )
    return parser

# Shell for the controllers ----------------------------------------------
class Simulator_Node(Node):
    def __init__(self, runtime_args: argparse.Namespace):
        super().__init__('Simulator_Node')
        self.runtime_args = runtime_args

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"/base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"/blind_state", 1)
        self.subscriber_control_signal = self.create_subscription(ControlSignal,"/quadruped_pympc_torques", self.get_torques_callback, 1)
        self.subscriber_trajectory_generator = self.create_subscription(TrajectoryGenerator,"/trajectory_generator", self.get_trajectory_generator_callback, 1)

        self.timer = self.create_timer(1.0 / self.runtime_args.scheduler_freq, self.compute_simulator_step_callback)

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None
        self.last_mpc_loop_time = 0.0


        # Mujoco env
        self.env = QuadrupedEnv(
            robot=cfg.robot,
            scene=cfg.simulation_params['scene'],
            sim_dt=1.0 / self.runtime_args.scheduler_freq,
            base_vel_command_type="human"
        )
        self.env.mjModel.opt.gravity[2] = -cfg.gravity_constant
        self.env.reset(random=False)
        
        self.last_render_time = time.time()
        self.render_enabled = not self.runtime_args.no_render
        if self.render_enabled:
            self.env.render()
            self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        # Torque vector
        self.desired_tau = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])

        # Desired PD 
        self.desired_joints_position = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])
        self.desired_joints_velocity = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])

    def _read_feet_contact_state(self) -> list[bool]:
        try:
            feet_contact_state, _, _ = self.env.feet_contact_state(ground_reaction_forces=True)
        except Exception:
            try:
                feet_contact_state, _ = self.env.feet_contact_state()
            except Exception:
                return [False, False, False, False]
        return [bool(contact) for contact in feet_contact_state]


    def get_torques_callback(self, msg):
        
        torques = np.array(msg.torques)

        self.desired_tau.FL = torques[0:3]
        self.desired_tau.FR = torques[3:6]
        self.desired_tau.RL = torques[6:9]
        self.desired_tau.RR = torques[9:12]




    def get_trajectory_generator_callback(self, msg):

        joints_position = np.array(msg.joints_position)

        self.desired_joints_position.FL = joints_position[0:3]
        self.desired_joints_position.FR = joints_position[3:6]
        self.desired_joints_position.RL = joints_position[6:9]
        self.desired_joints_position.RR = joints_position[9:12]
        


    def compute_simulator_step_callback(self):

        action = np.zeros(self.env.mjModel.nu)
        action[self.env.legs_tau_idx.FL] = self.desired_tau.FL.reshape(-1)
        action[self.env.legs_tau_idx.FR] = self.desired_tau.FR.reshape(-1)
        action[self.env.legs_tau_idx.RL] = self.desired_tau.RL.reshape(-1)
        action[self.env.legs_tau_idx.RR] = self.desired_tau.RR.reshape(-1)
        self.env.step(action=action)

        base_lin_vel = self.env.base_lin_vel(frame='world')
        base_ang_vel = self.env.base_ang_vel(frame='base')
        base_pos = self.env.base_pos
        feet_pos = self.env.feet_pos(frame='world')

        base_state_msg = BaseState()
        base_state_msg.pose.position = base_pos
        base_state_msg.pose.orientation = np.roll(self.env.mjData.qpos[3:7],-1)
        base_state_msg.velocity.linear = base_lin_vel
        base_state_msg.velocity.angular = base_ang_vel
        self.publisher_base_state.publish(base_state_msg)

        blind_state_msg = BlindState()
        blind_state_msg.joints_position = self.env.mjData.qpos[7:].tolist()
        blind_state_msg.joints_velocity = self.env.mjData.qvel[6:].tolist()
        blind_state_msg.feet_contact = self._read_feet_contact_state()
        blind_state_msg.current_feet_positions = np.concatenate(
            [
                np.asarray(feet_pos.FL).reshape(-1),
                np.asarray(feet_pos.FR).reshape(-1),
                np.asarray(feet_pos.RL).reshape(-1),
                np.asarray(feet_pos.RR).reshape(-1),
            ],
            axis=0,
        ).tolist()
        self.publisher_blind_state.publish(blind_state_msg)


        # Render only at a certain frequency -----------------------------------------------------------------
        if self.render_enabled and time.time() - self.last_render_time > 1.0 / self.runtime_args.render_freq:
            self.env.render()
            self.last_render_time = time.time()


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    print('Hello from the gym_quadruped simulator.')
    rclpy.init()

    simulator_node = Simulator_Node(args)

    rclpy.spin(simulator_node)
    simulator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
