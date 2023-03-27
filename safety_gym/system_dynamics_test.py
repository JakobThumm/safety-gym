"""Test the system dynamics."""
import argparse
import gym
import safety_gym  # noqa
import numpy as np  # noqa
from copy import deepcopy
import matplotlib.pyplot as plt

MASS = 0.00518879
GRAVITY = 9.81
FRICTION = 0.01
GEAR = 0.3
GEAR_TURN = 3


def force_2_acc(force):
    """Calculate the acceleration."""
    return GEAR * force / MASS


def dynamics(state, u, dt):
    """Calculate the next state of the system."""
    # x, y, v_x, v_y, theta
    x, y, v_x, v_y, theta = state
    # u = [force, steering]
    force, steering = u
    limit = 0.05
    force = np.clip(force, -limit, limit)
    steering = np.clip(steering, -1, 1)
    acc = force_2_acc(force)
    F_mu = FRICTION * MASS * GRAVITY
    sub_steps = 1
    for i in range(sub_steps):
        theta += GEAR_TURN * steering * dt/sub_steps
        v_x += (acc * np.cos(theta) - np.sign(v_x) * FRICTION * F_mu) *\
            dt/sub_steps
        v_y += (acc * np.sin(theta) - np.sign(v_y) * FRICTION * F_mu) *\
            dt/sub_steps
        x += v_x * dt/sub_steps
        y += v_y * dt/sub_steps
    return np.array([x, y, v_x, v_y, theta])


def quat_2_euler(q):
    """Convert quaternion to euler angles."""
    # q = [w, x, y, z]
    w, x, y, z = q
    # Calculate the euler angles
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    pitch = np.arcsin(2 * (w * y - z * x))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    return np.array([roll, pitch, yaw])


def run_dynamics_test(env_name):
    """Test the system dynamics."""
    env = gym.make(env_name)
    env.seed(1)
    obs = env.reset()
    state_0 = deepcopy(env.sim.data.get_body_xpos('robot')[0:2])
    theta_0 = quat_2_euler(deepcopy(env.sim.data.get_body_xquat('robot')))[2]
    length = 1000
    dt = 0.002
    true_states = np.zeros((length, 5))
    model_states = np.zeros((length, 5))
    done = False
    ep_ret = 0
    ep_cost = 0
    agent_inputs = np.array([0.05, 1])
    # x, y, v_x, v_y, theta
    p_true = deepcopy(env.sim.data.get_body_xpos('robot')[0:2]) - state_0
    v_true = deepcopy(env.sim.data.get_body_xvelp('robot'))
    theta_true = quat_2_euler(
        deepcopy(env.sim.data.get_body_xquat('robot'))
    )[2]
    true_states[0] = np.array(
        [p_true[0], p_true[1], v_true[0], v_true[1], theta_true]
    )
    s = np.array([0, 0, 0, 0, theta_0])
    model_states[0] = s
    for i in range(1, length):
        if i < length/3:
            agent_inputs = np.array([0.05, 0.5])
        else:
            # new failsafe strategy
            acc_opt = -1/dt * (
                v_true[0] * np.cos(theta_true) + v_true[1] * np.sin(theta_true)
            )
            if np.linalg.norm(v_true) > 0.01:
                u1 = acc_opt * MASS / GEAR
                delta_theta = np.arctan2(v_true[1], v_true[0]) - theta_true
            else:
                u1 = 0
                delta_theta = 0
                # robot_stopped = True
            if np.abs(delta_theta) > np.pi:
                delta_theta -= np.sign(delta_theta) * 2 * np.pi

            if abs(delta_theta) > 0.01 and np.linalg.norm(v_true) > 0.1:
                u2 = np.clip(delta_theta / (GEAR_TURN * dt), -1, 1)
            else:
                u2 = 0
            agent_inputs = np.array([u1, u2])
        # assert env.observation_space.contains(obs)
        act = agent_inputs
        # assert env.action_space.contains(act)
        obs, reward, done, info = env.step(act)
        p_true = deepcopy(env.sim.data.get_body_xpos('robot')[0:2]) - state_0
        v_true = deepcopy(env.sim.data.get_body_xvelp('robot'))
        theta_true = quat_2_euler(
            deepcopy(env.sim.data.get_body_xquat('robot')))[2]
        true_states[i] = np.array(
            [p_true[0], p_true[1], v_true[0], v_true[1], theta_true]
        )
        s = dynamics(s, act, dt)
        model_states[i] = s
        # print('reward', reward)
        ep_ret += reward
        ep_cost += info.get('cost', 0)
        env.render()
    # Plot y over x position
    plt.figure()
    plt.plot(true_states[:, 0], true_states[:, 1], label='True')
    plt.plot(model_states[:, 0], model_states[:, 1], label='Model')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend()
    # Plot v over time
    plt.figure()
    plt.plot(np.arange(length)*dt, true_states[:, 2], label='True v_x')
    plt.plot(np.arange(length)*dt, model_states[:, 2], label='Model v_x')
    plt.plot(np.arange(length)*dt, true_states[:, 3], label='True v_y')
    plt.plot(np.arange(length)*dt, model_states[:, 3], label='Model v_y')
    plt.xlabel('time [s]')
    plt.ylabel('v [m/s]')
    plt.legend()
    # plot theta over time
    plt.figure()
    plt.plot(np.arange(length)*dt, true_states[:, 4], label='True')
    plt.plot(np.arange(length)*dt, model_states[:, 4], label='Model')
    plt.xlabel('time [s]')
    plt.ylabel('theta [rad]')
    plt.legend()
    # plot delta theta over time
    plt.figure()
    plt.plot(np.arange(length)*dt,
             np.arctan2(true_states[:, 3], true_states[:, 2]) -
             true_states[:, 4],
             label='True')
    plt.plot(np.arange(length)*dt,
             np.arctan2(model_states[:, 3], model_states[:, 2]) -
             model_states[:, 4],
             label='Model')
    plt.xlabel('time [s]')
    plt.ylabel('delta theta [rad]')
    plt.legend()
    plt.show()
    # stop=0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', default='Safexp-PointGoal0-v0')
    args = parser.parse_args()
    run_dynamics_test(args.env)
