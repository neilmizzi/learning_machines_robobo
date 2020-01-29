#!/usr/bin/env python2
from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
matplotlib.use('Agg')
import os
import time
import numpy as np
from numpy import inf, random
import matplotlib.pyplot as plt
import pickle

import json

import robobo
import cv2
import sys
import signal
from pprint import pprint
import prey

import collections

use_simulation = True
run_test = False
speed = 20
dist = 500
rewards = []
fitness = []


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo("#0").connect(address='10.15.3.49', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="192.168.43.187")

    def get_sensor_info(direction):
        a = np.log(np.array(rob.read_irs())) / 10
        all_sensor_info = np.array([0 if x == inf else 1 + (-x / 2) - 0.2 for x in a]) if use_simulation \
            else np.array(np.log(rob.read_irs())) / 10
        all_sensor_info[all_sensor_info == inf] = 0
        all_sensor_info[all_sensor_info == -inf] = 0
        # [0, 1, 2, 3, 4, 5, 6, 7]
        if direction == 'front':
            return all_sensor_info[5]
        elif direction == 'back':
            return all_sensor_info[1]
        elif direction == 'front_left':
            return np.max(all_sensor_info[[6, 7]])
        elif direction == 'front_right':
            return np.max(all_sensor_info[[3, 4]])
        elif direction == 'back_left':
            return all_sensor_info[0]
        elif direction == 'back_right':
            return all_sensor_info[2]
        elif direction == 'all':
            return all_sensor_info
        else:
            raise Exception('Invalid direction')

    # safe, almost safe, not safe. combine with previous state of safe almost safe and not safe.
    # safe to almost safe is good, almost safe to safe is okay, safe to safe is neutral
    # s to a to r to s'.
    # Small steps for going left or right (left or right are only rotating and straight is going forward).
    # controller is the q values: the boundary for every sensor.

    def move_left():
        rob.move(-speed, speed, dist)

    def move_right():
        rob.move(speed, -speed, dist)

    def go_straight():
        rob.move(speed, speed, dist)

    def move_back():
        rob.move(-speed, -speed, dist)

    boundary = [0.5, 0.8] if not use_simulation else [0.75, 0.95]

    # A static collision-avoidance policy
    def static_policy(s):
        if get_sensor_info('front_left') >= s \
                and get_sensor_info('front_left') > get_sensor_info('front_right'):
            take_action('right')

        elif get_sensor_info('front_right') >= s \
                and get_sensor_info('front_right') > get_sensor_info('front_left'):
            take_action('left')
        else:
            take_action('straight')

    state_table = {}
    if os.path.exists('./src/state_table.json'):
        with open('./src/state_table.json') as f:
            state_table = json.load(f)

    def epsilon_policy(s, epsilon):
        s = str(s)
        # epsilon greedy
        """"
        ACTIONS ARE DEFINED AS FOLLOWS:
          NUM: ACTION
            ------------
            0: STRAIGHT
            1: LEFT
            2: RIGHT
            ------------
        """
        e = 0 if run_test else epsilon
        if e > random.random():
            return random.choice([0, 1, 2])
        else:
            return np.argmax(state_table[s])

    def take_action(action):
        if action == 1:
            move_left()
        elif action == 2:
            move_right()
        elif action == 0:
            go_straight()
        # elif action == 'back':
        #     move_back()

    def get_reward(current, new, action):
        if current == 0 and new == 0:
            return 0 if action == 0 else -1
        elif current == 0 and new == 1:
            return 1
        elif current == 0 and new == 2:
            return -10
        elif current == 1 and new == 0:
            return 1
        elif current == 1 and new == 1:
            return 1 if action == 0 else 0
        elif current == 1 and new == 2:
            return -10
        return 0
        # TODO give negative reward for repetitions

    def make_discrete(values, boundaries):
        discrete_list = []
        for x in values:
            if x > boundaries[1]:
                discrete_list.append(2)
            elif boundaries[1] > x > boundaries[0]:
                discrete_list.append(1)
            elif boundaries[0] > x:
                discrete_list.append(0)
        return discrete_list

    """
    REINFORCEMENT LEARNING PROCESS
    INPUT:  alpha    : learning rate
            gamma    : discount factor
            epsilon  : epsilon value for e-greedy
            episodes : no. of episodes
            act_lim  : no. of actions robot takes before ending episode
            qL       : True if you use Q-Learning
    """

    def rl(alpha, gamma, epsilon, episodes, act_lim, qL=False):
        for i in range(episodes):
            print('Episode ' + str(i))
            terminate = False
            if use_simulation:
                rob.play_simulation()

            current_state = make_discrete(get_sensor_info('all')[3:], boundary)

            if str(current_state) not in state_table.keys():
                state_table[str(current_state)] = [0 for _ in range(3)]

            action = epsilon_policy(current_state, epsilon)
            # initialise state if it doesn't exist, else retrieve the current q-value
            x = 0
            while not terminate:
                take_action(action)
                new_state = make_discrete(get_sensor_info('all')[3:], boundary)

                if str(new_state) not in state_table.keys():
                    state_table[str(new_state)] = [0 for _ in range(3)]

                new_action = epsilon_policy(new_state, epsilon)

                # Retrieve the max action if we use Q-Learning
                max_action = np.argmax(state_table[str(new_state)]) if qL else new_action

                # Get reward
                r = get_reward(np.max(current_state), np.max(new_state), action)

                normalized_r = ((r - -10) / (2 - -10)) * (1 - -1) + -1
                fitness.append(normalized_r * np.max(get_sensor_info("all")[3:]))
                # print(fitness)
                if rewards:
                    rewards.append(rewards[-1] + normalized_r)
                else:
                    rewards.append(normalized_r)

                # Update rule
                print("r: ", r)

                if not run_test:
                    print('update')
                    state_table[str(current_state)][action] += \
                        alpha * (r + (gamma *
                                      np.array(
                                          state_table[str(new_state)][max_action]))
                                 - np.array(state_table[str(current_state)][action]))

                # Stop episode if we get very close to an obstacle
                if max(new_state) == 2 or x == act_lim-1:
                    state_table[str(new_state)][new_action] = -10
                    terminate = True
                    print("done")
                    if not run_test:
                        print('writing json')
                        with open('./src/state_table.json', 'w') as json_file:
                            json.dump(state_table, json_file)

                    if use_simulation:
                        print("stopping the simulation")
                        rob.stop_world()
                        while not rob.is_sim_stopped():
                            print("waiting for the simulation to stop")
                        time.sleep(2)

                # update current state and action
                current_state = new_state
                action = new_action

                # increment action limit counter
                x += 1

    # alpha, gamma, epsilon, episodes, actions per episode
    rl(0.9, 0.9, 0.05, 30, 500, qL=True)

    pprint(state_table)

    if not run_test:
        all_rewards = []
        all_fits = []
        if os.path.exists('./src/rewards.csv'):
            with open('./src/rewards.csv') as f:
                all_rewards = pickle.load(f)

        if os.path.exists('./src/fitness.csv'):
            with open('./src/fitness.csv') as f:
                all_fits = pickle.load(f)

        all_rewards += rewards
        all_fits += fitness

        print(all_rewards)
        print(all_fits)

        with open('./src/rewards.csv', 'w') as f:
            pickle.dump(all_rewards, f)

        with open('./src/fitness.csv', 'w') as f:
            pickle.dump(all_fits, f)

        plt.figure('Rewards')
        plt.plot(all_rewards)
        plt.savefig("./src/plot_reward.png")
        plt.show()

        plt.figure('Fitness Values')
        plt.plot(all_fits)
        plt.savefig("./src/plot_fitness.png")


if __name__ == "__main__":
    main()
