#!/usr/bin/env python2
from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import numpy as np
from numpy import inf, random

import robobo
import cv2
import sys
import signal
from pprint import pprint
import prey

import collections

use_simulation = True
speed = 20
dist = 500


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo('#0').connect(address='192.168.1.2', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="192.168.43.187")

    if use_simulation:
        rob.play_simulation()

    def get_sensor_info(direction):
        #   TODO Fix Transformation
        a = np.log(np.array(rob.read_irs())) / 10
        all_sensor_info = np.array([0 if x == inf else 1 + (-x / 2) - 0.2 for x in a]) if use_simulation \
            else np.array(np.log(rob.read_irs())) / 10
        all_sensor_info[all_sensor_info == inf] = 0
        all_sensor_info[all_sensor_info == -inf] = 0
        print(format(all_sensor_info[3:]))
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

    def move(speed1, speed2, dist):
        rob.move(speed1, speed2, dist)

    def move_left():
        rob.move(-speed, speed, dist)

    def move_right():
        rob.move(speed, -speed, dist)

    def go_straight():
        rob.move(speed, speed, dist)

    def move_back():
        rob.move(-speed, -speed, dist)

    boundary = [0.6, 0.8] if not use_simulation else [0.75, 0.95]

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
    epsilon = 0.3

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

        if epsilon > random.random():
            return random.choice([0, 1, 2])
        else:
            print(state_table[s])
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

    def get_reward(current, new):
        if current == 0 and new == 0:
            return 0
        elif current == 0 and new == 1:
            return 2
        elif current == 1 and new == 0:
            return 1
        elif current == 1 and new == 1:
            return 1
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
            epsilon  : discount factor
            episodes : no. of episodes
            act_lim  : no. of actions robot takes before resetting space
    """
    def SARSA(alpha, epsilon):
        terminate = False
        for i in range(5):
            current_state = make_discrete(get_sensor_info('all')[3:], boundary)

            if str(current_state) not in state_table.keys():
                state_table[str(current_state)] = [0 for _ in range(3)]
                action = epsilon_policy(current_state, epsilon)
            # initialise state if it doesn't exist, else retrieve the current q-value
            while not terminate:
                take_action(action)
                new_state = make_discrete(get_sensor_info('all')[3:], boundary)
                new_action = epsilon_policy(current_state, epsilon)
                if str(new_state) not in state_table.keys():
                    state_table[str(new_state)] = [0 for _ in range(3)]

                r = get_reward(max(current_state), max(new_state))
                print("rrrrr", r)

                state_table[str(current_state)][action] += \
                    alpha * (r + epsilon *
                             np.array(state_table[str(new_state)][new_action])
                             - np.array(state_table[str(current_state)][action]))
                if max(new_state) == 2:
                    terminate = True
                    print("done")

                current_state = new_state
                action = new_action

    SARSA(0.9, 0.9)

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("../test_pictures.png", image)
    #
    # time.sleep(0.1)

    # IR reading

    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
    #     time.sleep(0.1)

    if use_simulation:
        # pause the simulation and read the collected food
        rob.pause_simulation()

        # Stopping the simulation resets the environment
        rob.stop_world()

    pprint(state_table)


if __name__ == "__main__":
    main()

