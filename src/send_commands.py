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
speed = 20 if use_simulation else 30
dist = 500 if use_simulation else 400
rewards = [0]
fitness = [0]


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='145.108.233.253', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="10.15.3.48")

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
            return all_sensor_info[6]
        elif direction == 'front_left_left':
            return all_sensor_info[7]
        elif direction == 'front_right':
            return all_sensor_info[4]
        elif direction == 'front_right_right':
            return all_sensor_info[3]
        elif direction == 'back_left':
            return all_sensor_info[0]
        elif direction == 'back_right':
            return all_sensor_info[2]
        elif direction == 'all':
            print(all_sensor_info[3:])
            return all_sensor_info
        elif direction == 'front_3':
            return [all_sensor_info[3]] + [all_sensor_info[5]] + [all_sensor_info[7]]
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

    boundary_sensor = 0.7 if not use_simulation else 0.95
    boundaries_color = [0.3, 0.7] if not use_simulation else [0.4, 0.95]

    # A static collision-avoidance policy
    def static_policy(s):
        if get_sensor_info('front_left') >= s \
                and get_sensor_info('front_left') > get_sensor_info('front_right'):
            return 2

        elif get_sensor_info('front_right') >= s \
                and get_sensor_info('front_right') > get_sensor_info('front_left'):
            return 1
        else:
            return 0

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

    def get_color_info():
        image = rob.get_image_front()

        def get_green_pixels(img):
            count = 0
            pix_count = 0
            b = 64
            for i in range(len(img)):
                for j in range(len(img[i])):
                    pixel = img[i][j]
                    pix_count += 1
                    if (pixel[0] > b or pixel[2] > b) and pixel[1] < b * 2 \
                            or (pixel[0] > b and pixel[1] > b and pixel[2] > b):
                        img[i][j] = [0, 0, 0]
                        count += 1
            return 1 - (count / pix_count)

        # rob.set_phone_tilt(109, 100)
        # image = rob.get_image_front()
        # print(image)

        left, middle_l, middle_r, right = np.hsplit(image, 4)
        middle = np.concatenate((middle_l, middle_r), axis=1)

        return get_green_pixels(left), get_green_pixels(middle), get_green_pixels(right)

    def get_reward(previous_state, new_state, action, prev_val, new_val):
        total_c_prev = sum(prev_val)
        total_c_new = sum(new_val)
        max_new_sensor = np.max(new_state[:3])
        max_prev_sensor = np.max(previous_state[:3])
        max_c_new = np.max(previous_state[3:])
        if max_c_new == 2:
            # TODO: check when actually hitting a green block, because sometimes gives +30 when not touching yet.
            #  Changing boundary doesn't help
            print("30")
            return 30
        elif total_c_prev < total_c_new:
            # This one is already fixed
            return 10
        elif max_prev_sensor < max_new_sensor:
            # TODO: change this in the non descretized values, would be way better!
            print("-20")
            return -20
        # The else is when it doesn't see anything, maybe we should add penalizing in one of the other if statements
        # as well for going left or right, but not sure
        else:
            return 1 if action == 0 else -1
        return 0
        # TODO give negative reward for repetitions

    # Returns list of values with discretized sensor values and color values.
    def make_discrete(values_s, boundary_s, values_c, boundaries_c):
        discrete_list_s = []
        discrete_list_c = []
        for x in values_s:
            if boundary_s > x:
                discrete_list_s.append(0)
            else:
                discrete_list_s.append(1)
        for y in values_c:
            if y < boundaries_c[0]:
                discrete_list_c.append(0)
            elif boundaries_c[0] < y < boundaries_c[1]:
                discrete_list_c.append(1)
            else:
                discrete_list_c.append(2)
        return discrete_list_s + discrete_list_c

    """
   REINFORCEMENT LEARNING PROCESS
   INPUT:  alpha    : learning rate
           gamma    : discount factor
           epsilon  : epsilon value for e-greedy
           episodes : no. of episodes
           act_lim  : no. of actions robot takes before ending episode
           qL       : True if you use Q-Learning
   """
    stat_fitness = [0]
    stat_rewards = [0]

    def run_static(lim):
        for _ in range(lim):
            if use_simulation:
                rob.play_simulation()

            current_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, get_color_info(),
                                          boundaries_color)

            if str(current_state) not in state_table.keys():
                state_table[str(current_state)] = [0 for _ in range(5)]

            action = static_policy(0.75)

            take_action(action)

            new_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, get_color_info(), boundaries_color)

            # r = get_reward(current_state, new_state, action, current_color_info, new_color_info)

            # normalized_r = ((r - -10) / (1 - -10)) * (1 - -1) + -1
            # stat_fitness.append(stat_fitness[-1] + (normalized_r * np.max(get_sensor_info("all")[3:])))
            # print(fitness)
            # if stat_rewards:
            #     stat_rewards.append(stat_rewards[-1] + normalized_r)
            # else:
            #     rewards.append(normalized_r)

    def rl(alpha, gamma, epsilon, episodes, act_lim, qL=False):
        for i in range(episodes):
            print('Episode ' + str(i))
            terminate = False
            if use_simulation:
                rob.play_simulation()

            current_color_info = get_color_info()
            current_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, current_color_info,
                                          boundaries_color)

            if str(current_state) not in state_table.keys():
                state_table[str(current_state)] = [0 for _ in range(5)]

            action = epsilon_policy(current_state, epsilon)
            # initialise state if it doesn't exist, else retrieve the current q-value
            x = 0
            while not terminate:

                take_action(action)
                new_color_info = get_color_info()
                new_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, new_color_info,
                                          boundaries_color)

                if str(new_state) not in state_table.keys():
                    state_table[str(new_state)] = [0 for _ in range(5)]

                new_action = epsilon_policy(new_state, epsilon)

                # Retrieve the max action if we use Q-Learning
                max_action = np.argmax(state_table[str(new_state)]) if qL else new_action

                # Get reward
                r = get_reward(current_state, new_state, action, current_color_info, new_color_info)
                print(r)

                # normalized_r = ((r - -10) / (1 - -10)) * (1 - -1) + -1
                # fitness.append(fitness[-1] + normalized_r * np.max(get_sensor_info("front_3")))
                # # print(fitness)
                # if rewards:
                #     rewards.append(rewards[-1] + normalized_r)
                # else:
                #     rewards.append(normalized_r)

                # Update rule

                if not run_test:
                    # print('update')
                    state_table[str(current_state)][action] += \
                        alpha * (r + (gamma *
                                      np.array(
                                          state_table[str(new_state)][max_action]))
                                 - np.array(state_table[str(current_state)][action]))

                # Stop episode if we get very close to an obstacle
                if (max(new_state[:3]) == 1 and max(new_state[3:]) != 2 and use_simulation) or x == act_lim - 1:
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
                current_color_info = new_color_info

                # increment action limit counter
                x += 1

    # alpha, gamma, epsilon, episodes, actions per episode
    # run_static(200)
    rl(0.9, 0.9, 0.08, 3, 500, qL=True)

    pprint(state_table)

    if run_test:
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

        # print(all_rewards)
        # print(all_fits)

        # with open('./src/rewards.csv', 'w') as f:
        #     pickle.dump(all_rewards, f)
        #
        # with open('./src/fitness.csv', 'w') as f:
        #     pickle.dump(all_fits, f)
        #
        # with open('./src/stat_rewards.csv', 'w') as f:
        #     pickle.dump(stat_rewards, f)
        #

        # with open('./src/stat_fitness.csv', 'w') as f:
        #     pickle.dump(stat_fitness, f)
        #
        # plt.figure('Rewards')
        # plt.plot(all_rewards, label='Q-Learning Controller')
        # plt.plot(stat_rewards, label='Static Controller')
        # plt.legend()
        # plt.savefig("./src/plot_reward.png")
        # plt.show()
        #
        # plt.figure('Fitness Values')
        # plt.plot(all_fits, label='Q-Learning Controller')
        # plt.plot(stat_fitness, label='Static Controller')
        # plt.legend()
        # plt.savefig("./src/plot_fitness.png")
        # plt.show()


def image_test():
    signal.signal(signal.SIGINT, terminate_program)
    rob = robobo.SimulationRobobo().connect(address='130.37.120.197', port=19997) if use_simulation \
         else robobo.HardwareRobobo(camera=True).connect(address="172.20.10.11")
    if use_simulation:
        rob.play_simulation()
    rob.set_phone_tilt(109, 90)

    print('taking pic')
    image = rob.get_image_front()
    cv2.imwrite("test_pictures.png", image)
    count = 0
    print(image)
    b = 64
    for i in range(len(image)):
        for j in range(len(image[i])):
            pixel = image[i][j]
            if (pixel[0] > b or pixel[2] > b) and pixel[1] < b * 2 \
                    or (pixel[0] > b and pixel[1] > b and pixel[2] > b):
                image[i][j] = [0, 0, 0]
                count += 1
    print(1 - (count / (640 * 480)))
    cv2.imwrite("test_img.png", image)

    if use_simulation:
        print('stopping the simulation')
        rob.stop_world()
        while not rob.is_sim_stopped():
            print("waiting for the simulation to stop")
        time.sleep(2)


if __name__ == "__main__":
    main()
    #image_test()