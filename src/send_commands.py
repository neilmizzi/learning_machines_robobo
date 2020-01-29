#!/usr/bin/env python2
from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
import numpy as np
from numpy import inf, random
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
run_test = True
speed = 20 if use_simulation else 50
dist = 500 if use_simulation else 400
rewards = [0]
fitness = list()
MIN_REWARD = -2.5
MAX_REWARD = 30


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    virtual_ip = '192.168.1.2'
    robot_ip = '10.15.3.48'

    rob = robobo.SimulationRobobo().connect(address=virtual_ip, port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address=robot_ip)
    rob.set_phone_tilt(45, 100) if use_simulation else rob.set_phone_tilt(100, 100)

    state_table = {}
    q_table_file = './src/state_table.json'
    if os.path.exists(q_table_file):
        with open(q_table_file) as g:
            state_table = json.load(g)

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

    boundary_sensor = [0.6, 0.8] if not use_simulation else [0.5, 0.95]
    boundaries_color = [0.01, 0.2] if not use_simulation else [0.001, 0.4]

    # A static collision-avoidance policy
    def static_policy(color_info):
        max_c = np.max(color_info)
        if max_c == color_info[0]:
            return 1
        elif max_c == color_info[1]:
            return 0
        elif max_c == color_info[2]:
            return 2
        return 0

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

        # Mask function
        def get_red_pixels(img):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_range = np.array([0, 50, 20])
            upper_range = np.array([5, 255, 255])
            mask = cv2.inRange(hsv, lower_range, upper_range)
            # print(get_green_pixels(image))
            cv2.imwrite('a.png', mask)
            a = b = 0
            for i in mask:
                for j in i:
                    b += 1
                    if j == 255:
                        a += 1
            return a / b
            # count = 0
            # pix_count = 0
            # b = 64
            # for i in range(len(img)):
            #     for j in range(len(img[i])):
            #         pixel = img[i][j]
            #         pix_count += 1
            #         if (pixel[0] > b or pixel[2] > b) and pixel[1] < b * 2 \
            #                 or (pixel[0] > b*2 and pixel[1] > b*2 and pixel[2] > b*2):
            #             # img[i][j] = [0, 0, 0]
            #             count += 1
            # return 1 - (count / pix_count)

        left, middle_l, middle_r, right = np.hsplit(image, 4)
        middle = np.concatenate((middle_l, middle_r), axis=1)
        return get_red_pixels(left), get_red_pixels(middle), get_red_pixels(right)

    def get_reward(previous_state, new_state,
                   previous_sensor, new_sensor,
                   prev_action, action,
                   prev_val, new_val):

        # Max no. of green in img, before and after
        # 0: No green pixels in img;    1: All img consists of green pixels

        prev_right, prev_mid, prev_left = prev_val
        sum_prev_val = sum(prev_val)
        new_right, new_mid, new_left = new_val
        sum_new_val = sum(new_val)
        max_new_sensor = np.max(new_sensor)
        max_prev_sensor = np.max(previous_sensor)
        max_c_prev = np.max(previous_state[3:])
        max_c_new = np.max(new_state[3:])

        # Encourages going towards prey
        if max_c_prev == 0 and max_c_new == 1:
            return 10 if action == 0 else 2

        # Massive payoff if we get super close to prey
        if max_c_prev == 1 and max_c_new == 2:
            return 30

        # Nothing happens if prey gets a little away
        if max_c_prev == 2 and max_c_new == 1:
            return 0

        # A LOT happens when prey is not in sight
        if max_c_prev == 1 and max_c_new == 0:
            return -3

        # Give good reward if we see more red than before
        if sum_prev_val < sum_new_val:
            return 5 if action == 0 else 0

        # If sensors detect enemy, then give good payoff.
        # If sensors detect wall, give bad payoff to steer clear
        if max_new_sensor > max_prev_sensor:
            return 15 if max_c_new >= 1 else -3

        if (prev_action == 1 and action == 2)\
                or (prev_action == 2 and action == 1)\
                and max_new_sensor == 0:
            return -5

        # if prev_action != 0 or action != 0:
        #     return -5

        # Give good payoff to encourage exploring (going straight)
        # Minor bad payoff for turning around, but not bad enough to discourage it
        return 0 if action == 0 else -1

    # Returns list of values with discretized sensor values and color values.
    def make_discrete(values_s, boundary_s, values_c, boundaries_c):
        discrete_list_s = []
        discrete_list_c = []

        for x in values_s:
            if boundary_s[0] > x:
                discrete_list_s.append(0)
            elif boundary_s[1] > x > boundary_s[0]:
                discrete_list_s.append(1)
            else:
                discrete_list_s.append(2)
        for y in values_c:
            if y < boundaries_c[0]:
                discrete_list_c.append(0)
            elif boundaries_c[0] < y < boundaries_c[1]:
                discrete_list_c.append(1)
            else:
                discrete_list_c.append(2)
        print('real c_values: ', values_c)
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
    stat_fitness = list()
    stat_rewards = [0]

    def normalize(reward, old_min, old_max, new_min=-1, new_max=1):
        return ((reward - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    # def run_static(lim, no_blocks=0):
    #     for i in range(lim):
    #         if use_simulation:
    #             rob.play_simulation()
    #
    #         a, b, c = get_color_info()
    #         current_color_info = a, b, c
    #         current_sensor_info = get_sensor_info('front_3')
    #
    #         current_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, current_color_info,
    #                                       boundaries_color)
    #
    #         if str(current_state) not in state_table.keys():
    #             state_table[str(current_state)] = [0 for _ in range(3)]
    #
    #         a, b, c = get_color_info()
    #         new_color_info = a, b, c
    #         # print(a, b, c, new_color_info)
    #
    #         action = static_policy(new_color_info)
    #
    #         take_action(action)
    #
    #         new_state = make_discrete(get_sensor_info('front_3'), boundary_sensor, new_color_info,
    #                                   boundaries_color)
    #         # TODO: make sure that current color info gets initialized the first time.
    #         r = get_reward(current_state, new_state, action, current_color_info, new_color_info, no_blocks)
    #         if r == 20:
    #             no_blocks += 1
    #
    #         norm_r = normalize(r, -30, 20)
    #
    #         if i != 0:
    #             stat_fitness.append(stat_fitness[-1] + (no_blocks / i))
    #         else:
    #             stat_fitness.append(float(0))
    #         print(fitness)
    #         if stat_rewards:
    #             stat_rewards.append(stat_rewards[-1] + norm_r)
    #         else:
    #             rewards.append(norm_r)
    #
    #         current_state = new_state
    #         current_color_info = new_color_info

    def rl(alpha, gamma, epsilon, episodes, act_lim, qL=False):

        fitness = list()
        rewards = [0]

        for i in range(episodes):
            print('Episode ' + str(i))
            terminate = False
            if use_simulation:
                rob.play_simulation()
                prey_robot = robobo.SimulationRoboboPrey().connect(address=virtual_ip, port=19989)
                prey_controller = prey.Prey(robot=prey_robot, level=2)
                prey_controller.start()
            current_color_space = get_color_info()
            current_sensor_info = get_sensor_info('front_3')
            current_state = make_discrete(current_sensor_info, boundary_sensor, current_color_space,
                                          boundaries_color)

            if str(current_state) not in state_table.keys():
                state_table[str(current_state)] = [0 for _ in range(3)]

            action = epsilon_policy(current_state, epsilon)
            x = 0
            while not terminate:

                take_action(action)
                # new_collected_food = rob.collected_food() if use_simulation else 0

                # Whole img extracted to get reward value
                # left, mid, right extracted to save state space accordingly

                new_color_space = get_color_info()
                new_sensor_info = get_sensor_info('front_3')
                new_state = make_discrete(new_sensor_info, boundary_sensor, new_color_space,
                                          boundaries_color)

                if str(new_state) not in state_table.keys():
                    state_table[str(new_state)] = [0 for _ in range(3)]

                new_action = epsilon_policy(new_state, epsilon)

                # Retrieve the max action if we use Q-Learning
                max_action = np.argmax(state_table[str(new_state)]) if qL else new_action

                # Get reward
                r = get_reward(current_state, new_state,
                               current_sensor_info, new_sensor_info,
                               action, new_action,
                               current_color_space, new_color_space)
                print("State and obtained Reward: ", new_state, r)

                norm_r = normalize(r, MIN_REWARD, MAX_REWARD)
                max_sensor_val = np.max(new_sensor_info)
                fitness.append(norm_r * (0.1 + max_sensor_val))

                # Update rule
                if not run_test:
                    # print('update')
                    state_table[str(current_state)][action] += \
                        alpha * (r + (gamma *
                                      np.array(
                                          state_table[str(new_state)][max_action]))
                                 - np.array(state_table[str(current_state)][action]))

                # Stop episode if we get very close to an obstacle
                if (max(new_state[:3]) == 2 and max(new_state[3:]) != 2 and use_simulation) or x == act_lim - 1:
                    state_table[str(new_state)][new_action] = -10
                    terminate = True
                    print("done")
                    if not run_test:
                        print('writing json')
                        with open(q_table_file, 'w') as json_file:
                            json.dump(state_table, json_file)

                    if use_simulation:
                        print("stopping the simulation")
                        prey_controller.stop()
                        prey_controller.join()
                        prey_robot.disconnect()
                        rob.stop_world()
                        while not rob.is_sim_stopped():
                            print("waiting for the simulation to stop")
                        time.sleep(2)

                # update current state and action
                current_state = new_state
                current_sensor_info = new_sensor_info
                action = new_action
                current_color_space = new_color_space

                # increment action limit counter
                x += 1

        return fitness, rewards


    # epsilons = [0.01, 0.08, 0.22]
    # gammas = [0.9]
    # param_tuples = [(epsilon, gamma) for epsilon in epsilons for gamma in gammas]
    experiments = 1 if not run_test else 1
    actions = 200 if not run_test else 10000
    eps = 30 if not run_test else 1
    epsilons = [0.08]
    for epsilon in epsilons:

        for run in range(experiments):
            print('======= RUNNING FOR epsilon ', epsilon, ' , run ', run)
            fitness, rewards = rl(0.9, 0.9, epsilon, eps, actions,
                                  qL=True)  # alpha, gamma, epsilon, episodes, actions per episode
            if not run_test:
                file_name_rewards = './src/rewards_epsilon' + str(epsilon) + '_run' + str(run) + '.csv'
                with open(file_name_rewards, 'wb') as f:
                    pickle.dump(rewards, f)

                file_name_fitness = './src/fitness_epsilon' + str(epsilon) + '_run' + str(run) + '.csv'
                with open(file_name_fitness, 'wb') as f:
                    pickle.dump(fitness, f)


if __name__ == "__main__":
    main()
