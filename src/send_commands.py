#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np
from numpy import inf

import robobo
import cv2
import sys
import signal
import prey

use_simulation = True
speed = 5
dist = 500


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.SimulationRobobo().connect(address='130.37.120.225', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="192.168.1.6")

    if use_simulation:
        rob.play_simulation()

    def get_sensor_info(direction):
        #   TODO Fix Transformation
        a = np.log(np.array(rob.read_irs())) / 10
        all_sensor_info = np.array([0 if x == inf else 1 + (-x / 2) - 0.2 for x in a]) if use_simulation \
            else np.array(rob.read_irs()) / 200
        all_sensor_info[all_sensor_info == inf] = 0
        #print(format(all_sensor_info))
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
        else:
            raise Exception('Invalid direction')

    def static_policy(s):  # To be replaced by Neural Network, this is static
        print('left: {:f}, right: {:f}'.format(get_sensor_info('front_left'),
              get_sensor_info('front_right')))
        if get_sensor_info('front_left') >= s \
                and get_sensor_info('front_left') > get_sensor_info('front_right'):
            take_action('right')

        elif get_sensor_info('front_right') >= s \
                and get_sensor_info('front_right') > get_sensor_info('front_left'):
            take_action('left')
        else:
            take_action('straight')

    def reward():
        pass

    def take_action(action):
        if action == 'left':
            move_left()
        elif action == 'right':
            move_right()
        elif action == 'straight':
            go_straight()
        elif action == 'back':
            move_back()

    def move_left():
        rob.move(-speed, speed, dist)

    def move_right():
        rob.move(speed, -speed, dist)

    def go_straight():
        rob.move(speed, speed, dist)

    def move_back():
        rob.move(-speed, -speed, dist)

    # TODO Replace with SARSA
    # Following code moves the robot
    for i in range(500000):
        # Q-Learning/SARSA goes here
        take_action(static_policy(0.75))

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("../test_pictures.png", image)
    #
    # time.sleep(0.1)

    # IR reading

    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
    #     time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simulation resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
