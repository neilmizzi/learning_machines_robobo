#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np

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

    rob = robobo.SimulationRobobo().connect(address='192.168.43.172', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="192.168.43.187")

    if use_simulation:
        rob.play_simulation()

    def get_sensor_info(direction):
        all_sensor_info = np.array([0 if not x else x for x in np.array(rob.read_irs())]) if use_simulation \
            else np.array(rob.read_irs()) / 200
        print(all_sensor_info)
        back_sensors = all_sensor_info[range(0, 4)]
        front_sensors = all_sensor_info[range(4, 8)]
        if direction == 'front':
            return np.max(front_sensors)
        elif direction == 'back':
            return np.max(back_sensors)
        elif direction == 'front_left':
            return np.max(front_sensors[range(0, 2)])
        elif direction == 'front_right':
            return np.max(front_sensors[range(2, 4)])
        elif direction == 'back_left':
            return np.max(back_sensors[range(0, 2)])
        elif direction == 'back_right':
            return np.max(back_sensors[range(2, 4)])
        else:
            raise Exception('Invalid direction')

    def policy(s):  # To be replaced by Neural Network, this is static
        if get_sensor_info('front_left') <= s \
                and get_sensor_info('front_left') < get_sensor_info('front_right') \
                and get_sensor_info('front') != 0:
            return 'right'
        elif get_sensor_info('front_right') <= s \
                and get_sensor_info('front_right') < get_sensor_info('front_left') \
                and get_sensor_info('front') != 0:
            return 'left'
        else:
            return 'straight'

    def reward():
        pass

    def take_action(action):
        if action == 'left':
            move_left()
        elif action == 'right':
            move_right()
        elif action == 'straight':
            go_straight()

    def move_left():
        rob.move(-speed, speed, 500)

    def move_right():
        rob.move(speed, -speed, 500)

    def go_straight():
        rob.move(speed, speed, 500)

    def move_back():
        rob.move(-speed, -speed, 500)

    # Following code moves the robot
    for i in range(500000):
        # Q-Learning/SARSA goes here
        take_action(policy(0.19))

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
