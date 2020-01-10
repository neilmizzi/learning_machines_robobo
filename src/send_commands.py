#!/usr/bin/env python2
from __future__ import print_function

import time
import numpy as np

import robobo
import cv2
import sys
import signal
import prey


def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)


def main():
    signal.signal(signal.SIGINT, terminate_program)

    rob = robobo.HardwareRobobo(camera=True).connect(address="10.15.3.48")
    #rob = robobo.SimulationRobobo().connect(address='172.20.10.5', port=19997)

    # rob.play_simulation()

    def get_sensor_info(direction):
        all_sensor_info = np.array(rob.read_irs())
        back_sensors = all_sensor_info[range(0, 3)]
        front_sensors = all_sensor_info[range(3, 6)]
        if direction == 'front':
            return np.max(front_sensors)
        elif direction == 'back':
            return np.max(back_sensors)
        else:
            raise Exception('Invalid direction')

    def move_left():
        rob.move(-15, 15, 10)

    def move_right():
        rob.move(15, -15, 10)

    def go_straight():
        rob.move(10, 10, 10)

    def move_back():
        rob.move(-5, -5, 10)

    # Following code moves the robot
    for i in range(500):
        print(get_sensor_info('front'))
        if get_sensor_info('front') >= 30:
            move_back()
            move_left()
        else:
            go_straight()


    # print("robobo is at {}".format(rob.position()))
    rob.sleep(1)

    # Following code moves the phone stand
    # rob.set_phone_pan(343, 100)
    # rob.set_phone_tilt(109, 100)
    # time.sleep(1)
    # rob.set_phone_pan(11, 100)
    # rob.set_phone_tilt(26, 100)

    # Following code makes the robot talk and be emotional
    # rob.set_emotion('happy')
    # rob.talk('Hi, my name is Robobo')
    # rob.sleep(1)
    # rob.set_emotion('sleepy')

    # Following code gets an image from the camera
    image = rob.get_image_front()
    cv2.imwrite("../test_pictures.png", image)

    time.sleep(0.1)

    # IR reading
    for i in range(1000000):
        print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))
        time.sleep(0.1)

    # pause the simulation and read the collected food
    rob.pause_simulation()

    # Stopping the simulation resets the environment
    rob.stop_world()


if __name__ == "__main__":
    main()
