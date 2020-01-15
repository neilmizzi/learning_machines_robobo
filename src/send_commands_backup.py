#!/usr/bin/env python2
from __future__ import print_function
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import numpy as np
from numpy import inf, random

import json

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


def main(iterations):
    rob = robobo.SimulationRobobo().connect(address='192.168.1.2', port=19997) if use_simulation \
        else robobo.HardwareRobobo(camera=True).connect(address="192.168.43.187")

    for i in range(iterations):
        signal.signal(signal.SIGINT, terminate_program)
        print("simulation stopped: {}".format(rob.is_sim_stopped()))
        rob.play_simulation()
        print(rob.get_sim_time())
        rob.move(20, 20, 5000)
        print(rob.get_sim_time())
        time.sleep(2)
        print(rob.get_sim_time())

        # print("This is iteration {}, trying to stop".format(i))
        rob.stop_world()
        print("stopped")
        while not rob.is_sim_stopped():
            print("waiting for the simulation to stop")
        time.sleep(2)

                        # Stopping the simulation resets the environment

    # Following code gets an image from the camera
    # image = rob.get_image_front()
    # cv2.imwrite("../test_pictures.png", image)
    #
    # time.sleep(0.1)

    # IR reading

    # for i in range(1000000):
    #     print("ROB Irs: {}".format(np.log(np.array(rob.read_irs())) / 10))


if __name__ == "__main__":
    iterations = 3
    main(iterations)
