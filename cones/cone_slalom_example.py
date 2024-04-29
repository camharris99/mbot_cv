#!/usr/bin/python3

import time
import lcm
import sys
from mbot_lcm_msgs.twist2D_t import twist2D_t
from mbot_lcm_msgs.path2D_t import path2D_t
from mbot_lcm_msgs.pose2D_t import pose2D_t
from mbot_lcm_msgs.mbot_cone_array_t import mbot_cone_array_t

# Moves forward for 5 seconds, then stops
# If nothing happens, try running "sudo systemctl stop mbot-motion-controller.service", then running this file again

class Slalom():
    def __init__(self):
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")
        self.path = path2D_t()
        self.nextPose = pose2D_t()
        self.cones = []
        self.currentCone = None
        self.nextCone = None
        cone_subscription = self.lc.subscribe("MBOT_CONE_ARRAY", self.find_cones)
        while (len(self.cones) == 0):
            self.lc.handle()
            print(len(self.cones))
        
    def find_cones(self, channel, data):
        msg = mbot_cone_array_t.decode(data)
        if msg.array_size == 0:
            print("No Detection")
        else:
            for detection in msg.detections:
                self.cones.append(detection)

my_slalom = Slalom()

# Edit these variables
# fwd_vel = 0.4
# turn_vel = 0.0
# move_time = 5

# command = twist2D_t() # A twist2D_t command encodes forward and rotational speeds of the bot
# command.vx = fwd_vel
# command.wz = turn_vel

# lc.publish("MBOT_VEL_CMD",command.encode())
# time.sleep(move_time)

# command.vx = 0
# command.wz = 0
# lc.publish("MBOT_VEL_CMD",command.encode())

# subscribe to message of apriltags
# sort them based on how close to the robot they are
# build the path
# assume first cone is red, start on right of it
# first part of path will be straight line, end halfway between first and second cone
# if next cone is same color as previous, go beyond it halfway between it and next next cone
# is next cone is opposite color, if it's green then go to the left, then go beyond it
                                # if it's red then go to the right, then go beyond it