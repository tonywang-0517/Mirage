#!/usr/bin/env python3
# coding: utf-8

from easydict import EasyDict

Config = EasyDict({
    "control_delay": False,
    "randomize_body_mass": False,
    "randomize_joint_parameters": False,
    "randomize_body_com": False,
    "randomize_actuator_gains": False,
    "randomize_rfi_lim": False,
    "random_push_robot": False,
    "randomize_physics_scene_gravity": False,
})